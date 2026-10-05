"""
crypto/lattice_conv.py

Pezzi di PRODUZIONE dello schema a UN CIPHERTEXT PER CANALE con reticolo
(vedi prototype_lattice_resample.py per la logica, test_lattice_resample_zeus.py
e test_stage_levels_lattice_zeus.py per le verifiche in HE vero).

Differenza rispetto a LatticeHE.conv3x3 dei test precedenti: quella versione
tiene vive TUTTE le rotazioni dei canali d'ingresso insieme (Cin x 9
ciphertext): va bene con 3-6 canali, con 256 sarebbero 2.304 ciphertext
(~200 GB). Qui si usa la forma misurata nel benchmark (bench_hoisted_zeus.py,
variante BLOCCO-8 con cache ciphertext 6 GiB: 5.4 ms per termine):

  per ogni blocco di B canali d'uscita:
      per ogni canale d'ingresso ci:
          ruota x[ci] nelle 9 posizioni del kernel (8 rotazioni, 9 copie vive)
          per ognuno dei B canali d'uscita: 9 prodotti per scalare + somme
          parcheggia x[ci] su RAM host (Offload), libera la memoria GPU
      bias + maschera, parcheggia le B uscite su RAM host

Working set sulla GPU: 9 copie ruotate + B accumulatori + 1 ingresso, a
prescindere da Cin e Cout. Costo extra: le rotazioni si rifanno Cout/B volte
e gli ingressi si ricaricano da host a ogni blocco (~12 ms l'uno).

Le operazioni Offload() / ricarica automatica sono quelle gia' usate e
verificate nel benchmark OFFLOAD.
"""

import os
import time
import numpy as np

# MASK_LAUNDER=1: dopo ogni EvalMult(ct, maschera) si fa EvalAdd(., 0.0) e si tiene solo il risultato. Ipotesi (H2, da
# confermare con probe_mask_leak_zeus.py): il risultato di una moltiplicazione per plaintext si porta dietro una copia
# del plaintext (~64 MiB) che Offload() non libera; il ciphertext 'lavato' non la porta. Costo: un'addizione, 0 livelli.
LAUNDER = os.environ.get('MASK_LAUNDER', '0') == '1'


def level_offsets(g, level):
    s = 1 << level
    return [(ky - 1) * s * g.Wg + (kx - 1) * s for ky in range(3) for kx in range(3)]


def sum_all_slots(cc, ct, N):
    """Somma di tutti gli N slot in OGNI slot: log2(N) passi rotazione+somma.
    Corretta perche' il batch CKKS e' esattamente N (il giro si chiude su N).
    Gli slot fuori dal reticolo valido sono zero (mascherati): non contribuiscono."""
    s = 1
    while s < N:
        ct = cc.EvalAdd(ct, cc.EvalRotate(ct, s))
        s *= 2
    return ct


class LHE:
    def __init__(self, cc, keys, g):
        self.cc, self.keys, self.g = cc, keys, g
        self._mask = {}

    # ---------- utilita' ----------
    def pt(self, vec):
        return self.cc.MakeCKKSPackedPlaintext([float(v) for v in vec])

    def enc(self, vec):
        return self.cc.Encrypt(self.keys.publicKey, self.pt(vec))

    def dec(self, ct):
        p = self.cc.Decrypt(self.keys.secretKey, ct)
        p.SetLength(self.g.N)
        return np.array(p.GetRealPackedValue())

    def mask_pt(self, level):
        if level not in self._mask:
            self._mask[level] = self.pt(self.g.valid_mask(level))
        return self._mask[level]

    def mask_mult(self, ct, mask):
        o = self.cc.EvalMult(ct, mask)
        if LAUNDER:
            o2 = self.cc.EvalAdd(o, 0.0)
            del o
            return o2
        return o

    def enc_offloaded(self, vecs, log=None):
        """Cifra una lista di vettori e parcheggia subito ogni ciphertext su host."""
        out = []
        for i, v in enumerate(vecs):
            ct = self.enc(v)
            ct.Offload()
            out.append(ct)
            if i % 8 == 7:
                self.cc.TrimGPUMemoryPool()
            if log and (i + 1) % 32 == 0:
                log(f"    cifrati {i + 1}/{len(vecs)}")
        self.cc.TrimGPUMemoryPool()
        return out

    # ---------- convoluzione a blocchi ----------
    def conv3x3_blocked(self, chans, w, b, level, stride2, B=8, offload_inputs=True):
        cc, g = self.cc, self.g
        offs = level_offsets(g, level)
        out_level = level + (1 if stride2 else 0)
        mask = self.mask_pt(out_level)
        Cin, Cout = len(chans), w.shape[0]
        outs = [None] * Cout
        for b0 in range(0, Cout, B):
            cos = list(range(b0, min(b0 + B, Cout)))
            acc = {co: None for co in cos}
            for ci in range(Cin):
                x = chans[ci]
                rot = [x if d == 0 else cc.EvalRotate(x, d) for d in offs]
                for co in cos:
                    a = acc[co]
                    for k in range(9):
                        t = cc.EvalMult(rot[k], float(w[co, ci, k // 3, k % 3]))
                        a = t if a is None else cc.EvalAdd(a, t)
                    acc[co] = a
                del rot
                if offload_inputs:
                    x.Offload()
                cc.TrimGPUMemoryPool()
            for co in cos:
                o = self.mask_mult(cc.EvalAdd(acc[co], float(b[co])), mask)
                o.Offload()
                outs[co] = o
            del acc
            cc.TrimGPUMemoryPool()
        return outs

    # ---------- norm + attivazione di una lista di canali gia' convoluti ----------
    def norm_act(self, conv, gamma, beta, out_level, coeffs, domain, act=(0.1, 1.0, 0.5), progress=None):
        """InstanceNorm (statistiche per canale, radice inversa Chebyshev+Newton) -> PolyAct -> maschera,
        un canale alla volta, con uscite e ingressi parcheggiati su host. progress(c) e' chiamata dopo ogni canale."""
        from crypto.fhe_ops_single_ciphertext import poly_act_fhe, isqrt_chebyshev_fhe
        cc, g = self.cc, self.g
        N = g.N
        Hh, Ww = g.size(out_level)
        n_valid = Hh * Ww
        mask = self.mask_pt(out_level)
        outs = []
        max_lvl = 0
        for c, x in enumerate(conv):
            total = sum_all_slots(cc, x, N)
            mean = cc.EvalMult(total, 1.0 / n_valid)
            total2 = sum_all_slots(cc, cc.EvalMult(x, x), N)
            mean_sq = cc.EvalMult(total2, 1.0 / n_valid)
            var = cc.EvalSub(mean_sq, cc.EvalMult(mean, mean))
            inv = isqrt_chebyshev_fhe(cc, var, coeffs, list(domain), post_iter=1)
            normalized = cc.EvalMult(cc.EvalSub(x, mean), inv)
            scaled = cc.EvalAdd(cc.EvalMult(normalized, float(gamma[c])), float(beta[c]))
            out = self.mask_mult(poly_act_fhe(cc, scaled, *act), mask)
            max_lvl = max(max_lvl, out.GetLevel())
            out.Offload()
            x.Offload()
            outs.append(out)
            if c % 4 == 3:
                cc.TrimGPUMemoryPool()
            if progress is not None:
                progress(c + 1)
        cc.TrimGPUMemoryPool()
        return outs, max_lvl

    # ---------- stadio completo: conv -> InstanceNorm -> PolyAct -> maschera ----------
    def stage(self, chans, w, b, gamma, beta, level, stride2, coeffs, domain,
              B=8, act=(0.1, 1.0, 0.5)):
        out_level = level + (1 if stride2 else 0)
        t0 = time.time()
        conv = self.conv3x3_blocked(chans, w, b, level, stride2, B=B)
        t_conv = time.time() - t0
        t0 = time.time()
        outs, max_lvl = self.norm_act(conv, gamma, beta, out_level, coeffs, domain, act)
        return outs, dict(t_conv=t_conv, t_norm=time.time() - t0, level=max_lvl)