"""
crypto/test_lattice_resample_zeus.py

Traduzione in HE VERO del prototipo prototype_lattice_resample.py (che
coincide con i riferimenti densi a precisione macchina): stride 2,
ConvTranspose2d(k=2,s=2), skip connection, conv 1x1 e una mini U-Net a 3
livelli, nello schema a UN CIPHERTEXT PER CANALE con pixel a "reticolo".

Caso piccolo: immagine 16x12, 3 livelli (passi 1,2,4). Wg = 14, Hg = 18,
N = 256 slot. Nessun bootstrap, nessuna normalizzazione (le statistiche sul
reticolo sono una somma su TUTTI gli slot del vettore mascherato: stessa
funzione gia' usata, verificata a parte in numpy).

Cosa si verifica che in numpy non si poteva verificare:
  - le rotazioni NEGATIVE e i livelli (passi) diversi con le chiavi vere;
  - che il vettore resti "zero fuori dal reticolo" con il rumore CKKS
    (l'upsampling NON usa maschere: si misura quanto valgono davvero gli
    slot fuori dal reticolo);
  - l'allineamento dei livelli nella somma della skip (EvalAdd tra
    ciphertext a livelli diversi);
  - il costo in livelli di uno strato conv + maschera + PolyAct.

REFRESH SIMULATO: senza bootstrap, 11 strati non entrano in 43 livelli.
Quando il livello supera una soglia, il test DECIFRA E RICIFRA (con la
chiave segreta). Non e' un bootstrap e la sua precisione non dice nulla sul
bootstrap vero; serve solo a non uscire dalla profondita'. Il numero di
refresh viene stampato.

Uso (sul repo, con prototype_lattice_resample.py nella stessa cartella):
  python3 crypto/test_lattice_resample_zeus.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import poly_act_fhe
from prototype_lattice_resample import Grid, conv_ref, upconv_ref, act_ref

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
H0, W0, L = 16, 12, 3
TOL = 1e-6
REFRESH_AT = 28


def level_offsets(g, level):
    s = 1 << level
    return [(ky - 1) * s * g.Wg + (kx - 1) * s for ky in range(3) for kx in range(3)]


def required_rotations(g):
    s = set()
    for lvl in range(L):
        s |= set(level_offsets(g, lvl))
    return sorted(r for r in s if r != 0)


def build_context(g, rot_list):
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(g.N)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetDevices([0])
    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, rot_list)
    cc.SetRotationKeyCache(4 * GiB)     # 8 chiavi per livello: entrano (lezione del benchmark)
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s", flush=True)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(6 * GiB)
    return cc, keys


class LatticeHE:
    def __init__(self, cc, keys, g):
        self.cc, self.keys, self.g = cc, keys, g
        self._mask = {}
        self.refreshes = 0
        self.max_level = 0
        self.offlattice_noise = None

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

    def rot(self, ct, d):
        return ct if d == 0 else self.cc.EvalRotate(ct, d)

    def lvl(self, chans):
        m = max(c.GetLevel() for c in chans)
        self.max_level = max(self.max_level, m)
        return m

    def maybe_refresh(self, chans):
        if self.lvl(chans) >= REFRESH_AT:
            self.refreshes += 1
            return [self.enc(self.dec(c)) for c in chans]
        return chans

    def unpack_all(self, chans, level):
        return np.stack([self.g.unpack(self.dec(c), level) for c in chans])

    # ---------- operazioni ----------
    def conv3x3(self, chans, w, b, level, stride2):
        cc, g = self.cc, self.g
        offs = level_offsets(g, level)
        rotated = [[self.rot(x, d) for d in offs] for x in chans]
        out_level = level + (1 if stride2 else 0)
        mask = self.mask_pt(out_level)
        outs = []
        for co in range(w.shape[0]):
            acc = None
            for ci in range(len(chans)):
                for k in range(9):
                    term = cc.EvalMult(rotated[ci][k], float(w[co, ci, k // 3, k % 3]))
                    acc = term if acc is None else cc.EvalAdd(acc, term)
            outs.append(cc.EvalMult(cc.EvalAdd(acc, float(b[co])), mask))
        cc.TrimGPUMemoryPool()
        return outs

    def upconv2x2(self, chans, w, b, level_in, probe=False):
        cc, g = self.cc, self.g
        lo = level_in - 1
        s = 1 << lo
        shifts = {(a, c): -(s * a * g.Wg + s * c) for a in (0, 1) for c in (0, 1)}
        rotated = [{ab: self.rot(x, d) for ab, d in shifts.items()} for x in chans]
        maskout = g.valid_mask(lo)
        outs = []
        for co in range(w.shape[1]):
            acc = None
            for ci in range(len(chans)):
                for (a, c) in shifts:
                    term = cc.EvalMult(rotated[ci][(a, c)], float(w[ci, co, a, c]))
                    acc = term if acc is None else cc.EvalAdd(acc, term)
            if probe and co == 0:
                v = self.dec(acc)
                self.offlattice_noise = float(np.abs(v * (1.0 - maskout)).max())
            outs.append(cc.EvalAdd(acc, self.pt(b[co] * maskout)))     # bias come plaintext: nessuna maschera ct
        cc.TrimGPUMemoryPool()
        return outs

    def conv1x1(self, chans, w, b):
        cc = self.cc
        outs = []
        for k in range(w.shape[0]):
            acc = None
            for ci in range(len(chans)):
                term = cc.EvalMult(chans[ci], float(w[k, ci]))
                acc = term if acc is None else cc.EvalAdd(acc, term)
            outs.append(cc.EvalAdd(acc, float(b[k])))
        return outs

    def polyact_masked(self, chans, level):
        cc = self.cc
        mask = self.mask_pt(level)
        outs = [cc.EvalMult(poly_act_fhe(cc, x, 0.1, 1.0, 0.5), mask) for x in chans]
        cc.TrimGPUMemoryPool()
        return outs

    def skip_add(self, a, b):
        return [self.cc.EvalAdd(x, y) for x, y in zip(a, b)]


def main():
    g = Grid(H0, W0, L)
    print(f"=== Operazioni sul reticolo in HE vero ===")
    print(f"Immagine {H0}x{W0}, Wg={g.Wg}, Hg={g.Hg}, {L} livelli, N={g.N} slot (batch CKKS)\n")
    assert g.N & (g.N - 1) == 0, "il batch CKKS deve essere una potenza di 2"

    rots = required_rotations(g)
    print(f"Costruzione contesto ({len(rots)} chiavi di rotazione: 8 per livello)...")
    t0 = time.time()
    cc, keys = build_context(g, rots)
    he = LatticeHE(cc, keys, g)
    print(f"Contesto pronto in {time.time()-t0:.1f}s.\n", flush=True)

    rng = np.random.default_rng(31)
    results = []

    def report(name, err, extra=""):
        ok = err < TOL
        results.append(ok)
        print(f"  {name:<56} errore {err:9.2e}  {'OK' if ok else 'ERRORE'}  {extra}", flush=True)

    def rand_in(C, level):
        Hl, Wl = g.size(level)
        imgs = rng.normal(size=(C, Hl, Wl)) * 0.5
        return imgs, [he.enc(g.pack(imgs[c], level)) for c in range(C)]

    print("=== Singole operazioni ===", flush=True)
    for level in range(L):
        imgs, ch = rand_in(3, level)
        w = rng.normal(size=(4, 3, 3, 3)) * 0.2
        b = rng.normal(size=4) * 0.1
        t0 = time.time()
        out = he.conv3x3(ch, w, b, level, stride2=False)
        err = np.abs(he.unpack_all(out, level) - conv_ref(imgs, w, b, 1)).max()
        report(f"conv 3x3 stride 1, livello {level} (passo {1 << level})", err,
               f"[{time.time()-t0:.1f}s, livello ct {he.lvl(out)}]")

    for level in range(L - 1):
        imgs, ch = rand_in(3, level)
        w = rng.normal(size=(4, 3, 3, 3)) * 0.2
        b = rng.normal(size=4) * 0.1
        out = he.conv3x3(ch, w, b, level, stride2=True)
        err = np.abs(he.unpack_all(out, level + 1) - conv_ref(imgs, w, b, 2)).max()
        report(f"conv 3x3 STRIDE 2, livello {level} -> {level + 1}", err, f"[livello ct {he.lvl(out)}]")

    for level_in in range(1, L):
        imgs, ch = rand_in(4, level_in)
        w = rng.normal(size=(4, 3, 2, 2)) * 0.3
        b = rng.normal(size=3) * 0.1
        out = he.upconv2x2(ch, w, b, level_in, probe=True)
        err = np.abs(he.unpack_all(out, level_in - 1) - upconv_ref(imgs, w, b)).max()
        report(f"ConvTranspose k2 s2, livello {level_in} -> {level_in - 1}", err,
               f"[slot fuori reticolo, senza maschera: |max| = {he.offlattice_noise:.1e}]")

    imgs, ch = rand_in(6, 0)
    w1 = rng.normal(size=(2, 6)) * 0.3
    b1 = rng.normal(size=2) * 0.1
    out = he.conv1x1(ch, w1, b1)
    ref = np.einsum('kc,chw->khw', w1, imgs) + b1[:, None, None]
    report("conv 1x1 (6 -> 2 canali)", np.abs(he.unpack_all(out, 0) - ref).max())

    print("\n=== Mini U-Net a 3 livelli (conv-PolyAct-maschera, stride, up + skip, 1x1) ===", flush=True)
    C0, C1, C2, NCLS = 3, 4, 5, 2
    x_img = rng.normal(size=(1, H0, W0))
    Wt = {
        'e0a': rng.normal(size=(C0, 1, 3, 3)) * 0.3, 'e0b': rng.normal(size=(C0, C0, 3, 3)) * 0.2,
        'e1a': rng.normal(size=(C1, C0, 3, 3)) * 0.2, 'e1b': rng.normal(size=(C1, C1, 3, 3)) * 0.2,
        'e2a': rng.normal(size=(C2, C1, 3, 3)) * 0.2, 'e2b': rng.normal(size=(C2, C2, 3, 3)) * 0.2,
        'u1': rng.normal(size=(C2, C1, 2, 2)) * 0.3, 'd1a': rng.normal(size=(C1, C1, 3, 3)) * 0.2,
        'u0': rng.normal(size=(C1, C0, 2, 2)) * 0.3, 'd0a': rng.normal(size=(C0, C0, 3, 3)) * 0.2,
        'out': rng.normal(size=(NCLS, C0)) * 0.3,
    }
    Bt = {k: rng.normal(size=(v.shape[1] if k in ('u1', 'u0') else v.shape[0])) * 0.05 for k, v in Wt.items()}

    r = act_ref(conv_ref(x_img, Wt['e0a'], Bt['e0a']))
    r = act_ref(conv_ref(r, Wt['e0b'], Bt['e0b'])); s0 = r
    r = act_ref(conv_ref(r, Wt['e1a'], Bt['e1a'], stride=2))
    r = act_ref(conv_ref(r, Wt['e1b'], Bt['e1b'])); s1 = r
    r = act_ref(conv_ref(r, Wt['e2a'], Bt['e2a'], stride=2))
    r = act_ref(conv_ref(r, Wt['e2b'], Bt['e2b']))
    r = upconv_ref(r, Wt['u1'], Bt['u1']) + s1
    r = act_ref(conv_ref(r, Wt['d1a'], Bt['d1a']))
    r = upconv_ref(r, Wt['u0'], Bt['u0']) + s0
    r = act_ref(conv_ref(r, Wt['d0a'], Bt['d0a']))
    ref_logits = np.einsum('kc,chw->khw', Wt['out'], r) + Bt['out'][:, None, None]

    he.refreshes = 0
    he.max_level = 0
    t0 = time.time()

    def layer(name, chans, wk, level, stride2=False, out_level=None):
        ol = out_level if out_level is not None else level + (1 if stride2 else 0)
        o = he.polyact_masked(he.conv3x3(chans, Wt[wk], Bt[wk], level, stride2), ol)
        print(f"    {name:<26} livello ct {he.lvl(o):2d}", flush=True)
        return he.maybe_refresh(o)

    x = [he.enc(g.pack(x_img[0], 0))]
    x = layer("enc0 conv a", x, 'e0a', 0)
    x = layer("enc0 conv b  -> skip0", x, 'e0b', 0); skip0 = x
    x = layer("enc1 conv a (stride 2)", x, 'e1a', 0, stride2=True)
    x = layer("enc1 conv b  -> skip1", x, 'e1b', 1); skip1 = x
    x = layer("enc2 conv a (stride 2)", x, 'e2a', 1, stride2=True)
    x = layer("enc2 conv b", x, 'e2b', 2)
    x = he.skip_add(he.upconv2x2(x, Wt['u1'], Bt['u1'], 2), skip1)
    print(f"    {'up1 + skip1':<26} livello ct {he.lvl(x):2d}", flush=True)
    x = he.maybe_refresh(x)
    x = layer("dec1 conv", x, 'd1a', 1)
    x = he.skip_add(he.upconv2x2(x, Wt['u0'], Bt['u0'], 1), skip0)
    print(f"    {'up0 + skip0':<26} livello ct {he.lvl(x):2d}", flush=True)
    x = he.maybe_refresh(x)
    x = layer("dec0 conv", x, 'd0a', 0)
    logits = he.unpack_all(he.conv1x1(x, Wt['out'], Bt['out']), 0)
    err = np.abs(logits - ref_logits).max()
    report("mini U-Net completa (logits vs riferimento denso)", err,
           f"[{time.time()-t0:.0f}s, {he.refreshes} refresh simulati, livello max {he.max_level}]")

    print("\n=== RIEPILOGO ===")
    if all(results):
        print("TUTTO COINCIDE in HE vero (a meno dei refresh simulati nella mini U-Net).")
    else:
        print("ATTENZIONE: qualche operazione NON coincide: guarda le righe ERRORE sopra.")
    print(f"Livelli per strato conv+maschera+PolyAct (senza norm): vedi la colonna 'livello ct' sopra;")
    print(f"con la normalizzazione (circa 20 livelli per stadio, misurati) lo strato vero costa molto di piu'.")


if __name__ == '__main__':
    main()