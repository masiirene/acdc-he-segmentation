"""
crypto/probe_norm_leak_zeus.py

DAI DATI DELLA SONDA PRECEDENTE (probe_gpu_memory_zeus.py, due giri):
  - parcheggiare e rileggere ciphertext: nessuna crescita
  - blocchi di convoluzione: si assesta (la cache si riempie e poi e' piatta)
  - norm + attivazione su 128 canali: la memoria cresce di circa 1 GiB ogni 16 canali
    (~64 MiB per canale), in modo identico con cache da 6+6 GiB e da 4+5 GiB: NON dipende
    dalle cache. Un ciphertext a livello ~11 pesa ~66 MiB: sospetto UN ciphertext per canale
    che non viene liberato (es. il risultato della radice inversa).
  - a 256 canali sarebbero +16 GB; su tutta la rete (~2.500 canali-norm per immagine) la memoria
    non basterebbe MAI. Va capito e risolto prima di qualunque run lungo.

QUESTA SONDA isola il passo. Per ogni variante elabora N canali (default 48) e ripete la catena
 fino a un certo punto, scartando il risultato (o parcheggiandolo, nelle ultime due):
   mult   EvalMult(x, 0.5)                                     controllo, nessuna crescita attesa
   sums   somme su tutti gli slot (16 rotazioni x2) + medie
   var    + varianza
   inv    + radice inversa (Chebyshev + 1 Newton)
   norm   + (x - media) * inv_std, gamma/beta
   act    + PolyAct
   out    + maschera finale
   out+Offload   come sopra ma l'uscita viene parcheggiata e CONSERVATA (come in norm_act)
   inv+sync      come 'inv' ma con una decifrazione (sincronizzazione) ogni 4 canali
Si misura la memoria dopo ogni 16 canali e la crescita per canale (esclusi i primi 16, di riscaldamento).

LETTURA: la PRIMA variante con crescita > 10 MiB/canale contiene il passo colpevole. Se 'inv+sync'
cresce meno di 'inv', e' un problema di liberazione asincrona (si risolve sincronizzando di tanto in tanto).

Uso (in tmux, GPU libera):   python3 crypto/probe_norm_leak_zeus.py
Durata: ~1.5 minuti di contesto + ~3 minuti di varianti.
"""

import os
import sys
import gc
import time
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fideslib_py as fhe
from prototype_lattice_resample import Grid
from lattice_conv import LHE, sum_all_slots
from test_stage_bootstrap_fullres_zeus import build_context

H0, W0 = 256, 224
LEVEL = 2
N_CH = 48
COEFFS, DOM = [1.2, -0.5, 0.15], [0.2, 1.3]


def used_mib():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                      text=True).strip()
        return int(out.split()[0])
    except Exception:
        return -1


def make_chain(he, n_valid):
    from crypto.fhe_ops_single_ciphertext import poly_act_fhe, isqrt_chebyshev_fhe
    cc, N = he.cc, he.g.N
    mask = he.mask_pt(LEVEL)

    def chain(x, upto):
        if upto == "mult":
            return cc.EvalMult(x, 0.5)
        total = sum_all_slots(cc, x, N)
        mean = cc.EvalMult(total, 1.0 / n_valid)
        total2 = sum_all_slots(cc, cc.EvalMult(x, x), N)
        mean_sq = cc.EvalMult(total2, 1.0 / n_valid)
        if upto == "sums":
            return mean_sq
        var = cc.EvalSub(mean_sq, cc.EvalMult(mean, mean))
        if upto == "var":
            return var
        inv = isqrt_chebyshev_fhe(cc, var, COEFFS, DOM, post_iter=1)
        if upto == "inv":
            return inv
        scaled = cc.EvalAdd(cc.EvalMult(cc.EvalMult(cc.EvalSub(x, mean), inv), 1.0), 0.0)
        if upto == "norm":
            return scaled
        act = poly_act_fhe(cc, scaled, 0.1, 1.0, 0.5)
        if upto == "act":
            return act
        return cc.EvalMult(act, mask)
    return chain


def run(cc, keys, g, n_ch=N_CH, log=print, mem=used_mib):
    he = LHE(cc, keys, g)
    n_valid = g.size(LEVEL)[0] * g.size(LEVEL)[1]
    rng = np.random.default_rng(7)
    base = he.enc_offloaded([g.pack(rng.normal(size=g.size(LEVEL)) * 0.5, LEVEL) for _ in range(4)])
    chain = make_chain(he, n_valid)
    # riscaldamento: carica chiavi di rotazione, plaintext e precalcolati (non vanno contati come crescita)
    r = chain(base[0], "out")
    he.dec(r)
    del r
    cc.TrimGPUMemoryPool()
    gc.collect()
    log(f"\n  memoria di partenza dopo il riscaldamento: {mem()} MiB\n")

    variants = [("mult", "mult", False, False), ("sums", "sums", False, False), ("var", "var", False, False),
                ("inv", "inv", False, False), ("norm", "norm", False, False), ("act", "act", False, False),
                ("out", "out", False, False), ("out+Offload", "out", True, False), ("inv+sync", "inv", False, True)]
    results = []
    for label, upto, keep, sync in variants:
        cc.TrimGPUMemoryPool()
        gc.collect()
        kept = []
        pts = []
        t0 = time.time()
        for i in range(n_ch):
            r = chain(base[i % 4], upto)
            if keep:
                r.Offload()
                kept.append(r)
            elif sync and i % 4 == 3:
                he.dec(r)
            del r
            cc.TrimGPUMemoryPool()
            if (i + 1) % 16 == 0:
                pts.append(mem())
        grow = (pts[-1] - pts[0]) / max(n_ch - 16, 1)
        results.append((label, pts, grow))
        log(f"  {label:<13} MiB dopo 16/32/48 canali: {' / '.join(str(p) for p in pts)}   crescita {grow:6.1f} MiB/canale   ({time.time() - t0:.0f}s)")
        del kept
        cc.TrimGPUMemoryPool()
        gc.collect()

    log("\n=== RIEPILOGO (crescita di memoria GPU per canale, esclusi i primi 16) ===")
    first = None
    for label, pts, grow in results:
        flag = ""
        if grow > 10 and first is None and label not in ("inv+sync", "out+Offload"):
            first = label
            flag = "   <-- PRIMO PASSO CHE CRESCE"
        log(f"  {label:<13}{grow:8.1f} MiB/canale{flag}")
    g_inv = dict((l, gr) for l, _, gr in results)
    log("\n  LETTURA:")
    if first is None:
        log("   - nessuna variante cresce: la crescita osservata nella sonda precedente nasce altrove "
            "(forse nel parcheggio delle uscite o nell'interazione con la convoluzione).")
    else:
        log(f"   - la crescita compare gia' in '{first}' ({g_inv[first]:.0f} MiB/canale): il passo colpevole e' quello "
            f"aggiunto rispetto alla variante precedente.")
    if g_inv.get("inv+sync") is not None and g_inv.get("inv") is not None and g_inv["inv"] > 10:
        if g_inv["inv+sync"] < 0.3 * g_inv["inv"]:
            log("   - con sincronizzazione ogni 4 canali la crescita cala molto: e' liberazione ASINCRONA; "
                "basta sincronizzare di tanto in tanto (una decifrazione o una barriera).")
        else:
            log("   - la sincronizzazione NON cambia nulla: non e' liberazione asincrona, il ciphertext resta davvero allocato.")
    return results


def main():
    g = Grid(H0, W0, LEVEL + 1)
    print(f"=== SONDA PERDITA NORM: {N_CH} canali per variante, cache ciphertext {os.environ.get('CT_CACHE_GIB', '6')} GiB, "
          f"rotazioni {os.environ.get('ROT_CACHE_GIB', '6')} GiB ===")
    cc, keys = build_context(g, LEVEL)
    run(cc, keys, g)


if __name__ == '__main__':
    main()