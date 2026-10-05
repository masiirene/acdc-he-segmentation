"""
crypto/test_bootstrap_outlier_zeus.py

DOMANDA: nelle reti allenate senza clamp i valori tipici delle attivazioni sono
piccoli (|x| ~ 1) ma in training si sono visti massimi di 340-600 (max_raw).
Se un canale ha QUALCHE slot enorme, il bootstrap ha lo stesso errore (~3e-3)
sugli slot tipici dello stesso ciphertext, oppure l'outlier "avvelena" tutto il
canale?

Il test di ampiezza (test_bootstrap_amplitude_zeus.py) non puo' rispondere: li'
tutti i dati scalavano insieme con A. Qui:
  - dati TIPICI: N(0, 0.5) troncati a +-2, su tutti gli slot del reticolo
  - K outlier: K slot casuali messi a +-A
  - un bootstrap (e tre di fila, con moltiplicazione per 0.999 come negli altri test)
  - errore misurato SEPARATAMENTE sugli slot tipici e sugli slot outlier

LETTURA:
  - errore sugli slot tipici ~ costante (~3e-3) qualunque A: il bootstrap ha errore
    assoluto per slot, i pochi valori grandi non danneggiano gli altri: contenere il
    massimo non e' necessario per la precisione;
  - errore sugli slot tipici che CRESCE con A: l'outlier avvelena tutto il ciphertext:
    il massimo delle attivazioni va tenuto basso (penalita' piu' stretta in training),
    perche' senza clamp non c'e' modo di tagliarlo sotto cifratura.

Uso (in tmux, GPU libera, DOPO le sonde di memoria):  python3 crypto/test_bootstrap_outlier_zeus.py
Durata: ~3 minuti piu' il setup (~1.5 minuti).
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_argv = sys.argv
sys.argv = sys.argv[:1]                       # l'altro modulo legge sys.argv all'import
from test_bootstrap_amplitude_zeus import make_context, gpu_mem, BATCH, REAL_LEN
sys.argv = _argv

CASES = [("nessun outlier", 0, 0.0), ("20 outlier a 30", 20, 30.0), ("20 outlier a 100", 20, 100.0),
         ("20 outlier a 300", 20, 300.0), ("20 outlier a 1000", 20, 1000.0), ("1 outlier a 1000", 1, 1000.0)]
K_ITERS = 3


def run(cc, keys, cases=CASES, k_iters=K_ITERS, log=print):
    rng = np.random.default_rng(29)

    def dec_full(ct):
        p = cc.Decrypt(keys.secretKey, ct)
        p.SetLength(BATCH)
        return np.array(p.GetRealPackedValue())

    rows = []
    for name, K, A in cases:
        data = np.zeros(BATCH)
        data[:REAL_LEN] = np.clip(rng.normal(size=REAL_LEN) * 0.5, -2, 2)
        out_idx = rng.choice(REAL_LEN, size=K, replace=False) if K else np.array([], dtype=int)
        if K:
            data[out_idx] = A * rng.choice([-1.0, 1.0], size=K)
        typical = np.ones(BATCH, dtype=bool)
        typical[REAL_LEN:] = False
        typical[out_idx] = False
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([float(v) for v in data]))
        ref = data.copy()
        res = {"name": name}
        for k in range(1, k_iters + 1):
            pre = cc.EvalMult(ct, 0.999)
            new = cc.EvalBootstrap(pre)
            ct.Offload()
            pre.Offload()
            cc.TrimGPUMemoryPool()
            ct = new
            ref = ref * 0.999
            if k == 1 or k == k_iters:
                err = np.abs(dec_full(ct) - ref)
                et = err[typical]
                eo = err[out_idx] if K else np.array([0.0])
                res[k] = (et.max(), et.mean(), eo.max())
        rows.append(res)
        log(f"  {name:<20} typical dopo 1 bootstrap: max {res[1][0]:.2e} medio {res[1][1]:.2e} | "
            f"dopo {k_iters}: max {res[k_iters][0]:.2e} medio {res[k_iters][1]:.2e} | errore sugli outlier (1 boot) {res[1][2]:.2e}")
        del ct
        cc.TrimGPUMemoryPool()
    return rows


def summarize(rows, k_iters=K_ITERS, log=print):
    base = rows[0][1]
    log("\n=== RIEPILOGO (errore sugli slot TIPICI, dopo 1 bootstrap) ===")
    log(f"  {'caso':<20}{'max':>11}{'medio':>11}{'max / senza outlier':>22}{'medio / senza outlier':>24}")
    worst_mean = 1.0
    for r in rows:
        e = r[1]
        rm, rmean = e[0] / base[0], e[1] / base[1]
        worst_mean = max(worst_mean, rmean)
        log(f"  {r['name']:<20}{e[0]:11.2e}{e[1]:11.2e}{rm:22.1f}x{rmean:23.1f}x")
    if worst_mean < 2.0:
        log("\n  -> gli outlier NON degradano gli slot tipici: l'errore del bootstrap e' per slot, non dipende dal massimo.")
    elif worst_mean < 10.0:
        log(f"\n  -> degrado MODERATO sugli slot tipici (fino a {worst_mean:.1f}x l'errore medio senza outlier).")
    else:
        log(f"\n  -> gli outlier AVVELENANO il ciphertext: errore medio sugli slot tipici fino a {worst_mean:.0f}x. "
            f"Il massimo delle attivazioni va contenuto in training (penalita' piu' stretta).")


def main():
    print("=== Bootstrap con outlier: precisione sugli slot tipici ===", flush=True)
    gpu_mem("inizio")
    cc, keys = make_context()
    rows = run(cc, keys)
    summarize(rows)
    gpu_mem("fine")


if __name__ == '__main__':
    main()