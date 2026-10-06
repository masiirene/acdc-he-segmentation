"""
crypto/probe_boot_retention_zeus.py   (sonda 1b, versione 2; circa 4 minuti)

COSA HA DETTO LA SONDA 1b (prima versione, Zeus, 5 ott):
  lotto 1: +126 MiB per ciphertext (di cui ~1 GiB una tantum per l'intero lotto), lotti 2 e 3: +64,0 MiB ciascuno.
  Eliminando le 48 uscite tenute: restituiti 0 MiB.   Con il lavaggio EvalAdd(., 0.0): 192 MiB per ciphertext.
  => non e' una cache che si riempie: ogni uscita di bootstrap parcheggiata con Offload() lascia 64 MiB sulla GPU.
  Pero' in uno stesso processo il pool di memoria puo' tenere i blocchi liberati e RIUSARLI: allora "0 MiB restituiti"
  non vuol dire perdita. Questa versione misura proprio il riuso, e se NON fare Offload evita il problema.

COSA FA (stesso contesto di prima; cache con CT_CACHE_GIB / ROT_CACHE_GIB):
  R1  2 lotti da 16 bootstrap con Offload + Trim, uscite tenute vive            (riprova il dato: ~126 poi ~64)
  R2  elimina le 32 uscite: quanta memoria torna?
  R4  DOPO l'eliminazione, altri 16 bootstrap con Offload e uscite tenute vive
        -> se la crescita e' ~0 il pool RIUSA i blocchi liberati: il massimo e' limitato, non e' una perdita cumulativa
        -> se e' ~64 per ciphertext: perdita cumulativa (a 3296 bootstrap sarebbero ~200 GiB: la rete intera non puo' finire)
  R5  16 bootstrap tenuti sulla GPU SENZA Offload, poi eliminati: quanto pesano e quanto tornano

Uso:   python3 crypto/probe_boot_retention_zeus.py 2>&1 | tee logs_zeus/1b_boot_retention_v2.txt
"""

import os
import sys
import gc
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fideslib_py as fhe  # noqa: F401  (build_context lo usa)
from prototype_lattice_resample import Grid
from lattice_conv import LHE
from test_stage_bootstrap_fullres_zeus import build_context

H0, W0 = 256, 224
LEVEL = 2
BATCH = 16
N_FULL = int(os.environ.get("N_BOOT_FULL", "3296"))        # bootstrap della rete intera (pack_5s14: 3296)


def used_mib():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                      text=True).strip()
        return int(out.split()[0])
    except Exception:
        return -1


def main():
    g = Grid(H0, W0, LEVEL + 1)
    print("=== SONDA 1b v2: le uscite del bootstrap, riuso della memoria e Offload ===")
    cc, keys = build_context(g, LEVEL)
    he = LHE(cc, keys, g)
    rng = np.random.default_rng(21)
    x0 = he.enc_offloaded([g.pack(rng.normal(size=g.size(LEVEL)) * 0.5, LEVEL)])[0]
    x = cc.EvalMult(cc.EvalMult(x0, 1.0001), 1.0001)
    he.dec(x)
    cc.TrimGPUMemoryPool()
    print(f"\n  ingresso a livello {x.GetLevel()}; memoria GPU all'inizio: {used_mib()} MiB")
    print("  riscaldamento (un bootstrap, non contato) ...", flush=True)
    w = cc.EvalBootstrap(x)
    w.Offload()
    del w
    gc.collect()
    cc.TrimGPUMemoryPool()
    base = used_mib()
    print(f"  memoria GPU dopo il riscaldamento: {base} MiB")

    def lotto(n, offload):
        lst = []
        for _ in range(n):
            c = cc.EvalBootstrap(x)
            if offload:
                c.Offload()
            lst.append(c)
        cc.TrimGPUMemoryPool()
        return lst

    prev = base
    kept = []
    print("\n  R1  due lotti con Offload, uscite tenute vive")
    per_r1 = []
    for b in range(2):
        kept += lotto(BATCH, True)
        now = used_mib()
        per_r1.append((now - prev) / BATCH)
        print(f"      lotto {b + 1}: {prev} -> {now} MiB   {per_r1[-1]:7.1f} MiB per ciphertext", flush=True)
        prev = now
    peak1 = prev

    print("\n  R2  eliminazione delle uscite tenute")
    kept.clear()
    gc.collect()
    cc.TrimGPUMemoryPool()
    after = used_mib()
    ret2 = peak1 - after
    print(f"      {peak1} -> {after} MiB   restituiti {ret2} MiB su {peak1 - base} cresciuti")

    print(f"\n  R4  dopo l'eliminazione: altri {BATCH} bootstrap con Offload, uscite tenute vive")
    m0 = used_mib()
    kept4 = lotto(BATCH, True)
    m1 = used_mib()
    per4 = (m1 - m0) / BATCH
    print(f"      {m0} -> {m1} MiB   {per4:7.1f} MiB per ciphertext", flush=True)
    kept4.clear()
    gc.collect()
    cc.TrimGPUMemoryPool()

    print(f"\n  R5  {BATCH} bootstrap tenuti sulla GPU SENZA Offload")
    m0 = used_mib()
    kept5 = lotto(BATCH, False)
    m1 = used_mib()
    per5 = (m1 - m0) / BATCH
    print(f"      {m0} -> {m1} MiB   {per5:7.1f} MiB per ciphertext", flush=True)
    kept5.clear()
    gc.collect()
    cc.TrimGPUMemoryPool()
    m2 = used_mib()
    ret5 = m1 - m2
    print(f"      dopo l'eliminazione: {m1} -> {m2} MiB   restituiti {ret5} MiB su {m1 - m0}")

    steady = per_r1[-1]
    print("\n=== LETTURA ===")
    print(f"  - regime: {steady:.0f} MiB per uscita di bootstrap parcheggiata con Offload (il primo lotto in piu' ~{(per_r1[0] - steady) * BATCH:.0f} MiB una tantum).")
    if per4 < 0.25 * steady:
        print(f"  - R4: la crescita dopo l'eliminazione e' {per4:.0f} MiB per ciphertext: il pool RIUSA i blocchi liberati. "
              f"Non e' una perdita cumulativa: il massimo e' limitato dal numero di uscite vive contemporaneamente.")
    elif per4 > 0.75 * steady:
        print(f"  - R4: la crescita e' di nuovo {per4:.0f} MiB per ciphertext: i blocchi liberati NON vengono riusati, la perdita e' "
              f"cumulativa. A {N_FULL} bootstrap sarebbero ~{N_FULL * steady / 1024:.0f} GiB: la rete intera non puo' finire cosi'.")
    else:
        print(f"  - R4: riuso parziale ({per4:.0f} MiB per ciphertext contro {steady:.0f}): da guardare nei numeri.")
    if per5 > 0:
        print(f"  - senza Offload un'uscita pesa {per5:.0f} MiB sulla GPU e alla distruzione se ne restituisce il "
              f"{100 * ret5 / max(1, m1 - m0):.0f}%: "
              + ("conviene NON parcheggiare le uscite del bootstrap, e tenerle sulla GPU." if (ret5 > 0.7 * (m1 - m0) and per5 <= steady + 10)
                 else "non e' una via d'uscita."))


if __name__ == '__main__':
    main()