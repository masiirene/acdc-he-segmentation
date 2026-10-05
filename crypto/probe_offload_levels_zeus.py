"""
crypto/probe_offload_levels_zeus.py

DAI DATI DELLA SONDA PRECEDENTE (probe_norm_leak_zeus.py) la crescita di memoria NON sta nella catena
norm+attivazione: con l'uscita scartata nessuna variante cresce. Cresce SOLO quando si fa Offload() dell'uscita
e la si CONSERVA: +64 MiB per ciphertext (due volte, in due sonde diverse). Eppure:
  - 128 ciphertext creati con EvalMult a livello ~1 e parcheggiati: nessuna crescita
  - 32 uscite di bootstrap (livello 21) parcheggiate: nessuna crescita
  - uscite di norm+attivazione (livello ~17) parcheggiate: +64 MiB ciascuna
Ipotesi (NON provata): Offload() libera la memoria GPU solo per ciphertext a certi livelli, e non per quelli
prodotti da molte moltiplicazioni. Se e' cosi', non e' una perdita illimitata: un ciphertext profondo e' piccolo
(2 x (44 - livello) MiB: 54 MiB a livello 17, ~12 MiB a livello 38).

QUESTA SONDA misura, per ogni livello L in {0, 2, 5, 10, 17, 25, 35}:
  - quanta memoria GPU resta occupata per ciphertext DOPO Offload() (e dopo Trim)
  - quanta ne occuperebbe senza Offload (controllo: deve valere ~2 x (44 - L) MiB)
  - se la memoria torna indietro quando si eliminano i ciphertext
La memoria si legge a multipli di 1 GiB: per questo ogni livello usa abbastanza ciphertext (circa 2.5 GiB in totale).

LETTURA: il livello piu' basso a cui Offload smette di liberare. Da li' in poi inutile parcheggiare (costa
un trasferimento e non libera): conviene tenere quei ciphertext sulla GPU, dove sono piccoli.

ATTENZIONE: la modalita' di default ("both", tutti i livelli nello stesso processo) e' FALSATA dal pool (vedi sopra):
nel primo uso ha dato valori negativi. Usare una misura per processo:

  for L in 2 17 35; do for M in offload keep; do
      OFFLOAD_LEVELS=$L OFFLOAD_MODE=$M python3 crypto/probe_offload_levels_zeus.py | grep "livello"
  done; done

Se per lo stesso livello 'offload' cresce come 'keep', Offload NON libera a quel livello; se cresce ~0, libera.
Durata: ~2 minuti per processo (6 processi).
"""

import os
import sys
import gc
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fideslib_py as fhe
from prototype_lattice_resample import Grid
from lattice_conv import LHE
from test_stage_bootstrap_fullres_zeus import build_context

H0, W0 = 256, 224
LEVEL = 2
MAX_LIMBS = 44                      # profondita' 43 + 1
LEVELS = [int(v) for v in os.environ.get("OFFLOAD_LEVELS", "0,2,5,10,17,25,35").split(",")]
MODE = os.environ.get("OFFLOAD_MODE", "both")        # both = vecchio comportamento (CONFONDIBILE dal pool); offload | keep = misura singola


def used_mib():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                      text=True).strip()
        return int(out.split()[0])
    except Exception:
        return -1


def run_single(cc, keys, g, level, mode, log=print, mem=used_mib):
    """UNA misura per processo (il pool della libreria non restituisce la memoria al driver: ogni misura
    successiva nello stesso processo riutilizzerebbe memoria gia' riservata e sarebbe falsata).
    mode = 'offload': ciphertext parcheggiati su host E conservati; 'keep': conservati sulla GPU."""
    he = LHE(cc, keys, g)
    rng = np.random.default_rng(13)
    x = he.enc_offloaded([g.pack(rng.normal(size=g.size(LEVEL)) * 0.5, LEVEL)])[0]
    for _ in range(level):
        x = cc.EvalMult(x, 1.0001)
    lvl = x.GetLevel()
    he.dec(x)
    cc.TrimGPUMemoryPool()
    size = 2 * (MAX_LIMBS - lvl)
    n = max(8, int(np.ceil(4096 / size)))
    m0 = mem()
    kept = []
    for i in range(n):
        c = cc.EvalAdd(x, 1e-6 * (i + 1))
        if mode == "offload":
            c.Offload()
        kept.append(c)
        if i % 8 == 7:
            cc.TrimGPUMemoryPool()
    cc.TrimGPUMemoryPool()
    m1 = mem()
    grow = (m1 - m0) / n
    log(f"  livello {lvl:>2}  modo {mode:<8}  {n:>3} ciphertext conservati: memoria {m0} -> {m1} MiB, "
        f"cresce {grow:7.1f} MiB per ciphertext (teorico a 1 MiB per limbo: {size} MiB)")
    return grow


def run(cc, keys, g, levels=LEVELS, log=print, mem=used_mib):
    if MODE in ("offload", "keep"):
        return [run_single(cc, keys, g, L, MODE, log, mem) for L in levels]
    he = LHE(cc, keys, g)
    rng = np.random.default_rng(13)
    base = he.enc_offloaded([g.pack(rng.normal(size=g.size(LEVEL)) * 0.5, LEVEL)])[0]
    rows = []
    for L in levels:
        x = base
        for _ in range(L):
            x = cc.EvalMult(x, 1.0001)                    # un livello per passo
        lvl = x.GetLevel()
        size = 2 * (MAX_LIMBS - lvl)                      # MiB attesi per ciphertext
        n = max(8, int(np.ceil(2560 / size)))
        he.dec(x)                                         # sincronizza e porta x a regime
        cc.TrimGPUMemoryPool()
        gc.collect()
        m0 = mem()
        kept = []
        for i in range(n):
            c = cc.EvalAdd(x, 1e-6 * (i + 1))             # nuovo ciphertext allo stesso livello
            c.Offload()
            kept.append(c)
            if i % 8 == 7:
                cc.TrimGPUMemoryPool()
        cc.TrimGPUMemoryPool()
        m1 = mem()
        del kept, c
        gc.collect()
        cc.TrimGPUMemoryPool()
        m2 = mem()
        kept2 = [cc.EvalAdd(x, 1e-6 * (i + 1)) for i in range(n)]      # controllo: SENZA Offload
        cc.TrimGPUMemoryPool()
        m3 = mem()
        del kept2
        gc.collect()
        cc.TrimGPUMemoryPool()
        retained = (m1 - m0) / n
        plain = (m3 - m2) / n
        back = m1 - m2
        rows.append((L, lvl, size, n, retained, plain, back))
        log(f"  livello {lvl:>2} (atteso {size:>3} MiB/ct, {n:>3} ct):  dopo Offload restano {retained:6.1f} MiB/ct   "
            f"senza Offload {plain:6.1f} MiB/ct   restituiti all'eliminazione {back:6d} MiB")
    log("\n=== RIEPILOGO ===")
    log(f"  {'livello':>8}{'atteso MiB/ct':>15}{'restano dopo Offload':>22}{'senza Offload':>16}   esito")
    first_bad = None
    for L, lvl, size, n, retained, plain, back in rows:
        frac = retained / size
        esito = "libera" if frac < 0.15 else ("NON libera" if frac > 0.6 else "libera in parte")
        if esito != "libera" and first_bad is None:
            first_bad = lvl
        log(f"  {lvl:>8}{size:>15}{retained:>22.1f}{plain:>16.1f}   {esito}")
    log("\n  LETTURA:")
    if first_bad is None:
        log("   - Offload libera la memoria a tutti i livelli provati: la crescita vista nella catena norm nasce da altro "
            "(per esempio dalla combinazione con la maschera o dal bootstrap): da indagare con la catena completa.")
    else:
        log(f"   - Offload smette di liberare a partire dal livello {first_bad}. Sopra quel livello conviene NON parcheggiare "
            f"i ciphertext (restano sulla GPU comunque) e tenerli: a livello 38 pesano ~12 MiB, quindi 256 canali = ~3 GB.")
    return rows


def main():
    g = Grid(H0, W0, LEVEL + 1)
    print("=== SONDA OFFLOAD PER LIVELLO ===")
    cc, keys = build_context(g, LEVEL)
    run(cc, keys, g)


if __name__ == '__main__':
    main()