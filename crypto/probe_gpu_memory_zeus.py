"""
crypto/probe_gpu_memory_zeus.py

PERCHE': la prova stadio-bootstrap-stadio con 128 canali (livello 2) e' andata in
'out of memory' all'inizio dello stadio 2, con 40.2 GB occupati su 46 dopo i
bootstrap. A 32 canali la stessa prova chiudeva a 28.9 GB. La memoria cresce con
il numero di canali anche se ogni ciphertext e' parcheggiato su RAM host con
Offload(): QUALCOSA non viene liberato. Una rete vera ha fino a 256 canali e
dura ore: va capito PRIMA, non a meta' di un run da 5 ore.

COSA FA (in ~5 minuti, 4 canali cifrati e 128 ciphertext derivati senza ricifrare):
  1  crea 128 ciphertext (EvalMult da 4 cifrati) e li parcheggia su host
  2  li rilegge e li riparcheggia (EvalMult 0.5 + Offload)          -> cresce con i ciphertext "toccati"?
  3  4 blocchi di convoluzione 128 -> 8 (come in uno stadio)          -> cresce con le uscite accumulate?
  4  norm + PolyAct su 128 canali (statistiche, Chebyshev+Newton, maschera) -> cresce con i canali elaborati?
  5  32 bootstrap (misura ogni 8)                                     -> il primo costa ~8 GB?
  6  libera tutto (del + TrimGPUMemoryPool)                           -> torna alla base? se no, e' una perdita
Per ogni passo: MiB occupati (nvidia-smi, include ~4.3 GB di un altro processo) e la differenza
dal passo precedente.

Variabili d'ambiente (le stesse della Prova 2): CT_CACHE_GIB (default 6), ROT_CACHE_GIB (default 6),
CONV_B (default 8, blocco di canali d'uscita). Si lancia due volte: con i default per riprodurre la
crescita, e con CT_CACHE_GIB=4 ROT_CACHE_GIB=5 CONV_B=4 per vedere cosa cambia.

Uso:  python3 crypto/probe_gpu_memory_zeus.py
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
N_CT = 128
CONV_B = int(os.environ.get("CONV_B", "8"))


def used_mib():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                      text=True).strip()
        return int(out.split()[0])
    except Exception:
        return -1


class Meter:
    def __init__(self, log=print):
        self.log = log
        self.prev = None
        self.base = None
        self.rows = []

    def __call__(self, label):
        u = used_mib()
        if self.base is None:
            self.base = u
        d = "" if self.prev is None else f"({u - self.prev:+d})"
        self.log(f"    [GPU] {label:<58}{u:>7} MiB {d:>9}   rispetto all'inizio {u - self.base:+d}")
        self.rows.append((label, u))
        self.prev = u
        return u


def run(cc, keys, g, log=print):
    he = LHE(cc, keys, g)
    mem = Meter(log)
    rng = np.random.default_rng(5)
    mem("dopo LoadContext (base)")

    # ---- 1: crea 128 ciphertext senza ricifrare ----
    log("\n[1] creazione di 128 ciphertext (4 cifrati + EvalMult) e parcheggio su host")
    base = he.enc_offloaded([g.pack(rng.normal(size=g.size(LEVEL)) * 0.5, LEVEL) for _ in range(4)])
    mem("4 ciphertext cifrati e parcheggiati")
    cts = []
    for i in range(N_CT):
        ct = cc.EvalMult(base[i % 4], 1.0 + 1e-3 * i)
        ct.Offload()
        cts.append(ct)
        if i % 8 == 7:
            cc.TrimGPUMemoryPool()
        if i % 32 == 31:
            mem(f"  {i + 1} ciphertext creati e parcheggiati")
    cc.TrimGPUMemoryPool()
    m1 = mem("128 ciphertext, dopo Trim")

    # ---- 2: rilettura e riparcheggio ----
    log("\n[2] rilettura di tutti i 128 (EvalMult 0.5 + Offload del risultato)")
    for i, ct in enumerate(cts):
        r = cc.EvalMult(ct, 0.5)
        r.Offload()
        ct.Offload()
        del r
        if i % 8 == 7:
            cc.TrimGPUMemoryPool()
        if i % 32 == 31:
            mem(f"  {i + 1} riletti")
    cc.TrimGPUMemoryPool()
    m2 = mem("dopo la rilettura, dopo Trim")

    # ---- 3: blocchi di convoluzione 128 -> 8, accumulando uscite ----
    log(f"\n[3] 4 blocchi di convoluzione 128 -> {CONV_B} (come in uno stadio), uscite accumulate su host")
    outs_all = []
    for blk in range(4):
        w = rng.normal(size=(CONV_B, N_CT, 3, 3)) * (0.5 / np.sqrt(N_CT * 9))
        b = rng.normal(size=CONV_B) * 0.05
        t0 = time.time()
        outs = he.conv3x3_blocked(cts, w, b, LEVEL, False, B=CONV_B)
        he.dec(outs[-1])
        outs_all += outs
        mem(f"  blocco {blk + 1} ({time.time() - t0:.0f}s), {len(outs_all)} uscite accumulate")
    m3 = mem("dopo i 4 blocchi")

    # ---- 4: catena norm + attivazione su 128 canali (come in uno stadio) ----
    log("\n[4] norm + attivazione su 128 canali: statistiche (16 chiavi), radice inversa Chebyshev+Newton, PolyAct, maschera")
    gamma, beta = np.ones(N_CT), np.zeros(N_CT)
    nouts, _ = he.norm_act(cts, gamma, beta, LEVEL, [1.2, -0.5, 0.15], [0.2, 1.3],
                           progress=lambda c: mem(f"  norm+act: {c} canali") if c % 16 == 0 else None)
    m4 = mem("dopo la norm+act su 128 canali, dopo Trim")

    # ---- 5: bootstrap ----
    log("\n[5] bootstrap (fino a 32)")
    boot = []
    n_boot = min(32, len(outs_all))                   # con CONV_B=4 le uscite sono 16, non 32
    for i in range(n_boot):
        nb = cc.EvalBootstrap(outs_all[i])
        outs_all[i].Offload()
        nb.Offload()
        boot.append(nb)
        if i == 0:
            mem("  dopo il PRIMO bootstrap")
        if i % 4 == 3:
            cc.TrimGPUMemoryPool()
        if i % 8 == 7:
            mem(f"  {i + 1} bootstrap")
    cc.TrimGPUMemoryPool()
    m5 = mem("dopo i 32 bootstrap, dopo Trim")

    # ---- 6: liberazione ----
    log("\n[6] liberazione di tutto")
    del cts, outs_all, boot, base, outs, nouts
    gc.collect()
    cc.TrimGPUMemoryPool()
    m6 = mem("dopo del + gc + Trim")

    base_mib = mem.base
    log("\n=== RIEPILOGO (MiB rispetto alla base dopo LoadContext) ===")
    for lab, v in (("128 ciphertext parcheggiati", m1), ("dopo rilettura", m2), ("dopo 4 blocchi conv", m3),
                   ("dopo norm+act su 128 canali", m4), ("dopo 32 bootstrap", m5), ("dopo la liberazione", m6)):
        log(f"  {lab:<30}{v - base_mib:+8d} MiB")
    log("\n  LETTURA:")
    if m1 - base_mib > 2000:
        log(f"   - parcheggiare 128 ciphertext lascia {m1 - base_mib} MiB sulla GPU: Offload() non libera subito "
            f"(circa {(m1 - base_mib) / N_CT:.0f} MiB per ciphertext).")
    if m2 - m1 > 2000:
        log(f"   - rileggere/riparcheggiare fa crescere di altri {m2 - m1} MiB.")
    if m3 - m2 > 2000:
        log(f"   - i blocchi di convoluzione aggiungono {m3 - m2} MiB.")
    if m5 - m4 > 4000:
        log(f"   - il bootstrap aggiunge {m5 - m4} MiB (in gran parte il primo: dati precalcolati del bootstrap).")
    if m6 - base_mib > 3000:
        log(f"   - DOPO la liberazione restano {m6 - base_mib} MiB sopra la base: perdita o pool non restituito.")
    else:
        log("   - dopo la liberazione la memoria torna vicina alla base: nessuna perdita, solo cache/pool.")


def main():
    g = Grid(H0, W0, LEVEL + 1)
    print(f"=== SONDA MEMORIA: caches ciphertext {os.environ.get('CT_CACHE_GIB', '6')} GiB, "
          f"rotazioni {os.environ.get('ROT_CACHE_GIB', '6')} GiB, blocco conv B={CONV_B} ===")
    cc, keys = build_context(g, LEVEL)
    run(cc, keys, g)


if __name__ == '__main__':
    main()