"""
crypto/test_conv_blocked_zeus.py     (PROVA 1)

Misura la convoluzione 3x3 nella forma di PRODUZIONE (a blocchi, vedi
lattice_conv.py) con canali d'ingresso realistici, a risoluzione piena
(256x224, 65.536 slot, un ciphertext per canale).

Per ogni Cin in {32, 64, 128, 256} si calcola UN BLOCCO di B=8 canali d'uscita
con TUTTI i Cin canali d'ingresso. Una convoluzione completa Cin->Cout e' fatta
di Cout/B blocchi identici, quindi il tempo completo si proietta con
    t_conv = (Cout / B) * t_blocco
senza eseguirla tutta (256->256 richiederebbe quasi un'ora).

Cosa si legge:
  - ms per termine (da confrontare con 5.4 ms BLOCCO-8 e 10.5 ms OFFLOAD del
    benchmark, fatto a Cin=8): se resta ~5-10 ms a Cin=256, la proiezione di
    4.8-9.3 ore per le convoluzioni e' credibile; se sale, il benchmark era
    ottimistico;
  - memoria GPU: deve restare piatta al crescere di Cin (ingressi su host);
  - errore su 2 canali d'uscita contro numpy: ~1e-9 o meno.

Cifratura degli ingressi: ~1 s per canale (256 canali = ~4-5 minuti); si
cifrano 256 canali UNA volta e le configurazioni minori ne usano un prefisso.

Uso (in tmux, GPU libera):  python3 crypto/test_conv_blocked_zeus.py
Durata attesa: ~10-12 minuti.
"""

import os
import sys
import time
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fideslib_py as fhe
from prototype_lattice_resample import Grid
from lattice_conv import LHE, level_offsets

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
H0, W0 = 256, 224
CONFIGS = [32, 64, 128, 256]          # Cin
B = 8
T_FAST, T_SLOW = 5.4, 10.5            # ms/termine misurati nel benchmark


def gpu_mem(label, log=print):
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                       '--format=csv,noheader,nounits'], text=True).strip()
        used, total = out.split(',')
        log(f"    [GPU MEM] {label}: {used.strip()} / {total.strip()} MiB")
    except Exception as e:
        log(f"    [GPU MEM] {label}: impossibile leggere ({e})")


def build_context(g):
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
    cc.EvalRotateKeyGen(keys.secretKey, sorted(r for r in level_offsets(g, 0) if r != 0))
    cc.SetRotationKeyCache(4 * GiB)                 # 8 chiavi * ~354 MB: entrano
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s", flush=True)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(6 * GiB)
    return cc, keys


def run(cc, keys, g, configs=CONFIGS, B=B, log=print):
    he = LHE(cc, keys, g)
    N = g.N
    rng = np.random.default_rng(3)
    cmax = max(configs)
    mask = g.valid_mask(0)
    offs = level_offsets(g, 0)

    log(f"\nCifratura di {cmax} canali d'ingresso (parcheggiati su host) ...")
    t0 = time.time()
    vecs = [g.pack(rng.normal(size=(H0, W0)) * 0.5, 0) for _ in range(cmax)]
    chans = he.enc_offloaded(vecs, log=log)
    log(f"  fatto in {time.time()-t0:.0f}s")
    gpu_mem("dopo la cifratura (gli ingressi stanno su host)", log)

    # costo di una sincronizzazione (decifrazione), da sottrarre
    t0 = time.time()
    he.dec(chans[0])
    sync_cost = time.time() - t0
    log(f"  (costo di una decifrazione/sincronizzazione: {sync_cost:.2f}s)")

    rows = []
    for Cin in configs:
        log(f"\n=== Cin = {Cin}, blocco di B = {B} canali d'uscita ===")
        w = rng.normal(size=(B, Cin, 3, 3)) * (0.5 / np.sqrt(Cin * 9))
        bias = rng.normal(size=B) * 0.05
        t0 = time.time()
        outs = he.conv3x3_blocked(chans[:Cin], w, bias, 0, False, B=B)
        he.dec(outs[-1])                                     # sincronizza
        t_block = time.time() - t0 - sync_cost
        terms = Cin * B * 9
        per_term = t_block / terms * 1000
        full = (Cin / B) * t_block                            # conv Cin -> Cin
        log(f"  tempo del blocco: {t_block:.1f}s   ({terms} termini, {per_term:.1f} ms/termine)")
        log(f"  proiezione conv completa {Cin}->{Cin} ({Cin // B} blocchi): {full / 60:.1f} min")
        gpu_mem("dopo il blocco", log)
        errs = []
        for co in (0, B - 1):
            ref = np.zeros(N)
            for ci in range(Cin):
                for k, off in enumerate(offs):
                    ref += w[co, ci, k // 3, k % 3] * np.roll(vecs[ci], -off)
            ref = (ref + bias[co]) * mask
            errs.append(np.abs(he.dec(outs[co]) - ref).max())
        log(f"  errore max su 2 canali d'uscita contro numpy: {max(errs):.2e}")
        rows.append((Cin, t_block, per_term, full / 60, max(errs)))
        del outs
        cc.TrimGPUMemoryPool()

    log("\n=== RIEPILOGO ===")
    log(f"  {'Cin':>4} {'blocco (s)':>11} {'ms/termine':>11} {'conv Cin->Cin (min)':>20} {'errore':>10}")
    for Cin, tb, pt_, full, e in rows:
        log(f"  {Cin:4d} {tb:11.1f} {pt_:11.1f} {full:20.1f} {e:10.1e}")
    log(f"\n  Riferimento benchmark (Cin=8): {T_FAST} ms/termine (BLOCCO-8) .. {T_SLOW} ms/termine (OFFLOAD).")
    worst = max(r[2] for r in rows)
    if worst <= 1.5 * T_SLOW:
        log("  Costo per termine in linea col benchmark anche a Cin grande: le stime di tempo per immagine reggono.")
    else:
        log(f"  Costo per termine fino a {worst:.1f} ms: PIU' ALTO del benchmark. Le stime di tempo per immagine "
            f"(7-14 h per le reti sum) vanno riviste al rialzo di un fattore ~{worst / T_SLOW:.1f}.")
    return rows


def main():
    g = Grid(H0, W0, 1)
    print(f"=== PROVA 1: convoluzione a blocchi, risoluzione piena {H0}x{W0}, N={g.N} slot ===")
    assert g.N == 65536
    gpu_mem("inizio")
    cc, keys = build_context(g)
    gpu_mem("dopo LoadContext")
    run(cc, keys, g)
    gpu_mem("fine")


if __name__ == '__main__':
    main()