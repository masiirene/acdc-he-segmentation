"""
crypto/run_he_network_zeus.py        (da lanciare su ZEUS, in tmux)

Fa girare in HE (CKKS, FIDESlib) la rete descritta da un pacchetto prodotto da pack_for_zeus.py:

    python3 crypto/run_he_network_zeus.py crypto/pack_narrow.npz        # prova a larghezza ridotta (pochi minuti)
    python3 crypto/run_he_network_zeus.py crypto/pack_5s14.npz          # rete vera, 5 stage a 14 norm (ore)

Variabili d'ambiente:
  CT_CACHE_GIB   cache ciphertext (default 3: basta ai livelli profondi; a 2 il primo stadio va in thrashing)
  ROT_CACHE_GIB  cache chiavi di rotazione (default 6: le 16 potenze di 2 delle somme ne vogliono ~5.7; a 5 la norm
                 va 3.7 volte piu' lenta)
  CONV_B         canali d'uscita per blocco della convoluzione (default 8)
  TRIM_SYNC      1 (predefinito) = Synchronize() prima di ogni TrimGPUMemoryPool: con la sincronizzazione la prova ridotta del
                 5 ott e' arrivata in fondo (picco 40,9 GB); senza, era andata fuori memoria a meta' (vedi he_network.PoolCC)
  AUX_CLEAR      1 (predefinito) = in piu' ClearAuxiliaryPolyPool(): nella prova ridotta del 5 ott 'aux 0' si leggeva DOPO la
                 pulizia; senza (prova sulla rete vera) il pool ausiliario cresce, 20 -> 58 voci in due stadi
  MEM_GUARD_MIB  limite di memoria GPU (predefinito 44500): oltre, si ferma e stampa il resoconto parziale (0 = spento)
  MASK_LAUNDER   1 = dopo ogni moltiplicazione per maschera si fa EvalAdd(., 0.0) (vedi probe_mask_leak_zeus.py)
  MARGIN         livelli di margine nei controlli del budget (default 1)
  CHECK          1 = confronta il canale 0 di ogni stadio con il riferimento (costa una decifrazione per stadio)
  STOP_AFTER     es. enc1: si ferma dopo quel blocco encoder (misura dei tempi senza fare tutta la rete)

Richiede nella stessa cartella: he_network.py, lattice_conv.py, prototype_lattice_resample.py.
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
import he_network as hn

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
LEVEL_BUDGET = [4, 4]
BSGS_DIM = [8, 8]


def used_mib():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                      text=True).strip()
        return int(out.split()[0])
    except Exception:
        return -1


def needed_rotations(g):
    rots = set()
    for l in range(g.n_levels):
        rots |= set(level_offsets(g, l))            # conv 3x3 e upsampling (gli offset negativi dell'upsampling sono inclusi)
    s = 1
    while s < g.N:
        rots.add(s)                                 # somme su tutti gli slot per le statistiche della norm
        s *= 2
    rots.discard(0)
    return sorted(rots)


def build_context(g):
    N = g.N
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(N)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetDevices([0])
    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    t0 = time.time()
    cc.EvalBootstrapSetup(LEVEL_BUDGET, BSGS_DIM, N)
    cc.EvalBootstrapKeyGen(keys.secretKey, N)
    print(f"  Bootstrap setup+keygen (bsgsDim={BSGS_DIM}): {time.time() - t0:.1f}s", flush=True)
    rots = needed_rotations(g)
    t0 = time.time()
    cc.EvalRotateKeyGen(keys.secretKey, rots)
    print(f"  chiavi di rotazione extra: {len(rots)} ({time.time() - t0:.1f}s)", flush=True)
    ct_gib = float(os.environ.get("CT_CACHE_GIB", "3"))
    rot_gib = float(os.environ.get("ROT_CACHE_GIB", "6"))
    cc.SetRotationKeyCache(int(rot_gib * GiB))
    cc.SetBootstrapCache(1 * GiB)
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time() - t0:.1f}s", flush=True)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(int(ct_gib * GiB))
    print(f"  cache: ciphertext {ct_gib} GiB, rotazioni {rot_gib} GiB", flush=True)
    return cc, keys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("uso: python3 crypto/run_he_network_zeus.py <pacchetto.npz>")
    pack = hn.load_pack(sys.argv[1])
    meta = pack["meta"]
    H0, W0 = pack["image"].shape[-2:]
    k = meta["k"]
    g = Grid(H0, W0, k)
    print(f"=== RETE IN HE: {sys.argv[1]} ===")
    print(f"  k={k}, filtri {meta['filters']}, norm bypassate {len(meta['bypass'])} ({','.join(meta['bypass'])})")
    print(f"  immagine {H0}x{W0}, N={g.N} slot, {k} livelli del reticolo; modo '{meta.get('mode')}', "
          f"fetta {meta.get('slice_index')} ({meta.get('split')})")
    costs = []
    for st in hn.plan(meta):
        if st["type"] == "stage":
            costs.append(hn.stage_cost(hn.get_scheme(meta, st["norm"]) if st["norm"] else None))
    print(f"  costo stimato per stadio (livelli): {costs}")
    print(f"  opzioni: MASK_LAUNDER={os.environ.get('MASK_LAUNDER', '0')} MARGIN={os.environ.get('MARGIN', '1')} "
          f"CONV_B={os.environ.get('CONV_B', '8')} CHECK={os.environ.get('CHECK', '0')} STOP_AFTER={os.environ.get('STOP_AFTER', '')} "
          f"TRIM_SYNC={os.environ.get('TRIM_SYNC', '1')} AUX_CLEAR={os.environ.get('AUX_CLEAR', '1')} MEM_GUARD_MIB={os.environ.get('MEM_GUARD_MIB', '44500')}")
    assert g.N == 65536, "il contesto e' costruito per 65.536 slot"
    print(f"    [GPU] inizio: {used_mib()} MiB")
    cc, keys = build_context(g)
    print(f"    [GPU] dopo LoadContext: {used_mib()} MiB", flush=True)
    aux_clear = os.environ.get('AUX_CLEAR', '1') == '1'
    sync_trim = os.environ.get('TRIM_SYNC', '1') == '1'
    cc = hn.PoolCC(cc, aux_clear, sync_trim)
    print(f"    pool: {hn.pool_info(cc) or 'nessuna statistica disponibile'} | TRIM_SYNC={int(sync_trim)} AUX_CLEAR={int(aux_clear)}", flush=True)
    he = LHE(cc, keys, g)
    runner = hn.HERunner(he, pack, log=lambda s: print(s, flush=True), mem=used_mib,
                         margin=int(os.environ.get("MARGIN", "1")), conv_b=int(os.environ.get("CONV_B", "8")),
                         check=os.environ.get("CHECK", "0") == "1", stop_after=os.environ.get("STOP_AFTER") or None,
                         mem_guard=int(os.environ.get("MEM_GUARD_MIB", "44500")))
    runner.run()


if __name__ == '__main__':
    main()