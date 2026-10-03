"""
crypto/test_keygen_only_diagnostic_zeus.py

Test DIAGNOSTICO, non il test vero: genera SOLO le chiavi (bootstrap +
le chiavi extra per la convoluzione a righe), senza cifrare nulla e
senza eseguire nessuna convoluzione. Serve a isolare con certezza se
il problema di oggi nasce gia' qui (generazione/caricamento chiavi)
o solo dopo, durante l'uso vero.

Stampa la memoria dopo OGNI singolo passo, cosi' se qualcosa va storto
sappiamo esattamente dove.

Uso: python3 crypto/test_keygen_only_diagnostic_zeus.py
"""

import sys
import math
import time
import subprocess

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
LEVEL_BUDGET = [4, 4]
BSGS_DIM = [4, 4]
CACHE_GIB = 1
CHANNELS = 32
WP = 226
K = 3


def print_gpu_mem(label):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'], text=True
        ).strip()
        used, total = out.split(',')
        print(f"    [GPU MEM] {label}: {used.strip()} / {total.strip()} MiB")
    except Exception as e:
        print(f"    [GPU MEM] {label}: impossibile leggere ({e})")


def main():
    n = CHANNELS
    real_len = WP * n
    batch_size = 1 << math.ceil(math.log2(real_len))

    print(f"=== DIAGNOSTICA: sola generazione chiavi (bootstrap + conv), {n} canali ===")
    print(f"batch={batch_size}\n")

    print_gpu_mem("inizio")

    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(batch_size)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetDevices([0])

    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    print_gpu_mem("dopo KeyGen base")

    t0 = time.time()
    cc.EvalBootstrapSetup(LEVEL_BUDGET, BSGS_DIM, batch_size)
    print(f"EvalBootstrapSetup: {time.time()-t0:.1f}s")
    print_gpu_mem("dopo EvalBootstrapSetup")

    t0 = time.time()
    cc.EvalBootstrapKeyGen(keys.secretKey, batch_size)
    print(f"EvalBootstrapKeyGen: {time.time()-t0:.1f}s")
    print_gpu_mem("dopo EvalBootstrapKeyGen (SOLO bootstrap, come nel test riuscito prima)")

    # Ora aggiungiamo le chiavi EXTRA per la convoluzione -- il pezzo
    # MAI testato insieme al bootstrap fino ad ora.
    rot_offsets = set()
    for kx in range(K):
        rot_offsets.add(kx * n)
    for d in range(1, n):
        rot_offsets.add(d)
        rot_offsets.add(d - n)
    rot_list = sorted(rot_offsets)
    print(f"\nGenerando {len(rot_list)} chiavi di rotazione EXTRA per la convoluzione...")
    t0 = time.time()
    cc.EvalRotateKeyGen(keys.secretKey, rot_list)
    print(f"EvalRotateKeyGen: {time.time()-t0:.1f}s")
    print_gpu_mem("dopo EvalRotateKeyGen (bootstrap + conv insieme -- IL PUNTO DA VERIFICARE)")

    cc.SetRotationKeyCache(CACHE_GIB * GiB)
    cc.SetBootstrapCache(CACHE_GIB * GiB)
    print_gpu_mem("dopo aver impostato le cache (ancora prima di LoadContext)")

    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"\nLoadContext: {time.time()-t0:.1f}s")
    print_gpu_mem("dopo LoadContext -- QUESTO E' IL NUMERO CHIAVE")

    cc.SetPlaintextCache(CACHE_GIB * GiB)
    cc.SetCiphertextCache(CACHE_GIB * GiB)
    print_gpu_mem("dopo le cache finali")

    print("\n=== Se sei arrivata fin qui, la generazione delle chiavi da sola NON e' il problema. ===")
    print("=== Il problema sarebbe allora nell'USO (le migliaia di operazioni della convoluzione). ===")


if __name__ == '__main__':
    main()