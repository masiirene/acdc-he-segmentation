"""
crypto/test_row_format_bootstrap_zeus.py

Passo successivo, prudente: il bootstrap vero (con tutto il lavoro di
ieri sera: bsgsDim=[4,4], Offload()/TrimGPUMemoryPool()) applicato al
NUOVO formato a righe (ieri il bootstrap lavorava su "un ciphertext
per canale, l'immagine intera dentro"; oggi il formato e' "un
ciphertext per riga, tutti i canali di quella riga dentro").

PRIMA larghezza REALE non giocattolo: Cin=Cout=32 (enc0/dec0, la piu'
stretta tra quelle vere) -- non ancora 256, un passo alla volta.

Questo test fa SOLO: cifra una riga, bootstrap, verifica che il
valore torni invariato (il bootstrap non dovrebbe cambiare il dato,
solo il livello/rumore). NON include ancora la convoluzione --
quello e' il passo successivo, una volta che questo regge.

Uso: python3 crypto/test_row_format_bootstrap_zeus.py
"""

import sys
import math
import time
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
LEVEL_BUDGET = [4, 4]
BSGS_DIM = [4, 4]
CACHE_GIB = 1

CHANNELS = 32  # larghezza vera di enc0/dec0 -- il primo passo, non 256
IMG_W = 224
HALO = 1
WP = IMG_W + 2 * HALO  # 226


def print_gpu_mem(label):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            text=True
        ).strip()
        used, total = out.split(',')
        print(f"    [GPU MEM] {label}: {used.strip()} / {total.strip()} MiB")
    except Exception as e:
        print(f"    [GPU MEM] {label}: impossibile leggere ({e})")


def build_context(batch_size):
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

    print("  Configurazione bootstrap (EvalBootstrapSetup + KeyGen)...")
    t0 = time.time()
    cc.EvalBootstrapSetup(LEVEL_BUDGET, BSGS_DIM, batch_size)
    print(f"    EvalBootstrapSetup (bsgsDim={BSGS_DIM}): {time.time()-t0:.1f}s")
    t0 = time.time()
    cc.EvalBootstrapKeyGen(keys.secretKey, batch_size)
    print(f"    EvalBootstrapKeyGen: {time.time()-t0:.1f}s")

    cc.SetRotationKeyCache(CACHE_GIB * GiB)
    cc.SetBootstrapCache(CACHE_GIB * GiB)
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s")
    cc.SetPlaintextCache(CACHE_GIB * GiB)
    cc.SetCiphertextCache(CACHE_GIB * GiB)

    return cc, keys


def main():
    real_len = WP * CHANNELS
    batch_size = 1 << math.ceil(math.log2(real_len))

    print(f"=== Bootstrap sul formato a righe, {CHANNELS} canali ===")
    print(f"Wp={WP}, canali={CHANNELS}, lunghezza reale={real_len}, batch (pot. 2)={batch_size}\n")

    print_gpu_mem("prima di costruire il contesto")
    t0 = time.time()
    cc, keys = build_context(batch_size)
    print(f"\nContesto pronto in {time.time()-t0:.1f}s totali.")
    print_gpu_mem("dopo LoadContext")

    rng = np.random.default_rng(3)
    data = rng.normal(size=real_len) * 0.5 + 1.0  # valori in un range ragionevole
    full = np.zeros(batch_size)
    full[:real_len] = data

    pt = cc.MakeCKKSPackedPlaintext(full.tolist())
    ct = cc.Encrypt(keys.publicKey, pt)
    print(f"\nRiga cifrata (livello {ct.GetLevel()}).")

    # Simuliamo un po' di "consumo" di profondita' prima del bootstrap
    # (altrimenti testiamo il bootstrap su un ciphertext troppo fresco,
    # non rappresentativo di dove servirebbe davvero nella pipeline)
    for _ in range(5):
        ct = cc.EvalMult(ct, 1.01)
    print(f"Dopo 5 moltiplicazioni di prova, livello: {ct.GetLevel()}")
    print_gpu_mem("prima del bootstrap")

    print("\nBootstrap...")
    t0 = time.time()
    ct_boot = cc.EvalBootstrap(ct)
    print(f"Completato in {time.time()-t0:.2f}s, livello dopo: {ct_boot.GetLevel()}")
    print_gpu_mem("dopo il bootstrap")

    pt_out = cc.Decrypt(keys.secretKey, ct_boot)
    pt_out.SetLength(real_len)
    result = np.array(pt_out.GetRealPackedValue())
    err = np.max(np.abs(result - data * (1.01 ** 5)))
    print(f"\nErrore massimo dopo bootstrap: {err:.6e}")

    if err < 1e-2:
        print("\n=== Bootstrap funziona correttamente sul formato a righe, larghezza reale. ===")
    else:
        print("\n=== ATTENZIONE: errore grande, da investigare. ===")


if __name__ == '__main__':
    main()