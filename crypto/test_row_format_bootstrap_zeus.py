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
CACHE_GIB = 1

# Da riga di comando, per fare uno sweep del SOLO bootstrap (minuti, non
# i 48 della convoluzione):
#   python3 crypto/test_row_format_bootstrap_zeus.py [CANALI [LB1 LB2 [BSGS1 BSGS2]]]
# BSGS 0 0 = lascia scegliere alla libreria. Caso che ha fallito:
#   python3 crypto/test_row_format_bootstrap_zeus.py 128 4 4 4 4
CHANNELS = int(sys.argv[1]) if len(sys.argv) > 1 else 128  # a 128 il bootstrap dopo la conv dava errore ~0.5 (prima del bootstrap: 5e-12)
LEVEL_BUDGET = [int(sys.argv[2]), int(sys.argv[3])] if len(sys.argv) > 3 else [4, 4]
BSGS_DIM = [int(sys.argv[4]), int(sys.argv[5])] if len(sys.argv) > 5 else [4, 4]
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

    def run_case(label, data):
        """Cifra, consuma 5 livelli, bootstrap, e analizza l'errore PER REGIONE
        (non solo il massimo): dati veri vs padding, primo/ultimo quarto dei
        blocchi-pixel, per capire se l'errore e' uniforme (rumore) o
        concentrato in una zona (problema strutturale)."""
        full = np.zeros(batch_size)
        full[:real_len] = data
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(full.tolist()))
        for _ in range(5):
            ct = cc.EvalMult(ct, 1.01)
        t0 = time.time()
        ct_boot = cc.EvalBootstrap(ct)
        print(f"\n[{label}] bootstrap in {time.time()-t0:.2f}s, livello dopo: {ct_boot.GetLevel()}", flush=True)
        pt_out = cc.Decrypt(keys.secretKey, ct_boot)
        pt_out.SetLength(batch_size)
        res = np.array(pt_out.GetRealPackedValue())
        expected = np.zeros(batch_size)
        expected[:real_len] = data * (1.01 ** 5)
        diff = np.abs(res - expected)
        print(f"[{label}] ampiezza dati: media|x|={np.mean(np.abs(data)):.3f}, max|x|={np.max(np.abs(data)):.3f}")
        print(f"[{label}] errore max sui dati veri: {diff[:real_len].max():.3e}  (medio {diff[:real_len].mean():.3e})")
        print(f"[{label}] errore max nel padding (atteso ~0): {diff[real_len:].max():.3e}")
        q = real_len // 4
        for i in range(4):
            seg = diff[i*q:(i+1)*q]
            print(f"[{label}]   quarto {i+1} dei dati: errore max {seg.max():.3e}")
        return diff[:real_len].max()

    # Caso A: dati "grandi" (come nei test precedenti riusciti a 32/64 canali)
    err_a = run_case("dati ~N(1,0.5)", rng.normal(size=real_len) * 0.5 + 1.0)
    # Caso B: dati "piccoli e centrati", come l'output vero della convoluzione
    # (pesi 0.1/sqrt(n*9), input N(0,1) -> output con std ~0.1)
    err_b = run_case("dati ~N(0,0.1) tipo conv", rng.normal(size=real_len) * 0.1)

    print_gpu_mem("fine")
    print(f"\nRIEPILOGO a {CHANNELS} canali (batch {batch_size}), level_budget={LEVEL_BUDGET}, bsgsDim={BSGS_DIM}: "
          f"errore caso A = {err_a:.3e}, caso B = {err_b:.3e}")
    print("Riferimento: a 32 e 64 canali il bootstrap in conv+boot dava ~5e-4.")


if __name__ == '__main__':
    main()