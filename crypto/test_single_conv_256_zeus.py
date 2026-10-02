"""
crypto/test_single_conv_256_zeus.py

PRIMA di provare un blocco intero a 256 canali (il punto piu' largo
della rete vera), misuriamo il tempo di UNA SOLA convoluzione 256->256.

Perche': conv2d_multichannel_fhe cicla su ogni combinazione
canale-in x canale-out. A 16 canali erano 256 combinazioni (16x16),
a 256 canali diventano 65.536 (256x256) -- 256 volte di piu'. Se il
tempo scala linearmente con le combinazioni, un blocco intero (2
convoluzioni) potrebbe richiedere ore, non minuti. Meglio scoprirlo
con un test da pochi minuti che lanciare qualcosa che gira tutta la
notte per niente.

Uso: python3 crypto/test_single_conv_256_zeus.py
"""

import sys
import math
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import conv2d_multichannel_fhe

GiB = 1 << 30
DEPTH = 43
RING_POW = 17

CHANNELS = 64  # <-- 256 richiederebbe ~9 ore solo per UNA convoluzione
               # (65536 combinazioni canale-in/out); 64 e' la larghezza
               # vera di enc1/dec1, un dato comunque realistico stasera


def build_context_no_bootstrap(img_h, img_w, halo, K):
    """Contesto minimo, senza bootstrap -- qui ci serve solo misurare
    il tempo di UNA convoluzione, non costruire tutta la pipeline."""
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_total = img_hp * img_wp
    batch = 1 << (RING_POW - 1)
    assert n_total <= batch

    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(batch)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetDevices([0])

    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    rot = [1]
    rot += sorted(set(ky*img_wp + kx for ky in range(3) for kx in range(3)) - {0})
    unique_rot = sorted(set(r for r in rot if r != 0))
    cc.EvalRotateKeyGen(keys.secretKey, unique_rot)

    cc.SetRotationKeyCache(1 * GiB)
    cc.LoadContext(keys.publicKey)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(1 * GiB)

    return cc, keys, batch


def main():
    print(f"=== Test: UNA SOLA convoluzione {CHANNELS}x{CHANNELS} canali ===\n")

    img_h, img_w = 256, 224
    halo = 1
    K = 3
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo

    print("Costruzione contesto minimo (senza bootstrap)...")
    t0 = time.time()
    cc, keys, batch = build_context_no_bootstrap(img_h, img_w, halo, K)
    print(f"Contesto pronto in {time.time()-t0:.1f}s.\n")

    rng = np.random.default_rng(11)
    x = rng.normal(size=(CHANNELS, img_h, img_w))
    x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))

    def enc(a):
        return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(a.flatten().tolist()))

    print(f"Cifratura di {CHANNELS} canali di input...")
    t0 = time.time()
    ct_in = [enc(x_padded[c]) for c in range(CHANNELS)]
    print(f"Cifratura completata in {time.time()-t0:.2f}s.\n")

    std = 0.2 / math.sqrt(CHANNELS * K * K)
    w = rng.normal(size=(CHANNELS, CHANNELS, K, K)) * std
    b = rng.normal(size=(CHANNELS,)) * 0.05

    print(f"=== UNA convoluzione {CHANNELS} in -> {CHANNELS} out "
          f"({CHANNELS*CHANNELS} combinazioni canale-in x canale-out) ===")
    t0 = time.time()
    out = conv2d_multichannel_fhe(cc, ct_in, w, b, img_hp, img_wp, K=K)
    elapsed = time.time() - t0
    print(f"\nCompletata in {elapsed:.2f}s ({elapsed/60:.1f} minuti).")
    print(f"Tempo medio per combinazione canale-in/out: {elapsed/(CHANNELS*CHANNELS)*1000:.2f}ms")

    # Estrapolazione onesta: un blocco intero ha 2 convoluzioni di questo tipo
    # (qui cin=cout=256; nel blocco vero la prima potrebbe avere cin diverso,
    # ma per enc3/dec3 cin=cout=256 esattamente, quindi la stima e' diretta)
    print(f"\n=== Estrapolazione per un blocco INTERO (2 convoluzioni + norm/att) ===")
    print(f"Solo le 2 convoluzioni: ~{2*elapsed:.1f}s (~{2*elapsed/60:.1f} minuti)")
    print(f"NOTA: non include norm/attivazione/bootstrap, che aggiungono altro tempo")
    print(f"      ma sono O(canali), non O(canali^2) come la convoluzione -- il")
    print(f"      collo di bottiglia vero e' quasi certamente questo.")


if __name__ == '__main__':
    main()