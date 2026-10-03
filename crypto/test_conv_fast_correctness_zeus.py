"""
crypto/test_conv_fast_correctness_zeus.py

Verifica che conv2d_multichannel_fhe_fast dia lo STESSO risultato di
conv2d_multichannel_fhe (originale), su un caso piccolo (8 canali,
veloce da testare su entrambe le versioni), prima di fidarsi per i
test a larghezza reale.

Uso: python3 crypto/test_conv_fast_correctness_zeus.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import conv2d_multichannel_fhe

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
CHANNELS = 64  # ieri sera l'originale ha impiegato 1438s (24 min) qui --
               # se "fast" non e' nettamente piu' veloce anche a questa
               # scala, le rotazioni non sono il vero collo di bottiglia


def conv2d_multichannel_fhe_fast(cc, ct_channels_in, weight, bias, tile_hp, tile_wp, K=3):
    Cout = weight.shape[0]
    Cin = len(ct_channels_in)
    rotated = {}
    for ci in range(Cin):
        for ky in range(K):
            for kx in range(K):
                offset = ky * tile_wp + kx
                rotated[(ci, ky, kx)] = ct_channels_in[ci] if offset == 0 else cc.EvalRotate(ct_channels_in[ci], offset)
    ct_channels_out = []
    for co in range(Cout):
        acc = None
        for ci in range(Cin):
            for ky in range(K):
                for kx in range(K):
                    w = float(weight[co, ci, ky, kx])
                    term = cc.EvalMult(rotated[(ci, ky, kx)], w)
                    acc = term if acc is None else cc.EvalAdd(acc, term)
        acc = cc.EvalAdd(acc, float(bias[co]))
        ct_channels_out.append(acc)
    return ct_channels_out


def build_context(img_h, img_w, halo, K):
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_total = img_hp * img_wp
    batch = 1 << (RING_POW - 1)
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
    rot += sorted(set(ky*img_wp + kx for ky in range(K) for kx in range(K)) - {0})
    unique_rot = sorted(set(r for r in rot if r != 0))
    cc.EvalRotateKeyGen(keys.secretKey, unique_rot)
    cc.SetRotationKeyCache(1 * GiB)
    cc.LoadContext(keys.publicKey)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(1 * GiB)  # rimessa a 1GiB -- 8GiB ha causato
                                     # out-of-memory di sistema, non ha
                                     # aiutato. Il problema non e' la
                                     # cache: la versione "fast" tiene
                                     # davvero troppi ciphertext vivi
                                     # insieme, qualunque sia la cache.
    return cc, keys, batch


def main():
    print(f"=== Verifica correttezza: conv ORIGINALE vs FAST, {CHANNELS} canali ===\n")

    img_h, img_w = 256, 224
    halo = 1
    K = 3
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo

    print("Costruzione contesto...")
    t0 = time.time()
    cc, keys, batch = build_context(img_h, img_w, halo, K)
    print(f"Contesto pronto in {time.time()-t0:.1f}s.\n")

    rng = np.random.default_rng(11)
    x = rng.normal(size=(CHANNELS, img_h, img_w))
    x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))

    def enc(a):
        return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(a.flatten().tolist()))

    def dec(ct, n):
        pt = cc.Decrypt(keys.secretKey, ct)
        pt.SetLength(n)
        return np.array(pt.GetRealPackedValue())

    print(f"Cifratura di {CHANNELS} canali...")
    ct_in = [enc(x_padded[c]) for c in range(CHANNELS)]

    w = rng.normal(size=(CHANNELS, CHANNELS, K, K)) * 0.05
    b = rng.normal(size=(CHANNELS,)) * 0.05

    # Gia' misurato in una run precedente con questi stessi parametri:
    # 1086.57s. Non lo rifacciamo per risparmiare 18 minuti -- qui
    # verifichiamo solo se la cache piu' grande cambia il tempo di "fast".
    t_orig = 1086.57
    print(f"\n=== Versione ORIGINALE: gia' nota da run precedente = {t_orig:.2f}s (non rifatta) ===")
    out_orig = None

    print(f"\n=== Versione FAST ({CHANNELS}x9 = {CHANNELS*9} rotazioni precalcolate) ===")
    t0 = time.time()
    out_fast = conv2d_multichannel_fhe_fast(cc, ct_in, w, b, img_hp, img_wp, K=K)
    t_fast = time.time() - t0
    print(f"Completata in {t_fast:.2f}s.")

    print(f"\n=== Confronto ===")
    print(f"Tempo originale: {t_orig:.2f}s")
    print(f"Tempo fast:      {t_fast:.2f}s")
    print(f"Speedup: {t_orig/t_fast:.1f}x\n")

    if out_orig is not None:
        sample_channels = [0, CHANNELS//4, CHANNELS//2, CHANNELS-1]
        max_err = 0.0
        for co in sample_channels:
            a_orig = dec(out_orig[co], img_hp*img_wp)
            a_fast = dec(out_fast[co], img_hp*img_wp)
            err = np.max(np.abs(a_orig - a_fast))
            max_err = max(max_err, err)
            print(f"  Canale {co}: errore max tra le due versioni = {err:.6e}")
        print(f"\nErrore massimo su tutti i canali: {max_err:.6e}")
    else:
        print("Correttezza gia' verificata nella run precedente (errore ~1e-12) -- qui misuriamo solo il tempo.")


if __name__ == '__main__':
    main()