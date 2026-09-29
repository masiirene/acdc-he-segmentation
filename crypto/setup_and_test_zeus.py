"""
crypto/setup_and_test_zeus.py

Script da lanciare per PRIMO su Zeus, appena la GPU e' libera:
1. Crea un contesto CKKS con N=2^17 (indicazione di Aurora), 65.536 slot.
2. Lancia il test diagnostico su broadcast_slot0 con n non potenza di 2
   (n=100) -- PRIMA di fidarsi della pipeline su un'immagine vera.
3. Se il test passa, prova un singolo ConvBlock su un'immagine piccola
   di prova (16x16, non ancora la vera 256x224) per verificare che
   tutto funzioni insieme prima di scalare alla dimensione reale.

Uso: python3 crypto/setup_and_test_zeus.py
"""

import sys
import numpy as np
sys.path.insert(0, '.')  # o il path corretto verso PyFIDESlib su Zeus

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import (
    poly_act_fhe, instance_stats_fhe, isqrt_chebyshev_fhe,
    fit_monotonic_isqrt_coeffs, instance_norm_fhe,
    conv2d_multichannel_fhe, crop_valid_fhe, embed_valid_into_padded_fhe,
    conv_block_fhe, broadcast_slot0, test_broadcast_nonpow2,
)


def build_context(depth=30, ring_pow=17):
    """
    N=2^17 (indicazione di Aurora): 65.536 slot per ciphertext,
    sufficienti per l'intera immagine 256x224=57.344 pixel in UN
    ciphertext, senza tiling.

    depth=30 e' un valore di partenza PRUDENTE per i primi test su
    Zeus (40GB VRAM) -- da alzare progressivamente seguendo lo stesso
    approccio bisezione gia' usato su Colab (tests/diag_bootstrap.py),
    ora con molto piu' margine di memoria a disposizione.
    """
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << ring_pow)
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(50)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetDevices([0])

    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    return cc, keys, params


def compute_rotation_indices_for_small_test(img_h, img_w, halo, K, n_pixels):
    """Chiavi di rotazione per un test piccolo (es. 16x16) -- calcolate
    in anticipo, come sempre richiesto dalla libreria prima di
    LoadContext."""
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    rot = [1]

    # Convoluzione K x K
    rot += sorted(set(ky*img_wp + kx for ky in range(K) for kx in range(K)) - {0})

    # Crop / embed (stride=1)
    for out_idx in range(img_h * img_w):
        r_out, c_out = divmod(out_idx, img_w)
        src_idx = r_out * img_wp + c_out
        rot.append(src_idx - out_idx)
    for r in range(img_h):
        for c in range(img_w):
            idx = r * img_w + c
            dest = (r + halo) * img_wp + (c + halo)
            rot.append(idx - dest)

    # AccumulateSum + broadcast (n_pixels, generalizzato a non-potenza-di-2)
    import math
    n_pow2 = 1 << math.ceil(math.log2(n_pixels))
    rot += fhe.accumulate_rotation_indices(n_pixels, stride=1)
    step = 1
    while step < n_pow2:
        rot.append(-step)
        step *= 2

    return sorted(set(r for r in rot if r != 0))


def main():
    print("=== Passo 1: crea il contesto (depth=30, N=2^17) ===")
    cc, keys, params = build_context(depth=30, ring_pow=17)
    print(f"Contesto creato (senza chiavi di rotazione ancora).\n")

    print("=== Passo 2: chiavi di rotazione per il test diagnostico (n=100) ===")
    import math
    n_test = 100
    n_pow2 = 1 << math.ceil(math.log2(n_test))
    rot_diag = fhe.accumulate_rotation_indices(n_test, stride=1)
    step = 1
    while step < n_pow2:
        rot_diag.append(-step)
        step *= 2
    cc.EvalRotateKeyGen(keys.secretKey, sorted(set(r for r in rot_diag if r != 0)))
    cc.LoadContext(keys.publicKey)
    print("Chiavi generate, contesto caricato.\n")

    print("=== Passo 3: test diagnostico broadcast_slot0 (n=100, non potenza di 2) ===")
    test_broadcast_nonpow2(cc, keys, fhe)
    print()

    print("=== FERMATI QUI e controlla il risultato del test sopra. ===")
    print("Se 'OK', procedi con un secondo script per il test del ConvBlock")
    print("su un'immagine piccola (16x16) prima di scalare a 256x224 vera.")


if __name__ == '__main__':
    main()