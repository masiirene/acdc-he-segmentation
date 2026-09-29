"""
crypto/test_convblock_small_zeus.py

Secondo test su Zeus: un intero ConvBlock (Conv->Norm->Act->Conv->Norm
->Act), versione a singolo ciphertext, su un'immagine PICCOLA (16x16,
2 canali) -- prima di scalare alla vera 256x224. Verifica end-to-end
contro un riferimento numpy, stessa metodologia di sempre.

Uso: python3 crypto/test_convblock_small_zeus.py
"""

import sys
import math
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import (
    poly_act_fhe, instance_stats_fhe, isqrt_chebyshev_fhe,
    fit_monotonic_isqrt_coeffs, instance_norm_fhe,
    conv2d_multichannel_fhe, crop_valid_fhe, embed_valid_into_padded_fhe,
    conv_block_fhe, broadcast_slot0,
)


def build_context_and_keys(img_h, img_w, halo, K, n_pixels, depth=30, ring_pow=17):
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

    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    rot = [1]

    # Convoluzione K x K
    rot += sorted(set(ky*img_wp + kx for ky in range(K) for kx in range(K)) - {0})

    # Crop / embed (stride=1) -- usati due volte (dopo ogni conv)
    for out_idx in range(img_h * img_w):
        r_out, c_out = divmod(out_idx, img_w)
        src_idx = r_out * img_wp + c_out
        rot.append(src_idx - out_idx)
    for r in range(img_h):
        for c in range(img_w):
            idx = r * img_w + c
            dest = (r + halo) * img_wp + (c + halo)
            rot.append(idx - dest)

    # AccumulateSum + broadcast generalizzato (n_pixels non potenza di 2)
    n_pow2 = 1 << math.ceil(math.log2(n_pixels))
    rot += fhe.accumulate_rotation_indices(n_pixels, stride=1)
    step = 1
    while step < n_pow2:
        rot.append(-step)
        step *= 2

    cc.EvalRotateKeyGen(keys.secretKey, sorted(set(r for r in rot if r != 0)))
    cc.LoadContext(keys.publicKey)

    return cc, keys


def main():
    print("=== Test ConvBlock completo, immagine piccola 16x16, 2 canali ===\n")

    img_h, img_w = 16, 16
    halo = 1
    K = 3
    Cin, Cout = 2, 2
    n_pixels = img_h * img_w  # 256, gia' potenza di 2 -- il caso facile
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo

    print("Costruzione contesto e chiavi...")
    cc, keys = build_context_and_keys(img_h, img_w, halo, K, n_pixels, depth=30)
    print("Contesto pronto.\n")

    rng = np.random.default_rng(42)
    x = rng.normal(size=(Cin, img_h, img_w))
    x_padded = np.pad(x, ((0, 0), (halo, halo), (halo, halo)))

    def encrypt_channel(arr2d):
        pt = cc.MakeCKKSPackedPlaintext(arr2d.flatten().tolist())
        return cc.Encrypt(keys.publicKey, pt)

    ct_channels_in = [encrypt_channel(x_padded[c]) for c in range(Cin)]
    print("Immagine di test cifrata (2 canali, 16x16 + halo).\n")

    conv1_w = rng.normal(size=(Cout, Cin, K, K)) * 0.2
    conv1_b = rng.normal(size=(Cout,)) * 0.05
    conv2_w = rng.normal(size=(Cout, Cout, K, K)) * 0.2
    conv2_b = rng.normal(size=(Cout,)) * 0.05
    gamma1 = [1.1, 0.9]
    beta1 = [0.1, -0.1]
    gamma2 = [1.0, 1.0]
    beta2 = [0.0, 0.0]
    a1, b1, c1 = 0.1, 1.0, 0.5
    a2, b2, c2 = 0.1, 1.0, 0.5

    x_min_test, x_max_test = 0.5, 4.0
    cheb_coeffs, shift = fit_monotonic_isqrt_coeffs(fhe, x_min_test, x_max_test, degree=4, extra_safety=1.2)
    cheb_domain = [x_min_test, x_max_test]
    post_iter = 2
    print(f"Coefficienti Chebyshev pronti (shift={shift:.4f}).\n")

    print("Eseguo il ConvBlock su ciphertext (potrebbe richiedere qualche secondo)...")
    out_channels = conv_block_fhe(
        cc, ct_channels_in,
        conv1_w, conv1_b, gamma1, beta1,
        conv2_w, conv2_b, gamma2, beta2,
        cheb_coeffs, cheb_domain, post_iter,
        a1, b1, c1, a2, b2, c2,
        img_h, img_w, halo=halo, K=K)
    print("ConvBlock completato.\n")

    def decrypt_compact(ct):
        pt = cc.Decrypt(keys.secretKey, ct)
        pt.SetLength(img_h * img_w)
        return np.array(pt.GetRealPackedValue()).reshape(img_h, img_w)

    he_out = [decrypt_compact(ct) for ct in out_channels]

    # --- Riferimento numpy ---
    def conv_ref(x_p, w, b, Cout_, Cin_):
        out = np.zeros((Cout_, img_h, img_w))
        for co in range(Cout_):
            for ci in range(Cin_):
                for r in range(img_h):
                    for cc_ in range(img_w):
                        for ky in range(K):
                            for kx in range(K):
                                out[co, r, cc_] += w[co, ci, ky, kx] * x_p[ci, r+ky, cc_+kx]
            out[co] += b[co]
        return out

    def norm_ref(x_, gamma, beta):
        mean, var = x_.mean(), x_.var()
        return gamma * (x_ - mean) / np.sqrt(var + 1e-5) + beta

    def act_ref(x_, a, b, c):
        return a*x_*x_ + b*x_ + c

    x1 = conv_ref(x_padded, conv1_w, conv1_b, Cout, Cin)
    x1_norm = np.stack([norm_ref(x1[co], gamma1[co], beta1[co]) for co in range(Cout)])
    x1_act = act_ref(x1_norm, a1, b1, c1)
    x1_padded = np.pad(x1_act, ((0, 0), (halo, halo), (halo, halo)))
    x2 = conv_ref(x1_padded, conv2_w, conv2_b, Cout, Cout)
    x2_norm = np.stack([norm_ref(x2[co], gamma2[co], beta2[co]) for co in range(Cout)])
    ref_out = act_ref(x2_norm, a2, b2, c2)

    print(f"{'Canale':>8s} {'Errore max':>14s} {'Varianza x1':>14s} {'Varianza x2':>14s}")
    all_ok = True
    for co in range(Cout):
        err = np.max(np.abs(he_out[co] - ref_out[co]))
        var1 = x1[co].var()
        var2 = x2[co].var()
        flag = "" if err < 0.1 else "  <-- CONTROLLA (varianza fuori range calibrato [0.5,4.0]?)"
        print(f"{co:8d} {err:14.6e} {var1:14.4f} {var2:14.4f}{flag}")
        if err >= 0.1:
            all_ok = False

    print()
    if all_ok:
        print("=== OK: ConvBlock a singolo ciphertext verificato su 16x16. ===")
        print("Prossimo passo: scalare a 256x224 (l'immagine vera).")
    else:
        print("=== Errore alto su almeno un canale -- controlla la varianza vs il range calibrato. ===")


if __name__ == '__main__':
    main()