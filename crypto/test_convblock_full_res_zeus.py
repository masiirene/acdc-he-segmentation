"""
crypto/test_convblock_full_res_zeus.py

Test del ConvBlock (versione corretta, mascheramento O(1)) alla
RISOLUZIONE VERA (256x224), 2 canali -- stessa struttura del test a
64x56 di ieri (dove sono stati trovati e risolti due bug: convenzione
di posizione della regione valida, e padding asimmetrico in ingresso).

Parametri gia' confermati funzionanti su Zeus (A40, RingDim=2^17):
- depth=45 (oltre questo valore, crash 'illegal memory access' dentro
  EvalRotate -- limite ancora da investigare, non ancora risolto)
- Chebyshev leggero: degree=3, post_iter=1 (varianze reali fuori dal
  range calibrato [0.5,4.0] danno comunque un errore accettabile,
  ~0.01-0.18, con questi parametri)

Uso: python3 crypto/test_convblock_full_res_zeus.py
"""

import sys
import math
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import (
    poly_act_fhe, isqrt_chebyshev_fhe, fit_monotonic_isqrt_coeffs,
    conv2d_multichannel_fhe, mask_border_fhe_precomputed,
    make_border_mask_plaintext, instance_stats_padded_fhe,
    broadcast_slot0,
)


def build_context_and_keys(img_h, img_w, halo, K, depth=45, ring_pow=17,
                            rotation_key_cache_gib=4):
    t0 = time.time()
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
    cc.SetRotationKeyCache(rotation_key_cache_gib * 1024**3)

    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_total = img_hp * img_wp
    print(f"  n_total (con halo): {n_total} slot (batch max disponibile: {1 << (ring_pow-1)})")
    assert n_total <= (1 << (ring_pow - 1)), \
        f"L'immagine ({n_total} slot) non entra nel ciphertext ({1 << (ring_pow-1)} slot disponibili)!"

    rot = [1]
    rot += sorted(set(ky*img_wp + kx for ky in range(K) for kx in range(K)) - {0})

    n_pow2 = 1 << math.ceil(math.log2(n_total))
    rot += fhe.accumulate_rotation_indices(n_total, stride=1)
    step = 1
    while step < n_pow2:
        rot.append(-step)
        step *= 2

    unique_rot = sorted(set(r for r in rot if r != 0))
    print(f"  Chiavi di rotazione uniche da generare: {len(unique_rot)}")

    t1 = time.time()
    cc.EvalRotateKeyGen(keys.secretKey, unique_rot)
    t2 = time.time()
    cc.LoadContext(keys.publicKey)
    t3 = time.time()

    print(f"  Setup contesto: {t1-t0:.1f}s | EvalRotateKeyGen: {t2-t1:.1f}s | LoadContext: {t3-t2:.1f}s")

    return cc, keys


def conv_block_fhe_timed(cc, ct_channels_in,
                          conv1_w, conv1_b, gamma1, beta1,
                          conv2_w, conv2_b, gamma2, beta2,
                          cheb_coeffs, cheb_domain, post_iter,
                          a1, b1_, c1, a2, b2_, c2,
                          img_h, img_w, halo, K):
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_valid = img_h * img_w
    n_total = img_hp * img_wp
    Cout = conv1_w.shape[0]

    pt_mask = make_border_mask_plaintext(cc, img_hp, img_wp, img_h, img_w, halo)

    def masked(ct):
        return mask_border_fhe_precomputed(cc, ct, pt_mask)

    t0 = time.time()
    x1 = conv2d_multichannel_fhe(cc, ct_channels_in, conv1_w, conv1_b, img_hp, img_wp, K=K)
    t1 = time.time()
    print(f"    conv1 (multi-canale): {t1-t0:.2f}s")

    x1_out = []
    for co in range(Cout):
        tc0 = time.time()
        x1_masked = masked(x1[co])
        mean, var = instance_stats_padded_fhe(cc, x1_masked, n_valid, n_total)
        inv_std = isqrt_chebyshev_fhe(cc, var, cheb_coeffs, cheb_domain, post_iter)
        centered = cc.EvalSub(x1_masked, mean)
        normalized = cc.EvalMult(centered, inv_std)
        scaled = cc.EvalMult(normalized, gamma1[co])
        norm_out = cc.EvalAdd(scaled, beta1[co])
        act_out = poly_act_fhe(cc, norm_out, a1, b1_, c1)
        x1_out.append(masked(act_out))
        print(f"    canale {co} (blocco 1): {time.time()-tc0:.2f}s")

    t2 = time.time()
    x2 = conv2d_multichannel_fhe(cc, x1_out, conv2_w, conv2_b, img_hp, img_wp, K=K)
    t3 = time.time()
    print(f"    conv2 (multi-canale): {t3-t2:.2f}s")

    out = []
    for co in range(Cout):
        tc0 = time.time()
        x2_masked = masked(x2[co])
        mean, var = instance_stats_padded_fhe(cc, x2_masked, n_valid, n_total)
        inv_std = isqrt_chebyshev_fhe(cc, var, cheb_coeffs, cheb_domain, post_iter)
        centered = cc.EvalSub(x2_masked, mean)
        normalized = cc.EvalMult(centered, inv_std)
        scaled = cc.EvalMult(normalized, gamma2[co])
        norm_out = cc.EvalAdd(scaled, beta2[co])
        out.append(poly_act_fhe(cc, norm_out, a2, b2_, c2))
        print(f"    canale {co} (blocco 2): {time.time()-tc0:.2f}s")

    t4 = time.time()
    print(f"    TOTALE ConvBlock: {t4-t0:.2f}s")

    return out, img_hp, img_wp


def main():
    print("=== Test ConvBlock RISOLUZIONE VERA (256x224), 2 canali ===\n")

    img_h, img_w = 256, 224
    halo = 1
    K = 3
    Cin, Cout = 2, 2
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo

    print(f"Dimensione: {img_h}x{img_w} = {img_h*img_w} pixel validi per canale\n")

    print("Costruzione contesto e chiavi...")
    t_ctx_start = time.time()
    cc, keys = build_context_and_keys(img_h, img_w, halo, K, depth=45)
    print(f"Contesto pronto in {time.time()-t_ctx_start:.1f}s totali.\n")

    rng = np.random.default_rng(42)
    x = rng.normal(size=(Cin, img_h, img_w))
    # Padding ASIMMETRICO (top-left) -- fix scoperto ieri, essenziale
    x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))

    def encrypt_channel(arr2d):
        pt = cc.MakeCKKSPackedPlaintext(arr2d.flatten().tolist())
        return cc.Encrypt(keys.publicKey, pt)

    t0 = time.time()
    ct_channels_in = [encrypt_channel(x_padded[c]) for c in range(Cin)]
    print(f"Immagine cifrata in {time.time()-t0:.2f}s.\n")

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
    cheb_coeffs, shift = fit_monotonic_isqrt_coeffs(fhe, x_min_test, x_max_test, degree=3, extra_safety=1.2)
    cheb_domain = [x_min_test, x_max_test]
    post_iter = 1
    print(f"Coefficienti Chebyshev leggeri pronti (shift={shift:.4f}).\n")

    def conv_ref(x_p, w, b, Cout_, Cin_, H, W):
        out = np.zeros((Cout_, H, W))
        for co in range(Cout_):
            for ci in range(Cin_):
                for r in range(H):
                    for cc_ in range(W):
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

    print("Calcolo il riferimento numpy (potrebbe richiedere un minuto, ciclo puro Python)...")
    t_ref0 = time.time()
    x1_ref = conv_ref(x_padded, conv1_w, conv1_b, Cout, Cin, img_h, img_w)
    print(f"[Rif.] varianza x1[0]: {x1_ref[0].var():.4f} "
          f"{'(FUORI range)' if not (x_min_test <= x1_ref[0].var() <= x_max_test) else '(dentro range)'}")

    x1_norm_ref = np.stack([norm_ref(x1_ref[co], gamma1[co], beta1[co]) for co in range(Cout)])
    x1_act_ref = act_ref(x1_norm_ref, a1, b1, c1)
    x1_padded_ref = np.pad(x1_act_ref, ((0, 0), (0, 2*halo), (0, 2*halo)))

    x2_ref = conv_ref(x1_padded_ref, conv2_w, conv2_b, Cout, Cout, img_h, img_w)
    print(f"[Rif.] varianza x2[0]: {x2_ref[0].var():.4f} "
          f"{'(FUORI range)' if not (x_min_test <= x2_ref[0].var() <= x_max_test) else '(dentro range)'}")

    x2_norm_ref = np.stack([norm_ref(x2_ref[co], gamma2[co], beta2[co]) for co in range(Cout)])
    ref_out = act_ref(x2_norm_ref, a2, b2, c2)
    print(f"Riferimento numpy calcolato in {time.time()-t_ref0:.1f}s.\n")

    print("Eseguo il ConvBlock su ciphertext (risoluzione vera):\n")
    out_channels, out_hp, out_wp = conv_block_fhe_timed(
        cc, ct_channels_in,
        conv1_w, conv1_b, gamma1, beta1,
        conv2_w, conv2_b, gamma2, beta2,
        cheb_coeffs, cheb_domain, post_iter,
        a1, b1, c1, a2, b2, c2,
        img_h, img_w, halo, K)
    print()

    def decrypt_full(ct):
        pt = cc.Decrypt(keys.secretKey, ct)
        pt.SetLength(out_hp * out_wp)
        return np.array(pt.GetRealPackedValue()).reshape(out_hp, out_wp)

    he_out_full = [decrypt_full(ct) for ct in out_channels]
    he_out = [f[0:img_h, 0:img_w] for f in he_out_full]

    print(f"{'Canale':>8s} {'Errore max':>14s} {'Var HE':>10s} {'Var rif.':>10s}")
    for co in range(Cout):
        err = np.max(np.abs(he_out[co] - ref_out[co]))
        print(f"{co:8d} {err:14.6e} {he_out[co].var():10.4f} {ref_out[co].var():10.4f}")

    print("\n=== Se l'errore e' nell'ordine 0.01-0.2 (coerente col test 64x56 di ieri), ===")
    print("=== il ConvBlock regge anche alla risoluzione vera. ===")


if __name__ == '__main__':
    main()