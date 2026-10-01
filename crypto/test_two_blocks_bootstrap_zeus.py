"""
crypto/test_two_blocks_bootstrap_zeus.py

Come test_two_blocks_skip_zeus.py, ma con un VERO EvalBootstrap() sui
ciphertext dopo la skip connection, prima di ConvBlock B -- dato che un
solo ConvBlock consuma ~35 livelli su un budget stabile di 45, due
blocchi in sequenza SENZA bootstrap esauriscono la profondita' (visto
ieri: segfault in ConvBlock B). Il bootstrap resetta il livello,
permettendo di incatenare blocchi arbitrariamente.

Ricetta di parametri (confermata funzionante fino a depth=45, sia su
2^17 che 2^18): ScalingModSize=59, FirstModSize=60, NumLargeDigits=3
-- presa da tests/diag_bootstrap.py.

Uso: python3 crypto/test_two_blocks_bootstrap_zeus.py
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
    skip_connection_sum_fhe,
)

DEPTH = 30
RING_POW = 17
LEVEL_BUDGET = [1, 1]


def build_context_with_bootstrap(img_h, img_w, halo, K, rotation_key_cache_gib=16):
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_total = img_hp * img_wp
    batch = 1 << (RING_POW - 1)  # slot totali disponibili (N/2)
    assert n_total <= batch, f"{n_total} > {batch}, l'immagine non entra"

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
    cc.SetRotationKeyCache(rotation_key_cache_gib * 1024**3)

    print("  Configurazione bootstrap (EvalBootstrapSetup + KeyGen)...")
    t0 = time.time()
    cc.EvalBootstrapSetup(LEVEL_BUDGET, [0, 0], batch)
    print(f"    EvalBootstrapSetup: {time.time()-t0:.1f}s")
    t0 = time.time()
    cc.EvalBootstrapKeyGen(keys.secretKey, batch)
    print(f"    EvalBootstrapKeyGen: {time.time()-t0:.1f}s")

    print("  Chiavi di rotazione per conv/norm...")
    rot = [1]
    rot += sorted(set(ky*img_wp + kx for ky in range(K) for kx in range(K)) - {0})
    n_pow2 = 1 << math.ceil(math.log2(n_total))
    rot += fhe.accumulate_rotation_indices(n_total, stride=1)
    step = 1
    while step < n_pow2:
        rot.append(-step)
        step *= 2
    unique_rot = sorted(set(r for r in rot if r != 0))
    print(f"    Chiavi extra (non-bootstrap): {len(unique_rot)}")
    t0 = time.time()
    cc.EvalRotateKeyGen(keys.secretKey, unique_rot)
    print(f"    EvalRotateKeyGen: {time.time()-t0:.1f}s")

    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s")

    return cc, keys, batch


def conv_block_fhe(cc, ct_channels_in, pt_mask,
                    conv1_w, conv1_b, gamma1, beta1,
                    conv2_w, conv2_b, gamma2, beta2,
                    cheb_coeffs, cheb_domain, post_iter,
                    a1, b1_, c1, a2, b2_, c2,
                    img_h, img_w, halo, K):
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_valid = img_h * img_w
    n_total = img_hp * img_wp
    Cout = conv1_w.shape[0]

    def masked(ct):
        return mask_border_fhe_precomputed(cc, ct, pt_mask)

    x1 = conv2d_multichannel_fhe(cc, ct_channels_in, conv1_w, conv1_b, img_hp, img_wp, K=K)
    x1_out = []
    for co in range(Cout):
        x1_masked = masked(x1[co])
        mean, var = instance_stats_padded_fhe(cc, x1_masked, n_valid, n_total)
        inv_std = isqrt_chebyshev_fhe(cc, var, cheb_coeffs, cheb_domain, post_iter)
        centered = cc.EvalSub(x1_masked, mean)
        normalized = cc.EvalMult(centered, inv_std)
        scaled = cc.EvalMult(normalized, gamma1[co])
        norm_out = cc.EvalAdd(scaled, beta1[co])
        act_out = poly_act_fhe(cc, norm_out, a1, b1_, c1)
        x1_out.append(masked(act_out))

    x2 = conv2d_multichannel_fhe(cc, x1_out, conv2_w, conv2_b, img_hp, img_wp, K=K)
    out = []
    for co in range(Cout):
        x2_masked = masked(x2[co])
        mean, var = instance_stats_padded_fhe(cc, x2_masked, n_valid, n_total)
        inv_std = isqrt_chebyshev_fhe(cc, var, cheb_coeffs, cheb_domain, post_iter)
        centered = cc.EvalSub(x2_masked, mean)
        normalized = cc.EvalMult(centered, inv_std)
        scaled = cc.EvalMult(normalized, gamma2[co])
        norm_out = cc.EvalAdd(scaled, beta2[co])
        out.append(poly_act_fhe(cc, norm_out, a2, b2_, c2))

    return out


def conv_ref(x_p, w, b, Cout_, Cin_, H, W, K):
    out = np.zeros((Cout_, H, W))
    for co in range(Cout_):
        for ci in range(Cin_):
            for r in range(H):
                for c in range(W):
                    for ky in range(K):
                        for kx in range(K):
                            out[co, r, c] += w[co, ci, ky, kx] * x_p[ci, r+ky, c+kx]
        out[co] += b[co]
    return out


def norm_ref(x_, gamma, beta):
    mean, var = x_.mean(), x_.var()
    return gamma * (x_ - mean) / np.sqrt(var + 1e-5) + beta


def act_ref(x_, a, b, c):
    return a*x_*x_ + b*x_ + c


def main():
    print("=== Test: 2 ConvBlock + skip + BOOTSTRAP VERO tra i due ===\n")

    img_h, img_w = 256, 224
    halo = 1
    K = 3
    Cin, Cout = 2, 2
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo

    print("Costruzione contesto (con bootstrap configurato)...")
    t0 = time.time()
    cc, keys, batch = build_context_with_bootstrap(img_h, img_w, halo, K)
    print(f"Contesto pronto in {time.time()-t0:.1f}s totali.\n")

    pt_mask = make_border_mask_plaintext(cc, img_hp, img_wp, img_h, img_w, halo)

    rng = np.random.default_rng(11)
    x = rng.normal(size=(Cin, img_h, img_w))
    x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))
    skip_data = rng.normal(size=(Cout, img_h, img_w)) * 0.5

    def enc(a):
        # padding a 'batch' slot totali (il resto zero)
        flat = a.flatten().tolist()
        return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(flat))

    def dec(ct, n):
        pt = cc.Decrypt(keys.secretKey, ct)
        pt.SetLength(n)
        return np.array(pt.GetRealPackedValue())

    t0 = time.time()
    ct_in = [enc(x_padded[c]) for c in range(Cin)]
    skip_padded = np.pad(skip_data, ((0, 0), (0, 2*halo), (0, 2*halo)))
    ct_skip = [enc(skip_padded[c]) for c in range(Cout)]
    print(f"Cifratura input completata in {time.time()-t0:.2f}s.\n")

    conv1_w = rng.normal(size=(Cout, Cin, K, K)) * 0.2
    conv1_b = rng.normal(size=(Cout,)) * 0.05
    conv2_w = rng.normal(size=(Cout, Cout, K, K)) * 0.2
    conv2_b = rng.normal(size=(Cout,)) * 0.05
    conv3_w = rng.normal(size=(Cout, Cout, K, K)) * 0.2
    conv3_b = rng.normal(size=(Cout,)) * 0.05
    conv4_w = rng.normal(size=(Cout, Cout, K, K)) * 0.2
    conv4_b = rng.normal(size=(Cout,)) * 0.05
    gamma = [1.1, 0.9]
    beta = [0.1, -0.1]
    a_, b_, c_ = 0.1, 1.0, 0.5

    x_min_test, x_max_test = 0.5, 4.0
    cheb_coeffs, shift = fit_monotonic_isqrt_coeffs(fhe, x_min_test, x_max_test, degree=2, extra_safety=1.1)
    cheb_domain = [x_min_test, x_max_test]
    post_iter = 1
    print(f"Coefficienti Chebyshev leggeri pronti (shift={shift:.4f}).\n")

    print("=== ConvBlock A (stride=1) ===")
    t0 = time.time()
    out_A = conv_block_fhe(cc, ct_in, pt_mask,
                            conv1_w, conv1_b, gamma, beta,
                            conv2_w, conv2_b, gamma, beta,
                            cheb_coeffs, cheb_domain, post_iter,
                            a_, b_, c_, a_, b_, c_,
                            img_h, img_w, halo, K)
    print(f"ConvBlock A completato in {time.time()-t0:.2f}s.")
    print(f"DEBUG livello out_A[0]: {out_A[0].GetLevel()}\n")

    print("=== Skip connection ===")
    out_A_masked = [mask_border_fhe_precomputed(cc, ct, pt_mask) for ct in out_A]
    ct_skip_masked = [mask_border_fhe_precomputed(cc, ct, pt_mask) for ct in ct_skip]

    # Allinea lo skip (fresco) al livello di out_A (consumato) prima di sommare
    level_deep = out_A_masked[0].GetLevel()
    for co in range(Cout):
        gap = level_deep - ct_skip_masked[co].GetLevel()
        for _ in range(gap):
            ct_skip_masked[co] = cc.EvalMult(ct_skip_masked[co], 1.0)

    summed = skip_connection_sum_fhe(cc, out_A_masked, ct_skip_masked)
    print(f"DEBUG livello summed[0] prima del bootstrap: {summed[0].GetLevel()}\n")

    print("=== BOOTSTRAP sui 2 canali ===")
    bootstrapped = []
    for co in range(Cout):
        t0 = time.time()
        ct_boot = cc.EvalBootstrap(summed[co])
        print(f"  Canale {co}: bootstrap in {time.time()-t0:.2f}s, "
              f"livello dopo: {ct_boot.GetLevel()}")
        bootstrapped.append(ct_boot)
    print()

    print("=== ConvBlock B (stride=1), dopo il bootstrap ===")
    t0 = time.time()
    out_B = conv_block_fhe(cc, bootstrapped, pt_mask,
                            conv3_w, conv3_b, gamma, beta,
                            conv4_w, conv4_b, gamma, beta,
                            cheb_coeffs, cheb_domain, post_iter,
                            a_, b_, c_, a_, b_, c_,
                            img_h, img_w, halo, K)
    print(f"ConvBlock B completato in {time.time()-t0:.2f}s.\n")

    print("Decifrazione e confronto col riferimento numpy...")
    he_out = [dec(ct, img_h*img_w).reshape(img_h, img_w) for ct in out_B]

    def block_ref(x_p, w1, b1, w2, b2, g, be, a, bb, c, H, W, Cin_, Cout_):
        y1 = conv_ref(x_p, w1, b1, Cout_, Cin_, H, W, K)
        y1n = np.stack([norm_ref(y1[co], g[co], be[co]) for co in range(Cout_)])
        y1a = act_ref(y1n, a, bb, c)
        y1p = np.pad(y1a, ((0, 0), (0, 2*halo), (0, 2*halo)))
        y2 = conv_ref(y1p, w2, b2, Cout_, Cout_, H, W, K)
        y2n = np.stack([norm_ref(y2[co], g[co], be[co]) for co in range(Cout_)])
        return act_ref(y2n, a, bb, c)

    print("Calcolo riferimento numpy...")
    t0 = time.time()
    ref_A = block_ref(x_padded, conv1_w, conv1_b, conv2_w, conv2_b, gamma, beta,
                       a_, b_, c_, img_h, img_w, Cin, Cout)
    ref_summed = ref_A + skip_data
    ref_summed_padded = np.pad(ref_summed, ((0, 0), (0, 2*halo), (0, 2*halo)))
    ref_B = block_ref(ref_summed_padded, conv3_w, conv3_b, conv4_w, conv4_b, gamma, beta,
                       a_, b_, c_, img_h, img_w, Cout, Cout)
    print(f"Riferimento calcolato in {time.time()-t0:.1f}s.\n")

    print(f"{'Canale':>8s} {'Errore max':>14s} {'Var HE':>10s} {'Var rif.':>10s}")
    for co in range(Cout):
        err = np.max(np.abs(he_out[co] - ref_B[co]))
        print(f"{co:8d} {err:14.6e} {he_out[co].var():10.4f} {ref_B[co].var():10.4f}")

    print("\n=== Se l'errore e' piccolo (0.01-0.5, coerente coi test precedenti), ===")
    print("=== il bootstrap tra blocchi FUNZIONA nella pipeline vera. ===")


if __name__ == '__main__':
    main()