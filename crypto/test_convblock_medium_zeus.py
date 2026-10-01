"""
crypto/test_convblock_medium_zeus.py (v5)

FIX METODOLOGICO IMPORTANTE rispetto a v3/v4: tutti i debug precedenti
leggevano gli slot [0:5], che nel layout row-major con halo sono
SEMPRE la prima riga dell'immagine con padding -- cioe' SEMPRE bordo,
mai un pixel valido interno. Il "quasi costante" osservato era un
artefatto del bordo (gia' azzerato o comunque non rappresentativo),
non necessariamente il vero bug.

Qui si decifra l'INTERO array a ogni checkpoint e si stampa una
piccola patch di pixel INTERNI (lontani dal bordo), confrontata
pixel-per-pixel col riferimento numpy alla STESSA posizione -- questo
localizza esattamente dove, nella catena, la variazione spaziale vera
si perde (se si perde).

Uso: python3 crypto/test_convblock_medium_zeus.py
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
                            rotation_key_cache_gib=8):
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


def debug_valid_patch(cc, keys, ct, img_hp, img_wp, halo, label, ref_2d=None,
                       patch_r=5, patch_c=5, patch_size=3):
    """
    Decifra l'INTERO ciphertext, lo mette in forma (img_hp, img_wp), e
    stampa una piccola patch di pixel INTERNI (partendo da patch_r,
    patch_c dentro la regione valida, non dal bordo) -- opzionalmente
    confrontata con la stessa patch del riferimento numpy (ref_2d, gia'
    nella sola regione valida img_h x img_w, senza halo).
    """
    n_total = img_hp * img_wp
    pt = cc.Decrypt(keys.secretKey, ct)
    pt.SetLength(n_total)
    arr = np.array(pt.GetRealPackedValue()).reshape(img_hp, img_wp)

    r0, c0 = patch_r, patch_c
    he_patch = arr[r0:r0+patch_size, c0:c0+patch_size]
    print(f"      DEBUG {label} -- patch HE (pixel validi, righe {r0}:{r0+patch_size}):")
    print(f"        {he_patch}")

    if ref_2d is not None:
        ref_patch = ref_2d[patch_r:patch_r+patch_size, patch_c:patch_c+patch_size]
        print(f"        Riferimento numpy alla stessa posizione:")
        print(f"        {ref_patch}")
        print(f"        Errore max su questa patch: {np.max(np.abs(he_patch - ref_patch)):.6e}")

    return arr


def conv_block_fhe_timed(cc, keys, ct_channels_in,
                          conv1_w, conv1_b, gamma1, beta1,
                          conv2_w, conv2_b, gamma2, beta2,
                          cheb_coeffs, cheb_domain, post_iter,
                          a1, b1_, c1, a2, b2_, c2,
                          img_h, img_w, halo, K,
                          x1_ref=None, x1_act_ref=None, x2_ref=None):
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_valid = img_h * img_w
    n_total = img_hp * img_wp
    Cout = conv1_w.shape[0]

    pt_mask = make_border_mask_plaintext(cc, img_hp, img_wp, img_h, img_w, halo)

    def masked(ct):
        return mask_border_fhe_precomputed(cc, ct, pt_mask)

    # ================= PRIMO BLOCCO =================
    t0 = time.time()
    x1 = conv2d_multichannel_fhe(cc, ct_channels_in, conv1_w, conv1_b, img_hp, img_wp, K=K)
    t1 = time.time()
    print(f"    conv1 (multi-canale): {t1-t0:.2f}s")

    print("\n  --- CHECKPOINT: x1 (uscita conv1, prima di norm), canale 0 ---")
    debug_valid_patch(cc, keys, x1[0], img_hp, img_wp, halo, "x1[0]",
                       ref_2d=x1_ref[0] if x1_ref is not None else None)

    x1_out = []
    for co in range(Cout):
        tc0 = time.time()
        x1_masked = masked(x1[co])
        tc1 = time.time()
        mean, var = instance_stats_padded_fhe(cc, x1_masked, n_valid, n_total)
        tc2 = time.time()
        inv_std = isqrt_chebyshev_fhe(cc, var, cheb_coeffs, cheb_domain, post_iter)
        tc3 = time.time()
        centered = cc.EvalSub(x1_masked, mean)
        normalized = cc.EvalMult(centered, inv_std)
        scaled = cc.EvalMult(normalized, gamma1[co])
        norm_out = cc.EvalAdd(scaled, beta1[co])
        tc4 = time.time()
        act_out = poly_act_fhe(cc, norm_out, a1, b1_, c1)
        tc5 = time.time()
        masked_act = masked(act_out)
        tc6 = time.time()
        x1_out.append(masked_act)
        print(f"    canale {co}: mask={tc1-tc0:.3f}s stats={tc2-tc1:.3f}s isqrt={tc3-tc2:.3f}s "
              f"affine={tc4-tc3:.3f}s act={tc5-tc4:.3f}s mask2={tc6-tc5:.3f}s")

    print("\n  --- CHECKPOINT: x1_out (uscita blocco 1 completo), canale 0 ---")
    debug_valid_patch(cc, keys, x1_out[0], img_hp, img_wp, halo, "x1_out[0]",
                       ref_2d=x1_act_ref[0] if x1_act_ref is not None else None)

    # ================= SECONDO BLOCCO =================
    t2 = time.time()
    x2 = conv2d_multichannel_fhe(cc, x1_out, conv2_w, conv2_b, img_hp, img_wp, K=K)
    t3 = time.time()
    print(f"    conv2 (multi-canale): {t3-t2:.2f}s")

    print("\n  --- CHECKPOINT: x2 (uscita conv2, prima di norm), canale 0 ---")
    debug_valid_patch(cc, keys, x2[0], img_hp, img_wp, halo, "x2[0]",
                       ref_2d=x2_ref[0] if x2_ref is not None else None)

    out = []
    for co in range(Cout):
        tc0 = time.time()
        x2_masked = masked(x2[co])
        if co == 0:
            debug_valid_patch(cc, keys, x2_masked, img_hp, img_wp, halo, "x2_masked[0]", ref_2d=x2_ref[0])
        tc1 = time.time()
        mean, var = instance_stats_padded_fhe(cc, x2_masked, n_valid, n_total)
        tc2 = time.time()
        inv_std = isqrt_chebyshev_fhe(cc, var, cheb_coeffs, cheb_domain, post_iter)
        tc3 = time.time()
        centered = cc.EvalSub(x2_masked, mean)
        normalized = cc.EvalMult(centered, inv_std)
        scaled = cc.EvalMult(normalized, gamma2[co])
        norm_out = cc.EvalAdd(scaled, beta2[co])
        tc4 = time.time()
        act_result = poly_act_fhe(cc, norm_out, a2, b2_, c2)
        tc5 = time.time()
        out.append(act_result)
        print(f"    canale {co}: mask={tc1-tc0:.3f}s stats={tc2-tc1:.3f}s isqrt={tc3-tc2:.3f}s "
              f"affine={tc4-tc3:.3f}s act={tc5-tc4:.3f}s")

    t4 = time.time()
    print(f"    TOTALE ConvBlock: {t4-t0:.2f}s")

    return out, img_hp, img_wp


def main():
    print("=== Test ConvBlock v5 (debug su pixel VALIDI, non bordo) ===\n")

    img_h, img_w = 64, 56
    halo = 1
    K = 3
    Cin, Cout = 2, 2
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo

    print(f"Dimensione: {img_h}x{img_w} pixel validi ({img_hp}x{img_wp} con halo)\n")

    print("Costruzione contesto e chiavi...")
    t_ctx_start = time.time()
    cc, keys = build_context_and_keys(img_h, img_w, halo, K, depth=45)
    print(f"Contesto pronto in {time.time()-t_ctx_start:.1f}s totali.\n")

    rng = np.random.default_rng(42)
    x = rng.normal(size=(Cin, img_h, img_w))
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
    print(f"Coefficienti Chebyshev leggeri pronti (shift={shift:.4f}).")
    print(f"ATTENZIONE: range calibrato [{x_min_test},{x_max_test}] -- verifica sotto se le")
    print(f"varianze reali ci rientrano.\n")

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

    x1_ref = conv_ref(x_padded, conv1_w, conv1_b, Cout, Cin, img_h, img_w)
    print(f"[Rif.] varianza x1[0]: {x1_ref[0].var():.4f} "
          f"{'(FUORI range calibrato!)' if not (x_min_test <= x1_ref[0].var() <= x_max_test) else '(dentro range)'}")

    x1_norm_ref = np.stack([norm_ref(x1_ref[co], gamma1[co], beta1[co]) for co in range(Cout)])
    x1_act_ref = act_ref(x1_norm_ref, a1, b1, c1)
    x1_padded_ref = np.pad(x1_act_ref, ((0, 0), (0, 2*halo), (0, 2*halo)))

    x2_ref = conv_ref(x1_padded_ref, conv2_w, conv2_b, Cout, Cout, img_h, img_w)
    print(f"[Rif.] varianza x2[0]: {x2_ref[0].var():.4f} "
          f"{'(FUORI range calibrato!)' if not (x_min_test <= x2_ref[0].var() <= x_max_test) else '(dentro range)'}\n")

    x2_norm_ref = np.stack([norm_ref(x2_ref[co], gamma2[co], beta2[co]) for co in range(Cout)])
    ref_out = act_ref(x2_norm_ref, a2, b2, c2)

    print("Eseguo il ConvBlock su ciphertext, con debug su pixel VALIDI:\n")
    out_channels, out_hp, out_wp = conv_block_fhe_timed(
        cc, keys, ct_channels_in,
        conv1_w, conv1_b, gamma1, beta1,
        conv2_w, conv2_b, gamma2, beta2,
        cheb_coeffs, cheb_domain, post_iter,
        a1, b1, c1, a2, b2, c2,
        img_h, img_w, halo, K,
        x1_ref=x1_ref, x1_act_ref=x1_act_ref, x2_ref=x2_ref)
    print()

    def decrypt_full(ct):
        pt = cc.Decrypt(keys.secretKey, ct)
        pt.SetLength(out_hp * out_wp)
        return np.array(pt.GetRealPackedValue()).reshape(out_hp, out_wp)

    he_out_full = [decrypt_full(ct) for ct in out_channels]
    he_out = [f[0:img_h, 0:img_w] for f in he_out_full]

    print("  --- CHECKPOINT FINALE: output completo, canale 0 ---")
    print(f"  Patch HE (righe 5:8, colonne 5:8):\n  {he_out[0][5:8, 5:8]}")
    print(f"  Riferimento numpy stessa posizione:\n  {ref_out[0][5:8, 5:8]}")

    print(f"\n{'Canale':>8s} {'Errore max':>14s} {'Var HE':>10s} {'Var rif.':>10s}")
    for co in range(Cout):
        err = np.max(np.abs(he_out[co] - ref_out[co]))
        print(f"{co:8d} {err:14.6e} {he_out[co].var():10.4f} {ref_out[co].var():10.4f}")


if __name__ == '__main__':
    main()