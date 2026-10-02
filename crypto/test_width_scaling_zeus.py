"""
crypto/test_width_scaling_zeus.py

Stessa pipeline di test_three_blocks_bootstrap_zeus.py (bsgsDim=[4,4],
Offload()/TrimGPUMemoryPool() a ogni livello), ma con CHANNELS
configurabile invece del giocattolo Cin=Cout=2 fisso.

Perche' conta: in questo schema ogni canale e' un ciphertext SEPARATO
(l'immagine intera sta in un solo ciphertext, uno per canale). Passare
da 2 a 16 canali non e' "un po' di dati in piu'" -- sono 8 volte piu'
ciphertext vivi contemporaneamente a ogni livello della rete. La
profondita' (N_BLOCKS, testata ieri fino a 11) e la larghezza (CHANNELS,
qui) sono due assi di scala indipendenti.

Si parte con N_BLOCKS=1 per isolare l'effetto della sola larghezza,
prima di combinarla con la profondita'.

Uso: python3 crypto/test_width_scaling_zeus.py
     (cambia CHANNELS e N_BLOCKS in cima a main())
"""

import sys
import math
import time
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import (
    poly_act_fhe, isqrt_chebyshev_fhe, fit_monotonic_isqrt_coeffs,
    conv2d_multichannel_fhe, mask_border_fhe_precomputed,
    make_border_mask_plaintext, instance_stats_padded_fhe,
)

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
LEVEL_BUDGET = [4, 4]
CACHE_GIB = 1
BSGS_DIM = [4, 4]


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


def build_context_with_bootstrap(img_h, img_w, halo, K, cache_gib=CACHE_GIB):
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_total = img_hp * img_wp
    batch = 1 << (RING_POW - 1)
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

    print("  Configurazione bootstrap (EvalBootstrapSetup + KeyGen)...")
    t0 = time.time()
    cc.EvalBootstrapSetup(LEVEL_BUDGET, BSGS_DIM, batch)
    print(f"    EvalBootstrapSetup (bsgsDim={BSGS_DIM}): {time.time()-t0:.1f}s")
    t0 = time.time()
    cc.EvalBootstrapKeyGen(keys.secretKey, batch)
    print(f"    EvalBootstrapKeyGen: {time.time()-t0:.1f}s")

    print("  Chiavi di rotazione per conv/norm (indipendenti dal numero di canali)...")
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

    cc.SetRotationKeyCache(cache_gib * GiB)
    cc.SetBootstrapCache(cache_gib * GiB)
    print(f"  Cache rotazione/bootstrap impostate a {cache_gib}GiB ciascuna")

    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s")

    cc.SetPlaintextCache(cache_gib * GiB)
    cc.SetCiphertextCache(cache_gib * GiB)
    print(f"  Cache plaintext/ciphertext impostate a {cache_gib}GiB ciascuna")

    return cc, keys, batch


def conv_block_fhe(cc, ct_channels_in, pt_mask,
                    conv1_w, conv1_b, gamma1, beta1,
                    conv2_w, conv2_b, gamma2, beta2,
                    cheb_coeffs, cheb_domain, post_iter,
                    a1, b1_, c1, a2, b2_, c2,
                    img_h, img_w, halo, K,
                    mid_bootstrap=False):
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_valid = img_h * img_w
    n_total = img_hp * img_wp
    Cout = conv1_w.shape[0]

    def masked(ct):
        return mask_border_fhe_precomputed(cc, ct, pt_mask)

    print(f"      [conv1: {conv1_w.shape[1]} canali in -> {Cout} canali out]")
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

    if mid_bootstrap:
        for ct in x1:
            ct.Offload()
        cc.TrimGPUMemoryPool()
        print("      [mid-block bootstrap]")
        for co in range(Cout):
            x1_out[co] = cc.EvalBootstrap(x1_out[co])
        cc.TrimGPUMemoryPool()
        x1_out = [masked(ct) for ct in x1_out]

    print(f"      [conv2: {Cout} canali in -> {Cout} canali out]")
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


def bootstrap_channels(cc, channels, label, offload_first=None):
    if offload_first:
        for ct in offload_first:
            ct.Offload()
        cc.TrimGPUMemoryPool()
        print(f"    (scaricati {len(offload_first)} ciphertext non piu' necessari prima del bootstrap)")
        print_gpu_mem(f"dopo il trim, prima di '{label}'")
    print(f"=== BOOTSTRAP ({label}) ===")
    out = []
    for co, ch in enumerate(channels):
        t0 = time.time()
        ct_boot = cc.EvalBootstrap(ch)
        if co % 4 == 0 or co == len(channels) - 1:
            print(f"  Canale {co}/{len(channels)}: {time.time()-t0:.2f}s, livello dopo: {ct_boot.GetLevel()}")
        out.append(ct_boot)
        ch.Offload()
        cc.TrimGPUMemoryPool()
    return out


CHANNELS = 16   # <-- la vera scala da testare (2 era il giocattolo di ieri)
N_BLOCKS = 11   # <-- il test finale: larghezza E profondita' vere insieme


def main():
    print(f"=== Test SCALA IN LARGHEZZA: {CHANNELS} canali, {N_BLOCKS} blocco/i ===\n")

    img_h, img_w = 256, 224
    halo = 1
    K = 3
    Cin, Cout = CHANNELS, CHANNELS
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo

    print_gpu_mem("prima di costruire il contesto")
    t0 = time.time()
    cc, keys, batch = build_context_with_bootstrap(img_h, img_w, halo, K)
    print(f"Contesto pronto in {time.time()-t0:.1f}s totali.")
    print_gpu_mem("dopo LoadContext (costo fisso: identico a ieri, non dipende dalla larghezza)")
    print()

    pt_mask = make_border_mask_plaintext(cc, img_hp, img_wp, img_h, img_w, halo)

    rng = np.random.default_rng(11)
    x = rng.normal(size=(Cin, img_h, img_w))
    x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))

    def enc(a):
        return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(a.flatten().tolist()))

    def dec(ct, n):
        pt = cc.Decrypt(keys.secretKey, ct)
        pt.SetLength(n)
        return np.array(pt.GetRealPackedValue())

    gamma = list(1.0 + 0.1*rng.normal(size=Cout))
    beta = list(0.1*rng.normal(size=Cout))
    a_, b_, c_ = 0.1, 1.0, 0.5

    x_min_test, x_max_test = 0.5, 4.0
    cheb_coeffs, shift = fit_monotonic_isqrt_coeffs(fhe, x_min_test, x_max_test, degree=3, extra_safety=1.2)
    cheb_domain = [x_min_test, x_max_test]
    post_iter = 1

    print(f"Cifratura di {Cin} canali di input...")
    t0 = time.time()
    current = [enc(x_padded[c]) for c in range(Cin)]
    print(f"Cifratura completata in {time.time()-t0:.2f}s.")
    print_gpu_mem("dopo cifratura input")

    for b_idx in range(1, N_BLOCKS + 1):
        print(f"\n=== ConvBlock {b_idx}/{N_BLOCKS} ({CHANNELS} canali) ===")
        cin_this = Cin if b_idx == 1 else Cout
        # Scala dei pesi proporzionata al fan-in (come una vera inizializzazione
        # tipo Kaiming) -- con 2 canali *0.2 andava bene, ma sommando 16
        # contributi invece di 2 la varianza in uscita cresce e i valori
        # escono dal dominio [0.5, 4.0] calibrato per il Chebyshev
        # dell'inverso della radice quadrata. Senza questo fix, il problema
        # non e' crittografico -- e' solo che i pesi casuali del test non
        # erano scalati per la larghezza.
        std1 = 0.2 / math.sqrt(cin_this * K * K)
        std2 = 0.2 / math.sqrt(Cout * K * K)
        w1 = rng.normal(size=(Cout, cin_this, K, K)) * std1
        b1 = rng.normal(size=(Cout,)) * 0.05
        w2 = rng.normal(size=(Cout, Cout, K, K)) * std2
        b2 = rng.normal(size=(Cout,)) * 0.05

        t0 = time.time()
        out_blk = conv_block_fhe(cc, current, pt_mask, w1, b1, gamma, beta, w2, b2, gamma, beta,
                                  cheb_coeffs, cheb_domain, post_iter, a_, b_, c_, a_, b_, c_,
                                  img_h, img_w, halo, K, mid_bootstrap=(b_idx > 1))
        print(f"Blocco {b_idx} completato in {time.time()-t0:.2f}s, livello: {out_blk[0].GetLevel()}")
        print_gpu_mem(f"dopo Blocco {b_idx} ({CHANNELS} canali)")

        out_masked = [mask_border_fhe_precomputed(cc, ct, pt_mask) for ct in out_blk]
        current = bootstrap_channels(cc, out_masked, f"dopo Blocco {b_idx}",
                                      offload_first=current + out_blk)
        print_gpu_mem(f"dopo bootstrap {b_idx} ({len(current)} canali bootstrappati)")
        for ct in out_masked:
            ct.Offload()
        cc.TrimGPUMemoryPool()

    he_out = dec(current[0], img_hp*img_wp).reshape(img_hp, img_wp)[0:img_h, 0:img_w]
    print(f"\nMedia/var output finale canale 0: {he_out.mean():.4f} / {he_out.var():.4f}")
    print(f"\n=== Se sei arrivata qui, {CHANNELS} canali x {N_BLOCKS} blocco/i funzionano. ===")


if __name__ == '__main__':
    main()