"""
crypto/test_three_blocks_bootstrap_zeus.py

Estende test_two_blocks_bootstrap_zeus.py a un TERZO blocco, per
rispondere alla domanda: la memoria GPU si stabilizza (costo fisso di
chiavi/precomputazione bootstrap) o continua a salire con ogni blocco
in piu' (qualcosa si accumula e va liberato esplicitamente)?

Due aggiunte rispetto alla v2:
1. print_gpu_mem(label) -- stampa la memoria GPU usata via nvidia-smi,
   cosi' il dato e' nel log invece che solo visibile a occhio su nvtop.
2. del esplicito + gc.collect() sui ciphertext dei blocchi gia' finiti,
   per capire se questo fa la differenza.

Uso: python3 crypto/test_three_blocks_bootstrap_zeus.py
"""

import sys
import gc
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
    skip_connection_sum_fhe,
)

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
LEVEL_BUDGET = [4, 4]
CACHE_GIB = 1  # abbassato ulteriormente da 2 a 1: vogliamo il massimo
               # margine libero possibile per i ciphertext di lavoro,
               # ora che testiamo un blocco in piu'


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
    # bsgsDim esplicito invece di [0,0] (= "scegli tu, ottimizzando la
    # velocita'"): valori piu' piccoli = meno chiavi di rotazione distinte
    # necessarie al bootstrap, a scapito di piu' operazioni di rotazione
    # (piu' lento, ma molto meno affamato di memoria). Deve essere <
    # ceil(log2(slots)) = 16 con batch=65536.
    BSGS_DIM = [4, 4]
    cc.EvalBootstrapSetup(LEVEL_BUDGET, BSGS_DIM, batch)
    print(f"    EvalBootstrapSetup (bsgsDim={BSGS_DIM}): {time.time()-t0:.1f}s")
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
        # x1 (risultato grezzo della prima convoluzione, pre-maschera/norm)
        # non serve piu': scaricalo e libera davvero la VRAM prima che il
        # bootstrap debba caricare tutte le sue chiavi in un colpo solo.
        for ct in x1:
            ct.Offload()
        cc.TrimGPUMemoryPool()
        print("    [mid-block bootstrap]")
        for co in range(Cout):
            x1_out[co] = cc.EvalBootstrap(x1_out[co])
        x1_out = [masked(ct) for ct in x1_out]
        # Pulizia della "spazzatura" accumulata dalla prima meta' del blocco
        # (tutti i ciphertext temporanei di conv1+norm1+att1, mai restituiti
        # esplicitamente finora) prima di iniziare la seconda meta', piu'
        # costosa.
        cc.TrimGPUMemoryPool()

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
        print(f"  Canale {co}: {time.time()-t0:.2f}s, livello dopo: {ct_boot.GetLevel()}")
        out.append(ct_boot)
        # L'input di QUESTO canale non serve piu' una volta bootstrappato --
        # scaricalo subito, prima di passare al canale successivo (non solo
        # prima dell'intero gruppo: il crash di ieri avveniva proprio tra
        # un canale e l'altro dello stesso bootstrap).
        ch.Offload()
        cc.TrimGPUMemoryPool()
    return out


def main():
    print("=== Test: 3 ConvBlock + skip + BOOTSTRAP, CACHE_GIB=1 ===\n")

    img_h, img_w = 256, 224
    halo = 1
    K = 3
    Cin, Cout = 2, 2
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo

    print_gpu_mem("prima di costruire il contesto")
    t0 = time.time()
    cc, keys, batch = build_context_with_bootstrap(img_h, img_w, halo, K)
    print(f"Contesto pronto in {time.time()-t0:.1f}s totali.")
    print_gpu_mem("dopo LoadContext (costo fisso: chiavi + precomp. bootstrap)")
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

    ct_in = [enc(x_padded[c]) for c in range(Cin)]

    def rand_weights():
        return (rng.normal(size=(Cout, Cout, K, K)) * 0.2, rng.normal(size=(Cout,)) * 0.05)

    conv_w = {}
    for i in range(1, 7):
        cin = Cin if i == 1 else Cout
        w = rng.normal(size=(Cout, cin, K, K)) * 0.2
        b = rng.normal(size=(Cout,)) * 0.05
        conv_w[i] = (w, b)

    gamma = [1.1, 0.9]
    beta = [0.1, -0.1]
    a_, b_, c_ = 0.1, 1.0, 0.5

    x_min_test, x_max_test = 0.5, 4.0
    cheb_coeffs, shift = fit_monotonic_isqrt_coeffs(fhe, x_min_test, x_max_test, degree=3, extra_safety=1.2)
    cheb_domain = [x_min_test, x_max_test]
    post_iter = 1

    # ===== BLOCCO A =====
    print("=== ConvBlock A ===")
    t0 = time.time()
    out_A = conv_block_fhe(cc, ct_in, pt_mask, *conv_w[1], gamma, beta, *conv_w[2], gamma, beta,
                            cheb_coeffs, cheb_domain, post_iter, a_, b_, c_, a_, b_, c_,
                            img_h, img_w, halo, K)
    print(f"Blocco A completato in {time.time()-t0:.2f}s, livello: {out_A[0].GetLevel()}")
    print_gpu_mem("dopo Blocco A")

    out_A_masked = [mask_border_fhe_precomputed(cc, ct, pt_mask) for ct in out_A]
    # ct_in e out_A non servono piu' -- scaricali per davvero (del+gc.collect()
    # da solo NON basta, FIDESlib tiene la VRAM nel suo pool finche' non lo
    # dici esplicitamente con Offload()+TrimGPUMemoryPool()).
    boot_A = bootstrap_channels(cc, out_A_masked, "dopo Blocco A",
                                 offload_first=ct_in + out_A)
    print_gpu_mem("dopo bootstrap A")
    for ct in out_A_masked:
        ct.Offload()
    cc.TrimGPUMemoryPool()

    # ===== BLOCCO B =====
    print("\n=== ConvBlock B ===")
    t0 = time.time()
    out_B = conv_block_fhe(cc, boot_A, pt_mask, *conv_w[3], gamma, beta, *conv_w[4], gamma, beta,
                            cheb_coeffs, cheb_domain, post_iter, a_, b_, c_, a_, b_, c_,
                            img_h, img_w, halo, K, mid_bootstrap=True)
    print(f"Blocco B completato in {time.time()-t0:.2f}s, livello: {out_B[0].GetLevel()}")
    print_gpu_mem("dopo Blocco B")

    out_B_masked = [mask_border_fhe_precomputed(cc, ct, pt_mask) for ct in out_B]
    # boot_A e out_B (grezzo, pre-maschera) non servono piu'.
    boot_B = bootstrap_channels(cc, out_B_masked, "dopo Blocco B",
                                 offload_first=boot_A + out_B)
    print_gpu_mem("dopo bootstrap B")
    for ct in out_B_masked:
        ct.Offload()
    cc.TrimGPUMemoryPool()

    # ===== BLOCCO C (il nuovo, il terzo) =====
    print("\n=== ConvBlock C (NUOVO -- terzo blocco) ===")
    t0 = time.time()
    out_C = conv_block_fhe(cc, boot_B, pt_mask, *conv_w[5], gamma, beta, *conv_w[6], gamma, beta,
                            cheb_coeffs, cheb_domain, post_iter, a_, b_, c_, a_, b_, c_,
                            img_h, img_w, halo, K, mid_bootstrap=True)
    print(f"Blocco C completato in {time.time()-t0:.2f}s, livello: {out_C[0].GetLevel()}")
    print_gpu_mem("dopo Blocco C (IL DATO CHE CI INTERESSA)")

    print("\nDecifrazione finale (solo per controllo di sanita', nessun riferimento qui)...")
    he_out = dec(out_C[0], img_hp*img_wp).reshape(img_hp, img_wp)[0:img_h, 0:img_w]
    print(f"Media/var output canale 0: {he_out.mean():.4f} / {he_out.var():.4f}")

    print("\n=== Se sei arrivata qui, 3 blocchi funzionano. Guarda i GPU MEM sopra: ===")
    print("=== stabile dopo Blocco A -> costo fisso, si puo' scalare oltre.      ===")
    print("=== continua a salire ad ogni blocco -> serve liberare qualcos'altro. ===")


if __name__ == '__main__':
    main()