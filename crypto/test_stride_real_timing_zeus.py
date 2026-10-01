"""
crypto/test_stride_real_timing_zeus.py

Misura il tempo REALE del primo stride della rete (256x224 -> 128x112),
usando downsample_stride2_fhe gia' verificata corretta (test piccola
scala, oggi). Verifica anche la correttezza a QUESTA scala, mai testata
finora -- per sicurezza, non per aspettarsi sorprese.

Uso: python3 crypto/test_stride_real_timing_zeus.py
"""

import sys
import math
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import (
    conv2d_multichannel_fhe, downsample_stride2_fhe,
)

img_h_in, img_w_in = 256, 224
halo = 1
K = 3
Cin, Cout = 2, 2
img_h_out, img_w_out = img_h_in // 2, img_w_in // 2  # 128, 112

img_hp_in, img_wp_in = img_h_in + 2*halo, img_w_in + 2*halo
img_hp_out, img_wp_out = img_h_out + 2*halo, img_w_out + 2*halo

print(f"Stride reale: {img_h_in}x{img_w_in} -> {img_h_out}x{img_w_out} "
      f"({img_h_out*img_w_out} pixel di output)\n")

params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_128_classic)
params.SetRingDim(1 << 17)
params.SetMultiplicativeDepth(45)
params.SetScalingModSize(50)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetDevices([0])
cc = fhe.GenCryptoContext(params)
for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
    cc.Enable(f)
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.SetRotationKeyCache(4 * 1024**3)

print("Calcolo le chiavi di rotazione necessarie (potrebbe richiedere un momento,")
print("le chiavi dello stride scalano con out_h*out_w, non con log)...")

t0 = time.time()
rot = [1]
rot += sorted(set(ky*img_wp_in + kx for ky in range(K) for kx in range(K)) - {0})

stride_shifts = set()
for out_idx in range(img_h_out * img_w_out):
    r_out, c_out = divmod(out_idx, img_w_out)
    src_idx = (r_out * 2) * img_wp_in + (c_out * 2)
    stride_shifts.add(src_idx - out_idx)
rot += list(stride_shifts)
print(f"  Shift unici richiesti dallo stride: {len(stride_shifts)} "
      f"(su {img_h_out*img_w_out} pixel di output -- {time.time()-t0:.1f}s per calcolarli)")

n_total_out = img_hp_out * img_wp_out
n_pow2 = 1 << math.ceil(math.log2(n_total_out))
rot += fhe.accumulate_rotation_indices(n_total_out, stride=1)
step = 1
while step < n_pow2:
    rot.append(-step)
    step *= 2

unique_rot = sorted(set(r for r in rot if r != 0))
print(f"  Totale chiavi di rotazione uniche: {len(unique_rot)}")

t0 = time.time()
cc.EvalRotateKeyGen(keys.secretKey, unique_rot)
t1 = time.time()
cc.LoadContext(keys.publicKey)
t2 = time.time()
print(f"  EvalRotateKeyGen: {t1-t0:.1f}s | LoadContext: {t2-t1:.1f}s\n")

rng = np.random.default_rng(7)
x = rng.normal(size=(Cin, img_h_in, img_w_in))
x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))


def enc(a):
    return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(a.flatten().tolist()))


def dec(ct, n):
    pt = cc.Decrypt(keys.secretKey, ct)
    pt.SetLength(n)
    return np.array(pt.GetRealPackedValue())


ct_in = [enc(x_padded[c]) for c in range(Cin)]

conv1_w = rng.normal(size=(Cout, Cin, K, K)) * 0.2
conv1_b = rng.normal(size=(Cout,)) * 0.05

t0 = time.time()
x1_full = conv2d_multichannel_fhe(cc, ct_in, conv1_w, conv1_b, img_hp_in, img_wp_in, K=K)
print(f"conv1 (senza stride, {img_h_in}x{img_w_in}): {time.time()-t0:.2f}s")

print("\nEseguo lo stride VERO (256x224 -> 128x112)...")
t0 = time.time()
x1_strided = [downsample_stride2_fhe(cc, x1_full[co], img_hp_in, img_wp_in, img_h_in, img_w_in, stride=2)
              for co in range(Cout)]
t_stride = time.time() - t0
print(f"STRIDE COMPLETATO in {t_stride:.1f}s ({t_stride/60:.2f} minuti) per {Cout} canali "
      f"({t_stride/Cout:.1f}s a canale)")

print("\nVerifica correttezza alla scala reale...")
he_strided = [dec(ct, img_h_out*img_w_out).reshape(img_h_out, img_w_out)
              for ct in x1_strided]

print("Calcolo il riferimento numpy (ciclo puro, puo' richiedere un minuto)...")
t0 = time.time()
ref_full = np.zeros((Cout, img_h_in, img_w_in))
for co in range(Cout):
    for ci in range(Cin):
        for r in range(img_h_in):
            for c in range(img_w_in):
                for ky in range(K):
                    for kx in range(K):
                        ref_full[co, r, c] += conv1_w[co, ci, ky, kx] * x_padded[ci, r+ky, c+kx]
    ref_full[co] += conv1_b[co]
ref_strided = ref_full[:, ::2, ::2]
print(f"Riferimento calcolato in {time.time()-t0:.1f}s.\n")

for co in range(Cout):
    err = np.max(np.abs(he_strided[co] - ref_strided[co]))
    print(f"Canale {co}: errore max = {err:.6e}")

print(f"\n=== RIEPILOGO: stride reale (256x224->128x112) = {t_stride:.1f}s per {Cout} canali ===")
print(f"Stima per l'intera rete (5 stride totali, dimensioni via via piu' piccole,")
print(f"quindi piu' veloci -- stima approssimativa, non lineare):")
print(f"  Primo stride (piu' costoso): {t_stride:.0f}s")
print(f"  Se gli altri 4 scalassero con out_h*out_w (dimezza ogni volta circa a 1/4):")
est_total = t_stride * (1 + 0.25 + 0.0625 + 0.0156 + 0.0039)
print(f"  Stima totale sui 5 stride: ~{est_total:.0f}s (~{est_total/60:.1f} minuti)")