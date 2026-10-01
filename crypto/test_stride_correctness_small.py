"""
crypto/test_stride_correctness_small.py

Verifica di CORRETTEZZA (non ancora velocita') dello stride reale:
conv1(stride=1 internamente) -> downsample_stride2_fhe, su
un'immagine piccola (16x14, cosi' anche l'operazione O(n) di
downsample resta questione di secondi, non minuti).

ATTENZIONE: questo NON e' ancora la versione veloce -- serve solo a
confermare che la logica dello stride sia giusta, prima di progettare
una versione O(log n) come si e' fatto per il mascheramento del bordo.

Uso: python3 crypto/test_stride_correctness_small.py
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

img_h_in, img_w_in = 16, 14   # dimensione INGRESSO (prima dello stride)
halo = 1
K = 3
Cin, Cout = 2, 2
img_h_out, img_w_out = img_h_in // 2, img_w_in // 2  # dimensione dopo stride=2

img_hp_in, img_wp_in = img_h_in + 2*halo, img_w_in + 2*halo
img_hp_out, img_wp_out = img_h_out + 2*halo, img_w_out + 2*halo

# --- Contesto: sicurezza disattivata (HEStd_NotSet), solo per questo
# test diagnostico di correttezza -- depth bassa, basta per conv+downsample ---
params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_NotSet)
params.SetRingDim(1 << 15)
params.SetMultiplicativeDepth(10)
params.SetScalingModSize(50)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetDevices([0])
cc = fhe.GenCryptoContext(params)
for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
    cc.Enable(f)
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)

rot = [1]
rot += sorted(set(ky*img_wp_in + kx for ky in range(K) for kx in range(K)) - {0})
for out_idx in range(img_h_out * img_w_out):
    r_out, c_out = divmod(out_idx, img_w_out)
    src_idx = (r_out * 2) * img_wp_in + (c_out * 2)
    rot.append(src_idx - out_idx)
n_total_out = img_hp_out * img_wp_out
n_pow2 = 1 << math.ceil(math.log2(n_total_out))
rot += fhe.accumulate_rotation_indices(n_total_out, stride=1)
step = 1
while step < n_pow2:
    rot.append(-step)
    step *= 2
cc.EvalRotateKeyGen(keys.secretKey, sorted(set(r for r in rot if r != 0)))
cc.LoadContext(keys.publicKey)
print(f"Contesto pronto. Chiavi: {len(set(r for r in rot if r != 0))}")

rng = np.random.default_rng(1)
x = rng.normal(size=(Cin, img_h_in, img_w_in))
x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))


def enc(a):
    return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(a.flatten().tolist()))


def dec_full(ct, n):
    pt = cc.Decrypt(keys.secretKey, ct)
    pt.SetLength(n)
    return np.array(pt.GetRealPackedValue())


ct_in = [enc(x_padded[c]) for c in range(Cin)]

conv1_w = rng.normal(size=(Cout, Cin, K, K)) * 0.2
conv1_b = rng.normal(size=(Cout,)) * 0.05

# ================= conv1 =================
t0 = time.time()
x1_full = conv2d_multichannel_fhe(cc, ct_in, conv1_w, conv1_b, img_hp_in, img_wp_in, K=K)
print(f"conv1 (stride=1 internamente): {time.time()-t0:.2f}s")

# --- DEBUG: verifica x1_full PRIMA del downsample ---
x1_full_he = [dec_full(x1_full[co], img_hp_in*img_wp_in).reshape(img_hp_in, img_wp_in)[0:img_h_in, 0:img_w_in]
              for co in range(Cout)]

ref_full_pre = np.zeros((Cout, img_h_in, img_w_in))
for co in range(Cout):
    for ci in range(Cin):
        for r in range(img_h_in):
            for c in range(img_w_in):
                for ky in range(K):
                    for kx in range(K):
                        ref_full_pre[co, r, c] += conv1_w[co, ci, ky, kx] * x_padded[ci, r+ky, c+kx]
    ref_full_pre[co] += conv1_b[co]

for co in range(Cout):
    err_pre = np.max(np.abs(x1_full_he[co] - ref_full_pre[co]))
    print(f"DEBUG: errore x1_full[{co}] PRIMA del downsample: {err_pre:.6e}")

# ================= downsample stride=2 =================
t0 = time.time()
x1_strided = [downsample_stride2_fhe(cc, x1_full[co], img_hp_in, img_wp_in, img_h_in, img_w_in, stride=2)
              for co in range(Cout)]
print(f"downsample stride=2 (O(n), {img_h_out*img_w_out} pixel di output): {time.time()-t0:.2f}s")

he_strided = [dec_full(ct, img_h_out*img_w_out).reshape(img_h_out, img_w_out)
              for ct in x1_strided]

ref_strided = ref_full_pre[:, ::2, ::2]

for co in range(Cout):
    err = np.max(np.abs(he_strided[co] - ref_strided[co]))
    print(f"Canale {co}: errore max (dopo downsample) = {err:.6e}")