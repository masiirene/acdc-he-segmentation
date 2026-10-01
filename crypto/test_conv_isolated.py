"""
crypto/test_conv_isolated.py
Test minimo: SOLO conv2d_multichannel_fhe, un canale, un kernel
ASIMMETRICO scelto apposta per rivelare errori di offset/direzione.
"""
import sys, numpy as np
sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import conv2d_multichannel_fhe

params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_128_classic)
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

K, halo = 3, 1
img_h, img_w = 5, 5
img_hp, img_wp = img_h+2*halo, img_w+2*halo
rot = sorted(set(ky*img_wp+kx for ky in range(K) for kx in range(K)) - {0})
cc.EvalRotateKeyGen(keys.secretKey, rot)
cc.LoadContext(keys.publicKey)

x = np.arange(1, img_h*img_w+1, dtype=float).reshape(img_h, img_w)
x_padded = np.pad(x, halo)
pt = cc.MakeCKKSPackedPlaintext(x_padded.flatten().tolist())
ct = cc.Encrypt(keys.publicKey, pt)

# Kernel ASIMMETRICO: solo l'angolo in alto a sinistra = 1, resto 0
weight = np.zeros((1, 1, K, K))
weight[0, 0, 0, 0] = 1.0
bias = np.array([0.0])

out = conv2d_multichannel_fhe(cc, [ct], weight, bias, img_hp, img_wp, K=K)
pt_out = cc.Decrypt(keys.secretKey, out[0])
pt_out.SetLength(img_hp*img_wp)
he_result = np.array(pt_out.GetRealPackedValue()).reshape(img_hp, img_wp)
he_valid = he_result[halo:halo+img_h, halo:halo+img_w]

# Riferimento: con kernel[0,0]=1 e resto 0, l'output atteso e'
# semplicemente l'input SHIFTATO -- out[r,c] = x_padded[r,c] (l'angolo
# top-left del kernel si applica alla posizione (r,c) del padded, che
# corrisponde a x_padded[r+0, c+0] = x_padded[r,c])
print("Input originale (5x5):")
print(x)
print("\nOutput HE (regione valida):")
print(he_valid)
print("\nAtteso (con kernel[0,0]=1, resto 0 -- vedi commento sopra):")
expected = x_padded[0:img_h, 0:img_w]  # shift verso l'angolo (0,0) del padded
print(expected)
print(f"\nErrore max: {np.max(np.abs(he_valid - expected)):.6e}")