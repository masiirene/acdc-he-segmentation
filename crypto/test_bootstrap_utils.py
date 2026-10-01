"""
crypto/test_bootstrap_utils.py

Verifica, su un contesto piccolo e veloce:
1. Che convenzione usa GetLevel() (crescente o decrescente verso
   l'esaurimento).
2. Se EvalAdd tra ciphertext a livelli diversi lancia un errore chiaro,
   o va gestito esplicitamente con align_for_add.

Uso: python3 crypto/test_bootstrap_utils.py
"""

import sys
sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe
from crypto.bootstrap_utils import ensure_level, align_for_add

DEPTH = 22
LEVEL_BUDGET = [3, 3]

params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_NotSet)
params.SetRingDim(1 << 15)
params.SetMultiplicativeDepth(DEPTH)
params.SetScalingModSize(59)
params.SetFirstModSize(60)
params.SetNumLargeDigits(3)
params.SetBatchSize(1 << 14)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
params.SetDevices([0])

cc = fhe.GenCryptoContext(params)
for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
    cc.Enable(f)
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.SetRotationKeyCache(2 * 1024**3)

slots = (1 << 14)
cc.EvalBootstrapSetup(LEVEL_BUDGET, [0, 0], slots)
cc.EvalBootstrapKeyGen(keys.secretKey, slots)
cc.LoadContext(keys.publicKey)
print("Contesto pronto.\n")

# --- Test 1: convenzione di GetLevel() ---
pt = cc.MakeCKKSPackedPlaintext([1.0] * slots)
ct = cc.Encrypt(keys.publicKey, pt)
print(f"Test 1 -- livello appena cifrato: {ct.GetLevel()} (depth totale: {DEPTH})")

ct_after_3_mults = ct
for i in range(3):
    ct_after_3_mults = cc.EvalMult(ct_after_3_mults, 1.0)
print(f"Test 1 -- livello dopo 3 moltiplicazioni: {ct_after_3_mults.GetLevel()}")
print("Se il numero e' SALITO rispetto a prima, GetLevel() cresce verso l'esaurimento")
print("(coerente con diag_bootstrap.py: 'level 21 of 22' = quasi esaurito).")
print("Se e' SCESO, la convenzione e' opposta -- ensure_level() andra' invertita.\n")

# --- Test 2: EvalAdd tra livelli diversi ---
ct_a = ct  # livello basso (fresco)
ct_b = ct_after_3_mults  # livello piu' alto (consumato)
print(f"Test 2 -- provo EvalAdd tra livelli diversi (a={ct_a.GetLevel()}, b={ct_b.GetLevel()})...")
try:
    result = cc.EvalAdd(ct_a, ct_b)
    pt_result = cc.Decrypt(keys.secretKey, result)
    pt_result.SetLength(4)
    print(f"EvalAdd diretto FUNZIONA (nessun errore). Risultato: {list(pt_result.GetRealPackedValue())[:4]}")
    print("(atteso: [2.0, 2.0, 2.0, 2.0] se la libreria allinea automaticamente)")
    print(">>> align_for_add() potrebbe NON essere necessaria.")
except Exception as e:
    print(f"EvalAdd diretto FALLISCE: {type(e).__name__}: {e}")
    print(">>> align_for_add() e' necessaria. Testo ora quella:")
    ct_a2, ct_b2 = align_for_add(cc, ct_a, ct_b, verbose=True)
    result = cc.EvalAdd(ct_a2, ct_b2)
    pt_result = cc.Decrypt(keys.secretKey, result)
    pt_result.SetLength(4)
    print(f"Con align_for_add: {list(pt_result.GetRealPackedValue())[:4]} (atteso: [2.0, 2.0, 2.0, 2.0])")

print("\nALL DONE")