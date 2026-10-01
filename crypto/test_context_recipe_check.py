"""
crypto/test_context_recipe_check.py

Verifica se il crash 'illegal memory access' visto ieri a RingDim=2^18
(e il tetto pratico di depth=45 su 2^17) fosse dovuto a una
combinazione di parametri di scala sbagliata, non a un limite vero
della libreria.

Ricetta corretta (da tests/diag_bootstrap.py di Alessandro, confermata
funzionante): ScalingModSize=59, FirstModSize=60 (non 50/default come
usato in tutti i nostri test di oggi).

Prova PROGRESSIVAMENTE piu' alta la profondita' su RingDim=2^18, con
la ricetta corretta, per trovare il vero tetto pratico -- se questo
risolve il crash, la strada di Aurora (anello piu' grande = piu' depth
= meno bootstrap) e' finalmente percorribile.

Uso: python3 crypto/test_context_recipe_check.py
"""

import sys
import time

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe


def try_context(depth, ring_pow=18, scaling_mod=59, first_mod=60):
    print(f"\n--- Provo: RingDim=2^{ring_pow}, depth={depth}, "
          f"ScalingModSize={scaling_mod}, FirstModSize={first_mod} ---")
    try:
        params = fhe.CCParams()
        params.SetSecurityLevel(fhe.HEStd_128_classic)
        params.SetRingDim(1 << ring_pow)
        params.SetMultiplicativeDepth(depth)
        params.SetScalingModSize(scaling_mod)
        params.SetFirstModSize(first_mod)
        params.SetNumLargeDigits(3)
        params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
        params.SetKeySwitchTechnique(fhe.HYBRID)
        params.SetDevices([0])

        cc = fhe.GenCryptoContext(params)
        for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
            cc.Enable(f)
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.SetRotationKeyCache(4 * 1024**3)

        # Poche chiavi di rotazione, solo per verificare che EvalRotate
        # non crashi (il punto esatto dove si era rotto ieri)
        cc.EvalRotateKeyGen(keys.secretKey, [1, 2, 4])
        cc.LoadContext(keys.publicKey)

        # Test minimo: cifra, ruota, decifra -- il punto che crashava ieri
        pt = cc.MakeCKKSPackedPlaintext([1.0, 2.0, 3.0, 4.0])
        ct = cc.Encrypt(keys.publicKey, pt)
        ct_rot = cc.EvalRotate(ct, 1)
        pt_out = cc.Decrypt(keys.secretKey, ct_rot)
        pt_out.SetLength(4)
        result = list(pt_out.GetRealPackedValue())

        print(f"  OK: contesto creato, EvalRotate funziona. Risultato: {result}")
        return True
    except Exception as e:
        print(f"  FALLITO: {type(e).__name__}: {e}")
        return False


# Prova diverse profondita' su 2^18 con la ricetta corretta
for depth_test in [50, 55, 60, 65]:
    ok = try_context(depth_test, ring_pow=18)
    if not ok:
        print(f"\n=== Si ferma a depth={depth_test} su RingDim=2^18 ===")
        break
else:
    print("\n=== Tutte le profondita' testate hanno funzionato su RingDim=2^18! ===")

print("\n\n--- Per confronto, riprovo anche RingDim=2^17 con la ricetta corretta ---")
for depth_test in [45, 50, 60, 70]:
    ok = try_context(depth_test, ring_pow=17)
    if not ok:
        print(f"\n=== Si ferma a depth={depth_test} su RingDim=2^17 (con ricetta corretta) ===")
        break