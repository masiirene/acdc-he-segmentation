"""
crypto/test_new_cache_api_zeus.py

Verifica la NUOVA API di cache di PyFIDESlib (aggiornata da Alessandro):
SetRotationKeyCache, SetBootstrapCache, SetPlaintextCache,
SetCiphertextCache -- permette di limitare esplicitamente quanto va in
VRAM per ciascuna categoria, lasciando il resto in RAM di sistema.

ATTENZIONE ALL'ORDINE (dal README aggiornato):
1. KeyGen, EvalMultKeyGen, EvalRotateKeyGen, EvalBootstrapSetup, EvalBootstrapKeyGen
2. SetRotationKeyCache, SetBootstrapCache  (PRIMA di LoadContext)
3. LoadContext
4. SetPlaintextCache, SetCiphertextCache  (DOPO LoadContext)

depth=43 e' il vero tetto di sicurezza a 128 bit (confermato da
Alessandro) -- non 45 come stimato empiricamente, ma vicino.

Valori di cache bassi (4GiB ciascuno) come punto di partenza prudente
(suggerimento di Alessandro) -- il picco reale di VRAM durante l'uso
puo' superare la somma dei 4 budget, quindi si parte bassi e si sale
monitorando nvidia-smi.

Uso: python3 crypto/test_new_cache_api_zeus.py
"""

import sys
import time

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
LEVEL_BUDGET = [4, 4]

# Dimensione reale della nostra immagine (256x224 + halo) -- serve il
# prossimo numero di slot utile; qui usiamo il batch massimo del nostro
# RingDim, coerente con quanto fatto finora.
SLOTS = 1 << (RING_POW - 1)  # 65536

CACHE_GIB = 4  # punto di partenza prudente per OGNI categoria


def main():
    print(f"=== Test nuova API di cache (depth={DEPTH}, slots={SLOTS}, "
          f"level_budget={LEVEL_BUDGET}) ===\n")

    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(SLOTS)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetDevices([0])

    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)

    keys = cc.KeyGen()
    print("KeyGen ok")
    cc.EvalMultKeyGen(keys.secretKey)
    print("EvalMultKeyGen ok")

    # Poche chiavi di rotazione, solo per verificare l'integrazione
    # (il test della pipeline vera arriva dopo, se questo passa)
    cc.EvalRotateKeyGen(keys.secretKey, [1, 2, 4, 8])
    print("EvalRotateKeyGen ok")

    t0 = time.time()
    cc.EvalBootstrapSetup(LEVEL_BUDGET, [0, 0], SLOTS)
    print(f"EvalBootstrapSetup ok ({time.time()-t0:.1f}s)")

    t0 = time.time()
    cc.EvalBootstrapKeyGen(keys.secretKey, SLOTS)
    print(f"EvalBootstrapKeyGen ok ({time.time()-t0:.1f}s)")

    # --- Cache PRIMA di LoadContext ---
    cc.SetRotationKeyCache(CACHE_GIB * GiB)
    cc.SetBootstrapCache(CACHE_GIB * GiB)
    print(f"Cache rotazione e bootstrap impostate a {CACHE_GIB}GiB ciascuna")

    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"LoadContext ok ({time.time()-t0:.1f}s)")

    # --- Cache DOPO LoadContext ---
    cc.SetPlaintextCache(CACHE_GIB * GiB)
    cc.SetCiphertextCache(CACHE_GIB * GiB)
    print(f"Cache plaintext e ciphertext impostate a {CACHE_GIB}GiB ciascuna\n")

    # --- Test reale: cifra, bootstrap, decifra ---
    data = [0.25 + 1e-4 * (i % 100) for i in range(SLOTS)]
    pt = cc.MakeCKKSPackedPlaintext(data)
    ct = cc.Encrypt(keys.publicKey, pt)
    print(f"Encrypt ok (livello {ct.GetLevel()})")

    t0 = time.time()
    bt = cc.EvalBootstrap(ct)
    print(f"EvalBootstrap ok ({time.time()-t0:.1f}s), livello dopo: {bt.GetLevel()}")

    pt_out = cc.Decrypt(keys.secretKey, bt)
    pt_out.SetLength(8)
    got = list(pt_out.GetRealPackedValue())
    print(f"Risultato: {got[:4]}")
    print(f"Atteso:    {data[:4]}")
    print(f"Errore max: {max(abs(a-b) for a,b in zip(got, data[:8])):.2e}")

    print("\n=== ALL OK -- se sei arrivata qui, la nuova cache funziona ===")
    print(f"=== Prossimo passo: riprova test_two_blocks_bootstrap_zeus.py ===")
    print(f"=== con depth=43, level_budget=[4,4] e le 4 chiamate di cache ===")


if __name__ == '__main__':
    main()