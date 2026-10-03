"""
crypto/test_row_packed_norm_small_zeus.py

Primo test HE VERO di InstanceNorm+PolyAct nel formato a righe --
isolato dalla convoluzione (quella e' gia' verificata in HE
separatamente). Verifica che la somma a passi (stride=n, nessuna
maschera di correzione necessaria) + Chebyshev + PolyAct funzionino
insieme, prima di unire tutto in un ConvBlock completo.

Caso piccolo apposta: Cin=Cout=6, immagine 4x4 (+alone), niente
bootstrap (profondita' minima).

Uso: python3 crypto/test_row_packed_norm_small_zeus.py
"""

import sys
import math
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import poly_act_fhe, isqrt_chebyshev_fhe, fit_monotonic_isqrt_coeffs

GiB = 1 << 30
DEPTH = 15
RING_POW = 17


def build_small_context(batch_size):
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(50)
    params.SetFirstModSize(55)
    params.SetBatchSize(batch_size)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetDevices([0])

    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    return cc, keys


def sum_within_row_strided_fhe(cc, ct_row, n, W, real_len, mask_cache):
    """
    Somma a passi DENTRO una riga, stride=n -- CORRETTA rispetto alla
    prima versione (che ignorava il padding a potenza di 2 del batch
    CKKS). Riusa la STESSA tecnica a due rotazioni + maschere gia'
    verificata per la convoluzione, ma con una dimensione di blocco
    DIVERSA: non n (un pixel), bensi' l'INTERA riga vera (real_len =
    Wp*n) -- il batch CKKS e' arrotondato a potenza di 2 (es. 64), ma
    i dati veri occupano solo real_len (es. 36) slot, e una singola
    EvalRotate gira sul batch intero, non sulla lunghezza vera.

    Bug trovato oggi, SOLO in HE vero: il test numpy non l'aveva mai
    scoperto perche' li' l'array aveva ESATTAMENTE lunghezza real_len,
    senza il padding a potenza di 2 che CKKS impone.
    """
    def rotate_correct(ct, shift):
        if shift == 0:
            return ct
        key = shift
        if key not in mask_cache:
            full_len = real_len
            # Maschera: posizione i e' "main" se i+shift < real_len
            # (nessun giro necessario), altrimenti "wrap".
            idx = np.arange(full_len)
            m_main = (idx + shift < full_len).astype(float)
            m_wrap = 1.0 - m_main
            full_main = np.zeros(1 << math.ceil(math.log2(full_len)))
            full_main[:full_len] = m_main
            full_wrap = np.zeros_like(full_main)
            full_wrap[:full_len] = m_wrap
            mask_cache[key] = (cc.MakeCKKSPackedPlaintext(full_main.tolist()),
                                cc.MakeCKKSPackedPlaintext(full_wrap.tolist()))
        m_main_pt, m_wrap_pt = mask_cache[key]
        r_main = cc.EvalRotate(ct, shift)
        r_wrap = cc.EvalRotate(ct, shift - real_len)
        t_main = cc.EvalMult(r_main, m_main_pt)
        t_wrap = cc.EvalMult(r_wrap, m_wrap_pt)
        return cc.EvalAdd(t_main, t_wrap)

    result = None
    partial = ct_row
    remaining = W
    shift_base = 0
    power = 1
    while remaining > 0:
        if remaining & 1:
            shifted = rotate_correct(partial, shift_base * n)
            result = shifted if result is None else cc.EvalAdd(result, shifted)
            shift_base += power
        remaining >>= 1
        if remaining > 0:
            shifted_p = rotate_correct(partial, power * n)
            partial = cc.EvalAdd(partial, shifted_p)
            power *= 2
    return result


def main():
    print("=== Primo test HE VERO: InstanceNorm+PolyAct nel formato a righe (isolato) ===\n")

    Cin = Cout = n = 6
    H, W = 4, 4
    halo = 1
    Hp, Wp = H + 2*halo, W + 2*halo
    real_len = Wp * n
    batch_size = 1 << math.ceil(math.log2(real_len))

    print(f"n={n}, H={H}, W={W} (+alone) -> Hp={Hp}, Wp={Wp}, batch={batch_size}\n")

    cc, keys = build_small_context(batch_size)

    real_len_for_keys = Wp * n
    rot_set = set()
    for k in [1, 2, 4]:
        shift = k * n
        if shift < batch_size:
            rot_set.add(shift)
            rot_set.add(shift - real_len_for_keys)  # per la correzione del wraparound
    rot_list = sorted(r for r in rot_set if r != 0)
    cc.EvalRotateKeyGen(keys.secretKey, rot_list)
    cc.SetRotationKeyCache(1 * GiB)
    cc.LoadContext(keys.publicKey)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(1 * GiB)
    print(f"Contesto pronto ({len(rot_list)} chiavi di rotazione).\n")

    rng = np.random.default_rng(9)
    x = rng.normal(size=(Cin, H, W)) * 1.5 + 0.5
    gamma = rng.normal(size=Cout) * 0.2 + 1.0
    beta = rng.normal(size=Cout) * 0.1
    a_, b_, c_ = 0.1, 1.0, 0.5

    def pack_row(r, data_padded):
        row = data_padded[:, r, :].T  # (Wp, n)
        full = np.zeros(batch_size)
        full[:Wp * n] = row.flatten()
        return full

    x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))

    def enc(a):
        return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(a.tolist()))

    rows_ct = [enc(pack_row(r, x_padded)) for r in range(Hp)]

    print("Calcolo media/varianza (somma a passi dentro riga + tra righe, SOLO righe valide)...")
    real_len = Wp * n
    mask_cache = {}
    t0 = time.time()
    row_sums = [sum_within_row_strided_fhe(cc, rows_ct[r], n, Wp, real_len, mask_cache) for r in range(H)]
    total_sum = row_sums[0]
    for r in row_sums[1:]:
        total_sum = cc.EvalAdd(total_sum, r)
    mean_ct = cc.EvalMult(total_sum, 1.0 / (H * W))

    rows_sq = [cc.EvalMult(rows_ct[r], rows_ct[r]) for r in range(H)]
    row_sums_sq = [sum_within_row_strided_fhe(cc, rows_sq[r], n, Wp, real_len, mask_cache) for r in range(H)]
    total_sum_sq = row_sums_sq[0]
    for r in row_sums_sq[1:]:
        total_sum_sq = cc.EvalAdd(total_sum_sq, r)
    mean_sq_ct = cc.EvalMult(total_sum_sq, 1.0 / (H * W))

    mean_sq_of_mean = cc.EvalMult(mean_ct, mean_ct)
    var_ct = cc.EvalSub(mean_sq_ct, mean_sq_of_mean)
    print(f"Statistiche calcolate in {time.time()-t0:.2f}s.\n")

    # Verifica media/varianza contro numpy PRIMA di proseguire
    pt = cc.Decrypt(keys.secretKey, mean_ct)
    pt.SetLength(n)
    mean_he = np.array(pt.GetRealPackedValue())
    pt2 = cc.Decrypt(keys.secretKey, var_ct)
    pt2.SetLength(n)
    var_he = np.array(pt2.GetRealPackedValue())

    mean_ref = np.array([x[ch].mean() for ch in range(Cout)])
    var_ref = np.array([x[ch].var() for ch in range(Cout)])
    print(f"Errore media: {np.max(np.abs(mean_he - mean_ref)):.6e}")
    print(f"Errore varianza: {np.max(np.abs(var_he - var_ref)):.6e}\n")

    print("=== DEBUG: mean_ct e' uguale in TUTTI i blocchi della riga, non solo il primo? ===")
    pt_full = cc.Decrypt(keys.secretKey, mean_ct)
    pt_full.SetLength(batch_size)
    mean_full_he = np.array(pt_full.GetRealPackedValue())[:Wp * n].reshape(Wp, n)
    print("Media HE, blocco per blocco (ogni riga = un pixel, dovrebbero essere TUTTE uguali):")
    print(mean_full_he)
    print(f"Atteso in ogni riga: {mean_ref}\n")

    print("Applicazione Chebyshev (radice inversa) + norm + PolyAct, su ogni riga valida...")
    x_min_t, x_max_t = max(0.01, var_ref.min() * 0.5), var_ref.max() * 2.0
    cheb_coeffs, shift = fit_monotonic_isqrt_coeffs(fhe, x_min_t, x_max_t, degree=3, extra_safety=1.2)
    inv_std_ct = isqrt_chebyshev_fhe(cc, var_ct, cheb_coeffs, [x_min_t, x_max_t], post_iter=1)

    gamma_tiled = np.tile(gamma, Wp)
    gamma_full = np.zeros(batch_size)
    gamma_full[:len(gamma_tiled)] = gamma_tiled
    beta_tiled = np.tile(beta, Wp)
    beta_full = np.zeros(batch_size)
    beta_full[:len(beta_tiled)] = beta_tiled
    gamma_pt = cc.MakeCKKSPackedPlaintext(gamma_full.tolist())
    beta_pt = cc.MakeCKKSPackedPlaintext(beta_full.tolist())

    out_rows = []
    for r in range(H):
        centered = cc.EvalSub(rows_ct[r], mean_ct)
        normalized = cc.EvalMult(centered, inv_std_ct)
        scaled = cc.EvalMult(normalized, gamma_pt)
        scaled = cc.EvalAdd(scaled, beta_pt)
        activated = poly_act_fhe(cc, scaled, a_, b_, c_)
        out_rows.append(activated)

    print("\nVerifica finale contro riferimento numpy...")
    ref_out = np.zeros((Cout, H, W))
    for ch in range(Cout):
        normalized_ref = (x[ch] - mean_ref[ch]) / np.sqrt(var_ref[ch] + 1e-5)
        scaled_ref = normalized_ref * gamma[ch] + beta[ch]
        ref_out[ch] = a_ * scaled_ref**2 + b_ * scaled_ref + c_

    max_err = 0.0
    for r in range(H):
        pt = cc.Decrypt(keys.secretKey, out_rows[r])
        pt.SetLength(batch_size)
        he_row = np.array(pt.GetRealPackedValue())[:Wp * n].reshape(Wp, n)[:W, :Cout]
        ref_row = ref_out[:, r, :].T
        err = np.max(np.abs(he_row - ref_row))
        max_err = max(max_err, err)
        print(f"  Riga {r}: errore max = {err:.6e}")

    print(f"\nErrore massimo: {max_err:.6e}")
    if max_err < 1e-2:
        print("\n=== FUNZIONA: InstanceNorm+PolyAct nel formato a righe, in HE vero. ===")
    else:
        print("\n=== ATTENZIONE: errore grande -- da investigare. ===")


if __name__ == '__main__':
    main()