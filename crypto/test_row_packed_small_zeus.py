"""
crypto/test_row_packed_small_zeus.py

Primo test in HE VERO (non piu' numpy) dello schema a righe + metodo
delle diagonali, verificato oggi in tre passaggi su numpy
(prototype_diagonal_matvec.py, prototype_diagonal_conv_spatial.py,
prototype_row_packed_conv.py + la correzione della rotazione a
blocchi in test_block_rotation_issue.py).

CASO VOLUTAMENTE PICCOLO (Cin=4, Cout=6, immagine 4x4) per evitare
out-of-memory come stamattina -- qui l'obiettivo e' solo verificare
che la LOGICA regga anche in HE vero, non misurare velocita' o
scalare alla larghezza reale.

Dettaglio importante: il batch CKKS e' impostato ESATTAMENTE alla
lunghezza usata (Wp*n), non al massimo 65536 come negli altri script
-- la correzione della rotazione a blocchi (due rotazioni + maschere)
assume che il "giro" della rotazione avvenga esattamente li', non a
65536. Se si lasciasse batch=65536 la maschera sarebbe calcolata per
il punto sbagliato.

Niente bootstrap in questo test -- la profondita' richiesta e'
minima (poche moltiplicazioni per diagonale), e l'obiettivo e' solo
la correttezza.

Uso: python3 crypto/test_row_packed_small_zeus.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe

GiB = 1 << 30
DEPTH = 15  # profondita' minima, niente bootstrap, caso piccolo
RING_POW = 14  # ring piccolo, sufficiente per poche decine di slot -- piu' veloce da costruire


def build_small_context(batch_size):
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(50)
    params.SetFirstModSize(55)
    params.SetBatchSize(batch_size)  # <-- ESATTAMENTE Wp*n, non 65536
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetDevices([0])

    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # Chiavi di rotazione: tutti gli shift che useremo -- gli offset
    # verticali/orizzontali del kernel (via indice di riga/colonna) e
    # tutti i possibili d (0..n-1) e (d-n) per la correzione a blocchi.
    return cc, keys


def correct_block_rotate_fhe(cc, ct, d, block_size, total_len, mask_main_pt, mask_wrap_pt):
    """Versione HE della correzione vista in test_block_rotation_issue.py."""
    if d == 0:
        return ct
    rotated_main = cc.EvalRotate(ct, d)
    rotated_wrap = cc.EvalRotate(ct, d - block_size)
    term_main = cc.EvalMult(rotated_main, mask_main_pt)
    term_wrap = cc.EvalMult(rotated_wrap, mask_wrap_pt)
    return cc.EvalAdd(term_main, term_wrap)


def make_masks(cc, total_len, block_size, d):
    rel_pos = np.arange(total_len) % block_size
    mask_main = (rel_pos < (block_size - d)).astype(float).tolist()
    mask_wrap = (1.0 - np.array(mask_main)).tolist()
    return cc.MakeCKKSPackedPlaintext(mask_main), cc.MakeCKKSPackedPlaintext(mask_wrap)


def diagonal_mix_channels_fhe(cc, ct_row, W_k, Cin, Cout, Wp, n):
    W_padded = np.zeros((n, n))
    W_padded[:Cout, :Cin] = W_k
    total_len = Wp * n

    acc = None
    for d in range(n):
        diag = np.array([W_padded[i, (i + d) % n] for i in range(n)])
        diag_tiled = cc.MakeCKKSPackedPlaintext(np.tile(diag, Wp).tolist())

        if d == 0:
            row_rot = ct_row
        else:
            mask_main_pt, mask_wrap_pt = make_masks(cc, total_len, n, d)
            row_rot = correct_block_rotate_fhe(cc, ct_row, d, n, total_len, mask_main_pt, mask_wrap_pt)

        term = cc.EvalMult(row_rot, diag_tiled)
        acc = term if acc is None else cc.EvalAdd(acc, term)

    return acc


def conv2d_row_packed_fhe(cc, rows_in_ct, weight, bias, Cin, Cout, Wp, n, halo=1, K=3):
    Hp = len(rows_in_ct)
    rows_out = [None] * Hp

    for r_out in range(Hp):
        for ky in range(K):
            r_in = r_out + ky
            if not (0 <= r_in < Hp):
                continue
            for kx in range(K):
                # shift orizzontale: ruota la riga di kx PIXEL = kx*n SLOT
                # (usando la stessa correzione a blocchi, ma con
                # block_size=n e offset kx*n -- shift di un numero
                # INTERO di blocchi, quindi NON attraversa un confine
                # di blocco a meta' -- una singola rotazione globale
                # basta qui, senza bisogno della correzione).
                shift_slots = kx * n
                row_shifted = ct_row = rows_in_ct[r_in] if shift_slots == 0 else cc.EvalRotate(rows_in_ct[r_in], shift_slots)

                W_k = weight[:, :, ky, kx]
                contrib = diagonal_mix_channels_fhe(cc, row_shifted, W_k, Cin, Cout, Wp, n)
                rows_out[r_out] = contrib if rows_out[r_out] is None else cc.EvalAdd(rows_out[r_out], contrib)

    # Aggiungi il bias (un plaintext ripetuto per ogni pixel della riga)
    bias_padded = np.zeros(n)
    bias_padded[:Cout] = bias
    bias_tiled = cc.MakeCKKSPackedPlaintext(np.tile(bias_padded, Wp).tolist())
    for r in range(Hp):
        if rows_out[r] is not None:
            rows_out[r] = cc.EvalAdd(rows_out[r], bias_tiled)

    return rows_out


def main():
    print("=== Primo test HE VERO: schema a righe + metodo delle diagonali (caso piccolo) ===\n")

    Cin, Cout = 4, 6
    H, W_img = 4, 4
    halo = 1
    K = 3
    Hp, Wp = H + 2*halo, W_img + 2*halo
    n = max(Cin, Cout)
    total_len = Wp * n

    print(f"Cin={Cin}, Cout={Cout}, immagine {H}x{W_img} (+halo) -> Hp={Hp}, Wp={Wp}")
    print(f"n=max(Cin,Cout)={n}, lunghezza per riga (batch CKKS) = Wp*n = {total_len}\n")

    print("Costruzione contesto piccolo...")
    t0 = time.time()
    cc, keys = build_small_context(total_len)

    rot_offsets = set()
    for kx in range(K):
        rot_offsets.add(kx * n)
    for d in range(1, n):
        rot_offsets.add(d)
        rot_offsets.add(d - n)
    rot_list = sorted(rot_offsets)
    cc.EvalRotateKeyGen(keys.secretKey, rot_list)
    cc.SetRotationKeyCache(1 * GiB)
    cc.LoadContext(keys.publicKey)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(1 * GiB)
    print(f"Contesto pronto in {time.time()-t0:.1f}s ({len(rot_list)} chiavi di rotazione).\n")

    rng = np.random.default_rng(7)
    x = rng.normal(size=(Cin, H, W_img))
    x_padded = np.pad(x, ((0, 0), (halo, halo), (halo, halo)))
    weight = rng.normal(size=(Cout, Cin, K, K)) * 0.1
    bias = rng.normal(size=(Cout,)) * 0.05

    def pack_row(r):
        row = x_padded[:, r, :].T  # (Wp, Cin)
        row_padded = np.zeros((Wp, n))
        row_padded[:, :Cin] = row
        return row_padded.flatten()

    print("Cifratura delle righe...")
    rows_in_ct = []
    for r in range(Hp):
        pt = cc.MakeCKKSPackedPlaintext(pack_row(r).tolist())
        rows_in_ct.append(cc.Encrypt(keys.publicKey, pt))
    print(f"{Hp} righe cifrate.\n")

    print("Convoluzione HE (schema a righe + diagonali)...")
    t0 = time.time()
    rows_out_ct = conv2d_row_packed_fhe(cc, rows_in_ct, weight, bias, Cin, Cout, Wp, n, halo=halo, K=K)
    print(f"Completata in {time.time()-t0:.2f}s.\n")

    def decrypt_row(ct, length):
        pt = cc.Decrypt(keys.secretKey, ct)
        pt.SetLength(length)
        return np.array(pt.GetRealPackedValue())

    print("Decifrazione e confronto col riferimento numpy diretto...")

    def conv2d_direct(x, W, b, K=3):
        Cin_, Hp_, Wp_ = x.shape
        Cout_ = W.shape[0]
        out = np.zeros((Cout_, Hp_, Wp_))
        for co in range(Cout_):
            for ci in range(Cin_):
                for ky in range(K):
                    for kx in range(K):
                        shifted = np.roll(np.roll(x[ci], -ky, axis=0), -kx, axis=1)
                        out[co] += W[co, ci, ky, kx] * shifted
        for co in range(Cout_):
            out[co] += b[co]
        return out

    ref = conv2d_direct(x_padded, weight, bias, K=K)

    max_err = 0.0
    for r in range(H):  # solo righe VALIDE (convenzione top-left)
        if rows_out_ct[r] is None:
            print(f"  Riga {r}: NESSUN CONTRIBUTO CALCOLATO (bug)")
            continue
        he_row = decrypt_row(rows_out_ct[r], total_len).reshape(Wp, n)[:W_img, :Cout]
        ref_row = ref[:, r, :W_img].T
        err = np.max(np.abs(he_row - ref_row))
        max_err = max(max_err, err)
        print(f"  Riga {r}: errore max = {err:.6e}")

    print(f"\nErrore massimo su tutte le righe valide: {max_err:.6e}")
    if max_err < 1e-3:
        print("\n=== FUNZIONA ANCHE IN HE VERO. Il concetto e' verificato end-to-end. ===")
    else:
        print("\n=== ATTENZIONE: differenza significativa -- da investigare prima di procedere. ===")


if __name__ == '__main__':
    main()