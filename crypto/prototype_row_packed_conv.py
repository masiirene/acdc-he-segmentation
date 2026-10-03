"""
crypto/prototype_row_packed_conv.py

PROTOTIPO IN CHIARO -- terzo e (si spera) definitivo passo dello schema
di impacchettamento: un ciphertext per RIGA dell'immagine (non per
canale, non per pixel), con tutti i Cin canali di quella riga
impacchettati negli slot.

Perche' le righe e non i pixel singoli: lo spostamento orizzontale del
kernel (kx) resta DENTRO lo stesso ciphertext (rotazione di Cin
posizioni = un pixel intero); lo spostamento verticale (ky) combina
righe ADIACENTI, gia' separate in modo naturale -- niente piu' il
problema dei "confini sporchi" tra pezzi che aveva reso impraticabile
il vecchio sistema a tile (eliminato settimane fa).

Con Wp=226 e Cin fino a 256 (il caso piu' largo reale): 226*256=57.856
slot -- entra in un solo ciphertext da 65.536 slot, con margine.

Questo script:
1. Impacchetta un'immagine (Cin, Hp, Wp) in una lista di Hp "righe",
   ciascuna un vettore di lunghezza Wp*Cin (layout: [pixel0_canale0,
   pixel0_canale1, ..., pixel0_canaleN, pixel1_canale0, ...]).
2. Implementa la convoluzione su questo schema: per ogni posizione di
   kernel (ky,kx), combina la riga giusta (shift verticale = quale
   riga della lista) con uno shift ORIZZONTALE dentro la riga
   (rotazione di kx*Cin posizioni), poi mescola i canali con il
   metodo delle diagonali.
3. Verifica contro la convoluzione diretta (stessa di prima).
4. Conta quanti ciphertext servono in totale e quanti vivi
   CONTEMPORANEAMENTE (non tutti insieme: bastano K righe alla volta).

Uso: python3 crypto/prototype_row_packed_conv.py
"""

import numpy as np


def conv2d_direct(x, W, b, K=3):
    Cin, Hp, Wp = x.shape
    Cout = W.shape[0]
    out = np.zeros((Cout, Hp, Wp))
    for co in range(Cout):
        for ci in range(Cin):
            for ky in range(K):
                for kx in range(K):
                    shifted = np.roll(np.roll(x[ci], -ky, axis=0), -kx, axis=1)
                    out[co] += W[co, ci, ky, kx] * shifted
    for co in range(Cout):
        out[co] += b[co]
    return out


def pack_into_rows(x):
    """(Cin, Hp, Wp) -> lista di Hp vettori, ciascuno (Wp*Cin,).
    Layout per riga: [pixel0_c0..pixel0_cN, pixel1_c0..pixel1_cN, ...]
    -- canali CONSECUTIVI per ogni pixel (necessario per il metodo
    delle diagonali, che ruota lungo l'asse canale)."""
    Cin, Hp, Wp = x.shape
    rows = []
    for r in range(Hp):
        row = x[:, r, :].T.flatten()  # (Wp, Cin) -> flatten riga per riga di pixel
        rows.append(row)
    return rows


def unpack_from_rows(rows, Cout, Wp):
    Hp = len(rows)
    out = np.zeros((Cout, Hp, Wp))
    for r in range(Hp):
        row_reshaped = rows[r].reshape(Wp, Cout)  # (Wp, Cout)
        out[:, r, :] = row_reshaped.T
    return out


def correct_block_rotate_flat(flat_vec, d, block_size):
    """Rotazione CORRETTA 'a blocchi indipendenti', usando SOLO
    operazioni che EvalRotate puo' davvero fare (rotazione GLOBALE
    dell'intero vettore piatto) -- vedi test_block_rotation_issue.py
    per la dimostrazione del problema e la verifica di questa
    correzione. Combina due rotazioni globali con due maschere
    complementari."""
    total_len = len(flat_vec)
    if d == 0:
        return flat_vec.copy()
    rotated_main = np.roll(flat_vec, -d)
    rotated_wrap = np.roll(flat_vec, -(d - block_size))
    rel_pos = np.arange(total_len) % block_size
    mask_main = (rel_pos < (block_size - d)).astype(float)
    mask_wrap = 1.0 - mask_main
    return mask_main * rotated_main + mask_wrap * rotated_wrap


def diagonal_mix_channels(row_vec, W_k, Cin, Cout, Wp):
    """Mescola i canali di UNA riga (gia' impacchettata, Wp*Cin slot)
    secondo la matrice W_k (Cout,Cin) per UNA posizione di kernel,
    usando il metodo delle diagonali -- applicato a TUTTI i Wp pixel
    della riga simultaneamente (broadcasting = SIMD).

    CORRETTO rispetto alla prima versione: la rotazione "dentro ogni
    blocco pixel" ora usa correct_block_rotate_flat (due rotazioni
    GLOBALI + maschere), non un comodo-ma-finto np.roll(axis=1) che
    in HE vero non corrisponde a nessuna operazione reale."""
    n = max(Cin, Cout)
    W_padded = np.zeros((n, n))
    W_padded[:Cout, :Cin] = W_k

    row_padded = np.zeros(Wp * n)
    row_reshaped = row_vec.reshape(Wp, Cin)
    row_padded_reshaped = row_padded.reshape(Wp, n)
    row_padded_reshaped[:, :Cin] = row_reshaped
    row_padded = row_padded_reshaped.flatten()  # vettore PIATTO, come un vero ciphertext

    out_flat = np.zeros(Wp * n)
    n_rot, n_pmult = 0, 0
    for d in range(n):
        diag = np.array([W_padded[i, (i + d) % n] for i in range(n)])
        diag_tiled = np.tile(diag, Wp)  # stessa diagonale ripetuta per ogni pixel della riga

        row_rot = correct_block_rotate_flat(row_padded, d, block_size=n)
        out_flat += diag_tiled * row_rot

        n_rot += 0 if d == 0 else 2  # 2 rotazioni globali per d!=0 (0 per d=0, e' l'identita')
        n_pmult += 1  # la maschera e' un plaintext, costa poco, ma la contiamo comunque

    out_reshaped = out_flat.reshape(Wp, n)
    return out_reshaped[:, :Cout].flatten(), n_rot, n_pmult


def conv2d_row_packed(rows_in, W, b, Cin, Cout, Wp, halo=1, K=3):
    Hp = len(rows_in)
    rows_out = [np.zeros(Wp * Cout) for _ in range(Hp)]
    total_rot, total_pmult = 0, 0

    for r_out in range(Hp):
        for ky in range(K):
            # NIENTE "-halo" qui: stessa convenzione di
            # conv2d_multichannel_fhe (offset = ky*tile_wp + kx, ky/kx
            # da 0 a K-1 SENZA centratura) -- l'offset del kernel va
            # "in avanti", non e' centrato sul pixel.
            r_in = r_out + ky
            if not (0 <= r_in < Hp):
                continue  # fuori dai bordi -- contributo zero (come il padding)
            for kx in range(K):
                # Shift ORIZZONTALE dentro la riga: stessa convenzione,
                # nessuna centratura.
                row_shifted = np.roll(rows_in[r_in].reshape(Wp, Cin), -kx, axis=0).flatten()
                W_k = W[:, :, ky, kx]
                contrib, n_rot, n_pmult = diagonal_mix_channels(row_shifted, W_k, Cin, Cout, Wp)
                rows_out[r_out] += contrib
                total_rot += n_rot
                total_pmult += n_pmult

    for r in range(Hp):
        rows_out[r] = rows_out[r].reshape(Wp, Cout)
        rows_out[r] += b[None, :]
        rows_out[r] = rows_out[r].flatten()

    return rows_out, total_rot, total_pmult


def main():
    print("=== Test: schema a RIGHE (un ciphertext per riga) vs convoluzione diretta ===\n")

    rng = np.random.default_rng(7)
    Cin, Cout = 6, 10
    H, W_img = 12, 10
    halo = 1
    K = 3
    Hp, Wp = H + 2*halo, W_img + 2*halo

    x = rng.normal(size=(Cin, H, W_img))
    x_padded = np.pad(x, ((0, 0), (halo, halo), (halo, halo)))

    weight = rng.normal(size=(Cout, Cin, K, K)) * 0.1
    bias = rng.normal(size=(Cout,)) * 0.05

    print(f"Immagine: Cin={Cin}, Cout={Cout}, {H}x{W_img} (+halo={halo}) -> Hp={Hp}, Wp={Wp}\n")

    out_direct = conv2d_direct(x_padded, weight, bias, K=K)

    rows_in = pack_into_rows(x_padded)
    rows_out, n_rot, n_pmult = conv2d_row_packed(rows_in, weight, bias, Cin, Cout, Wp, halo=halo, K=K)
    out_row_packed = unpack_from_rows(rows_out, Cout, Wp)

    # Convenzione del progetto: la regione VALIDA e' in ALTO A SINISTRA
    # (0:H, 0:W), non centrata -- stessa convenzione gia' stabilita
    # settimane fa per conv2d_multichannel_fhe. Confrontiamo quella.
    inner = out_direct[:, 0:H, 0:W_img]
    inner_rp = out_row_packed[:, 0:H, 0:W_img]
    err = np.max(np.abs(inner - inner_rp))

    print(f"Errore massimo (regione valida, convenzione top-left): {err:.2e}")
    print(f"Operazioni costose (rotazioni): {n_rot}")
    print(f"Moltiplicazioni plaintext: {n_pmult}")

    if err < 1e-8:
        print("\n=== COINCIDONO nella regione interna: la logica di base e' corretta. ===")
    else:
        print("\n=== ATTENZIONE: differenza significativa anche nella regione interna. ===")
        print("Valori diretto (regione interna, canale 0):")
        print(inner[0])
        print("Valori row-packed (regione interna, canale 0):")
        print(inner_rp[0])

    print(f"\n=== Quanti ciphertext servono, ai canali reali della rete ===")
    print(f"{'Stage':<8} {'Cin':>5} {'Cout':>5} {'Slot/riga (Cin)':>16} {'Slot/riga (Cout)':>17} {'Entra in 65536?':>16}")
    real_stages = [
        ("enc0", 1, 32), ("enc1", 32, 64), ("enc2", 64, 128),
        ("enc3", 128, 256), ("enc4", 256, 128),
        ("dec3", 256, 256), ("dec2", 128, 128), ("dec1", 64, 64), ("dec0", 32, 32),
    ]
    real_Wp = 226  # 224 + 2*halo
    for name, cin, cout in real_stages:
        slots_in = real_Wp * cin
        slots_out = real_Wp * cout
        fits = "SI" if max(slots_in, slots_out) <= 65536 else "NO"
        print(f"{name:<8} {cin:>5} {cout:>5} {slots_in:>16} {slots_out:>17} {fits:>16}")
    print(f"\nNumero di ciphertext per l'INTERA immagine a ogni stage: Hp = 258 (sempre, indipendente da Cin)")
    print(f"Ciphertext VIVI CONTEMPORANEAMENTE per calcolare una riga di output: K={K} (solo le righe adiacenti)")


if __name__ == '__main__':
    main()