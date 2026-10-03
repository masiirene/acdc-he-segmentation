"""
crypto/prototype_row_packed_convblock.py

PROTOTIPO IN CHIARO -- unisce i due pezzi verificati separatamente
oggi (prototype_row_packed_conv.py: convoluzione spaziale col metodo
delle diagonali + correzione a blocchi; prototype_row_packed_norm.py:
InstanceNorm+PolyAct) in un vero ConvBlock completo a DUE stadi
(Conv->Norm->Att->Conv->Norm->Att), tutto nel formato a righe.

Verificato contro il calcolo di riferimento standard (array per
canale), stessa convenzione "valida in alto a sinistra" gia'
stabilita nel progetto.

Uso: python3 crypto/prototype_row_packed_convblock.py
"""

import numpy as np


# ============================================================
# Pezzi della CONVOLUZIONE (da prototype_row_packed_conv.py,
# gia' verificati -- vedi quel file per i dettagli/commenti)
# ============================================================

def correct_block_rotate_flat(flat_vec, d, block_size):
    if d == 0:
        return flat_vec.copy()
    total_len = len(flat_vec)
    rotated_main = np.roll(flat_vec, -d)
    rotated_wrap = np.roll(flat_vec, -(d - block_size))
    rel_pos = np.arange(total_len) % block_size
    mask_main = (rel_pos < (block_size - d)).astype(float)
    mask_wrap = 1.0 - mask_main
    return mask_main * rotated_main + mask_wrap * rotated_wrap


def diagonal_mix_channels(row_vec, W_k, Cin, Cout, Wp, n):
    W_padded = np.zeros((n, n))
    W_padded[:Cout, :Cin] = W_k
    row_padded = np.zeros(Wp * n)
    row_reshaped = row_vec.reshape(Wp, n)  # row_vec e' GIA' alla larghezza n (gestito dal chiamante)
    row_padded = row_reshaped.flatten()

    out_flat = np.zeros(Wp * n)
    for d in range(n):
        diag = np.array([W_padded[i, (i + d) % n] for i in range(n)])
        diag_tiled = np.tile(diag, Wp)
        row_rot = correct_block_rotate_flat(row_padded, d, block_size=n)
        out_flat += diag_tiled * row_rot

    return out_flat.reshape(Wp, n)[:, :Cout].flatten()


def conv2d_row_packed(rows_in, weight, bias, Cin, Cout, Wp, n, halo=1, K=3):
    Hp = len(rows_in)
    rows_out = [np.zeros(Wp * n) for _ in range(Hp)]
    for r_out in range(Hp):
        for ky in range(K):
            r_in = r_out + ky
            if not (0 <= r_in < Hp):
                continue
            for kx in range(K):
                row_shifted = np.roll(rows_in[r_in].reshape(Wp, n), -kx, axis=0).flatten()
                W_k = weight[:, :, ky, kx]
                contrib = diagonal_mix_channels(row_shifted, W_k, Cin, Cout, Wp, n)
                contrib_full = np.zeros(Wp * n)
                contrib_full[:len(contrib)] = contrib
                rows_out[r_out] += contrib_full
    for r in range(Hp):
        rows_out[r] = rows_out[r].reshape(Wp, n)
        rows_out[r][:, :Cout] += bias[None, :]
        rows_out[r] = rows_out[r].flatten()
    return rows_out


# ============================================================
# Pezzi della NORMALIZZAZIONE (da prototype_row_packed_norm.py,
# gia' verificati)
# ============================================================

def sum_within_row_strided(row_vec, n, W):
    result = np.zeros_like(row_vec)
    partial = row_vec.copy()
    remaining = W
    shift_base = 0
    power = 1
    while remaining > 0:
        if remaining & 1:
            shifted = np.roll(partial, -shift_base * n)
            result = result + shifted
            shift_base += power
        remaining >>= 1
        if remaining > 0:
            partial = partial + np.roll(partial, -power * n)
            power *= 2
    return result


def compute_norm_stats(rows, n, W, n_valid_pixels):
    row_sums = [sum_within_row_strided(r, n, W) for r in rows]
    total_sum = row_sums[0].copy()
    for r in row_sums[1:]:
        total_sum = total_sum + r
    mean = total_sum / n_valid_pixels

    rows_sq = [r ** 2 for r in rows]
    row_sums_sq = [sum_within_row_strided(r, n, W) for r in rows_sq]
    total_sum_sq = row_sums_sq[0].copy()
    for r in row_sums_sq[1:]:
        total_sum_sq = total_sum_sq + r
    mean_sq = total_sum_sq / n_valid_pixels

    variance = mean_sq - mean ** 2
    return mean[:n], variance[:n]


def apply_norm_and_act(rows, mean, variance, gamma, beta, a, b, c, n, eps=1e-5):
    inv_std = 1.0 / np.sqrt(variance + eps)
    n_blocks = len(rows[0]) // n
    gamma_full = np.tile(gamma, n_blocks)
    beta_full = np.tile(beta, n_blocks)
    inv_std_full = np.tile(inv_std, n_blocks)
    mean_full = np.tile(mean, n_blocks)

    out_rows = []
    for row in rows:
        centered = row - mean_full
        normalized = centered * inv_std_full
        scaled = normalized * gamma_full + beta_full
        activated = a * scaled**2 + b * scaled + c
        out_rows.append(activated)
    return out_rows


def mask_border_rows(rows, n, Wp, W_valid, H_valid=None):
    """Azzera il bordo COMPLETO dell'alone: le colonne oltre W_valid
    DENTRO ogni riga (bordo orizzontale) E, se H_valid e' dato, le
    RIGHE INTERE oltre H_valid (bordo verticale) -- bug trovato oggi:
    senza azzerare anche le righe intere, il loro contenuto "naturale"
    (non zero) contaminava la convoluzione successiva, mentre il
    riferimento assume esplicitamente zero li' (np.pad con zeri prima
    del secondo conv)."""
    mask = np.zeros(Wp * n)
    mask_reshaped = mask.reshape(Wp, n)
    mask_reshaped[:W_valid, :] = 1.0
    mask = mask_reshaped.flatten()
    out = []
    for i, r in enumerate(rows):
        if H_valid is not None and i >= H_valid:
            out.append(np.zeros_like(r))  # riga intera dell'alone verticale -> zero
        else:
            out.append(r * mask)
    return out


# ============================================================
# Riferimento standard (array per canale), stessa convenzione
# "valida in alto a sinistra" del progetto
# ============================================================

def conv_ref(x_p, w, b, Cout, Cin, H, W, K):
    out = np.zeros((Cout, H, W))
    for co in range(Cout):
        for ci in range(Cin):
            for r in range(H):
                for c in range(W):
                    for ky in range(K):
                        for kx in range(K):
                            out[co, r, c] += w[co, ci, ky, kx] * x_p[ci, r+ky, c+kx]
        out[co] += b[co]
    return out


def norm_act_ref(x, gamma, beta, a, b, c, eps=1e-5):
    C = x.shape[0]
    out = np.zeros_like(x)
    for ch in range(C):
        mean, var = x[ch].mean(), x[ch].var()
        normalized = (x[ch] - mean) / np.sqrt(var + eps)
        scaled = normalized * gamma[ch] + beta[ch]
        out[ch] = a * scaled**2 + b * scaled + c
    return out


def unpack_rows(rows, C, n, W):
    H = len(rows)
    out = np.zeros((C, H, W))
    for r in range(H):
        out[:, r, :] = rows[r].reshape(-1, n)[:, :C].T
    return out


def pack_rows(x, n, halo):
    C, H, W = x.shape
    x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))
    Hp, Wp = H + 2*halo, W + 2*halo
    rows = []
    for r in range(Hp):
        row = np.zeros((Wp, n))
        row[:, :C] = x_padded[:, r, :].T
        rows.append(row.flatten())
    return rows, Hp, Wp


def main():
    print("=== ConvBlock COMPLETO (2 stadi) nel formato a righe, verificato ===\n")

    rng = np.random.default_rng(13)
    Cin, Cout = 4, 6
    H, W = 5, 5
    halo = 1
    K = 3
    n = max(Cin, Cout)

    x = rng.normal(size=(Cin, H, W))
    conv1_w = rng.normal(size=(Cout, Cin, K, K)) * 0.15
    conv1_b = rng.normal(size=(Cout,)) * 0.05
    conv2_w = rng.normal(size=(Cout, Cout, K, K)) * 0.15
    conv2_b = rng.normal(size=(Cout,)) * 0.05
    gamma1 = rng.normal(size=Cout) * 0.2 + 1.0
    beta1 = rng.normal(size=Cout) * 0.1
    gamma2 = rng.normal(size=Cout) * 0.2 + 1.0
    beta2 = rng.normal(size=Cout) * 0.1
    a_, b_, c_ = 0.1, 1.0, 0.5

    # ---- Riferimento standard ----
    x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))
    y1 = conv_ref(x_padded, conv1_w, conv1_b, Cout, Cin, H, W, K)
    y1n = norm_act_ref(y1, gamma1, beta1, a_, b_, c_)
    y1n_padded = np.pad(y1n, ((0, 0), (0, 2*halo), (0, 2*halo)))
    y2 = conv_ref(y1n_padded, conv2_w, conv2_b, Cout, Cout, H, W, K)
    ref_out = norm_act_ref(y2, gamma2, beta2, a_, b_, c_)

    # ---- Formato a righe ----
    rows_in, Hp, Wp = pack_rows(x, n, halo)

    # NOTA IMPORTANTE (bug trovato e corretto): mask_border_rows azzera
    # solo il bordo ORIZZONTALE (le colonne oltre W, dentro ogni riga).
    # Le righe dell'ALONE VERTICALE (oltre H, cioe' le righe r>=H fino
    # a Hp-1) hanno comunque un output di convoluzione non-zero, mai
    # azzerato -- vanno semplicemente ESCLUSE dalla somma per le
    # statistiche di norm, non azzerate (la convenzione "valida in alto
    # a sinistra" del progetto: le righe valide sono SEMPRE 0..H-1).
    rows_c1 = conv2d_row_packed(rows_in, conv1_w, conv1_b, Cin, Cout, Wp, n, halo=halo, K=K)
    rows_c1 = mask_border_rows(rows_c1, n, Wp, W, H_valid=H)
    mean1, var1 = compute_norm_stats(rows_c1[:H], n, Wp, n_valid_pixels=H*W)
    rows_n1 = apply_norm_and_act(rows_c1, mean1, var1, gamma1, beta1, a_, b_, c_, n)
    rows_n1 = mask_border_rows(rows_n1, n, Wp, W, H_valid=H)  # azzera ANCHE le righe intere

    rows_c2 = conv2d_row_packed(rows_n1, conv2_w, conv2_b, Cout, Cout, Wp, n, halo=halo, K=K)
    rows_c2 = mask_border_rows(rows_c2, n, Wp, W, H_valid=H)
    mean2, var2 = compute_norm_stats(rows_c2[:H], n, Wp, n_valid_pixels=H*W)
    rows_n2 = apply_norm_and_act(rows_c2, mean2, var2, gamma2, beta2, a_, b_, c_, n)

    out_row_packed = unpack_rows(rows_n2, Cout, n, Wp)[:, :H, :W]

    err = np.max(np.abs(out_row_packed - ref_out))
    print(f"Errore massimo, ConvBlock completo (2 stadi): {err:.2e}")
    if err < 1e-8:
        print("\n=== COINCIDE: il ConvBlock completo nel formato a righe e' corretto. ===")
    else:
        print("\n=== ATTENZIONE: differenza significativa. ===")
        print("Riferimento (canale 0):")
        print(ref_out[0])
        print("Row-packed (canale 0):")
        print(out_row_packed[0])


if __name__ == '__main__':
    main()