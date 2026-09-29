"""
crypto/fhe_ops_single_ciphertext.py

Versione SEMPLIFICATA della pipeline HE (FIDESlib), che elimina del
tutto il concetto di griglia multi-tile -- indicazione di Aurora:
usare N=2^17 (131.072), che da' 65.536 slot per ciphertext, sufficienti
a contenere l'intera immagine (256x224 = 57.344 pixel) in UN SOLO
ciphertext per canale, invece dei 2 tile usati finora.

Cosa e' cambiato rispetto alla versione a 2 tile:
- SPARISCE exchange_halo_2tiles_fhe (nessun bordo condiviso tra tile
  da riempire -- il bordo dell'immagine e' sempre e solo il vero
  bordo esterno, gia' gestito dallo zero-padding).
- SPARISCE combined_tile_stats_fhe (le statistiche di InstanceNorm si
  calcolano su un solo ciphertext, non piu' sommando due tile).
- RESTANO INVARIATE: poly_act_fhe, isqrt_chebyshev_fhe,
  fit_monotonic_isqrt_coeffs, conv2d_multichannel_fhe,
  downsample_stride2_fhe, expand_scatter_fhe,
  conv_transpose2d_single_channel_fhe, skip_connection_sum_fhe --
  non dipendevano dal numero di tile, solo dalle dimensioni passate.
- RESTA NECESSARIO il ritaglio/reinserimento dell'halo (crop_valid_fhe,
  embed_valid_into_padded_fhe): non e' un problema di tiling, e'
  dovuto al fatto che la convoluzione "sporca" comunque il bordo per
  il wraparound delle rotazioni, anche con un solo ciphertext che
  contiene l'intera immagine con il suo halo.

*** AVVISO IMPORTANTE, DA VERIFICARE SU ZEUS, NON DA DARE PER SCONTATO ***
L'immagine e' 256x224 = 57.344 pixel per canale -- NON una potenza di
2. broadcast_slot0 qui sotto e' stata generalizzata a ceil(log2(n))
passi per gestire n arbitrario, ma finora e' stata testata solo su
potenze di 2 esatte (8, 16). Il comportamento su un n reale non-potenza-
di-2, con un ciphertext che ha PIU' slot totali (65.536) di quanti ne
usa il dato reale (57.344), va verificato con un test dedicato PRIMA
di fidarsene nel resto della pipeline -- vedi test_broadcast_nonpow2()
in fondo al file.
"""

import math


# ============================================================
# PolyAct -- invariata rispetto alla versione a tile
# ============================================================

def poly_act_fhe(cc, ct_x, a, b, c):
    """a*x^2 + b*x + c. Nessuna dipendenza da tile/dimensioni."""
    x_squared = cc.EvalMult(ct_x, ct_x)
    ax2 = cc.EvalMult(x_squared, a)
    bx = cc.EvalMult(ct_x, b)
    result = cc.EvalAdd(ax2, bx)
    result = cc.EvalAdd(result, c)
    return result


# ============================================================
# Broadcast e statistiche -- generalizzate a n arbitrario (non piu'
# solo potenze di 2 piccole come nei test di ieri)
# ============================================================

def broadcast_slot0(cc, ct, n_elements):
    """
    Porta il valore nello slot 0 su tutti gli n_elements slot.

    Generalizzata: usa ceil(log2(n_elements)) passi di raddoppio,
    invece di richiedere che n_elements sia esattamente una potenza
    di 2. *** DA VERIFICARE SU ZEUS con n_elements=57344 (il caso
    reale) prima di fidarsene nel resto della pipeline. ***
    """
    n_steps = math.ceil(math.log2(n_elements)) if n_elements > 1 else 0
    n_slots_covered = 1 << n_steps  # prossima potenza di 2 >= n_elements

    mask = [1.0] + [0.0] * (n_slots_covered - 1)
    pt_mask = cc.MakeCKKSPackedPlaintext(mask)
    ct_masked = cc.EvalMult(ct, pt_mask)

    step = 1
    while step < n_slots_covered:
        ct_masked = cc.EvalAdd(ct_masked, cc.EvalRotate(ct_masked, -step))
        step *= 2

    return ct_masked


def instance_stats_fhe(cc, ct_x, n_pixels):
    """
    Media e varianza per-istanza su un SOLO ciphertext (l'intera
    immagine per quel canale, non piu' un tile). Stessa logica di
    ieri (AccumulateSum + broadcast), ma senza bisogno di sommare
    contributi da piu' tile.
    """
    sum_x = cc.AccumulateSum(ct_x, n_pixels, stride=1)
    sum_x = broadcast_slot0(cc, sum_x, n_pixels)
    mean = cc.EvalMult(sum_x, 1.0 / n_pixels)

    x_squared = cc.EvalMult(ct_x, ct_x)
    sum_x2 = cc.AccumulateSum(x_squared, n_pixels, stride=1)
    sum_x2 = broadcast_slot0(cc, sum_x2, n_pixels)
    mean_sq = cc.EvalMult(sum_x2, 1.0 / n_pixels)

    mean_squared = cc.EvalMult(mean, mean)
    variance = cc.EvalSub(mean_sq, mean_squared)

    return mean, variance


# ============================================================
# Radice inversa -- invariata (non dipendeva mai dai tile)
# ============================================================

def isqrt_chebyshev_fhe(cc, ct_var, cheb_coeffs, cheb_domain, post_iter):
    """
    Inizializzazione via EvalChebyshevSeries (che si aspetta input in
    [-1,1] -- il mapping va fatto esplicitamente) + rifinitura Newton.
    Identica alla versione di ieri, invariata dal tiling.
    """
    x_min, x_max = cheb_domain
    scale = 2.0 / (x_max - x_min)
    offset = -(x_min + x_max) / (x_max - x_min)

    ct_t = cc.EvalMult(ct_var, scale)
    ct_t = cc.EvalAdd(ct_t, offset)

    y = cc.EvalChebyshevSeries(ct_t, cheb_coeffs, -1.0, 1.0)

    for _ in range(post_iter):
        y_sq = cc.EvalMult(y, y)
        xy2 = cc.EvalMult(ct_var, y_sq)
        term = cc.EvalMult(xy2, -0.5)
        term = cc.EvalAdd(term, 1.5)
        y = cc.EvalMult(y, term)

    return y


def fit_monotonic_isqrt_coeffs(fhe_module, x_min, x_max, degree, extra_safety=1.2, n_check=500):
    """
    Genera coefficienti Chebyshev per 1/sqrt(x), garantiti compatibili
    con EvalChebyshevSeries (dominio [-1,1]). Identica a ieri -- non
    riceve piu' il modulo 'fhe' come import globale del notebook, ma
    come parametro esplicito, dato che qui siamo in un vero modulo .py
    (import fideslib_py as fhe va fatto nello script chiamante).
    """
    def target(t):
        x = x_min + (t + 1.0) / 2.0 * (x_max - x_min)
        return 1.0 / math.sqrt(x)

    raw_coeffs = fhe_module.GetChebyshevCoefficients(target, -1.0, 1.0, degree)

    def clenshaw_unit(coeffs, t):
        b1 = b2 = 0.0
        for c in reversed(coeffs[1:]):
            b1, b2 = 2.0 * t * b1 - b2 + c, b1
        return t * b1 - b2 + coeffs[0] / 2.0

    ts_check = [-1.0 + i * 2.0 / (n_check - 1) for i in range(n_check)]
    overshoot = max(clenshaw_unit(raw_coeffs, t) - target(t) for t in ts_check)
    shift = max(0.0, overshoot) * extra_safety

    if shift > 0:
        coeffs = fhe_module.GetChebyshevCoefficients(lambda t: target(t) - shift, -1.0, 1.0, degree)
    else:
        coeffs = raw_coeffs

    return coeffs, shift


def instance_norm_fhe(cc, ct_x, n_pixels, gamma, beta, cheb_coeffs, cheb_domain, post_iter):
    """InstanceNorm completa su un solo ciphertext (un canale, l'intera
    immagine)."""
    mean, variance = instance_stats_fhe(cc, ct_x, n_pixels)
    inv_std = isqrt_chebyshev_fhe(cc, variance, cheb_coeffs, cheb_domain, post_iter)

    centered = cc.EvalSub(ct_x, mean)
    normalized = cc.EvalMult(centered, inv_std)
    scaled = cc.EvalMult(normalized, gamma)
    result = cc.EvalAdd(scaled, beta)

    return result


# ============================================================
# Convoluzione -- invariata (la "griglia" ora e' semplicemente
# l'immagine intera con il suo halo, non un pezzo di essa)
# ============================================================

def conv2d_multichannel_fhe(cc, ct_channels_in, weight, bias, tile_hp, tile_wp, K=3):
    """
    Convoluzione multi-canale, algoritmo SISO (rotazione + moltiplica-
    zione scalare + somma). Identica a ieri -- qui tile_hp/tile_wp sono
    semplicemente H+2*halo, W+2*halo dell'INTERA immagine, non di un
    pezzo di essa.
    """
    Cout = weight.shape[0]
    Cin = len(ct_channels_in)
    ct_channels_out = []

    for co in range(Cout):
        acc = None
        for ci in range(Cin):
            for ky in range(K):
                for kx in range(K):
                    w = float(weight[co, ci, ky, kx])
                    offset = ky * tile_wp + kx
                    shifted = ct_channels_in[ci] if offset == 0 else cc.EvalRotate(ct_channels_in[ci], offset)
                    term = cc.EvalMult(shifted, w)
                    acc = term if acc is None else cc.EvalAdd(acc, term)
        acc = cc.EvalAdd(acc, float(bias[co]))
        ct_channels_out.append(acc)

    return ct_channels_out


# ============================================================
# Ritaglio / reinserimento dell'halo -- ANCORA NECESSARI: non sono
# un problema di tiling, ma del wraparound della convoluzione sul
# bordo del rettangolo (che ora e' il vero bordo immagine, non piu'
# un confine tra tile).
# ============================================================

def crop_valid_fhe(cc, ct, tile_hp, tile_wp, tile_h, tile_w):
    """Estrae la regione valida [0:H, 0:W] dall'immagine con halo,
    compattandola in H*W slot (stride=1, nessun sottocampionamento)."""
    return downsample_stride2_fhe(cc, ct, tile_hp, tile_wp, tile_h, tile_w, stride=1)


def downsample_stride2_fhe(cc, ct, tile_hp, tile_wp, tile_h, tile_w, stride=2):
    """Estrae/sottocampiona [0:tile_h,0:tile_w] da un'immagine con halo
    tile_hp x tile_wp, compattando l'output nei primi
    (tile_h//stride)*(tile_w//stride) slot. Invariata da ieri."""
    out_h, out_w = tile_h // stride, tile_w // stride
    out_len = out_h * out_w
    result = None
    for out_idx in range(out_len):
        r_out, c_out = divmod(out_idx, out_w)
        src_idx = (r_out * stride) * tile_wp + (c_out * stride)
        shift = src_idx - out_idx
        shifted = cc.EvalRotate(ct, shift) if shift != 0 else ct
        mask = [1.0 if i == out_idx else 0.0 for i in range(out_len)]
        pt_mask = cc.MakeCKKSPackedPlaintext(mask)
        term = cc.EvalMult(shifted, pt_mask)
        result = term if result is None else cc.EvalAdd(result, term)
    return result


def embed_valid_into_padded_fhe(cc, ct_compact, tile_h, tile_w, tile_hp, tile_wp, halo=1):
    """Inverso di crop_valid_fhe: rimette i valori compatti dentro
    un'immagine H+2h x W+2h, bordo lasciato a zero (il vero bordo
    esterno -- resta zero, niente scambio con nessuno). Invariata."""
    out_len = tile_hp * tile_wp
    result = None
    for r in range(tile_h):
        for c in range(tile_w):
            idx = r * tile_w + c
            dest = (r + halo) * tile_wp + (c + halo)
            shift = idx - dest
            shifted = cc.EvalRotate(ct_compact, shift) if shift != 0 else ct_compact
            mask = [1.0 if k == dest else 0.0 for k in range(out_len)]
            pt_mask = cc.MakeCKKSPackedPlaintext(mask)
            term = cc.EvalMult(shifted, pt_mask)
            result = term if result is None else cc.EvalAdd(result, term)
    return result


# ============================================================
# Stride 2 e ConvTranspose2d -- invariate, gia' generiche
# ============================================================

def expand_scatter_fhe(cc, ct_in, in_h, in_w, ky, kx, stride=2):
    out_h, out_w = in_h * stride, in_w * stride
    out_len = out_h * out_w
    result = None
    for i in range(in_h):
        for j in range(in_w):
            idx = i * in_w + j
            dest = (i * stride + ky) * out_w + (j * stride + kx)
            shift = idx - dest
            shifted = cc.EvalRotate(ct_in, shift) if shift != 0 else ct_in
            mask = [1.0 if k == dest else 0.0 for k in range(out_len)]
            pt_mask = cc.MakeCKKSPackedPlaintext(mask)
            term = cc.EvalMult(shifted, pt_mask)
            result = term if result is None else cc.EvalAdd(result, term)
    return result


def conv_transpose2d_single_channel_fhe(cc, ct_in, weight, bias, in_h, in_w, stride=2):
    """kernel==stride, nessuna sovrapposizione (vedi packing.py)."""
    acc = None
    for ky in range(stride):
        for kx in range(stride):
            w = float(weight[ky][kx])
            scattered = expand_scatter_fhe(cc, ct_in, in_h, in_w, ky, kx, stride)
            term = cc.EvalMult(scattered, w)
            acc = term if acc is None else cc.EvalAdd(acc, term)
    acc = cc.EvalAdd(acc, bias)
    return acc


# ============================================================
# Skip connection -- invariata (EvalAdd puro, mai dipeso dai tile)
# ============================================================

def skip_connection_sum_fhe(cc, ct_upsampled_channels, ct_skip_channels):
    assert len(ct_upsampled_channels) == len(ct_skip_channels)
    return [cc.EvalAdd(up, skip) for up, skip in zip(ct_upsampled_channels, ct_skip_channels)]


# ============================================================
# ConvBlock completo -- SEMPLIFICATO: niente piu' loop su
# top/bottom, niente scambio halo tra tile.
# ============================================================

def conv_block_fhe(cc, ct_channels_in,
                    conv1_w, conv1_b, gamma1, beta1,
                    conv2_w, conv2_b, gamma2, beta2,
                    cheb_coeffs, cheb_domain, post_iter,
                    a1, b1_, c1, a2, b2_, c2,
                    img_h, img_w, halo=1, K=3):
    """
    ConvBlock completo (Conv->Norm->Act->Conv->Norm->Act), multi-
    canale, SU UN SOLO CIPHERTEXT PER CANALE (l'intera immagine).

    Molto piu' semplice della versione a 2 tile di ieri: nessun loop
    su top/bottom, nessuno scambio di halo -- il bordo e' sempre e
    solo il vero bordo esterno dell'immagine (zero-padding, mai
    'condiviso' con nessun vicino).

    ct_channels_in: lista di Cin ciphertext, GIA' con halo (immagine
    img_h+2*halo x img_w+2*halo, flatten row-major).

    Returns: lista di Cout ciphertext, COMPATTI (img_h x img_w, SENZA
    halo) -- da re-impacchettare con embed_valid_into_padded_fhe prima
    del prossimo ConvBlock, esattamente come nella versione a tile.
    """
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_pixels = img_h * img_w
    Cout = conv1_w.shape[0]

    # --- Primo conv + norm + act ---
    x1 = conv2d_multichannel_fhe(cc, ct_channels_in, conv1_w, conv1_b, img_hp, img_wp, K=K)

    x1_padded = []
    for co in range(Cout):
        x1_valid = crop_valid_fhe(cc, x1[co], img_hp, img_wp, img_h, img_w)
        x1_norm = instance_norm_fhe(cc, x1_valid, n_pixels, gamma1[co], beta1[co],
                                     cheb_coeffs, cheb_domain, post_iter)
        x1_act = poly_act_fhe(cc, x1_norm, a1, b1_, c1)
        x1_padded.append(embed_valid_into_padded_fhe(cc, x1_act, img_h, img_w, img_hp, img_wp, halo))

    # --- Secondo conv + norm + act ---
    x2 = conv2d_multichannel_fhe(cc, x1_padded, conv2_w, conv2_b, img_hp, img_wp, K=K)

    out = []
    for co in range(Cout):
        x2_valid = crop_valid_fhe(cc, x2[co], img_hp, img_wp, img_h, img_w)
        x2_norm = instance_norm_fhe(cc, x2_valid, n_pixels, gamma2[co], beta2[co],
                                     cheb_coeffs, cheb_domain, post_iter)
        out.append(poly_act_fhe(cc, x2_norm, a2, b2_, c2))

    return out  # compatti, senza halo -- il chiamante decide se re-impacchettare


# ============================================================
# Test diagnostico da lanciare PER PRIMO su Zeus, prima di fidarsi
# del resto: verifica broadcast_slot0 su un n realistico, vicino
# (ma piu' piccolo, per velocita') alla vera dimensione dell'immagine.
# ============================================================

def test_broadcast_nonpow2(cc, keys, fhe_module):
    """
    Verifica broadcast_slot0 su n=100 (non potenza di 2, piccolo
    abbastanza da essere veloce) prima di fidarsene su n=57344.
    Chiamare cosi': test_broadcast_nonpow2(cc, keys, fhe)
    """
    import random
    n = 100
    test_values = [random.uniform(1, 10) for _ in range(n)]
    # padding a potenza di 2 successiva per la codifica (128 >= 100)
    n_pow2 = 1 << math.ceil(math.log2(n))
    padded = test_values + [0.0] * (n_pow2 - n)

    pt = cc.MakeCKKSPackedPlaintext(padded)
    ct = cc.Encrypt(keys.publicKey, pt)

    sum_ct = cc.AccumulateSum(ct, n, stride=1)
    broadcast_ct = broadcast_slot0(cc, sum_ct, n)

    pt_result = cc.Decrypt(keys.secretKey, broadcast_ct)
    pt_result.SetLength(n)
    result = pt_result.GetRealPackedValue()

    expected_sum = sum(test_values)
    max_err = max(abs(r - expected_sum) for r in result)
    print(f"Test broadcast n={n} (non potenza di 2): somma attesa={expected_sum:.4f}")
    print(f"Valori HE (primi 5): {result[:5]}")
    print(f"Errore massimo su tutti gli {n} slot: {max_err:.6e}")
    if max_err > 1e-3:
        print("\u26a0\ufe0f  ATTENZIONE: errore alto, broadcast_slot0 potrebbe non "
              "gestire correttamente n non potenza di 2 -- indagare prima di procedere.")
    else:
        print("OK: broadcast_slot0 funziona correttamente anche per n non potenza di 2.")