"""
crypto/fhe_ops_single_ciphertext.py (v2)

AGGIORNAMENTO IMPORTANTE rispetto alla v1: crop_valid_fhe ed
embed_valid_into_padded_fhe erano O(n_pixel) -- un ciclo con una
rotazione+maschera+somma per OGNI pixel. Misurato su Zeus (A40): a
64x56 (3.584 pixel), un solo crop+embed per canale costava ~920
secondi -- a piena risoluzione (256x224, 16x piu' pixel) sarebbe
costato ore per un solo ConvBlock, del tutto impraticabile per
un'intera rete di 11 blocchi.

LA CORREZIONE: crop ed embed servivano solo ad azzerare il bordo
(halo) "sporco" dal wraparound della convoluzione, prima che
contaminasse norm/act. Questo si ottiene con UNA SOLA moltiplicazione
per una maschera precalcolata (mask_border_fhe) -- costo O(1), non
O(n_pixel) -- invece di ritagliare e ricompattare i dati pixel per
pixel. Il ciphertext resta sempre alla dimensione "piena" (con halo);
solo i valori al bordo vengono azzerati.

Le statistiche di InstanceNorm si adattano di conseguenza
(instance_stats_padded_fhe): si somma su TUTTO l'array (bordo gia'
zero, contribuisce 0), dividendo pero' per il numero di pixel VALIDI,
non per il totale -- nessun ritaglio necessario nemmeno li'.

Le funzioni vecchie (crop_valid_fhe, downsample_stride2_fhe,
embed_valid_into_padded_fhe) restano nel file: downsample_stride2_fhe
serve ancora per il vero stride 2 (sottocampionamento reale, non un
semplice mascheramento -- li' l'operazione per pixel e' inevitabile,
anche se andra' probabilmente ottimizzata allo stesso modo piu' avanti
se risultasse un collo di bottiglia simile). crop/embed restano come
riferimento storico, ma conv_block_fhe (la versione corrente) non le
usa piu'.
"""

import math


# ============================================================
# PolyAct -- invariata
# ============================================================

def poly_act_fhe(cc, ct_x, a, b, c):
    """a*x^2 + b*x + c."""
    x_squared = cc.EvalMult(ct_x, ct_x)
    ax2 = cc.EvalMult(x_squared, a)
    bx = cc.EvalMult(ct_x, b)
    result = cc.EvalAdd(ax2, bx)
    result = cc.EvalAdd(result, c)
    return result


# ============================================================
# Broadcast -- invariata, generalizzata a n arbitrario (verificato
# funzionante su n=100 non potenza di 2, su Zeus)
# ============================================================

def broadcast_slot0(cc, ct, n_elements):
    """Porta il valore nello slot 0 su tutti gli n_elements slot."""
    n_steps = math.ceil(math.log2(n_elements)) if n_elements > 1 else 0
    n_slots_covered = 1 << n_steps

    mask = [1.0] + [0.0] * (n_slots_covered - 1)
    pt_mask = cc.MakeCKKSPackedPlaintext(mask)
    ct_masked = cc.EvalMult(ct, pt_mask)

    step = 1
    while step < n_slots_covered:
        ct_masked = cc.EvalAdd(ct_masked, cc.EvalRotate(ct_masked, -step))
        step *= 2

    return ct_masked


# ============================================================
# NUOVO: mascheramento del bordo, O(1) invece di O(n_pixel)
# ============================================================

def mask_border_fhe(cc, ct, tile_hp, tile_wp, tile_h, tile_w, halo=1):
    """
    Azzera il bordo (halo) di un'immagine con padding, con UNA SOLA
    moltiplicazione per una maschera precalcolata -- sostituisce sia
    crop_valid_fhe SIA embed_valid_into_padded_fhe (che costavano
    O(tile_h*tile_w) operazioni, una per pixel).

    Il dato NON viene mai spostato/compattato: resta sempre nella
    forma "piena" (tile_hp x tile_wp, flatten row-major) -- solo i
    valori al bordo vengono azzerati. La maschera (un plaintext) va
    ricalcolata solo se cambiano le dimensioni, non a ogni chiamata
    in un ciclo -- se questa funzione viene chiamata molte volte con
    le STESSE dimensioni, conviene precalcolare la maschera una volta
    fuori e passarla, invece di ricostruirla ogni volta (vedi
    mask_border_fhe_precomputed sotto per quel caso).
    """
    mask = []
    for r in range(tile_hp):
        for c in range(tile_wp):
            valid = (0 <= r < tile_h) and (0 <= c < tile_w)
            mask.append(1.0 if valid else 0.0)
    pt_mask = cc.MakeCKKSPackedPlaintext(mask)
    return cc.EvalMult(ct, pt_mask)


def make_border_mask_plaintext(cc, tile_hp, tile_wp, tile_h, tile_w, halo=1):
    """
    Precalcola la maschera di bordo come plaintext, da riusare su piu'
    chiamate senza doverla ricostruire ogni volta (utile quando si
    processano molti canali/blocchi con le stesse dimensioni, come
    nell'intera rete -- la maschera per una data risoluzione e' sempre
    la stessa).
    """
    mask = []
    for r in range(tile_hp):
        for c in range(tile_wp):
            valid = (0 <= r < tile_h) and (0 <= c < tile_w)
            mask.append(1.0 if valid else 0.0)
    return cc.MakeCKKSPackedPlaintext(mask)


def mask_border_fhe_precomputed(cc, ct, pt_mask):
    """Versione che riusa una maschera gia' precalcolata (vedi
    make_border_mask_plaintext) -- preferibile quando si processano
    molti ciphertext con le stesse dimensioni, per non rifare il ciclo
    Python di costruzione della maschera ogni volta (quel ciclo e' in
    chiaro, quindi economico, ma non c'e' motivo di ripeterlo)."""
    return cc.EvalMult(ct, pt_mask)


# ============================================================
# NUOVO: statistiche di InstanceNorm su ciphertext "pieno" (con
# bordo gia' azzerato) -- nessun ritaglio necessario.
# ============================================================

def instance_stats_padded_fhe(cc, ct_x_masked, n_valid_pixels, n_total_slots):
    """
    Media e varianza per-istanza, calcolate su un ciphertext GIA'
    mascherato (bordo a zero, vedi mask_border_fhe) di dimensione
    n_total_slots (l'intera immagine CON halo).

    Il bordo azzerato contribuisce 0 alla somma e alla somma dei
    quadrati -- sommando su TUTTO l'array e dividendo per
    n_valid_pixels (non n_total_slots) si ottiene la media/varianza
    corretta sui soli pixel validi, senza alcun ritaglio.
    """
    sum_x = cc.AccumulateSum(ct_x_masked, n_total_slots, stride=1)
    sum_x = broadcast_slot0(cc, sum_x, n_total_slots)
    mean = cc.EvalMult(sum_x, 1.0 / n_valid_pixels)

    x_squared = cc.EvalMult(ct_x_masked, ct_x_masked)
    sum_x2 = cc.AccumulateSum(x_squared, n_total_slots, stride=1)
    sum_x2 = broadcast_slot0(cc, sum_x2, n_total_slots)
    mean_of_sq = cc.EvalMult(sum_x2, 1.0 / n_valid_pixels)

    mean_squared = cc.EvalMult(mean, mean)
    variance = cc.EvalSub(mean_of_sq, mean_squared)

    return mean, variance


# ============================================================
# Radice inversa -- invariata
# ============================================================

def isqrt_chebyshev_fhe(cc, ct_var, cheb_coeffs, cheb_domain, post_iter):
    """Inizializzazione via EvalChebyshevSeries (dominio [-1,1], mapping
    esplicito) + rifinitura Newton."""
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
    """Genera coefficienti Chebyshev per 1/sqrt(x), dominio [-1,1]."""
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


# ============================================================
# Convoluzione -- invariata
# ============================================================

def conv2d_multichannel_fhe(cc, ct_channels_in, weight, bias, tile_hp, tile_wp, K=3):
    """Convoluzione multi-canale, algoritmo SISO."""
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
# VECCHIE crop/embed -- MANTENUTE come riferimento storico, ma NON
# PIU' USATE da conv_block_fhe. Restano O(n_pixel), troppo lente per
# uso ripetuto su immagini grandi -- vedi mask_border_fhe sopra.
# ============================================================

def downsample_stride2_fhe(cc, ct, tile_hp, tile_wp, tile_h, tile_w, stride=2):
    out_h, out_w = tile_h // stride, tile_w // stride
    out_len = out_h * out_w
    n_total = tile_hp * tile_wp  # <-- NUOVO: lunghezza vera del ciphertext
    result = None
    for out_idx in range(out_len):
        r_out, c_out = divmod(out_idx, out_w)
        src_idx = (r_out * stride) * tile_wp + (c_out * stride)
        shift = src_idx - out_idx
        shifted = cc.EvalRotate(ct, shift) if shift != 0 else ct
        mask = [1.0 if i == out_idx else 0.0 for i in range(n_total)]  # <-- FIX: n_total, non out_len
        pt_mask = cc.MakeCKKSPackedPlaintext(mask)
        term = cc.EvalMult(shifted, pt_mask)
        result = term if result is None else cc.EvalAdd(result, term)
    return result


def crop_valid_fhe(cc, ct, tile_hp, tile_wp, tile_h, tile_w):
    """DEPRECATA per l'uso 'azzera il bordo' -- usa mask_border_fhe.
    Mantenuta solo come riferimento storico (O(n_pixel), lenta)."""
    return downsample_stride2_fhe(cc, ct, tile_hp, tile_wp, tile_h, tile_w, stride=1)


def embed_valid_into_padded_fhe(cc, ct_compact, tile_h, tile_w, tile_hp, tile_wp, halo=1):
    """DEPRECATA -- usa mask_border_fhe (che non richiede nemmeno
    questo passo: il dato resta sempre alla dimensione piena)."""
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


def instance_stats_fhe(cc, ct_x, n_pixels):
    """DEPRECATA per l'uso principale -- usa instance_stats_padded_fhe.
    Mantenuta per compatibilita' con codice che lavora gia' su
    ciphertext compatti (senza halo), es. test isolati."""
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


def instance_norm_fhe(cc, ct_x, n_pixels, gamma, beta, cheb_coeffs, cheb_domain, post_iter):
    """DEPRECATA per l'uso principale (lavora su ciphertext compatto,
    senza halo) -- il ConvBlock corrente usa la logica equivalente ma
    inline con instance_stats_padded_fhe, per lavorare direttamente sul
    ciphertext pieno senza mai ritagliare. Mantenuta per compatibilita'."""
    mean, variance = instance_stats_fhe(cc, ct_x, n_pixels)
    inv_std = isqrt_chebyshev_fhe(cc, variance, cheb_coeffs, cheb_domain, post_iter)

    centered = cc.EvalSub(ct_x, mean)
    normalized = cc.EvalMult(centered, inv_std)
    scaled = cc.EvalMult(normalized, gamma)
    result = cc.EvalAdd(scaled, beta)

    return result


# ============================================================
# Stride 2 e ConvTranspose2d -- invariate
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
    """kernel==stride, nessuna sovrapposizione."""
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
# Skip connection -- invariata
# ============================================================

def skip_connection_sum_fhe(cc, ct_upsampled_channels, ct_skip_channels):
    assert len(ct_upsampled_channels) == len(ct_skip_channels)
    return [cc.EvalAdd(up, skip) for up, skip in zip(ct_upsampled_channels, ct_skip_channels)]


# ============================================================
# ConvBlock -- NUOVA VERSIONE, O(1) invece di O(n_pixel) per la
# gestione del bordo. Il ciphertext resta sempre alla dimensione
# piena (img_hp x img_wp); mai ritagliato/reimpacchettato.
# ============================================================

def conv_block_fhe(cc, ct_channels_in,
                    conv1_w, conv1_b, gamma1, beta1,
                    conv2_w, conv2_b, gamma2, beta2,
                    cheb_coeffs, cheb_domain, post_iter,
                    a1, b1_, c1, a2, b2_, c2,
                    img_h, img_w, halo=1, K=3,
                    pt_mask=None):
    """
    ConvBlock completo (Conv->Norm->Act->Conv->Norm->Act), multi-
    canale, su un solo ciphertext per canale (l'intera immagine, CON
    halo, sempre alla dimensione piena img_hp x img_wp).

    pt_mask: maschera di bordo GIA' precalcolata (vedi
    make_border_mask_plaintext) -- se fornita, evita di ricostruirla
    ogni volta che questa funzione viene chiamata (utile quando si
    processano molti blocchi con le stesse dimensioni, come nella rete
    intera). Se None, la calcola internamente (comodo per test isolati).

    Returns: lista di Cout ciphertext, dimensione PIENA (img_hp x
    img_wp) -- il bordo dell'ULTIMO output non e' necessariamente zero
    (l'ultima attivazione puo' produrre valori non-zero al bordo); se
    il prossimo passo e' un'altra convoluzione, va comunque mascherato
    di nuovo PRIMA di quella (la maschera va applicata subito dopo ogni
    conv, non dopo l'ultima activation di un blocco, a meno che serva
    esplicitamente un output pulito).
    """
    img_hp, img_wp = img_h + 2*halo, img_w + 2*halo
    n_valid = img_h * img_w
    n_total = img_hp * img_wp
    Cout = conv1_w.shape[0]

    if pt_mask is None:
        pt_mask = make_border_mask_plaintext(cc, img_hp, img_wp, img_h, img_w, halo)

    def masked(ct):
        return mask_border_fhe_precomputed(cc, ct, pt_mask)

    # --- Primo conv + norm + act ---
    x1 = conv2d_multichannel_fhe(cc, ct_channels_in, conv1_w, conv1_b, img_hp, img_wp, K=K)

    x1_out = []
    for co in range(Cout):
        x1_masked = masked(x1[co])
        mean, var = instance_stats_padded_fhe(cc, x1_masked, n_valid, n_total)
        inv_std = isqrt_chebyshev_fhe(cc, var, cheb_coeffs, cheb_domain, post_iter)
        centered = cc.EvalSub(x1_masked, mean)
        normalized = cc.EvalMult(centered, inv_std)
        scaled = cc.EvalMult(normalized, gamma1[co])
        norm_out = cc.EvalAdd(scaled, beta1[co])
        act_out = poly_act_fhe(cc, norm_out, a1, b1_, c1)
        # Ri-azzera il bordo (norm/act possono averlo reso non-zero)
        # prima della prossima convoluzione, che altrimenti farebbe
        # wraparound su valori sporchi.
        x1_out.append(masked(act_out))

    # --- Secondo conv + norm + act ---
    x2 = conv2d_multichannel_fhe(cc, x1_out, conv2_w, conv2_b, img_hp, img_wp, K=K)

    out = []
    for co in range(Cout):
        x2_masked = masked(x2[co])
        mean, var = instance_stats_padded_fhe(cc, x2_masked, n_valid, n_total)
        inv_std = isqrt_chebyshev_fhe(cc, var, cheb_coeffs, cheb_domain, post_iter)
        centered = cc.EvalSub(x2_masked, mean)
        normalized = cc.EvalMult(centered, inv_std)
        scaled = cc.EvalMult(normalized, gamma2[co])
        norm_out = cc.EvalAdd(scaled, beta2[co])
        out.append(poly_act_fhe(cc, norm_out, a2, b2_, c2))

    return out