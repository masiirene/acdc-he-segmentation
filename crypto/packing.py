"""
crypto/packing.py

Prototipo numpy dello schema di packing per l'inferenza HE (CKKS) della
HEFriendlyUNet -- Fase 3, punto A dello schema di lavoro (Aurora).

Un ciphertext CKKS e' modellato come un vettore di lunghezza fissa su cui
sono disponibili solo tre operazioni: somma, moltiplicazione (per un altro
ciphertext o per una costante in chiaro), rotazione ciclica. Tutte le
funzioni in questo modulo sono scritte usando SOLO queste tre operazioni
(np.roll per la rotazione, + e * per somma/moltiplicazione), cosi' da
essere direttamente traducibili in chiamate a una libreria HE reale
(FIDESlib) in una fase successiva.

--------------------------------------------------------------------------
DESIGN: perche' il tiling con griglia fissa
--------------------------------------------------------------------------
Un singolo ciphertext ha un numero di slot limitato (N/2, con N il grado
dell'anello ciclotomico -- tipicamente qualche migliaio a qualche decina
di migliaia di slot). Una slice 2D dell'immagine ACDC a piena risoluzione
(256x224 = 57.344 pixel) non entra in un solo ciphertext con i parametri
CKKS tipici.

Soluzione adottata: dividere ogni canale in TILE (blocchi spaziali), ognuno
dei quali entra in un ciphertext. Ogni tile porta con se' un piccolo bordo
extra ("halo") che fornisce il contesto necessario alla convoluzione ai
bordi del tile, evitando output errati.

Il punto critico (segnalato esplicitamente da Aurora: "il packing deve
garantire continuita' tra i layer") e' che la rete ha diversi stride-2
(dimezzano la risoluzione spaziale ad ogni stage dell'encoder) e diversi
ConvTranspose2d stride-2 nel decoder (la raddoppiano). Se il numero di tile
cambiasse ad ogni stage, servirebbe repacking tra un layer e il successivo
-- operazione costosa che Aurora chiede esplicitamente di evitare.

La soluzione: fissare la GRIGLIA di tile (numero di tile, non la loro
dimensione in pixel) una volta per tutte, dimensionata sulla risoluzione
piu' alta della rete (l'input). La stessa identica griglia si applica a
OGNI stage: quando la risoluzione scende (stride-2), ogni tile diventa
piu' piccolo in pixel ma il LORO NUMERO resta invariato; quando risale
(ConvTranspose2d), i tile ricrescono simmetricamente. Questo elimina il
bisogno di repacking, e rende naturali anche le skip connection (un tile
dell'encoder e il tile "corrispondente" del decoder condividono lo stesso
indice di griglia).

Costo noto di questa scelta: nei layer piu' profondi (risoluzione bassa,
molti canali) i tile usano solo una piccola frazione della capacita' del
ciphertext -- inefficienza nota, da ottimizzare in una fase successiva
(non blocca la correttezza).

Verificato numericamente (vedi test_packing.py): tutte le funzioni qui
sotto producono output identici (a meno di arrotondamento float32) alle
corrispondenti operazioni PyTorch/numpy "in chiaro".
"""

import numpy as np


# ---------------------------------------------------------------------------
# Convoluzione 2D (encoder: enc0..enc5, stride 1 o 2, kernel 3x3, padding 1)
# ---------------------------------------------------------------------------

def tiled_conv2d(x, weight, bias, n_tiles_h, n_tiles_w, stride=1, halo=1, K=3):
    """
    Convoluzione 2D multi-canale, calcolata a tile con griglia fissa,
    usando solo rotazione (np.roll) + moltiplicazione scalare + somma
    (le uniche operazioni disponibili su un ciphertext CKKS).

    Ogni tile e' elaborato in modo indipendente con l'algoritmo "SISO"
    (Single-Input Single-Output, cfr. HyPHEN et al.): per ogni posizione
    (ky, kx) del kernel, il tile viene ruotato dell'offset corrispondente,
    moltiplicato per il peso scalare, e accumulato.

    Args:
        x:       (Cin, H, W) -- feature map di input
        weight:  (Cout, Cin, K, K) -- come nn.Conv2d.weight
        bias:    (Cout,)
        n_tiles_h, n_tiles_w: dimensioni della griglia di tile FISSA
                  (deve essere la stessa in tutta la rete, vedi docstring
                  del modulo)
        stride:  1 o 2
        halo:    contesto extra ai bordi di ogni tile (= padding, per K=3)
        K:       dimensione del kernel (3 per i blocchi conv del modello)

    Returns:
        (Cout, H//stride, W//stride)
    """
    Cin, H, W = x.shape
    Cout = weight.shape[0]
    assert H % n_tiles_h == 0 and W % n_tiles_w == 0, \
        "La risoluzione deve essere divisibile per la griglia di tile"

    tile_h, tile_w = H // n_tiles_h, W // n_tiles_w
    out_h, out_w = H // stride, W // stride
    out_tile_h, out_tile_w = out_h // n_tiles_h, out_w // n_tiles_w

    out = np.zeros((Cout, out_h, out_w), dtype=np.float32)

    for ti in range(n_tiles_h):
        for tj in range(n_tiles_w):
            row_start, col_start = ti * tile_h, tj * tile_w
            tile_hp, tile_wp = tile_h + 2 * halo, tile_w + 2 * halo

            for co in range(Cout):
                acc = np.zeros(tile_h * tile_w, dtype=np.float32)
                for ci in range(Cin):
                    # In HE: x[ci] sarebbe gia' un ciphertext per canale/tile;
                    # qui il padding globale simula il contesto ai bordi
                    # dell'immagine (in HE andrebbe gestito esplicitamente,
                    # es. con zero-padding incluso nel packing).
                    x_padded_global = np.pad(x[ci], halo)
                    tile_with_halo = x_padded_global[
                        row_start:row_start + tile_hp,
                        col_start:col_start + tile_wp
                    ]
                    flat = tile_with_halo.flatten()  # <- il "ciphertext" del tile

                    for ky in range(K):
                        for kx in range(K):
                            w = weight[co, ci, ky, kx]
                            offset = ky * tile_wp + kx
                            # rotazione: np.roll <-> EvalRotate su ciphertext
                            shifted = np.roll(flat, -offset)
                            shifted_2d = shifted.reshape(tile_hp, tile_wp)[:tile_h, :tile_w]
                            # moltiplicazione scalare + somma: cMult + Add
                            acc += w * shifted_2d.flatten()

                full_res_tile_out = acc.reshape(tile_h, tile_w) + bias[co]
                # Lo stride si implementa come sotto-campionamento del
                # risultato: in HE richiede una maschera + rotazioni per
                # "compattare" gli slot selezionati (non prototipato qui,
                # e' un'ottimizzazione successiva).
                strided_tile_out = full_res_tile_out[::stride, ::stride]

                out_row_start = ti * out_tile_h
                out_col_start = tj * out_tile_w
                out[co,
                    out_row_start:out_row_start + out_tile_h,
                    out_col_start:out_col_start + out_tile_w] = strided_tile_out

    return out


# ---------------------------------------------------------------------------
# ConvTranspose2d (decoder: up0..up4, kernel=stride=2 -> nessun overlap)
# ---------------------------------------------------------------------------

def tiled_conv_transpose2d(x, weight, bias, n_tiles_h, n_tiles_w, K=2, stride=2):
    """
    ConvTranspose2d con kernel_size=stride=2 (come nel decoder del modello):
    caso particolarmente semplice perche' kernel==stride implica che i
    blocchi di output NON si sovrappongono mai -- quindi, a differenza di
    tiled_conv2d, non serve alcun halo: ogni tile di output dipende
    ESCLUSIVAMENTE dal tile di input con lo stesso indice di griglia.

    Ogni pixel di input viene "distribuito" in un blocco stride x stride
    di output, pesato dal kernel trasposto.

    Args:
        x:      (Cin, H, W)
        weight: (Cin, Cout, K, K) -- come nn.ConvTranspose2d.weight
        bias:   (Cout,)
        n_tiles_h, n_tiles_w: STESSA griglia usata nel resto della rete

    Returns:
        (Cout, H*stride, W*stride)
    """
    Cin, H, W = x.shape
    Cout = weight.shape[1]
    tile_h, tile_w = H // n_tiles_h, W // n_tiles_w
    out_tile_h, out_tile_w = tile_h * stride, tile_w * stride
    out_h, out_w = H * stride, W * stride

    out = np.zeros((Cout, out_h, out_w), dtype=np.float32)

    for ti in range(n_tiles_h):
        for tj in range(n_tiles_w):
            row_start, col_start = ti * tile_h, tj * tile_w
            x_tile = x[:, row_start:row_start + tile_h, col_start:col_start + tile_w]

            out_tile = np.zeros((Cout, out_tile_h, out_tile_w), dtype=np.float32)
            for co in range(Cout):
                for ky in range(K):
                    for kx in range(K):
                        sub = np.zeros((tile_h, tile_w), dtype=np.float32)
                        for ci in range(Cin):
                            sub += weight[ci, co, ky, kx] * x_tile[ci]
                        # "Interleaving": ogni posizione del kernel finisce
                        # in slot diversi e non sovrapposti dell'output.
                        out_tile[co, ky::stride, kx::stride] = sub
                out_tile[co] += bias[co]

            out_row_start, out_col_start = ti * out_tile_h, tj * out_tile_w
            out[:,
                out_row_start:out_row_start + out_tile_h,
                out_col_start:out_col_start + out_tile_w] = out_tile

    return out


# ---------------------------------------------------------------------------
# Layer pointwise (PolyAct, InstanceNorm in eval mode)
# ---------------------------------------------------------------------------

def poly_act(x, a=0.1, b=1.0, c=0.5, clamp_value=None):
    """
    PolyAct: a*x^2 + b*x + c, applicata elemento per elemento.

    Compatibile nativamente con il tiling: essendo pointwise, non serve
    alcuna rotazione ne' contesto tra tile diversi -- ogni ciphertext/tile
    si trasforma in modo completamente indipendente.

    clamp_value: se None (default), nessun clamp -- il polinomio puro,
    utile per isolare la correttezza matematica del tiling dalla questione
    del clamp (vedi packing_match_test.py). Se specificato, applica
    np.clip(out, -clamp_value, clamp_value) -- riproduce il comportamento
    REALE del modello allenato in eval mode: lo straight-through estimator
    (models/he_friendly.py:PolyAct.forward) restituisce SEMPRE il valore
    clampato nel forward (il gradiente e' "straight-through", ma il valore
    numerico no) -- quindi il modello salvato, anche in inferenza pura,
    clippa davvero. NON e' direttamente traducibile in HE cosi' com'e'
    (CKKS non supporta operazioni di confronto/clamp senza approssimazioni
    polinomiali dedicate) -- da affrontare separatamente in Fase 3D.
    """
    out = a * x * x + b * x + c
    if clamp_value is not None:
        out = np.clip(out, -clamp_value, clamp_value)
    return out


def instance_norm_eval(x, running_mean, running_var, gamma, beta, eps=1e-5):
    """
    InstanceNorm2d in eval mode CON STATISTICHE DI POPOLAZIONE CONGELATE
    (running_mean/running_var accumulate durante il training in plaintext).

    ATTENZIONE -- SUPERATA, TENUTA SOLO COME RIFERIMENTO STORICO/CONFRONTO:
    questa e' la modalita' che il progetto ha diagnosticato come causa
    strutturale del bisogno di clamp in Fase III (vedi CONTESTO_PROGETTO_
    TESI: con statistiche di popolazione, 46/46 batch di validazione
    esplodono in NaN/Inf a clamp disattivato; con statistiche per-istanza,
    0/46). Il modello che si intende portare in HE ora usa
    norm_mode='per_instance' -- usare instance_norm_per_instance() qui
    sotto, non questa funzione, per l'implementazione HE reale.

    In questa modalita' la normalizzazione si riduce a una trasformazione
    affine y = gamma*(x-mean)/sqrt(var+eps) + beta con costanti fisse per
    canale -- x*A + B, HE-compatibile nativamente e molto economica (nessun
    calcolo aggiuntivo in inferenza). E' il vantaggio di costo che si perde
    passando a instance_norm_per_instance(), da confrontare col vantaggio
    di stabilita' (vedi discussione con Aurora sul trade-off).

    Args:
        x: (C, H, W)
        running_mean, running_var, gamma, beta: (C,) -- dal checkpoint
           allenato in plaintext
    """
    C = x.shape[0]
    out = np.zeros_like(x)
    for c in range(C):
        out[c] = gamma[c] * (x[c] - running_mean[c]) / np.sqrt(running_var[c] + eps) + beta[c]
    return out


def evaluate_chebyshev_poly(x, coeffs, domain):
    """
    Valuta un polinomio di Chebyshev (nella base standard di Chebyshev,
    coefficienti come restituiti da numpy.polynomial.chebyshev.Chebyshev.fit(...).coef)
    su un valore x, usando la ricorrenza di Clenshaw -- che richiede
    SOLO moltiplicazioni e somme (nessuna divisione, nessun confronto),
    quindi e' direttamente traducibile in operazioni CKKS.

    Il mapping affine da [domain[0], domain[1]] a [-1, 1] (t = (2x -
    (xmax+xmin)) / (xmax-xmin)) e' anch'esso solo moltiplicazione per
    costante + somma di costante -- HE-native, nessuna operazione nuova
    rispetto a quelle gia' usate altrove in questo modulo.

    Costo in livelli di profondita': la ricorrenza di Clenshaw fa una
    moltiplicazione ciphertext-per-ciphertext (t*b_k1) per ogni grado del
    polinomio, in sequenza -- quindi un polinomio di grado n costa
    APPROSSIMATIVAMENTE n livelli in questa forma "sequenziale" (la
    stessa assunzione, pessimistica, gia' usata in
    crypto/simulate_chebyshev_init_isqrt.py per il confronto costo/
    beneficio). Schemi di valutazione "ad albero" potrebbero ridurre
    questo a circa log2(n) livelli -- da verificare con la libreria HE
    reale, non assunto qui.

    Args:
        x: valore (scalare o array) su cui valutare il polinomio
        coeffs: coefficienti di Chebyshev, dal grado 0 al grado n
        domain: [x_min, x_max] usato in fase di calibrazione (fit del
           polinomio) -- deve coincidere con quello usato per calcolare
           'coeffs', altrimenti il mapping affine e' sbagliato

    Returns:
        Approssimazione di f(x) (qui sempre 1/sqrt(x)), stessa shape di x.
    """
    x_min, x_max = domain
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)  # mappa a [-1, 1]: mult+add, HE-native

    n = len(coeffs) - 1
    b_k1 = 0.0
    b_k2 = 0.0
    for k in range(n, 0, -1):
        b_k = coeffs[k] + 2.0 * t * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    return coeffs[0] + t * b_k1 - b_k2


def newton_raphson_isqrt(x, y0, n_iter):
    """
    Approssimazione di 1/sqrt(x) tramite iterazione di Newton-Raphson,
    usando SOLO le operazioni disponibili su un ciphertext CKKS
    (moltiplicazione ciphertext-ciphertext, moltiplicazione per costante,
    somma):

        y_{n+1} = y_n * (1.5 - 0.5 * x * y_n^2)

    Ogni iterazione costa 3 moltiplicazioni ciphertext-ciphertext (y*y,
    x*(y*y), y*(1.5 - 0.5*...)) -- quindi 3 livelli di profondita'
    moltiplicativa per iterazione. Nessuna operazione di confronto,
    divisione o radice: solo mult/add, HE-native.

    y0: punto di partenza -- puo' essere uno SCALARE fisso (schema
        originale, calibrato sul massimo per layer -- vedi
        crypto/simulate_newton_raphson_isqrt_v2.py) oppure un ARRAY della
        stessa shape di x (schema Chebyshev: un punto di partenza diverso
        per ogni valore, gia' vicino al risultato vero -- vedi
        evaluate_chebyshev_poly() sopra). In entrambi i casi l'iterazione
        successiva e' identica: e' solo il valore/i valori di partenza a
        cambiare.

        Con y0 scalare, DEVE essere calibrato su 1/sqrt(x_max), non sulla
        media del range atteso: Newton-Raphson per questa funzione diverge
        se capita un campione con x molto piu' grande di quello su cui y0
        e' stato calibrato. Partire da 1/sqrt(x_max) garantisce invece
        x*y0^2 <= 1 su ogni campione della calibrazione, quindi
        convergenza monotona senza rischio di overshoot.

    n_iter: numero di iterazioni, calibrato per layer (e per schema:
        4-9 con y0 scalare, tipicamente 0-2 con y0 da Chebyshev).

    Args:
        x: array (qualunque shape) di valori su cui approssimare 1/sqrt(x)
        y0: float o array, punto di partenza per questo layer
        n_iter: int, numero di iterazioni per questo layer

    Returns:
        Approssimazione di 1/sqrt(x), stessa shape di x.
    """
    y = np.asarray(y0, dtype=np.float64) * np.ones_like(x, dtype=np.float64)
    for _ in range(n_iter):
        y = y * (1.5 - 0.5 * x * y * y)
    return y


def instance_norm_per_instance(x, gamma, beta, n_tiles_h, n_tiles_w, eps=1e-5,
                                isqrt_y0=None, isqrt_n_iter=None,
                                isqrt_cheb_coeffs=None, isqrt_cheb_domain=None,
                                isqrt_cheb_post_iter=None):
    """
    InstanceNorm2d con statistiche PER-ISTANZA: media e varianza calcolate
    LIVE sull'immagine corrente (il singolo paziente cifrato), non piu'
    costanti di popolazione precalcolate. Questa e' ora la normalizzazione
    di riferimento per il modello (norm_mode='per_instance' in models/
    he_friendly.py) -- vedi CONTESTO_PROGETTO_TESI per la motivazione
    completa: la normalizzazione di popolazione era la causa strutturale
    dell'instabilita' di Fase III.

    REALIZZAZIONE HE (solo primitive rotazione/somma/moltiplicazione),
    in 4 passi -- ognuno commentato con l'operazione HE reale che
    rappresenta, perche' la traduzione a FIDESlib (Fase 3D) sia diretta:

    1. SOMMA ENTRO TILE: la somma di tutti i pixel di un tile (un
       ciphertext) si ottiene con una riduzione rotate-and-add (EvalSum:
       log2(n_slot) rotazioni + addizioni, non un operatore nativo di
       "somma totale"). Qui usiamo np.sum: il RISULTATO numerico e'
       identico indipendentemente da come si esegue la somma, cambia solo
       il numero di operazioni HE necessarie -- corretto per un prototipo
       di correttezza, la conta delle operazioni si affronta in Fase 3D.
    2. SOMMA TRA TILE: dato che la griglia e' fissa (stessa forma per ogni
       tile, vedi docstring del modulo), sommare gli scalari "somma-di-
       tile" di canali corrispondenti tra tile diversi e' una normale
       addizione tra ciphertext -- nessuna operazione nuova rispetto a
       quelle gia' usate in tiled_conv2d.
    3. VARIANZA: var = E[x^2] - E[x]^2. Richiede x*x (moltiplicazione
       ciphertext-ciphertext, un livello moltiplicativo in piu' rispetto
       al solo calcolo della media) prima della stessa riduzione del
       punto 1-2.
    4. 1/sqrt(var+eps): approssimato con Newton-Raphson. Due schemi di
       inizializzazione disponibili, mutuamente esclusivi (priorita' a
       Chebyshev se entrambi forniti per lo stesso layer):

       (a) SCHEMA ORIGINALE -- isqrt_y0 (scalare fisso per layer,
           calibrato sul massimo osservato) + isqrt_n_iter. Vedi
           crypto/simulate_newton_raphson_isqrt_v2.py.

       (b) SCHEMA CHEBYSHEV -- isqrt_cheb_coeffs + isqrt_cheb_domain
           (un polinomio di Chebyshev, diverso per layer, che approssima
           1/sqrt(x) sull'intero range calibrato) usato per calcolare un
           punto di partenza y0 GIA' VICINO al valore vero per OGNI
           valore di varianza, non solo per il caso peggiore -- seguito
           da isqrt_cheb_post_iter iterazioni di Newton-Raphson
           (tipicamente 0-2, molte meno delle 4-9 richieste dallo schema
           (a)). Vedi crypto/simulate_chebyshev_init_isqrt.py per la
           derivazione: riduce la profondita' totale della
           normalizzazione del ~58% rispetto allo schema (a), sulla rete
           aggressiva+sum. Ispirato da CryptoInvSqrt (PP-STAT, arXiv:
           2508.12093): l'idea e' identica (Chebyshev SOLO per
           l'inizializzazione, Newton-Raphson per la convergenza finale),
           adattata qui al caso per-layer della nostra rete invece che a
           un singolo range fisso.

           ATTENZIONE: il polinomio di Chebyshev estrapola molto male
           fuori dal range di calibrazione (isqrt_cheb_domain) -- var_eps
           viene quindi CLIPPATO in quel range solo per il calcolo di
           y0_cheb (non per l'iterazione Newton-Raphson successiva, che
           lavora sempre sul valore vero) -- stesso clip gia' applicato
           in fase di simulazione/calibrazione
           (crypto/simulate_chebyshev_init_isqrt.py). Senza questo clip,
           un paziente con varianza leggermente fuori dal range calibrato
           produce un y0 sbagliato, senza che le iterazioni di rifinitura
           (spesso 0 per i layer dove il polinomio da solo bastava)
           abbiano modo di correggerlo -- bug osservato empiricamente in
           crypto/packing_match_test.py (2 pazienti su 3 in mismatch,
           risolto aggiungendo questo clip).

       Se ne' (a) ne' (b) sono forniti, ricade su np.sqrt esatto (solo
       per test con pesi sintetici, vedi test_packing.py).

    Args:
        x: (C, H, W) -- feature map della SINGOLA istanza (un paziente)
        gamma, beta: (C,) -- parametri affine allenati (sempre HE-nativi,
           invariati rispetto alla versione a statistiche di popolazione)
        n_tiles_h, n_tiles_w: STESSA griglia fissa usata nel resto della rete
        eps: costante di stabilita' numerica
        isqrt_y0, isqrt_n_iter: schema (a), vedi sopra
        isqrt_cheb_coeffs, isqrt_cheb_domain, isqrt_cheb_post_iter: schema
           (b), vedi sopra -- tutti e tre richiesti insieme se si usa
           questo schema

    Returns:
        (C, H, W), normalizzato con statistiche calcolate su x stesso
    """
    C, H, W = x.shape
    tile_h, tile_w = H // n_tiles_h, W // n_tiles_w
    n_pixels = H * W  # costante nota a priori -> moltiplicazione per 1/n_pixels e' un cMult

    out = np.zeros_like(x)
    for c in range(C):
        sum_x = 0.0
        sum_x2 = 0.0
        for ti in range(n_tiles_h):
            for tj in range(n_tiles_w):
                tile = x[c,
                         ti * tile_h:(ti + 1) * tile_h,
                         tj * tile_w:(tj + 1) * tile_w]
                flat = tile.flatten()
                # Passo 1: somma entro tile (in HE: EvalSum via rotate-and-add)
                sum_x += np.sum(flat)
                # x*x e' una cMult ciphertext-ciphertext; poi stessa riduzione
                sum_x2 += np.sum(flat * flat)
        # Passo 2 (somma tra tile) e' gia' inclusa sopra come normale
        # accumulo scalare -- in HE, addizione tra i ciphertext "somma-di-
        # tile" di ciascun tile (tutti della stessa forma per costruzione).
        mean = sum_x / n_pixels
        mean_sq = sum_x2 / n_pixels
        var = mean_sq - mean * mean  # Passo 3
        var_eps = var + eps

        # Passo 4: Newton-Raphson, con inizializzazione (a) o (b) -- vedi
        # docstring. Priorita' a Chebyshev se entrambi gli schemi sono
        # forniti per lo stesso layer.
        if isqrt_cheb_coeffs is not None:
            # Schema (b): Chebyshev per l'inizializzazione, poche iterazioni
            # di rifinitura. y0 calcolato su var_eps CLIPPATO nel range di
            # calibrazione (il polinomio estrapola male fuori da li'); le
            # iterazioni Newton-Raphson successive lavorano invece sul
            # valore VERO di var_eps, non su quello clippato.
            var_eps_for_y0 = np.clip(var_eps, isqrt_cheb_domain[0], isqrt_cheb_domain[1])
            y0_cheb = evaluate_chebyshev_poly(var_eps_for_y0, isqrt_cheb_coeffs, isqrt_cheb_domain)
            y0_cheb = max(float(y0_cheb), 1e-6)  # sicurezza: mai negativo/zero
            inv_std = float(newton_raphson_isqrt(np.array(var_eps), y0_cheb, isqrt_cheb_post_iter))
        elif isqrt_y0 is not None:
            # Schema (a): y0 scalare fisso per layer
            inv_std = float(newton_raphson_isqrt(np.array(var_eps), isqrt_y0, isqrt_n_iter))
        else:
            # Nessuna approssimazione richiesta: sqrt esatto (solo per test
            # con pesi sintetici, vedi test_packing.py)
            inv_std = 1.0 / np.sqrt(var_eps)

        out[c] = gamma[c] * (x[c] - mean) * inv_std + beta[c]

    return out