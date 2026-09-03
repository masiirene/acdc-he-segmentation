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

def poly_act(x, a=0.1, b=1.0, c=0.5):
    """
    PolyAct: a*x^2 + b*x + c, applicata elemento per elemento.

    Compatibile nativamente con il tiling: essendo pointwise, non serve
    alcuna rotazione ne' contesto tra tile diversi -- ogni ciphertext/tile
    si trasforma in modo completamente indipendente. Nota: il clamp
    interno che abbiamo introdotto in plaintext (models/he_friendly.py,
    fix instabilita' numerica) NON e' direttamente traducibile in HE
    (CKKS non supporta operazioni di confronto/clamp senza approssimazioni
    polinomiali dedicate) -- da affrontare separatamente in Fase 3D.
    """
    return a * x * x + b * x + c


def instance_norm_eval(x, running_mean, running_var, gamma, beta, eps=1e-5):
    """
    InstanceNorm2d in eval mode: usa running_mean/running_var CONGELATE
    (accumulate durante il training in plaintext), non calcolate al volo.

    In questa modalita' la normalizzazione si riduce a una trasformazione
    affine y = gamma*(x-mean)/sqrt(var+eps) + beta con costanti fisse per
    canale -- x*A + B, HE-compatibile nativamente. Come PolyAct, e'
    pointwise: compatibile col tiling senza bisogno di rotazioni.

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