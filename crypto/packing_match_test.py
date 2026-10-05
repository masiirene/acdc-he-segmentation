"""
crypto/packing_match_test.py

Fase 3, punto C (schema di Aurora): "packing match test" -- verifica che
l'intera pipeline HE-style (crypto/packing.py, tiling a griglia fissa)
riproduca ESATTAMENTE l'inferenza del vero modello allenato (HEFriendlyUNet,
PyTorch), non solo su pesi sintetici (vedi test_packing.py) ma sul
checkpoint reale e su un'immagine reale del validation set.

DIFFERENZA rispetto a test_packing.py:
- test_packing.py verifica la CORRETTEZZA MATEMATICA delle singole
  funzioni (tiled_conv2d, instance_norm_per_instance, ecc.) con pesi
  casuali e reti giocattolo -- risponde a "l'algoritmo e' giusto?"
- Questo script verifica che l'INTERA RETE, con i pesi VERI del training
  di oggi, dia lo stesso identico output (a meno di arrotondamento) sia
  in PyTorch sia nella pipeline a tile -- risponde a "il modello allenato
  e' davvero portabile in HE cosi' com'e'?"

NOTA SUL CLAMP: il modello PyTorch reale, anche in eval mode, applica
ancora il clamp interno per-layer (STE: il valore numerico restituito nel
forward e' sempre quello clampato, solo il gradiente e' "straight-
through" -- quindi il clamp non si disattiva mai da solo passando a eval
mode). crypto/packing.py:poly_act() invece calcola il polinomio puro,
senza clamp. Per un confronto equo, questo script disattiva il clamp
anche nel modello PyTorch di riferimento (stessa tecnica di
check_inference_stability.py) -- cosi' si isola la correttezza della
matematica del tiling dalla questione, separata e ancora aperta con
Aurora, se il clamp serva o meno in produzione HE.

NOTA SU WEIGHT STANDARDIZATION: se il checkpoint e' stato allenato con
--weight_standardization (WSConv2d), i pesi grezzi salvati nello
state_dict NON sono quelli usati nella convoluzione -- WSConv2d li
standardizza (media, deviazione standard) e li riscala con un gain
appreso per canale ad OGNI forward pass, prima di convolvere. E' una
trasformazione sui pesi in chiaro (nessun costo HE aggiuntivo), ma va
riprodotta esplicitamente nel lato numpy, altrimenti il confronto usa
pesi diversi da quelli realmente usati dal modello PyTorch. Passa
--weight_standardization qui SE E SOLO SE il checkpoint testato e' stato
allenato con quel flag.

NOTA SULLO SKIP_MODE: deve corrispondere a come e' stato allenato il
checkpoint ('concat' o 'sum') -- vedi --skip_mode. Con 'sum' i canali del
decoder NON raddoppiano (skip e upsampling sono sommati, non affiancati),
quindi i pesi delle Conv2d dei blocchi decoder hanno una forma diversa
rispetto a 'concat' con la stessa lista di filters.

NOTA SUI LAYER DI NORMALIZZAZIONE BYPASSATI (--bypass_norm_layers):
per i checkpoint in cui alcune InstanceNorm sono state rimosse in modo
strutturale e permanente durante il fine-tuning (vedi crypto/finetune_
without_6_instancenorm.py -- 6 layer su 22, Dice 0.8744, superiore al
checkpoint a piena normalizzazione), il modello PyTorch di riferimento
applica gia' un forward_hook che sostituisce l'output di quei layer con
il loro input (identita' pura). Il lato numpy deve fare lo stesso: NON
calcolare affatto la normalizzazione per quei layer (ne' sqrt esatto ne'
approssimazione), altrimenti il confronto sarebbe contro un comportamento
che la rete reale non ha mai avuto. Passa i nomi via --bypass_norm_layers,
separati da virgola, con lo stesso formato usato in BYPASS_LAYERS di
finetune_without_6_instancenorm.py (es. 'enc4.block.1,enc5.block.4').

NOTA SULLA RADICE QUADRATA: per default questo script usa 1/sqrt esatto
su entrambi i lati (isola tiling/PolyAct dalla questione della radice
quadrata). Sono disponibili DUE schemi di approssimazione alternativi,
mutuamente esclusivi (se entrambi forniti ha priorita' Chebyshev):

- --isqrt_json: Newton-Raphson con y0 SCALARE fisso per layer (calibrato
  sul massimo osservato, vedi crypto/simulate_newton_raphson_isqrt_v2.py).
- --cheb_isqrt_json: inizializzazione tramite un polinomio di Chebyshev
  per layer (diverso per ogni valore di varianza, non un valore fisso),
  seguito da poche iterazioni di rifinitura Newton-Raphson. Ispirato da
  CryptoInvSqrt (PP-STAT, arXiv:2508.12093).

In entrambi i casi il lato numpy usa l'approssimazione, il lato PyTorch
di riferimento continua SEMPRE a usare sqrt esatto (rappresenta il
modello "ideale", non un'approssimazione HE) -- quindi una diff residua
maggiore di zero e' ATTESA (e' l'errore di approssimazione, non un bug
del tiling) -- l'obiettivo e' verificare che la diff resti CONTENUTA,
non che sia zero. I layer in --bypass_norm_layers sono esclusi da
entrambi gli schemi (nessuna approssimazione applicata, vedi sopra).

GRIGLIA DI TILE: 2x1 (2 righe, 1 colonna). Non e' una scelta arbitraria:
e' la stessa individuata nel dimensionamento di Fase 3A -- la larghezza
224 si divide "pulita" solo per 1 o 7, quindi 1 colonna e' la scelta piu'
efficiente disponibile; 2 righe perche' 256 (e tutte le risoluzioni
successive fino al bottleneck 8x7) restano divisibili per 2. Questa
griglia dipende solo da (H, W) e dal numero di stage stride-2 -- NON dal
numero di canali per stage ne' dallo skip_mode, quindi e' la stessa
griglia in ogni configurazione.

AVVISO SULLE PRESTAZIONI: tiled_conv2d non e' vettorizzato sui canali (e'
un prototipo di CORRETTEZZA, non di velocita'), quindi il forward numpy
completo su un'immagine puo' richiedere diversi minuti. Normale;
l'ottimizzazione delle prestazioni e' un problema separato, da affrontare
solo quando si passa a FIDESlib.

USO:
    # Piena larghezza, concat, senza WS, senza approssimazione radice:
    python3 -m crypto.packing_match_test --checkpoint <path> --n_samples 1

    # Checkpoint aggressivo+sum, Newton-Raphson puro calibrato per layer:
    python3 -m crypto.packing_match_test --checkpoint <path_aggressivo_sum> \\
        --filters 32 64 128 256 128 64 --skip_mode sum \\
        --isqrt_json crypto/newton_raphson_simulation_v2.json \\
        --clamp_values_json crypto/calibrated_clamp_values.json \\
        --sample_indices 0,180,352

    # Checkpoint con 6 InstanceNorm rimosse, Chebyshev monotono sui 16 rimasti:
    python3 -m crypto.packing_match_test --checkpoint <path_no6norm> \\
        --filters 32 64 128 256 128 64 --skip_mode sum \\
        --cheb_isqrt_json crypto/chebyshev_calibration_no6norm.json \\
        --bypass_norm_layers "enc4.block.1,enc5.block.1,dec0.block.1,enc5.block.4,enc3.block.1,enc4.block.4" \\
        --clamp_values_json crypto/calibrated_clamp_values.json \\
        --sample_indices 0,20,60,100,140,180,220,260,300,340,352
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet, PolyAct
from training.dataset import ACDCDataset, load_splits
from crypto.packing import (
    tiled_conv2d,
    tiled_conv_transpose2d,
    poly_act,
    instance_norm_per_instance,
)

N_TILES_H, N_TILES_W = 2, 1  # griglia fissa, vedi docstring del modulo sopra


def identity_bypass_hook(module, inputs, output):
    """Sostituisce l'output del modulo con il suo input -- identita' pura.
    Registrato sul modello PyTorch di riferimento per i layer indicati in
    --bypass_norm_layers, cosi' il suo comportamento coincide esattamente
    con quello del checkpoint reale (vedi crypto/finetune_without_6_
    instancenorm.py, dove lo stesso hook e' usato durante il fine-tuning)."""
    return inputs[0]


def apply_weight_standardization(weight, raw_gain, gain_floor=0.05, eps=1e-5):
    """
    Riproduce in numpy esattamente cio' che WSConv2d.forward fa in
    PyTorch prima di ogni convoluzione: standardizza il peso per canale
    di uscita (media e deviazione standard di popolazione su
    in_channels*kh*kw), poi lo riscala con un gain appreso per canale
    (gain_floor + softplus(raw_gain)).

    Applicata UNA SOLA VOLTA qui, sul peso in chiaro, prima di passarlo a
    tiled_conv2d -- nessun costo HE aggiuntivo, nessuna operazione nuova
    su ciphertext.

    Args:
        weight: (Cout, Cin, K, K) o (Cout, Cin, 1, 1) -- peso grezzo dal
           checkpoint, cosi' come salvato in state_dict
        raw_gain: (Cout,) -- parametro raw_gain letto dal checkpoint
           (chiave '{layer}.raw_gain')
        gain_floor: float -- costante FISSA (non appresa), 0.05 nel
           codice di training attuale
        eps: stabilita' numerica, coerente con quella usata in WSConv2d

    Returns:
        (Cout, Cin, K, K), peso gia' standardizzato e riscalato
    """
    out_ch = weight.shape[0]
    w_flat = weight.reshape(out_ch, -1).astype(np.float64)
    mean = w_flat.mean(axis=1, keepdims=True)
    std = w_flat.std(axis=1, keepdims=True)  # popolazione, ddof=0
    w_std = (w_flat - mean) / (std + eps)
    effective_gain = gain_floor + np.log1p(np.exp(raw_gain.astype(np.float64)))  # softplus
    w_std = w_std * effective_gain.reshape(-1, 1)
    return w_std.reshape(weight.shape).astype(np.float32)


def extract_conv_block_weights(state_dict, prefix, clamp_values=None, isqrt_values=None,
                                cheb_values=None, weight_standardization=False, gain_floor=0.05,
                                bypass_norm_layers=None):
    """
    Estrae i pesi di un ConvBlock (Conv->Norm->Act->Conv->Norm->Act, indici
    0..5 nel nn.Sequential) come numpy array, nel formato atteso dalle
    funzioni di crypto/packing.py.

    clamp_values: dizionario opzionale {nome_layer: soglia}. Se fornito,
    le soglie per 'act1' e 'act2' di questo blocco vengono lette da li'
    (chiavi '{prefix}.block.2' e '{prefix}.block.5'); altrimenti None
    (nessun clamp applicato da poly_act).

    isqrt_values: dizionario opzionale {nome_layer: {"y0":..., "iterations_
    needed":...}}. Schema Newton-Raphson puro (y0 scalare fisso per layer).

    cheb_values: dizionario opzionale {nome_layer: {"cheb_coeffs":...,
    "cheb_domain":..., "post_iter":...}}. Schema Chebyshev -- se fornito
    insieme a isqrt_values per lo stesso layer, ha PRIORITA'.

    weight_standardization, gain_floor: se weight_standardization=True,
    i pesi delle due Conv2d di questo blocco vengono standardizzati prima
    di essere restituiti -- DEVE corrispondere a come e' stato allenato
    il checkpoint.

    bypass_norm_layers: set opzionale di nomi di layer (es.
    'enc4.block.1') da bypassare COMPLETAMENTE -- nessuna normalizzazione
    calcolata (ne' sqrt esatto ne' approssimazione), coerente con un
    forward_hook a identita' registrato sul modello PyTorch di
    riferimento. Vedi crypto/finetune_without_6_instancenorm.py.

    NOTA: assume act_type='poly' con 'a' parametro libero (max_a_poly=None).
    """
    def npy(key):
        return state_dict[key].detach().cpu().numpy()

    bypass_norm_layers = bypass_norm_layers or set()

    act1_clamp = clamp_values.get(f'{prefix}.block.2') if clamp_values else None
    act2_clamp = clamp_values.get(f'{prefix}.block.5') if clamp_values else None

    norm1_isqrt = isqrt_values.get(f'{prefix}.block.1') if isqrt_values else None
    norm2_isqrt = isqrt_values.get(f'{prefix}.block.4') if isqrt_values else None

    norm1_cheb = cheb_values.get(f'{prefix}.block.1') if cheb_values else None
    norm2_cheb = cheb_values.get(f'{prefix}.block.4') if cheb_values else None

    norm1_bypass = f'{prefix}.block.1' in bypass_norm_layers
    norm2_bypass = f'{prefix}.block.4' in bypass_norm_layers

    conv1_w = npy(f'{prefix}.block.0.weight')
    conv2_w = npy(f'{prefix}.block.3.weight')
    if weight_standardization:
        conv1_w = apply_weight_standardization(
            conv1_w, npy(f'{prefix}.block.0.raw_gain'), gain_floor)
        conv2_w = apply_weight_standardization(
            conv2_w, npy(f'{prefix}.block.3.raw_gain'), gain_floor)

    return {
        'conv1_w': conv1_w,
        'conv1_b': npy(f'{prefix}.block.0.bias'),
        'norm1_gamma': npy(f'{prefix}.block.1.weight'),
        'norm1_beta': npy(f'{prefix}.block.1.bias'),
        'norm1_isqrt': norm1_isqrt,
        'norm1_cheb': norm1_cheb,
        'norm1_bypass': norm1_bypass,
        'act1_a': float(npy(f'{prefix}.block.2.a')),
        'act1_b': float(npy(f'{prefix}.block.2.b')),
        'act1_c': float(npy(f'{prefix}.block.2.c')),
        'act1_clamp': act1_clamp,
        'conv2_w': conv2_w,
        'conv2_b': npy(f'{prefix}.block.3.bias'),
        'norm2_gamma': npy(f'{prefix}.block.4.weight'),
        'norm2_beta': npy(f'{prefix}.block.4.bias'),
        'norm2_isqrt': norm2_isqrt,
        'norm2_cheb': norm2_cheb,
        'norm2_bypass': norm2_bypass,
        'act2_a': float(npy(f'{prefix}.block.5.a')),
        'act2_b': float(npy(f'{prefix}.block.5.b')),
        'act2_c': float(npy(f'{prefix}.block.5.c')),
        'act2_clamp': act2_clamp,
    }


def run_conv_block_numpy(x, w, stride):
    """
    Un ConvBlock completo (Conv->Norm->Act->Conv->Norm->Act) con le
    funzioni HE-style di crypto/packing.py.

    Se un layer di normalizzazione e' in bypass_norm_layers (vedi
    extract_conv_block_weights), la normalizzazione viene saltata DEL
    TUTTO (identita' pura) -- coerente con il forward_hook registrato sul
    modello PyTorch di riferimento per quei layer. Altrimenti, se e'
    presente sia lo schema Newton-Raphson puro sia quello Chebyshev, ha
    PRIORITA' Chebyshev.
    """
    x = tiled_conv2d(x, w['conv1_w'], w['conv1_b'], N_TILES_H, N_TILES_W, stride=stride)
    if not w['norm1_bypass']:
        n1 = w['norm1_isqrt']
        n1_cheb = w['norm1_cheb']
        x = instance_norm_per_instance(
            x, w['norm1_gamma'], w['norm1_beta'], N_TILES_H, N_TILES_W,
            isqrt_y0=n1['y0'] if (n1 and not n1_cheb) else None,
            isqrt_n_iter=n1['iterations_needed'] if (n1 and not n1_cheb) else None,
            isqrt_cheb_coeffs=n1_cheb['cheb_coeffs'] if n1_cheb else None,
            isqrt_cheb_domain=n1_cheb['cheb_domain'] if n1_cheb else None,
            isqrt_cheb_post_iter=n1_cheb['post_iter'] if n1_cheb else None)
    x = poly_act(x, a=w['act1_a'], b=w['act1_b'], c=w['act1_c'], clamp_value=w['act1_clamp'])

    x = tiled_conv2d(x, w['conv2_w'], w['conv2_b'], N_TILES_H, N_TILES_W, stride=1)
    if not w['norm2_bypass']:
        n2 = w['norm2_isqrt']
        n2_cheb = w['norm2_cheb']
        x = instance_norm_per_instance(
            x, w['norm2_gamma'], w['norm2_beta'], N_TILES_H, N_TILES_W,
            isqrt_y0=n2['y0'] if (n2 and not n2_cheb) else None,
            isqrt_n_iter=n2['iterations_needed'] if (n2 and not n2_cheb) else None,
            isqrt_cheb_coeffs=n2_cheb['cheb_coeffs'] if n2_cheb else None,
            isqrt_cheb_domain=n2_cheb['cheb_domain'] if n2_cheb else None,
            isqrt_cheb_post_iter=n2_cheb['post_iter'] if n2_cheb else None)
    x = poly_act(x, a=w['act2_a'], b=w['act2_b'], c=w['act2_c'], clamp_value=w['act2_clamp'])
    return x


def run_full_model_numpy(x_in, state_dict, clamp_values=None, isqrt_values=None,
                          cheb_values=None, weight_standardization=False, gain_floor=0.05,
                          skip_mode='concat', bypass_norm_layers=None, verbose=False):
    """
    Ricostruisce l'intero forward pass di HEFriendlyUNet usando SOLO le
    funzioni HE-style di crypto/packing.py, con i pesi reali del
    checkpoint. Rispecchia esattamente HEFriendlyUNet.forward(): 6 stage
    encoder, 5 stage decoder con skip connection (concatenate O sommate,
    a seconda di skip_mode), conv finale 1x1.

    clamp_values, isqrt_values, cheb_values, bypass_norm_layers: vedi
    extract_conv_block_weights().
    weight_standardization, gain_floor: vedi extract_conv_block_weights()
       -- si applica anche a out_conv.
    skip_mode: 'concat' (skip e upsampling affiancati, canali raddoppiati
       in ingresso al primo conv di ogni blocco decoder) o 'sum' (skip e
       upsampling sommati canale per canale, nessun raddoppio).
    """
    def npy(key):
        return state_dict[key].detach().cpu().numpy()

    enc_blocks = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5']
    enc_strides = [1, 2, 2, 2, 2, 2]

    # --- Encoder ---
    e = [None] * 6
    x = x_in
    for i in range(6):
        t0 = time.time()
        w = extract_conv_block_weights(state_dict, enc_blocks[i], clamp_values, isqrt_values,
                                        cheb_values, weight_standardization, gain_floor,
                                        bypass_norm_layers)
        x = run_conv_block_numpy(x, w, stride=enc_strides[i])
        e[i] = x
        if verbose:
            print(f'    {enc_blocks[i]}: shape={x.shape}  ({time.time()-t0:.1f}s)')

    # --- Decoder (dec4..dec0, skip da e4..e0) ---
    dec_blocks = ['dec4', 'dec3', 'dec2', 'dec1', 'dec0']
    up_names = ['up4', 'up3', 'up2', 'up1', 'up0']
    skip_indices = [4, 3, 2, 1, 0]  # e4, e3, e2, e1, e0

    d = e[5]  # bottleneck (output di enc5)
    for dec_name, up_name, skip_idx in zip(dec_blocks, up_names, skip_indices):
        t0 = time.time()
        # up0..up4 sono ConvTranspose2d, MAI wrappate in WSConv2d nel
        # modello -- nessuna standardizzazione qui, in nessun caso.
        up_w = npy(f'{up_name}.weight')  # (Cin, Cout, 2, 2)
        up_b = npy(f'{up_name}.bias')
        d = tiled_conv_transpose2d(d, up_w, up_b, N_TILES_H, N_TILES_W)

        if skip_mode == 'concat':
            d = np.concatenate([d, e[skip_idx]], axis=0)  # skip connection affiancata
        elif skip_mode == 'sum':
            d = d + e[skip_idx]  # skip connection sommata canale per canale
        else:
            raise ValueError(f"skip_mode sconosciuto: {skip_mode!r} (atteso 'concat' o 'sum')")

        dw = extract_conv_block_weights(state_dict, dec_name, clamp_values, isqrt_values,
                                         cheb_values, weight_standardization, gain_floor,
                                         bypass_norm_layers)
        d = run_conv_block_numpy(d, dw, stride=1)
        if verbose:
            print(f'    {dec_name}: shape={d.shape}  ({time.time()-t0:.1f}s)')

    # --- Conv finale 1x1 (out_conv): kernel=1, nessun padding -> halo=0 ---
    out_w = npy('out_conv.weight')  # (num_classes, Cin, 1, 1)
    out_b = npy('out_conv.bias')
    if weight_standardization:
        out_w = apply_weight_standardization(out_w, npy('out_conv.raw_gain'), gain_floor)
    logits = tiled_conv2d(d, out_w, out_b, N_TILES_H, N_TILES_W, stride=1, halo=0, K=1)

    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--filters', type=int, nargs=6, default=None,
                        help='6 interi [enc0..enc5]: DEVE corrispondere esattamente ai filters con '
                             'cui e\' stato allenato --checkpoint. Omesso (default) = piena '
                             'larghezza [32,64,128,256,512,512].')
    parser.add_argument('--skip_mode', default='concat', choices=['concat', 'sum'],
                        help='DEVE corrispondere a come e\' stato allenato --checkpoint.')
    parser.add_argument('--weight_standardization', action='store_true',
                        help='DEVE corrispondere a come e\' stato allenato --checkpoint.')
    parser.add_argument('--gain_floor', type=float, default=0.05,
                        help='Costante gain_floor di WSConv2d. Ignorato se '
                             '--weight_standardization non e\' passato.')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Quante slice di validazione testare, prendendo le PRIME n in ordine. '
                             'Ignorato se --sample_indices e\' fornito.')
    parser.add_argument('--sample_indices', default=None,
                        help='Lista di indici espliciti separati da virgola (es. "0,180,352"). '
                             'Ha priorita\' su --n_samples se entrambi sono forniti.')
    parser.add_argument('--verbose', action='store_true',
                        help='Stampa la forma e il tempo di ogni stage encoder/decoder mentre gira.')
    parser.add_argument('--clamp_values_json', default=None,
                        help='Path al JSON con le soglie di clamp calibrate per layer. Se fornito, '
                             'il confronto include il clamp REALE del modello su entrambi i lati.')
    parser.add_argument('--isqrt_json', default=None,
                        help='Path al JSON con y0/iterazioni Newton-Raphson calibrati per layer. '
                             'Alternativo a --cheb_isqrt_json -- se entrambi forniti, ha priorita\' '
                             'Chebyshev.')
    parser.add_argument('--cheb_isqrt_json', default=None,
                        help='Path al JSON con la calibrazione Chebyshev per layer -- '
                             'inizializzazione via polinomio di Chebyshev seguita da poche '
                             'iterazioni di rifinitura Newton-Raphson.')
    parser.add_argument('--bypass_norm_layers', default='',
                        help='Lista separata da virgola di layer di normalizzazione da bypassare '
                             'COMPLETAMENTE (identita\' pura, nessuna approssimazione), coerente '
                             'con un checkpoint dove quei layer sono stati rimossi in modo '
                             'permanente durante il fine-tuning (vedi crypto/finetune_without_6_'
                             'instancenorm.py). Es: "enc4.block.1,enc5.block.1,dec0.block.1,'
                             'enc5.block.4,enc3.block.1,enc4.block.4". Vuoto (default): nessun '
                             'layer bypassato.')
    args = parser.parse_args()

    device = torch.device('cpu')  # confronto in chiaro, CPU sufficiente e piu' riproducibile
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Griglia di tile: {N_TILES_H}x{N_TILES_W}')
    filters = args.filters if args.filters else [32, 64, 128, 256, 512, 512]
    print(f'Filters: {filters}')
    print(f'Skip mode: {args.skip_mode}')
    print(f'Weight standardization: {args.weight_standardization}')

    bypass_norm_layers = set(l.strip() for l in args.bypass_norm_layers.split(',') if l.strip())
    if bypass_norm_layers:
        print(f'Layer di normalizzazione bypassati (identita\' pura, {len(bypass_norm_layers)}): '
              f'{sorted(bypass_norm_layers)}')
    print('ATTENZIONE: il forward numpy non e\' vettorizzato sui canali -- '
          'puo\' richiedere diversi minuti per immagine. Usa --verbose per '
          'seguire il progresso stage per stage.\n')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly',
                           norm_type='instance', norm_mode='per_instance',
                           weight_standardization=args.weight_standardization,
                           skip_mode=args.skip_mode,
                           filters=filters).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  \u26a0\ufe0f  Chiavi mancanti: {missing}')
    if unexpected:
        print(f'  \u26a0\ufe0f  Chiavi impreviste (verifica --filters/--weight_standardization/'
              f'--skip_mode): {unexpected}')
    model.eval()

    # Registra sul modello PyTorch di riferimento lo STESSO bypass a
    # identita' usato in fase di fine-tuning per i layer indicati --
    # senza questo, il confronto sarebbe contro un comportamento che la
    # rete reale (col checkpoint dato) non ha mai avuto.
    if bypass_norm_layers:
        modules_dict = dict(model.named_modules())
        for layer_name in bypass_norm_layers:
            if layer_name not in modules_dict:
                print(f'  \u26a0\ufe0f  Layer da bypassare non trovato nel modello: {layer_name}')
                continue
            modules_dict[layer_name].register_forward_hook(identity_bypass_hook)

    clamp_values = None
    if args.clamp_values_json:
        with open(args.clamp_values_json) as f:
            clamp_values = json.load(f)
        n_set = 0
        for name, m in model.named_modules():
            if isinstance(m, PolyAct) and name in clamp_values:
                m.clamp_value = clamp_values[name]
                n_set += 1
        print(f'Soglie di clamp calibrate applicate al modello di riferimento: '
              f'{n_set}/{sum(1 for _, m in model.named_modules() if isinstance(m, PolyAct))} layer')
        print('Confronto CON clamp attivo (configurazione reale di produzione) su entrambi i lati.\n')
    else:
        for name, m in model.named_modules():
            if isinstance(m, PolyAct):
                m.clamp_value = float('inf')
        print('Nessun --clamp_values_json fornito: clamp interno disattivato su ENTRAMBI i lati')
        print('(PyTorch e numpy) -- isola la correttezza del tiling dalla questione del clamp.\n')

    isqrt_values = None
    if args.isqrt_json:
        with open(args.isqrt_json) as f:
            isqrt_values = json.load(f)
        n_layers = sum(1 for k in isqrt_values if not k.startswith('_'))
        print(f'Schema Newton-Raphson puro caricato da: {args.isqrt_json} ({n_layers} layer)')

    cheb_values = None
    if args.cheb_isqrt_json:
        with open(args.cheb_isqrt_json) as f:
            cheb_values = json.load(f)
        print(f'Schema Chebyshev caricato da: {args.cheb_isqrt_json} ({len(cheb_values)} layer)')

    if isqrt_values or cheb_values:
        print('Lato numpy: approssimazione HE-style. Lato PyTorch: sqrt esatto (modello ideale).')
        print('Una diff residua maggiore di zero e\' ATTESA (errore di approssimazione, non un bug).\n')
    else:
        print('Nessuna approssimazione della radice fornita: sqrt esatto su entrambi i lati -- isola')
        print('tiling/PolyAct dalla questione della radice quadrata.\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)

    n = min(args.n_samples, len(val_ds))
    if args.sample_indices:
        indices = [int(x.strip()) for x in args.sample_indices.split(',')]
        for idx in indices:
            assert 0 <= idx < len(val_ds), f'Indice {idx} fuori range (0..{len(val_ds)-1})'
    else:
        indices = list(range(n))

    print(f'Test su {len(indices)} slice del validation set (su {len(val_ds)} totali), '
          f'indici: {indices}')
    if not args.sample_indices:
        print('  (indici consecutivi -- verifica tu stessa se appartengono allo stesso paziente,')
        print('   vedi val_ds.slices[i], oppure usa --sample_indices per scegliere a mano)\n')
    else:
        print()

    max_diffs = []
    for i in indices:
        img, _ = val_ds[i]  # img: tensor (1, 256, 224)
        img_np = img.numpy().astype(np.float32)  # (Cin=1, H, W)
        img_path, _, slice_idx = val_ds.slices[i]
        patient_name = os.path.basename(img_path)

        print(f'--- Slice indice {i} ({patient_name}, slice {slice_idx}) ---')
        t0 = time.time()
        with torch.no_grad():
            logits_torch = model(img.unsqueeze(0)).squeeze(0).numpy()  # (num_classes, H, W)
        print(f'  PyTorch:  {time.time()-t0:.2f}s')

        t0 = time.time()
        logits_numpy = run_full_model_numpy(
            img_np, state, clamp_values=clamp_values, isqrt_values=isqrt_values,
            cheb_values=cheb_values, weight_standardization=args.weight_standardization,
            gain_floor=args.gain_floor, skip_mode=args.skip_mode,
            bypass_norm_layers=bypass_norm_layers, verbose=args.verbose)
        print(f'  numpy tiled: {time.time()-t0:.1f}s')

        diff = np.abs(logits_torch - logits_numpy)
        max_diff = diff.max()
        rel_diff = max_diff / (np.abs(logits_torch).max() + 1e-8)
        max_diffs.append(max_diff)

        threshold = 1e-1 if (isqrt_values or cheb_values) else 1e-2
        status = "OK" if max_diff < threshold else "MISMATCH"
        print(f'  [{status}] diff max={max_diff:.4e}  relativo={rel_diff:.4e}  '
              f'range logits torch=[{logits_torch.min():.2f}, {logits_torch.max():.2f}]\n')

    threshold = 1e-1 if (isqrt_values or cheb_values) else 1e-2
    print(f'Diff massima su {len(max_diffs)} slice: {max(max_diffs):.4e} '
          f'(soglia di accettazione: {threshold:.0e})')
    if max(max_diffs) < threshold:
        if cheb_values:
            print('=> La pipeline HE-style (packing.py) con inizializzazione Chebyshev riproduce')
            print('   il modello reale entro l\'errore di approssimazione atteso.')
        elif isqrt_values:
            print('=> La pipeline HE-style (packing.py) con Newton-Raphson calibrato riproduce')
            print('   il modello reale entro l\'errore di approssimazione atteso.')
        else:
            print('=> La pipeline HE-style (packing.py) riproduce FEDELMENTE il modello reale.')
    else:
        print('=> ATTENZIONE: divergenza superiore alla soglia attesa, da investigare prima di')
        print('   procedere a FIDESlib.')


if __name__ == '__main__':
    main()