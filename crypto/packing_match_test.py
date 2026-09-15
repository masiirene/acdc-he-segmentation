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
  di oggi (22 PolyAct, 6 encoder + 5 decoder stage, skip connection),
  dia lo stesso identico output (a meno di arrotondamento) sia in
  PyTorch sia nella pipeline a tile -- risponde a "il modello allenato
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

GRIGLIA DI TILE: 2x1 (2 righe, 1 colonna). Non e' una scelta arbitraria:
e' la stessa individuata nel dimensionamento di Fase 3A (vedi CONTESTO_
PROGETTO_TESI, mail ad Aurora del 3 settembre) -- la larghezza 224 si
divide "pulita" solo per 1 o 7, quindi 1 colonna e' la scelta piu'
efficiente disponibile; 2 righe perche' 256 (e tutte le risoluzioni
successive fino al bottleneck 8x7) restano divisibili per 2.

AVVISO SULLE PRESTAZIONI: a differenza dei test sintetici in
test_packing.py (2-8 canali), qui i canali arrivano fino a 512
(bottleneck enc5/dec4). tiled_conv2d non e' vettorizzato sui canali (e'
un prototipo di CORRETTEZZA, non di velocita' -- vedi nota
sull'inefficienza nota nella docstring di packing.py), quindi il forward
numpy completo su un'immagine puo' richiedere diversi minuti. Normale;
l'ottimizzazione delle prestazioni e' un problema separato, da affrontare
solo quando si passa a FIDESlib.

USO:
    python3 -m crypto.packing_match_test --checkpoint <path> --n_samples 1
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


def extract_conv_block_weights(state_dict, prefix, clamp_values=None):
    """
    Estrae i pesi di un ConvBlock (Conv->Norm->Act->Conv->Norm->Act, indici
    0..5 nel nn.Sequential -- coerente con la numerazione gia' usata nella
    diagnostica di training, es. 'dec0.block.5') come numpy array, nel
    formato atteso dalle funzioni di crypto/packing.py.

    clamp_values: dizionario opzionale {nome_layer: soglia} (stesso formato
    di crypto/calibrated_clamp_values.json). Se fornito, le soglie per
    'act1' e 'act2' di questo blocco vengono lette da li' (chiavi
    '{prefix}.block.2' e '{prefix}.block.5'); altrimenti None (nessun
    clamp applicato da poly_act).

    NOTA: assume act_type='poly' con 'a' parametro libero (max_a_poly=None),
    coerente con tutti i checkpoint vincenti di oggi (nessuno ha usato
    --max_a_poly). Se in futuro si usa un checkpoint con max_a_poly
    impostato, la chiave sarebbe 'raw_a' invece di 'a' e andrebbe applicata
    la trasformazione tanh -- non gestito qui, da estendere se necessario.
    """
    def npy(key):
        return state_dict[key].detach().cpu().numpy()

    act1_clamp = clamp_values.get(f'{prefix}.block.2') if clamp_values else None
    act2_clamp = clamp_values.get(f'{prefix}.block.5') if clamp_values else None

    return {
        'conv1_w': npy(f'{prefix}.block.0.weight'),
        'conv1_b': npy(f'{prefix}.block.0.bias'),
        'norm1_gamma': npy(f'{prefix}.block.1.weight'),
        'norm1_beta': npy(f'{prefix}.block.1.bias'),
        'act1_a': float(npy(f'{prefix}.block.2.a')),
        'act1_b': float(npy(f'{prefix}.block.2.b')),
        'act1_c': float(npy(f'{prefix}.block.2.c')),
        'act1_clamp': act1_clamp,
        'conv2_w': npy(f'{prefix}.block.3.weight'),
        'conv2_b': npy(f'{prefix}.block.3.bias'),
        'norm2_gamma': npy(f'{prefix}.block.4.weight'),
        'norm2_beta': npy(f'{prefix}.block.4.bias'),
        'act2_a': float(npy(f'{prefix}.block.5.a')),
        'act2_b': float(npy(f'{prefix}.block.5.b')),
        'act2_c': float(npy(f'{prefix}.block.5.c')),
        'act2_clamp': act2_clamp,
    }


def run_conv_block_numpy(x, w, stride):
    """Un ConvBlock completo (Conv->Norm->Act->Conv->Norm->Act) con le
    funzioni HE-style di crypto/packing.py."""
    x = tiled_conv2d(x, w['conv1_w'], w['conv1_b'], N_TILES_H, N_TILES_W, stride=stride)
    x = instance_norm_per_instance(x, w['norm1_gamma'], w['norm1_beta'], N_TILES_H, N_TILES_W)
    x = poly_act(x, a=w['act1_a'], b=w['act1_b'], c=w['act1_c'], clamp_value=w['act1_clamp'])

    x = tiled_conv2d(x, w['conv2_w'], w['conv2_b'], N_TILES_H, N_TILES_W, stride=1)
    x = instance_norm_per_instance(x, w['norm2_gamma'], w['norm2_beta'], N_TILES_H, N_TILES_W)
    x = poly_act(x, a=w['act2_a'], b=w['act2_b'], c=w['act2_c'], clamp_value=w['act2_clamp'])
    return x


def run_full_model_numpy(x_in, state_dict, clamp_values=None, verbose=False):
    """
    Ricostruisce l'intero forward pass di HEFriendlyUNet usando SOLO le
    funzioni HE-style di crypto/packing.py, con i pesi reali del
    checkpoint. Rispecchia esattamente HEFriendlyUNet.forward()
    (models/he_friendly.py): 6 stage encoder, 5 stage decoder con skip
    connection, conv finale 1x1.

    clamp_values: vedi extract_conv_block_weights().
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
        w = extract_conv_block_weights(state_dict, enc_blocks[i], clamp_values)
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
        up_w = npy(f'{up_name}.weight')  # (Cin, Cout, 2, 2)
        up_b = npy(f'{up_name}.bias')
        d = tiled_conv_transpose2d(d, up_w, up_b, N_TILES_H, N_TILES_W)
        d = np.concatenate([d, e[skip_idx]], axis=0)  # skip connection

        dw = extract_conv_block_weights(state_dict, dec_name, clamp_values)
        d = run_conv_block_numpy(d, dw, stride=1)
        if verbose:
            print(f'    {dec_name}: shape={d.shape}  ({time.time()-t0:.1f}s)')

    # --- Conv finale 1x1 (out_conv): kernel=1, nessun padding -> halo=0 ---
    out_w = npy('out_conv.weight')  # (num_classes, Cin, 1, 1)
    out_b = npy('out_conv.bias')
    logits = tiled_conv2d(d, out_w, out_b, N_TILES_H, N_TILES_W, stride=1, halo=0, K=1)

    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Quante slice di validazione testare, prendendo le PRIME n in ordine '
                             '(val_ds[0], val_ds[1], ...). ATTENZIONE: slice consecutive spesso '
                             'appartengono allo STESSO paziente (es. slice 0,1,2 = tre slice di '
                             'profondita\' diversa dello stesso paziente, non tre pazienti diversi) '
                             '-- usa --sample_indices per scegliere esplicitamente slice di '
                             'pazienti differenti. Ignorato se --sample_indices e\' fornito.')
    parser.add_argument('--sample_indices', default=None,
                        help='Lista di indici espliciti separati da virgola (es. "0,180,352"), per '
                             'testare pazienti scelti a mano invece delle prime N slice in ordine. '
                             'Ha priorita\' su --n_samples se entrambi sono forniti.')
    parser.add_argument('--verbose', action='store_true',
                        help='Stampa la forma e il tempo di ogni stage encoder/decoder mentre gira '
                             '(utile per capire a che punto e\' arrivato durante l\'attesa)')
    parser.add_argument('--clamp_values_json', default=None,
                        help='Path al JSON con le soglie di clamp calibrate per layer (lo stesso '
                             'usato per allenare il checkpoint, es. crypto/calibrated_clamp_values.json). '
                             'Se fornito, il confronto include il clamp REALE del modello (la '
                             'configurazione con cui verra\' davvero usato in produzione) -- test '
                             'piu\' rigoroso del semplice "clamp disattivato". Se omesso (default), '
                             'il clamp resta disattivato su entrambi i lati (test della sola '
                             'matematica del tiling, indipendente dal clamp).')
    args = parser.parse_args()

    device = torch.device('cpu')  # confronto in chiaro, CPU sufficiente e piu' riproducibile
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Griglia di tile: {N_TILES_H}x{N_TILES_W}')
    print('ATTENZIONE: il forward numpy non e\' vettorizzato sui canali (fino a 512 nel '
          'bottleneck) -- puo\' richiedere diversi minuti per immagine. Usa --verbose per '
          'seguire il progresso stage per stage.\n')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly',
                           norm_type='instance', norm_mode='per_instance').to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  \u26a0\ufe0f  Chiavi mancanti: {missing}')
    model.eval()

    clamp_values = None
    if args.clamp_values_json:
        with open(args.clamp_values_json) as f:
            clamp_values = json.load(f)
        # Applica le STESSE soglie al modello PyTorch di riferimento, cosi'
        # il confronto e' con la configurazione REALE con cui il modello
        # verrebbe usato in produzione (clamp attivo, per-layer) -- non
        # solo la variante "clamp disattivato" (test piu' debole, isola
        # solo la matematica del tiling).
        n_set = 0
        for name, m in model.named_modules():
            if isinstance(m, PolyAct) and name in clamp_values:
                m.clamp_value = clamp_values[name]
                n_set += 1
        print(f'Soglie di clamp calibrate applicate al modello di riferimento: '
              f'{n_set}/{sum(1 for _, m in model.named_modules() if isinstance(m, PolyAct))} layer')
        print('Confronto CON clamp attivo (configurazione reale di produzione) su entrambi i lati.\n')
    else:
        # IMPORTANTE: clamp_value NON e' salvato nello state_dict (e' un
        # attributo Python impostato solo alla costruzione, da
        # --clamp_values_json). Lo STE clampa SEMPRE nel forward (anche in
        # eval: il valore numerico restituito e' letteralmente quello
        # clampato, solo il gradiente e' "straight-through"). Se non
        # forniamo --clamp_values_json qui, disattiviamo il clamp su
        # ENTRAMBI i lati per un confronto equo -- isola la correttezza
        # della matematica del tiling dalla questione del clamp.
        for name, m in model.named_modules():
            if isinstance(m, PolyAct):
                m.clamp_value = float('inf')
        print('Nessun --clamp_values_json fornito: clamp interno disattivato su ENTRAMBI i lati')
        print('(PyTorch e numpy) -- isola la correttezza del tiling dalla questione del clamp.\n')

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
        logits_numpy = run_full_model_numpy(img_np, state, clamp_values=clamp_values, verbose=args.verbose)
        print(f'  numpy tiled: {time.time()-t0:.1f}s')

        diff = np.abs(logits_torch - logits_numpy)
        max_diff = diff.max()
        rel_diff = max_diff / (np.abs(logits_torch).max() + 1e-8)
        max_diffs.append(max_diff)

        status = "OK" if max_diff < 1e-2 else "MISMATCH"
        print(f'  [{status}] diff max={max_diff:.4e}  relativo={rel_diff:.4e}  '
              f'range logits torch=[{logits_torch.min():.2f}, {logits_torch.max():.2f}]\n')

    print(f'Diff massima su {len(max_diffs)} slice: {max(max_diffs):.4e}')
    if max(max_diffs) < 1e-2:
        print('=> La pipeline HE-style (packing.py) riproduce FEDELMENTE il modello reale.')
    else:
        print('=> ATTENZIONE: divergenza significativa, da investigare prima di procedere a FIDESlib.')


if __name__ == '__main__':
    main()