"""
crypto/ablate_instance_norm_combined.py

Test combinato: bypassa PIU' layer di InstanceNorm CONTEMPORANEAMENTE
(non uno alla volta come in ablate_instance_norm.py) per verificare se
l'effetto cumulativo sul Dice e' coerente con la somma dei singoli delta,
o se emergono interazioni non lineari (es. rimuovendo insieme piu' layer
"quasi gratis" singolarmente, l'effetto combinato potrebbe essere peggiore
del previsto).

Include anche una verifica di STABILITA' REALE (clamp disattivato) sulla
combinazione scelta -- il test precedente (ablate_instance_norm.py) girava
con clamp attivo, quindi non garantisce che i layer rimossi non introducano
instabilita' mascherata dal clamp, come gia' visto piu' volte nel progetto.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits
from training.train import dice_score


def evaluate_dice(model, val_loader, device):
    model.eval()
    dice_rv, dice_myo, dice_lv = [], [], []
    n_exploded = 0
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            logits = model(imgs)
            if not torch.isfinite(logits).all():
                n_exploded += 1
                continue
            preds = logits.argmax(dim=1)
            scores = dice_score(preds, segs)
            dice_rv.append(scores[1]); dice_myo.append(scores[2]); dice_lv.append(scores[3])
    if not dice_rv:
        return None, n_exploded
    mean_dice = (sum(dice_rv)/len(dice_rv) + sum(dice_myo)/len(dice_myo) + sum(dice_lv)/len(dice_lv)) / 3
    return mean_dice, n_exploded


def check_stability_with_bypass(model, val_loader, device, bypass_layers):
    """
    Come check_inference_stability.py, ma con alcuni layer di InstanceNorm
    bypassati -- disattiva il clamp interno e conta i batch che esploderebbero
    in NaN/Inf SENZA la rete di sicurezza, sulla combinazione di layer rimossi.
    """
    from models.he_friendly import PolyAct
    original_clamps = {}
    for name, m in model.named_modules():
        if isinstance(m, PolyAct):
            original_clamps[name] = m.clamp_value
            m.clamp_value = float('inf')

    handles = []
    modules_dict = dict(model.named_modules())
    for layer_name in bypass_layers:
        module = modules_dict[layer_name]

        def bypass_hook(mod, inputs, output):
            return inputs[0]

        handles.append(module.register_forward_hook(bypass_hook))

    model.eval()
    n_batches = 0
    n_exploded = 0
    max_val_seen = 0.0
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            n_batches += 1
            if not torch.isfinite(logits).all():
                n_exploded += 1
            else:
                max_val_seen = max(max_val_seen, logits.abs().max().item())

    for h in handles:
        h.remove()
    for name, m in model.named_modules():
        if isinstance(m, PolyAct) and name in original_clamps:
            m.clamp_value = original_clamps[name]

    return n_exploded, n_batches, max_val_seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--filters', type=int, nargs=6, required=True)
    parser.add_argument('--skip_mode', default='sum', choices=['concat', 'sum'])
    parser.add_argument('--weight_standardization', action='store_true')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--layers', type=str, required=True,
                        help="Lista di layer da bypassare insieme, separati da virgola. "
                             "Es: 'enc4.block.1,enc5.block.1,dec0.block.1,enc5.block.4,"
                             "enc3.block.1,enc4.block.4'")
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    bypass_layers = [l.strip() for l in args.layers.split(',')]
    print(f'Layer da bypassare insieme ({len(bypass_layers)}): {bypass_layers}\n')

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode=args.skip_mode,
        weight_standardization=args.weight_standardization, filters=args.filters,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    print(f'Checkpoint caricato: {args.checkpoint}')

    modules_dict = dict(model.named_modules())
    for l in bypass_layers:
        assert l in modules_dict, f"Layer '{l}' non trovato nel modello -- controlla il nome."

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Baseline (nessun layer bypassato) ---
    baseline_dice, _ = evaluate_dice(model, val_loader, device)
    print(f'BASELINE (nessun layer bypassato): Dice={baseline_dice:.4f}\n')

    # --- Bypass combinato: Dice con clamp attivo ---
    handles = []
    for layer_name in bypass_layers:
        module = modules_dict[layer_name]

        def bypass_hook(mod, inputs, output):
            return inputs[0]

        handles.append(module.register_forward_hook(bypass_hook))

    combined_dice, n_exploded_clamp_on = evaluate_dice(model, val_loader, device)
    for h in handles:
        h.remove()

    print(f'COMBINATO ({len(bypass_layers)} layer bypassati, clamp attivo): '
          f'Dice={combined_dice:.4f} (delta={combined_dice - baseline_dice:+.4f})')

    # Confronto con la somma "ingenua" dei delta singoli, se disponibile
    single_ablation_path = 'crypto/instance_norm_ablation.json'
    if os.path.exists(single_ablation_path):
        with open(single_ablation_path) as f:
            single_results = json.load(f)
        sum_of_deltas = sum(single_results[l]['delta'] for l in bypass_layers if l in single_results)
        predicted_dice = baseline_dice + sum_of_deltas
        print(f'Somma "ingenua" dei delta singoli: {sum_of_deltas:+.4f} '
              f'(Dice previsto se additivo: {predicted_dice:.4f})')
        interaction = combined_dice - predicted_dice
        if abs(interaction) > 0.01:
            print(f'\u26a0\ufe0f  Interazione non trascurabile rilevata: {interaction:+.4f} '
                  f'(l\'effetto combinato NON e\' semplicemente additivo)')
        else:
            print(f'Interazione trascurabile ({interaction:+.4f}): l\'effetto e\' approssimativamente additivo')
    print()

    # --- Verifica di stabilita' reale (clamp disattivato) sulla combinazione ---
    print('Verifica di stabilita\' con clamp COMPLETAMENTE disattivato...')
    n_exploded, n_batches, max_val = check_stability_with_bypass(
        model, val_loader, device, bypass_layers)
    print(f'Batch con NaN/Inf nei logit finali: {n_exploded}/{n_batches}')
    print(f'Valore assoluto massimo osservato (batch finiti): {max_val:.2f}')

    if n_exploded == 0:
        print('\n=> STABILE: la combinazione di layer rimossi non introduce esplosioni numeriche.')
    else:
        print(f'\n\u26a0\ufe0f  ATTENZIONE: {n_exploded} batch esplodono senza clamp -- questa '
              f'combinazione di rimozioni introduce instabilita\' mascherata dal clamp.')


if __name__ == '__main__':
    main()