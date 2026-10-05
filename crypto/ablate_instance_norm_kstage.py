"""
crypto/ablate_instance_norm_kstage.py

Ablazione per singolo layer di InstanceNorm su un UNetKStage (k=5,4,3) --
stesso principio di crypto/ablate_instance_norm.py (bypass a identita',
un layer alla volta, misura il delta di Dice), generalizzato per
funzionare su qualunque profondita' ridotta, scoprendo i layer
dinamicamente invece di avere un elenco fisso di 22 nomi.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from crypto.remove_deep_stages import UNetKStage
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--k', type=int, required=True, choices=[3, 4, 5])
    parser.add_argument('--filters', type=int, nargs='+', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_json', required=True)
    args = parser.parse_args()

    assert len(args.filters) == args.k

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    model = UNetKStage(args.k, args.filters, clamp_values=clamp_values).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    print(f'Checkpoint caricato: {args.checkpoint} (k={args.k})')

    norm_layers = [name for name, m in model.named_modules() if isinstance(m, nn.InstanceNorm2d)]
    print(f'Trovati {len(norm_layers)} layer di InstanceNorm da testare.\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    baseline_dice, n_exploded0 = evaluate_dice(model, val_loader, device)
    print(f'BASELINE (nessun layer bypassato): Dice={baseline_dice:.4f}, '
          f'batch esplosi={n_exploded0}\n')

    print(f"{'Layer bypassato':16s} {'Dice':>8s} {'Delta vs baseline':>18s} {'Batch esplosi':>14s}")
    print("-" * 60)

    results = {'_baseline': baseline_dice}
    modules_dict = dict(model.named_modules())
    for layer_name in norm_layers:
        module = modules_dict[layer_name]

        def bypass_hook(mod, inputs, output):
            return inputs[0]

        handle = module.register_forward_hook(bypass_hook)
        dice, n_exploded = evaluate_dice(model, val_loader, device)
        handle.remove()

        if dice is None:
            print(f"{layer_name:16s} {'--':>8s} {'ESPLOSO (NaN/Inf)':>18s} {n_exploded:14d}")
            results[layer_name] = {"dice": None, "delta": None, "n_exploded": n_exploded}
        else:
            delta = dice - baseline_dice
            marker = "  <-- poco impattante" if abs(delta) < 0.02 else ""
            print(f"{layer_name:16s} {dice:8.4f} {delta:+18.4f} {n_exploded:14d}{marker}")
            results[layer_name] = {"dice": dice, "delta": delta, "n_exploded": n_exploded}

    print("\n" + "=" * 60)
    print("RIEPILOGO -- layer ordinati per impatto crescente sul Dice")
    print("=" * 60)
    ranked = sorted(
        [(k, v) for k, v in results.items() if k != '_baseline' and v['dice'] is not None],
        key=lambda kv: abs(kv[1]['delta'])
    )
    for name, v in ranked:
        print(f"  {name:16s} delta={v['delta']:+.4f}")

    exploded_layers = [k for k, v in results.items() if k != '_baseline' and v['dice'] is None]
    if exploded_layers:
        print(f"\n\u26a0\ufe0f  Layer la cui rimozione fa ESPLODERE la rete: {exploded_layers}")

    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDettaglio completo salvato in: {args.out_json}")


if __name__ == '__main__':
    main()