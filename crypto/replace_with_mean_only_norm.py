"""
crypto/replace_with_mean_only_norm.py

Sostituisce ogni InstanceNorm2d del modello con una versione "mean-only"
(sottrae la media per-istanza/per-canale, NON divide per la deviazione
standard) -- ispirata alla "mean-only batch normalization" di Salimans &
Kingma (2016), la tecnica accompagnatoria alla Weight Normalization
suggerita da Aurora come possibile "normalizzazione senza denominatore".

VANTAGGIO CHIAVE per HE: elimina del tutto il bisogno di approssimare
1/sqrt(varianza) -- nessun Chebyshev, nessun Newton, nessun fallback per
i layer difficili (dec0.block.4, enc0.block.1). La sottrazione della
media e' un'operazione ESATTA in CKKS (nessun errore di approssimazione),
e la riduzione (AccumulateSum) non consuma nemmeno un livello di
profondita' moltiplicativa (e' una somma, non una moltiplicazione).

Il warm-start e' DIRETTO: weight e bias hanno la stessa forma esatta di
InstanceNorm2d, quindi si copiano senza alcuna conversione -- solo la
formula del forward cambia (niente piu' divisione).
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits
from training.train import DiceCELoss, dice_score


class MeanOnlyNorm2d(nn.Module):
    """
    y = gamma * (x - mean_per_instance_per_channel) + beta

    NESSUNA divisione, NESSUNA varianza calcolata -- a differenza di
    InstanceNorm2d, che normalizza anche per la scala (1/sqrt(var+eps)).
    """
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        return self.weight.view(1, -1, 1, 1) * (x - mean) + self.bias.view(1, -1, 1, 1)


def replace_instancenorm_with_meanonly(module):
    for name, child in module.named_children():
        if isinstance(child, nn.InstanceNorm2d):
            device = child.weight.device
            new_norm = MeanOnlyNorm2d(child.num_features).to(device)
            new_norm.weight.data.copy_(child.weight.data)
            new_norm.bias.data.copy_(child.bias.data)
            setattr(module, name, new_norm)
        else:
            replace_instancenorm_with_meanonly(child)


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
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint sorgente GIA\' allenato con InstanceNorm normale '
                             '(es. il checkpoint aggressivo+sum, Dice 0.863)')
    parser.add_argument('--filters', type=int, nargs=6, required=True)
    parser.add_argument('--skip_mode', default='sum')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--out_dir', default='results/test_mean_only_norm')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode=args.skip_mode,
        weight_standardization=False, filters=args.filters,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    print(f'Checkpoint sorgente caricato: {args.checkpoint}')

    n_before = sum(1 for m in model.modules() if isinstance(m, nn.InstanceNorm2d))
    replace_instancenorm_with_meanonly(model)
    n_after = sum(1 for m in model.modules() if isinstance(m, MeanOnlyNorm2d))
    print(f'Sostituiti {n_before} InstanceNorm2d con MeanOnlyNorm2d (verificato: {n_after})\n')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = DiceCELoss(num_classes=4)

    train_cases, val_cases = load_splits(args.splits_path, fold=args.fold)
    train_ds = ACDCDataset(args.data_dir, train_cases, patch_size=(256, 224), augment=True)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    dice0, n_exploded0 = evaluate_dice(model, val_loader, device)
    print(f'Dice PRIMA del fine-tuning (solo sostituzione, warm-start diretto): '
          f'{dice0:.4f}, batch esplosi: {n_exploded0}\n')

    best_dice = dice0
    best_path = os.path.join(args.out_dir, 'best_model.pth')
    torch.save(model.state_dict(), best_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for imgs, segs in train_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            if not torch.isfinite(logits).all():
                continue
            loss = criterion(logits, segs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        val_dice, n_exploded = evaluate_dice(model, val_loader, device)
        dt = time.time() - t0

        marker = ""
        if val_dice is not None and val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)
            marker = "  \u2192 saved best model"

        print(f'Epoch {epoch:3d} | loss {avg_loss:.4f} | val_dice {val_dice:.4f} | '
              f'batch esplosi {n_exploded} | {dt:.1f}s{marker}')

    print(f'\nFine-tuning completato. Best Dice (mean-only, nessuna divisione): {best_dice:.4f}')
    print(f'Checkpoint migliore: {best_path}')


if __name__ == '__main__':
    main()