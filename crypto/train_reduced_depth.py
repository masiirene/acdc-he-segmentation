"""
crypto/train_reduced_depth.py

Allena DA CAPO una NStageUNet a profondita' ridotta (richiesta esplicita
di Aurora, punto 3: "allenare una rete via via piu' piccola" invece di
simulare la rimozione su una rete gia' allenata).
"""

import os
import sys
import time
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models.nstage_unet import NStageUNet
from training.dataset import ACDCDataset, load_splits
from training.train import DiceCELoss, dice_score


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
    parser.add_argument('--n_stages', type=int, required=True, choices=[3, 4, 5, 6])
    parser.add_argument('--filters', type=int, nargs='+', required=True,
                        help='n_stages interi, canali per stage (es. per 4 stage: 32 64 128 256)')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    assert len(args.filters) == args.n_stages

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}, n_stages={args.n_stages}, filters={args.filters}')
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    model = NStageUNet(args.n_stages, args.filters, clamp_values=clamp_values).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parametri: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = DiceCELoss(num_classes=4)

    train_cases, val_cases = load_splits(args.splits_path, fold=args.fold)
    train_ds = ACDCDataset(args.data_dir, train_cases, patch_size=(256, 224), augment=True)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_dice = 0.0
    best_path = os.path.join(args.out_dir, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
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

    print(f'\nn_stages={args.n_stages}: Best Dice = {best_dice:.4f} ({n_params:,} parametri)')


if __name__ == '__main__':
    main()