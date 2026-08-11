
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.he_friendly import HEFriendlyUNet
from training.dataset import ACDCDataset, load_splits
from tools.load_pretrained import load_pretrained_conv


class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(self.num_classes):
            p = probs[:, c]
            t = (targets == c).float()
            intersection = (p * t).sum()
            loss += 1 - (2 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth)
        return loss / self.num_classes


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.ce(logits, targets)


def dice_score(pred, target, num_classes=4):
    scores = {}
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        score = (2 * intersection + 1e-5) / (p.sum() + t.sum() + 1e-5)
        scores[c] = score.item()
    return scores


def train(args):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    splits_path = os.path.join(args.data_dir, '..', 'splits_final.json')
    if os.path.exists(splits_path):
        train_cases, val_cases = load_splits(splits_path, fold=args.fold)
    else:
        patients = sorted(os.listdir(args.data_dir))
        all_cases = []
        for p in patients:
            pdir = os.path.join(args.data_dir, p)
            if not os.path.isdir(pdir):
                continue
            for f in os.listdir(pdir):
                if f.endswith('.nii.gz') and '_gt' not in f and '_4d' not in f:
                    frame = f.replace(p + '_frame', '').replace('.nii.gz', '')
                    all_cases.append(f'{p}_frame{frame}')
        n = len(all_cases)
        train_cases = all_cases[:int(n * 0.8)]
        val_cases   = all_cases[int(n * 0.8):]

    print(f'Train cases: {len(train_cases)}, Val cases: {len(val_cases)}')

    train_ds = ACDCDataset(args.data_dir, train_cases,
                           patch_size=(256, 224), augment=True)
    val_ds   = ACDCDataset(args.data_dir, val_cases,
                           patch_size=(256, 224), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f'Train slices: {len(train_ds)}, Val slices: {len(val_ds)}')

    model = HEFriendlyUNet(
        in_channels=1,
        num_classes=4,
        act_type=args.act,
        norm_type=args.norm
    ).to(device)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_dice = ckpt['best_dice']
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}, best dice {best_dice:.3f}')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: act={args.act}, norm={args.norm}, params={n_params:,}')

    if args.pretrained:
        model = load_pretrained_conv(model, args.pretrained)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    def get_lr(epoch):
        return args.lr

    criterion = DiceCELoss(num_classes=4)

    run_name = f'act={args.act}_norm={args.norm}_bs{args.batch_size}_lr{args.lr}' 
    if args.warmup > 0:
        run_name += f'_warmup{args.warmup}'
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    best_dice = 0.0
    history   = []
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        # Aggiorna lr
        current_lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.train()
        train_loss = 0.0
        for imgs, segs in tqdm(train_loader,
                               desc=f'Epoch {epoch}/{args.epochs} [train]',
                               leave=False):
            imgs = imgs.to(device)
            segs = segs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, segs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        dice_rv, dice_myo, dice_lv = [], [], []

        with torch.no_grad():
            for imgs, segs in val_loader:
                imgs = imgs.to(device)
                segs = segs.to(device)
                logits = model(imgs)
                loss   = criterion(logits, segs)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                scores = dice_score(preds, segs)
                dice_rv.append(scores[1])
                dice_myo.append(scores[2])
                dice_lv.append(scores[3])

        val_loss /= len(val_loader)

        if torch.isnan(torch.tensor(val_loss)):
            val_loss = 999.0  # sostituisce NaN ma non skippa l'epoca

        rv  = sum(dice_rv)  / len(dice_rv)
        myo = sum(dice_myo) / len(dice_myo)
        lv  = sum(dice_lv)  / len(dice_lv)
        mean_dice = (rv + myo + lv) / 3

        print(f'Epoch {epoch:3d} | loss {train_loss:.4f} | val_loss {val_loss:.4f} | '
              f'RV {rv:.3f} MYO {myo:.3f} LV {lv:.3f} | mean {mean_dice:.3f} | lr {current_lr:.2e}')

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'dice_rv': rv, 'dice_myo': myo, 'dice_lv': lv,
            'mean_dice': mean_dice, 'lr': current_lr
        })

        if mean_dice > best_dice:
            best_dice = mean_dice
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
            print(f'  → saved best model (mean dice {best_dice:.3f})')
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f'\nEarly stopping at epoch {epoch}')
                break

        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'history': history,
            }, os.path.join(out_dir, f'checkpoint_epoch{epoch}.pth'))
            print(f'  → checkpoint saved at epoch {epoch}')    


    torch.save(model.state_dict(), os.path.join(out_dir, 'final_model.pth'))
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nDone. Best mean Dice: {best_dice:.3f}')
    print(f'Results saved in: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--out_dir',  default='results')
    parser.add_argument('--act',      default='poly', choices=['identity', 'linear', 'squared', 'poly'])
    parser.add_argument('--norm',     default='none', choices=['none', 'batch', 'instance', 'group', 'poly'])
    parser.add_argument('--epochs',   type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--fold',     type=int, default=0)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--warmup',   type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    train(args)