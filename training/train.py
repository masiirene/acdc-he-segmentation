import os
import json
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
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


def get_param_groups(model):
    """
    Raggruppa i parametri del modello per tipo, per permettere di
    congelare/scongelare selettivamente gruppi durante il training a fasi:
      - 'conv': pesi delle Conv2d/ConvTranspose2d (pretrained dal baseline)
      - 'act':  coefficienti a,b,c di ogni PolyAct
      - 'norm': gamma,beta di InstanceNorm2d/BatchNorm2d
    """
    from models.he_friendly import PolyAct
    groups = {'conv': [], 'act': [], 'norm': []}
    seen = set()
    for name, module in model.named_modules():
        if isinstance(module, PolyAct):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups['act'].append(p)
                    seen.add(id(p))
        elif isinstance(module, (nn.InstanceNorm2d, nn.BatchNorm2d)):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups['norm'].append(p)
                    seen.add(id(p))
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups['conv'].append(p)
                    seen.add(id(p))
    return groups


def apply_freeze(model, freeze_arg):
    """
    Congela i gruppi di parametri elencati in freeze_arg (stringa CSV,
    es. 'conv' o 'conv,norm'). Ritorna la lista dei parametri allenabili.
    """
    groups = get_param_groups(model)
    freeze_groups = [g.strip() for g in freeze_arg.split(',') if g.strip()] if freeze_arg else []

    for g in freeze_groups:
        if g not in groups:
            raise ValueError(f"Gruppo di freeze sconosciuto: '{g}'. Validi: {list(groups.keys())}")
        for p in groups[g]:
            p.requires_grad = False

    if freeze_groups:
        print(f'Gruppi congelati: {freeze_groups}')

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'Parametri allenabili: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)')
    return trainable


def dice_score(pred, target, num_classes=4):
    scores = {}
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        score = (2 * intersection + 1e-5) / (p.sum() + t.sum() + 1e-5)
        scores[c] = score.item()
    return scores


def safe_clamp_logits(logits, clamp_value=50.0):
    """
    Clampa i logits in [-clamp_value, clamp_value] come rete di sicurezza
    contro l'esplosione numerica di PolyAct (ax^2+bx+c, non limitata) che
    puo' ancora verificarsi nei layer interni anche con input clippato.

    Ritorna (logits_clampati, n_valori_clampati) per poter monitorare
    quanto spesso interviene.
    """
    n_clamped = (~torch.isfinite(logits) | (logits.abs() > clamp_value)).sum().item()
    # Prima sostituisci eventuali NaN/Inf con 0, poi clippa il resto nel range
    logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
    logits = torch.clamp(logits, -clamp_value, clamp_value)
    return logits, n_clamped


def train(args):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # Fix random seed for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random seed fixed to 42')

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

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: act={args.act}, norm={args.norm}, params={n_params:,}')

    # --- Caricamento pesi iniziali ---
    # --pretrained: solo le Conv2d dal baseline nnU-Net (usato in Fase II,
    #               partendo da un modello HEFriendlyUNet non ancora addestrato)
    # --init_from:  l'intero modello da un checkpoint di una fase precedente
    #               del training a fasi (es. Fase II -> Fase III)
    if args.pretrained:
        model = load_pretrained_conv(model, args.pretrained)
        print(f'Pesi conv caricati da: {args.pretrained}')
    if args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(state)
        print(f'Modello inizializzato da: {args.init_from}')

    # --- Freeze selettivo per il training a fasi ---
    trainable_params = apply_freeze(model, args.freeze)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 1
    best_dice = 0.0
    history   = []
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_dice = ckpt['best_dice']
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}, best dice {best_dice:.3f}')

    def get_lr(epoch):
        if args.lr_schedule == 'constant':
            return args.lr
        elif args.lr_schedule == 'cosine':
            # Warmup lineare (se args.warmup > 0), poi decadimento coseno
            # fino a lr_min all'ultima epoca.
            if args.warmup > 0 and epoch <= args.warmup:
                return args.lr * epoch / args.warmup
            total = max(1, args.epochs - args.warmup)
            progress = (epoch - args.warmup) / total
            progress = min(max(progress, 0.0), 1.0)
            return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))
        elif args.lr_schedule == 'step':
            # Dimezza (o moltiplica per lr_decay) il lr ogni lr_step_size epoche
            n_decays = (epoch - 1) // args.lr_step_size
            return max(args.lr * (args.lr_decay ** n_decays), args.lr_min)
        else:
            raise ValueError(f'lr_schedule sconosciuto: {args.lr_schedule}')

    criterion = DiceCELoss(num_classes=4)

    run_name = f'act={args.act}_norm={args.norm}_bs{args.batch_size}_lr{args.lr}'
    if args.lr_schedule != 'constant':
        run_name += f'_sched-{args.lr_schedule}'
    if args.freeze:
        run_name += f'_freeze-{args.freeze.replace(",", "-")}'
    if args.warmup > 0:
        run_name += f'_warmup{args.warmup}'
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        # Aggiorna lr
        current_lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.train()
        train_loss = 0.0
        n_clamped_train = 0
        for imgs, segs in tqdm(train_loader,
                               desc=f'Epoch {epoch}/{args.epochs} [train]',
                               leave=False):
            imgs = imgs.to(device)
            segs = segs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            logits, n_clamp = safe_clamp_logits(logits)
            n_clamped_train += n_clamp
            loss   = criterion(logits, segs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if n_clamped_train > 0:
            print(f'  \u26a0\ufe0f  Clamp attivato {n_clamped_train} volte nei logits di training')

        # ---------------------------------------------------------------
        # VALIDAZIONE con diagnostica NaN/Inf nei logits
        # ---------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        n_valid_batches = 0
        n_clamped_val = 0
        dice_rv, dice_myo, dice_lv = [], [], []
        nan_cases_this_epoch = []

        with torch.no_grad():
            for batch_idx, (imgs, segs) in enumerate(val_loader):
                imgs = imgs.to(device)
                segs = segs.to(device)
                logits = model(imgs)

                # --- DIAGNOSTICA: quali sample nel batch hanno logits non finiti ---
                # NB: ACDCDataset ha una entry per slice 2D (non per paziente/frame),
                # quindi il mapping corretto passa da val_ds.slices, non da val_cases.
                bad_mask = ~torch.isfinite(logits).all(dim=(1, 2, 3))
                if bad_mask.any():
                    start = batch_idx * val_loader.batch_size
                    bad_local_idxs = bad_mask.nonzero(as_tuple=True)[0].tolist()
                    for li in bad_local_idxs:
                        global_idx = start + li
                        if global_idx < len(val_ds.slices):
                            img_path, _, slice_idx = val_ds.slices[global_idx]
                            # img_path tipo '.../patient002/patient002_frame01.nii.gz'
                            fname = os.path.basename(img_path).replace('.nii.gz', '')
                            case_id = f'{fname}_slice{slice_idx}'
                        else:
                            case_id = f'idx_{global_idx}'
                        nan_cases_this_epoch.append(case_id)

                # --- CLAMP di sicurezza sui logits, prima della loss ---
                logits, n_clamp = safe_clamp_logits(logits)
                n_clamped_val += n_clamp

                loss = criterion(logits, segs)
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    n_valid_batches += 1

                preds = logits.argmax(dim=1)
                scores = dice_score(preds, segs)
                dice_rv.append(scores[1])
                dice_myo.append(scores[2])
                dice_lv.append(scores[3])

        val_loss = val_loss / n_valid_batches if n_valid_batches > 0 else 999.0

        if nan_cases_this_epoch:
            unique_cases = sorted(set(nan_cases_this_epoch))
            print(f'  \u26a0\ufe0f  NaN/Inf nei logits (prima del clamp) — {len(unique_cases)} casi: {unique_cases[:10]}'
                  f'{" ..." if len(unique_cases) > 10 else ""}')
        if n_clamped_val > 0:
            print(f'  \u26a0\ufe0f  Clamp attivato {n_clamped_val} volte nei logits di validazione')

        rv  = sum(dice_rv)  / len(dice_rv)
        myo = sum(dice_myo) / len(dice_myo)
        lv  = sum(dice_lv)  / len(dice_lv)
        mean_dice = (rv + myo + lv) / 3

        print(f'Epoch {epoch:3d} | loss {train_loss:.4f} | val_loss {val_loss:.4f} | '
              f'RV {rv:.3f} MYO {myo:.3f} LV {lv:.3f} | mean {mean_dice:.3f} | lr {current_lr:.2e}')

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'dice_rv': rv, 'dice_myo': myo, 'dice_lv': lv,
            'mean_dice': mean_dice, 'lr': current_lr,
            'nan_cases': sorted(set(nan_cases_this_epoch))
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
    parser.add_argument('--lr_schedule', default='constant', choices=['constant', 'cosine', 'step'],
                        help="'constant' (default, comportamento originale): lr fisso per tutto il training. "
                             "'cosine': warmup lineare (se --warmup>0) poi decadimento a coseno fino a lr_min. "
                             "'step': dimezza (o *lr_decay) il lr ogni lr_step_size epoche.")
    parser.add_argument('--lr_min',   type=float, default=1e-6,
                        help='lr minimo raggiungibile con schedule cosine/step')
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='ogni quante epoche decade il lr con schedule step')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='fattore moltiplicativo del decadimento con schedule step')
    parser.add_argument('--fold',     type=int, default=0)
    parser.add_argument('--pretrained', default=None,
                        help='Path a pesi conv-only del baseline (usato in Fase II)')
    parser.add_argument('--init_from', default=None,
                        help='Path a un checkpoint completo di una fase precedente (usato in Fase III)')
    parser.add_argument('--freeze', default=None,
                        help="Gruppi di parametri da congelare, CSV tra 'conv','act','norm'. "
                             "Es: --freeze conv (Fase II, congela le conv, allena act+norm) "
                             "oppure --freeze norm (Fase III, congela IN, allena conv+act)")
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--warmup',   type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    train(args)