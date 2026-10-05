"""
crypto/finetune_without_6_instancenorm.py

Fine-tuning del checkpoint aggressivo+sum con 6 InstanceNorm rimosse
strutturalmente (bypassate in modo permanente, sia in training che in
eval), a partire dal risultato di crypto/ablate_instance_norm_combined.py:
Dice 0.8255 con clamp attivo, 0/23 batch esplosi a clamp disattivato,
PRIMA di qualunque fine-tuning.

Layer rimossi (i 6 con impatto singolo trascurabile, vedi
crypto/instance_norm_ablation.json): enc4.block.1, enc5.block.1,
dec0.block.1, enc5.block.4, enc3.block.1, enc4.block.4.

METODO: nessuna selezione di canali necessaria (rimuovere una
InstanceNorm non cambia alcuna shape di peso) -- carichiamo il checkpoint
sorgente al 100% via load_state_dict, poi registriamo hook permanenti
(forward_hook che sostituisce l'output con l'input, cioe' un'identita')
sui 6 layer bersaglio. Gli hook restano attivi per TUTTO il fine-tuning
(sia model.train() che model.eval()), cosi' la rete impara a compensare
la loro assenza invece di limitarsi a "subirla" come nel test di ablazione.

Le 12 coppie di parametri gamma/beta di quei 6 layer restano nello
state_dict (per compatibilita' di caricamento) ma diventano INUTILIZZATE
dal forward pass -- le escludiamo esplicitamente dall'ottimizzatore, cosi'
non vengono aggiornate a vuoto durante il fine-tuning (sarebbe uno spreco
di calcolo, e potrebbe confondere la lettura dei log di training).
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits
from training.train import DiceCELoss, dice_score

BYPASS_LAYERS = [
    'enc4.block.1', 'enc5.block.1', 'dec0.block.1',
    'enc5.block.4', 'enc3.block.1', 'enc4.block.4',
]


def identity_bypass_hook(module, inputs, output):
    return inputs[0]


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
                        help='Checkpoint sorgente aggressivo+sum (Dice 0.863)')
    parser.add_argument('--filters', type=int, nargs=6, required=True)
    parser.add_argument('--skip_mode', default='sum', choices=['concat', 'sum'])
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--out_dir', default='results/test_sum_aggressive_no6norm')
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Checkpoint sorgente caricato: {args.checkpoint}')
    print(f'  missing={len(missing)} unexpected={len(unexpected)} (attesi entrambi 0: '
          f'nessuna modifica di shape, solo comportamento del forward)')

    # --- Bypass permanente dei 6 layer, attivo sia in train che in eval ---
    modules_dict = dict(model.named_modules())
    frozen_param_names = set()
    for layer_name in BYPASS_LAYERS:
        assert layer_name in modules_dict, f"Layer '{layer_name}' non trovato nel modello"
        module = modules_dict[layer_name]
        module.register_forward_hook(identity_bypass_hook)
        frozen_param_names.add(f'{layer_name}.weight')  # gamma
        frozen_param_names.add(f'{layer_name}.bias')    # beta
    print(f'\n{len(BYPASS_LAYERS)} layer InstanceNorm bypassati permanentemente: {BYPASS_LAYERS}')

    # --- Esclude gamma/beta dei layer bypassati dall'ottimizzatore ---
    # (restano nello state_dict per compatibilita', ma non ricevono piu'
    # update -- il forward pass non li usa comunque, dato il bypass).
    trainable_params = [p for name, p in model.named_parameters()
                        if name not in frozen_param_names]
    n_frozen = sum(1 for name, _ in model.named_parameters() if name in frozen_param_names)
    print(f'Parametri esclusi dall\'ottimizzatore (gamma/beta dei layer bypassati): {n_frozen}')

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = DiceCELoss(num_classes=4)

    train_cases, val_cases = load_splits(args.splits_path, fold=args.fold)
    train_ds = ACDCDataset(args.data_dir, train_cases, patch_size=(256, 224), augment=True)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'Train cases: {len(train_cases)}, Val cases: {len(val_cases)}\n')

    dice0, n_exploded0 = evaluate_dice(model, val_loader, device)
    print(f'Dice PRIMA del fine-tuning (con i 6 layer gia\' bypassati): {dice0:.4f}, '
          f'batch esplosi: {n_exploded0}\n')

    best_dice = dice0
    best_path = os.path.join(args.out_dir, 'best_model.pth')
    torch.save(model.state_dict(), best_path)  # baseline come primo "best", in caso il ft non migliori

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
                continue  # batch instabile, salta (raro, ma coerente con la cautela del progetto)
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

    final_path = os.path.join(args.out_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f'\nFine-tuning completato. Best Dice: {best_dice:.4f}')
    print(f'Checkpoint migliore: {best_path}')
    print(f'Checkpoint finale (ultima epoca): {final_path}')


if __name__ == '__main__':
    main()