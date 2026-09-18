"""
crypto/check_ws_sum_norm_collapse.py

Diagnosi mirata: nei test WS + skip_mode=sum (trial50), la varianza minima
di InstanceNorm crolla progressivamente (da ~1e-3 a 1e-13/1e-14) sui layer
dec1.block.1 / dec2.block.1, in parallelo alla crescita esplosiva del clamp
sui logits finali (da 0 a 30k+ interventi entro l'epoch 15).

Ipotesi da verificare: il collasso di varianza PRECEDE l'esplosione del
clamp (causa, non conseguenza) -- e se sì, quale canale specifico collassa
per primo, e cosa succede al gain WS del layer Conv2d immediatamente a
monte in quello stesso canale.

Replica lo stesso identico training loop di training/train.py (stessa loss,
stesso ordine safe_clamp_logits/penalty/backward), ma:
  1. registra un hook per-CANALE (non solo per-istanza aggregato come
     register_instance_norm_variance_hooks in train.py) su OGNI
     InstanceNorm2d, per identificare quale canale specifico collassa
  2. appena una varianza per-canale scende sotto --collapse_threshold,
     stampa un report immediato: layer, indice canale, valore di gain WS
     e norma dei pesi del Conv2d immediatamente a monte in quel canale
  3. si ferma da solo dopo il primo collasso confermato (o dopo
     --max_epochs se non si presenta), invece di continuare alla cieca

USO:
    python3 crypto/check_ws_sum_norm_collapse.py \
        --pretrained ~/Desktop/tesi_acdc/baseline_weights.pth \
        --data_dir ~/Desktop/tesi_acdc/training \
        --splits_path ~/Desktop/tesi_acdc/splits_final.json \
        --clamp_values_json crypto/calibrated_clamp_values.json \
        --collapse_threshold 1e-8 \
        --max_epochs 15
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet, WSConv2d, calibrate_ws_gain
from training.dataset import ACDCDataset, load_splits
from training.train import DiceCELoss, safe_clamp_logits, activation_penalty
from tools.load_pretrained import load_pretrained_conv


def find_preceding_conv(model, norm_name):
    """
    Dato il nome di una InstanceNorm2d nello stile 'dec1.block.1' (indice
    1 nel nn.Sequential del ConvBlock), risale al Conv2d/WSConv2d
    immediatamente precedente nello stesso blocco (indice 0), per poter
    correlare un canale che collassa con il gain/i pesi che lo producono.
    Ritorna (nome_conv, modulo_conv) o (None, None) se non trovato.
    """
    parts = norm_name.rsplit('.', 1)
    if len(parts) != 2:
        return None, None
    block_path, idx_str = parts
    try:
        idx = int(idx_str)
    except ValueError:
        return None, None
    conv_idx = idx - 1  # Conv sta sempre immediatamente prima della Norm
    conv_name = f'{block_path}.{conv_idx}'
    for name, m in model.named_modules():
        if name == conv_name and isinstance(m, nn.Conv2d):
            return name, m
    return None, None


def register_per_channel_variance_hooks(model, min_var_per_channel, collapse_log,
                                        collapse_threshold, epoch_holder):
    """
    Come register_instance_norm_variance_hooks in training/train.py, ma
    tiene traccia della varianza minima PER SINGOLO CANALE (non solo il
    minimo aggregato sull'intero layer) -- necessario per identificare
    ESATTAMENTE quale canale collassa, non solo in quale layer.

    Appena un canale scende sotto collapse_threshold per la prima volta,
    lo registra in collapse_log con epoca, batch e valore -- one-shot per
    canale (non riempie il log ad ogni batch successivo).
    """
    handles = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            x = inputs[0]
            if x.dim() != 4:
                return
            # var per (N, C) su (H, W) -- stessa formula di nn.InstanceNorm2d
            var = x.var(dim=[2, 3], unbiased=False)  # shape (N, C)
            min_per_channel = var.min(dim=0).values  # shape (C,) -- minimo sul batch

            if layer_name not in min_var_per_channel:
                min_var_per_channel[layer_name] = min_per_channel.detach().clone()
            else:
                min_var_per_channel[layer_name] = torch.minimum(
                    min_var_per_channel[layer_name], min_per_channel.detach())

            below = (min_per_channel < collapse_threshold).nonzero(as_tuple=True)[0]
            for ch in below.tolist():
                key = (layer_name, ch)
                if key not in collapse_log:
                    collapse_log[key] = {
                        'epoch': epoch_holder['epoch'],
                        'variance': min_per_channel[ch].item(),
                    }
        return hook

    for name, m in model.named_modules():
        if isinstance(m, nn.InstanceNorm2d):
            handles.append(m.register_forward_hook(make_hook(name)))

    return handles


def report_collapse(model, layer_name, channel_idx, variance_value, epoch):
    """
    Stampa un report immediato al primo collasso rilevato: correlazione
    con il gain WS e la norma dei pesi del Conv2d a monte, per quel canale
    specifico -- per capire se e' il gain a spingere quel canale verso
    un'uscita quasi costante (varianza per-istanza ~0).
    """
    print(f'\n{"="*70}')
    print(f'COLLASSO RILEVATO: {layer_name}, canale {channel_idx}, '
          f'epoch {epoch}, varianza={variance_value:.3e}')
    print(f'{"="*70}')

    conv_name, conv_module = find_preceding_conv(model, layer_name)
    if conv_module is None:
        print(f'  \u26a0\ufe0f  Conv2d a monte non trovato per {layer_name} (nome atteso non risolto)')
        return

    print(f'  Conv2d a monte: {conv_name}')
    if isinstance(conv_module, WSConv2d):
        gain_val = conv_module.gain[channel_idx].item()
        w_flat = conv_module.weight[channel_idx].reshape(-1)
        w_std = w_flat.std(unbiased=False).item()
        w_mean = w_flat.mean().item()
        print(f'  WS gain[{channel_idx}] = {gain_val:.6f}')
        print(f'  Peso grezzo canale {channel_idx}: mean={w_mean:.6f}  std={w_std:.6f}')
        print(f'  Peso EFFETTIVO dopo standardizzazione = gain * (w-mean)/std, quindi la')
        print(f'  scala di uscita di questo filtro e\' governata quasi interamente da gain.')
        if abs(gain_val) < 1e-3:
            print(f'  \u26a0\ufe0f  gain vicino a ZERO -- il filtro produce un\'uscita quasi costante')
            print(f'     per costruzione, indipendentemente dall\'input: la varianza per-istanza')
            print(f'     DEVE essere quasi nulla. Ipotesi confermata per questo canale.')
    else:
        print(f'  (non e\' un WSConv2d -- gain non applicabile)')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--clamp_values_json', default=None)
    parser.add_argument('--act_penalty_weight', type=float, default=1.0)
    parser.add_argument('--collapse_threshold', type=float, default=1e-8,
                        help='Soglia di varianza per-canale sotto cui si considera "collassato". '
                             'Default 1e-8: ben sotto il range normale (~1e-3), ben sopra il '
                             'rumore numerico float32 tipico (~1e-12 visto nei run falliti).')
    parser.add_argument('--max_epochs', type=int, default=15,
                        help='Si ferma comunque dopo questo numero di epoche anche senza collasso.')
    parser.add_argument('--out_json', default='crypto/ws_sum_collapse_report.json')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    torch.manual_seed(42)

    train_cases, val_cases = load_splits(args.splits_path, fold=args.fold)
    train_ds = ACDCDataset(args.data_dir, train_cases, patch_size=(256, 224), augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f'Train slices: {len(train_ds)}')

    clamp_values = None
    if args.clamp_values_json:
        with open(args.clamp_values_json) as f:
            clamp_values = json.load(f)

    model = HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance',
        skip_mode='sum', weight_standardization=True,
    ).to(device)

    model = load_pretrained_conv(model, args.pretrained)
    calibrate_ws_gain(model)
    print('WS gain calibrato sui pesi pretrained caricati')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DiceCELoss(num_classes=4)

    min_var_per_channel = {}
    collapse_log = {}
    epoch_holder = {'epoch': 0}
    handles = register_per_channel_variance_hooks(
        model, min_var_per_channel, collapse_log,
        args.collapse_threshold, epoch_holder)

    reported = set()
    stop = False

    for epoch in range(1, args.max_epochs + 1):
        epoch_holder['epoch'] = epoch
        model.train()
        n_clamped_train = 0

        for imgs, segs in train_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            logits, n_clamp = safe_clamp_logits(logits)
            n_clamped_train += n_clamp
            loss = criterion(logits, segs)
            if args.act_penalty_weight > 0:
                pen = activation_penalty(model, penalty_threshold=50.0)
                loss = loss + args.act_penalty_weight * pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Controllo immediato, non solo a fine epoca: appena un nuovo
            # collasso appare nel log, lo riportiamo e ci fermiamo.
            for key, info in collapse_log.items():
                if key not in reported:
                    reported.add(key)
                    layer_name, ch = key
                    report_collapse(model, layer_name, ch, info['variance'], info['epoch'])
                    stop = True
            if stop:
                break

        print(f'Epoch {epoch:3d} | clamp_train={n_clamped_train} | '
              f'canali collassati finora: {len(collapse_log)}')

        if stop:
            print(f'\nFermato all\'epoch {epoch} dopo il primo collasso rilevato.')
            break

    for h in handles:
        h.remove()

    # Salva il report completo (tutti i minimi per-canale, non solo i collassati)
    report = {
        layer: values.tolist()
        for layer, values in min_var_per_channel.items()
    }
    with open(args.out_json, 'w') as f:
        json.dump({'min_var_per_channel': report,
                   'collapses': {f'{k[0]}[{k[1]}]': v for k, v in collapse_log.items()}}, f, indent=2)
    print(f'\nReport completo salvato in: {args.out_json}')

    if not collapse_log:
        print(f'\nNessun collasso sotto {args.collapse_threshold:.0e} in {args.max_epochs} epoche.')
        print('Il pattern osservato nel trial50 potrebbe richiedere piu\' epoche per manifestarsi,')
        print('o essere sensibile al seed/all\'ordine dei batch -- valuta di alzare --max_epochs.')


if __name__ == '__main__':
    main()