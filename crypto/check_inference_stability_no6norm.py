"""
crypto/check_inference_stability_no6norm.py

Variante di check_inference_stability.py per il checkpoint con 6
InstanceNorm rimosse strutturalmente (crypto/finetune_without_6_
instancenorm.py, Dice 0.8744 dopo fine-tuning). Applica lo STESSO
identico bypass permanente usato durante il fine-tuning, prima di
disattivare il clamp e misurare la stabilita' -- senza questo, lo script
tratterebbe quei 6 layer come InstanceNorm normali ancora attive, dando
una misura non coerente con la configurazione reale del modello.

Layer bypassati (identici a finetune_without_6_instancenorm.py):
enc4.block.1, enc5.block.1, dec0.block.1, enc5.block.4, enc3.block.1,
enc4.block.4.
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from models.he_friendly import PolyAct
from training.dataset import ACDCDataset, load_splits

BYPASS_LAYERS = [
    'enc4.block.1', 'enc5.block.1', 'dec0.block.1',
    'enc5.block.4', 'enc3.block.1', 'enc4.block.4',
]


def identity_bypass_hook(module, inputs, output):
    return inputs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--filters', type=int, nargs=6, required=True)
    parser.add_argument('--skip_mode', default='sum', choices=['concat', 'sum'])
    parser.add_argument('--weight_standardization', action='store_true')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        norm_mode='per_instance', skip_mode=args.skip_mode,
        weight_standardization=args.weight_standardization, filters=args.filters,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Checkpoint: {args.checkpoint}')
    if missing or unexpected:
        print(f'  \u26a0\ufe0f  missing={len(missing)} unexpected={len(unexpected)} '
              f'(atteso 0/0 -- verifica filters/skip_mode)')

    # --- Bypass permanente dei 6 layer, IDENTICO al fine-tuning ---
    modules_dict = dict(model.named_modules())
    for layer_name in BYPASS_LAYERS:
        assert layer_name in modules_dict, f"Layer '{layer_name}' non trovato nel modello"
        modules_dict[layer_name].register_forward_hook(identity_bypass_hook)
    print(f'{len(BYPASS_LAYERS)} layer InstanceNorm bypassati (coerente col checkpoint): {BYPASS_LAYERS}')

    # --- Clamp interno DISATTIVATO su ogni PolyAct, per il vero test di stabilita' ---
    for name, m in model.named_modules():
        if isinstance(m, PolyAct):
            m.clamp_value = float('inf')
    print('Clamp interno DISATTIVATO per la misura (nessun limite)\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'Validation: {len(val_cases)} pazienti, {len(val_ds)} slice\n')

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

    print(f'Batch con NaN/Inf nei LOGITS FINALI (vero criterio di fallimento): '
          f'{n_exploded}/{n_batches} ({100*n_exploded/n_batches:.1f}%)')
    print(f'Valore assoluto massimo osservato (batch finiti): {max_val_seen:.2f}\n')

    print("=== Interpretazione ===")
    if n_exploded == 0:
        print('NESSUNA esplosione NaN/Inf su tutto il validation set, clamp completamente')
        print('disattivato, 6 InstanceNorm rimosse -- il modello resta stabile anche senza')
        print('quei layer, coerente con quanto atteso dall\'ablazione.')
    else:
        print(f'\u26a0\ufe0f  {n_exploded} batch esplodono senza clamp -- la rimozione di questi 6')
        print('   layer introduce instabilita\' che il clamp stava mascherando.')


if __name__ == '__main__':
    main()