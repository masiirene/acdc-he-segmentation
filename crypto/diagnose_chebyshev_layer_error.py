"""
crypto/diagnose_chebyshev_layer_error.py

Diagnostica mirata: confronta, LAYER PER LAYER (non solo sui logit
finali), l'output di ogni InstanceNorm quando usa lo schema Chebyshev
rispetto a quando usa sqrt esatto -- sullo stesso identico paziente,
stesso input in ingresso a ciascun layer (calcolato dal modello PyTorch
reale, cosi' l'unica differenza possibile e' l'approssimazione stessa,
non un accumulo di errori precedenti).

Obiettivo: capire SE l'errore e' concentrato su pochi layer specifici
(soluzione: aumentare la precisione solo li') o distribuito su tutti
(soluzione: il problema e' nello schema in generale, non in un layer).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits
from crypto.packing import evaluate_chebyshev_poly, newton_raphson_isqrt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--filters', type=int, nargs=6, required=True)
    parser.add_argument('--skip_mode', default='sum')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--cheb_json', default='crypto/chebyshev_calibration_final.json')
    parser.add_argument('--sample_index', type=int, default=180,
                        help="Indice del paziente da diagnosticare (usa quello con l'errore piu' grande).")
    args = parser.parse_args()

    device = torch.device('cpu')
    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)
    with open(args.cheb_json) as f:
        cheb_values = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode=args.skip_mode,
        weight_standardization=False, filters=args.filters,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Hook: cattura l'INPUT reale di ogni InstanceNorm durante un vero
    # forward pass PyTorch -- cosi' testiamo l'approssimazione sul valore
    # ESATTO che quel layer vedrebbe nella rete reale, non un valore
    # sintetico.
    captured = {}

    def make_hook(name):
        def hook(module, inputs):
            captured[name] = inputs[0].detach().numpy()
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.InstanceNorm2d):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    img, _ = val_ds[args.sample_index]

    with torch.no_grad():
        model(img.unsqueeze(0))

    for h in handles:
        h.remove()

    print(f"{'Layer':16s} {'var reale':>12s} {'in range?':>10s} {'y0_cheb':>10s} "
          f"{'1/sqrt vero':>12s} {'errore y0':>10s} {'post_iter':>10s} {'errore finale':>14s}")
    print("-" * 100)

    for name, cheb in sorted(cheb_values.items()):
        if name not in captured:
            continue
        x = captured[name]  # (1, C, H, W)
        # stessa formula esatta usata in instance_norm_per_instance
        var = x.var(axis=(2, 3), ddof=0) + 1e-5  # (1, C)
        var = var.flatten()

        x_min, x_max = cheb['cheb_domain']
        in_range = np.mean((var >= x_min) & (var <= x_max))

        var_clipped = np.clip(var, x_min, x_max)
        y0 = evaluate_chebyshev_poly(var_clipped, cheb['cheb_coeffs'], cheb['cheb_domain'])
        y0 = np.maximum(y0, 1e-6)

        true_isqrt = 1.0 / np.sqrt(var)
        y0_err = np.abs(y0 - true_isqrt) / true_isqrt

        final = newton_raphson_isqrt(var, y0, cheb['post_iter'])
        final_err = np.abs(final - true_isqrt) / true_isqrt

        worst_idx = np.argmax(final_err)
        print(f"{name:16s} {var[worst_idx]:12.4f} {100*in_range:9.1f}% {y0[worst_idx]:10.4f} "
              f"{true_isqrt[worst_idx]:12.4f} {100*y0_err[worst_idx]:9.2f}% {cheb['post_iter']:10d} "
              f"{100*final_err[worst_idx]:13.2f}%")


if __name__ == '__main__':
    main()