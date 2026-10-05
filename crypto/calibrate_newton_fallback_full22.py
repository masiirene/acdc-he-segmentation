"""
crypto/calibrate_newton_fallback_full22.py

Calibrazione Newton-Raphson pura per i 2 layer che non convergono con
Chebyshev nella configurazione a 22 normalizzazioni attive (aggressivo+
sum): enc0.block.1, dec0.block.4 -- stessi due layer "difficili" gia'
scoperti nella configurazione a 16 norm, confermando che il problema e'
strutturale (range di varianza troppo ampio), non specifico di una
singola configurazione.

A differenza di crypto/calibrate_newton_fallback_no6norm.py, qui NESSUN
layer viene bypassato (checkpoint a piena normalizzazione).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits

TARGET_LAYERS = ['enc0.block.1', 'dec0.block.4']


def collect_raw_variance_samples(model, val_loader, device, target_names):
    per_layer_values = {}

    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0]
            var = x.var(dim=[2, 3], unbiased=False)
            per_layer_values.setdefault(name, []).append(var.detach().flatten().cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.InstanceNorm2d) and name in target_names:
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs = imgs.to(device)
            model(imgs)

    for h in handles:
        h.remove()

    return {name: torch.cat(vals).numpy() for name, vals in per_layer_values.items()}


def newton_raphson_isqrt(x, y0, n_iter):
    y = np.full_like(x, y0, dtype=np.float64)
    for _ in range(n_iter):
        y = y * (1.5 - 0.5 * x * y * y)
    return y


def relative_error(y, x):
    y_true = 1.0 / np.sqrt(x)
    return np.abs(y - y_true) / y_true


def find_min_iter_scalar(x, y0, max_iter, target):
    for n in range(1, max_iter + 1):
        y = newton_raphson_isqrt(x, y0, n)
        err = relative_error(y, x)
        if np.mean(err < target) >= 0.999:
            return n, np.percentile(err, 99.9)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--filters', type=int, nargs=6, required=True)
    parser.add_argument('--skip_mode', default='sum')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--max_iter', type=int, default=15)
    parser.add_argument('--target_rel_error', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode=args.skip_mode,
        weight_standardization=False, filters=args.filters,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    print(f'Checkpoint caricato: {args.checkpoint}')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    print(f'Calibrazione Newton-Raphson pura per: {TARGET_LAYERS}\n')
    raw = collect_raw_variance_samples(model, val_loader, device, set(TARGET_LAYERS))

    print(f"{'Layer':16s} {'x_max_calib':>12s} {'y0':>10s} {'iter':>6s} {'err@iter':>10s}")
    print("-" * 60)

    result = {}
    for name in TARGET_LAYERS:
        x = raw[name].astype(np.float64)
        x = x[x > 0]
        x_max_calib = np.percentile(x, 99.9) * 1.2
        y0 = 1.0 / np.sqrt(x_max_calib)

        n_iter, err = find_min_iter_scalar(x, y0, args.max_iter, args.target_rel_error)
        if n_iter is None:
            print(f"{name:16s} \u26a0\ufe0f  non converge nemmeno con Newton puro entro max_iter={args.max_iter}")
            continue
        result[name] = {"y0": float(y0), "iterations_needed": n_iter}
        print(f"{name:16s} {x_max_calib:12.4f} {y0:10.4f} {n_iter:6d} {100*err:9.3f}%")

    with open('crypto/newton_fallback_full22.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\nSalvato in: crypto/newton_fallback_full22.json")


if __name__ == '__main__':
    main()