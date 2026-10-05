"""
crypto/find_min_monotonic_margin_no6norm.py

Variante di find_min_monotonic_margin.py per il checkpoint con 6
InstanceNorm rimosse (crypto/finetune_without_6_instancenorm.py, Dice
0.8744). Applica lo stesso bypass permanente usato nel fine-tuning PRIMA
di raccogliere i campioni di varianza -- cosi' la distribuzione vista
dai 16 layer rimanenti riflette il comportamento REALE della rete
(l'assenza delle 6 normalizzazioni cambia anche la distribuzione a valle
negli altri layer, non solo in se stessa).

I 6 layer bypassati vengono ESCLUSI dalla calibrazione (nessun hook
registrato su di loro) -- non serve calibrarli, non vengono mai usati
nella traduzione HE di questa configurazione.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from numpy.polynomial import chebyshev as C

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits

BYPASS_LAYERS = [
    'enc4.block.1', 'enc5.block.1', 'dec0.block.1',
    'enc5.block.4', 'enc3.block.1', 'enc4.block.4',
]


def identity_bypass_hook(module, inputs, output):
    return inputs[0]


def collect_raw_variance_samples(model, val_loader, device, skip_names):
    per_layer_values = {}

    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0]
            var = x.var(dim=[2, 3], unbiased=False)
            per_layer_values.setdefault(name, []).append(var.detach().flatten().cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.InstanceNorm2d) and name not in skip_names:
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs = imgs.to(device)
            model(imgs)

    for h in handles:
        h.remove()

    return {name: torch.cat(vals).numpy() for name, vals in per_layer_values.items()}


def fit_chebyshev_isqrt(x_min, x_max, degree):
    n_fit_points = max(degree * 4, 50)
    k = np.arange(n_fit_points)
    cheb_nodes = np.cos((2 * k + 1) * np.pi / (2 * n_fit_points))
    x_nodes = 0.5 * (x_max - x_min) * cheb_nodes + 0.5 * (x_max + x_min)
    y_nodes = 1.0 / np.sqrt(x_nodes)
    return C.Chebyshev.fit(x_nodes, y_nodes, deg=degree, domain=[x_min, x_max])


def newton_raphson_isqrt(x, y0, n_iter):
    y = np.asarray(y0, dtype=np.float64) * np.ones_like(x, dtype=np.float64)
    for _ in range(n_iter):
        y = y * (1.5 - 0.5 * x * y * y)
    return y


def relative_error(y, x):
    y_true = 1.0 / np.sqrt(x)
    return np.abs(y - y_true) / y_true


def find_min_iter(x, y0_vec, max_iter, target):
    # Minimo 2 iterazioni sempre -- fix gia' validato: con 0-1 iterazioni
    # l'effetto domino tra layer in cascata puo' far fallire pazienti mai
    # visti in calibrazione, anche con margine di sicurezza ampio.
    for n in range(2, max_iter + 1):
        y = newton_raphson_isqrt(x, y0_vec, n)
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
    parser.add_argument('--cheb_degrees', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    parser.add_argument('--max_iter', type=int, default=15)
    parser.add_argument('--target_rel_error', type=float, default=0.01)
    parser.add_argument('--extra_safety', type=float, default=1.2)
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

    # Bypass permanente dei 6 layer, IDENTICO al fine-tuning -- cosi' la
    # distribuzione vista dagli altri 16 layer riflette la rete reale.
    modules_dict = dict(model.named_modules())
    for layer_name in BYPASS_LAYERS:
        modules_dict[layer_name].register_forward_hook(identity_bypass_hook)
    print(f'{len(BYPASS_LAYERS)} layer bypassati (esclusi dalla calibrazione): {BYPASS_LAYERS}\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    print(f'Raccolgo varianza su tutto il validation set ({len(val_ds)} slice)...\n')
    raw = collect_raw_variance_samples(model, val_loader, device, skip_names=set(BYPASS_LAYERS))
    print(f'Layer calibrati: {len(raw)} (atteso: 22 - 6 = 16)\n')

    print(f"{'Layer':16s} {'overshoot max':>14s} {'shift finale':>14s} {'deg':>5s} {'iter':>5s} {'profondita':>11s}")
    print("-" * 78)

    final_calibration = {}
    total_depth = 0

    for name in sorted(raw.keys()):
        x = raw[name].astype(np.float64)
        x = x[x > 0]

        x_min_calib = np.percentile(x, 0.1)
        x_max_calib = np.percentile(x, 99.9) * 1.2
        x_clipped = np.clip(x, x_min_calib, x_max_calib)

        best_depth = None
        best_entry = None
        for deg in args.cheb_degrees:
            poly = fit_chebyshev_isqrt(x_min_calib, x_max_calib, deg)
            y_true = 1.0 / np.sqrt(x_clipped)
            overshoot = np.max(poly(x_clipped) - y_true)
            shift = max(0.0, overshoot) * args.extra_safety

            coeffs = poly.coef.copy()
            coeffs[0] -= shift
            mono_poly = C.Chebyshev(coeffs, domain=poly.domain)
            y0_vec = np.maximum(mono_poly(x_clipped), 1e-6)

            n_iter, err = find_min_iter(x, y0_vec, args.max_iter, args.target_rel_error)
            if n_iter is None:
                continue
            total_d = 3 * n_iter + deg
            if best_depth is None or total_d < best_depth:
                best_depth = total_d
                best_entry = {
                    "cheb_coeffs": mono_poly.coef.tolist(),
                    "cheb_domain": [float(x_min_calib), float(x_max_calib)],
                    "post_iter": n_iter,
                    "poly_degree": deg,
                    "total_depth": total_d,
                    "shift_applied": float(shift),
                }

        # IMPORTANTE: stampa e somma SOLO qui, UNA VOLTA per layer, dopo
        # aver provato tutti i gradi -- non dentro il ciclo "for deg".
        if best_entry:
            final_calibration[name] = best_entry
            total_depth += best_entry['total_depth']
            print(f"{name:16s} {best_entry['shift_applied']:14.4f} "
                  f"{'--':>14s} {best_entry['poly_degree']:5d} "
                  f"{best_entry['post_iter']:5d} {best_entry['total_depth']:11d}")
        else:
            print(f"{name:16s} \u26a0\ufe0f  NESSUNA configurazione converge entro max_iter="
                  f"{args.max_iter} (range x: [{x.min():.4f}, {x.max():.4f}], "
                  f"rapporto max/min: {x.max()/x.min():.1f}x)")

    print("-" * 78)
    print(f"\nPROFONDITA' TOTALE normalizzazione (16 layer, Chebyshev monotono): {total_depth}")

    with open('crypto/chebyshev_calibration_no6norm.json', 'w') as f:
        json.dump(final_calibration, f, indent=2)
    print("Calibrazione salvata in: crypto/chebyshev_calibration_no6norm.json")


if __name__ == '__main__':
    main()