"""
crypto/calibrate_kstage_isqrt.py

Calibrazione Chebyshev monotona (con fallback automatico a Newton puro
per i layer che non convergono) per un UNetKStage con k arbitrario --
generalizza il lavoro gia' fatto per la rete a 6 stage (find_min_
monotonic_margin_no6norm.py + calibrate_newton_fallback_no6norm.py),
ma in un solo passaggio, e senza bypass (qui NESSUNA normalizzazione e'
rimossa -- e' un esperimento diverso, sulla PROFONDITA' della rete).

I nomi dei layer sono scoperti dinamicamente da model.named_modules(),
non hardcoded -- funziona identico per k=5, k=4, k=3.
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
from crypto.remove_deep_stages import UNetKStage
from training.dataset import ACDCDataset, load_splits

ORIGINAL_FILTERS = [32, 64, 128, 256, 128, 64]


def collect_raw_variance_samples(model, val_loader, device):
    """Scopre TUTTI i layer InstanceNorm2d nel modello (qualunque sia il
    loro nome/percorso) e raccoglie la varianza vista da ciascuno."""
    per_layer_values = {}

    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0]
            var = x.var(dim=[2, 3], unbiased=False)
            per_layer_values.setdefault(name, []).append(var.detach().flatten().cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.InstanceNorm2d):
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
    k_arr = np.arange(n_fit_points)
    cheb_nodes = np.cos((2 * k_arr + 1) * np.pi / (2 * n_fit_points))
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


def find_min_iter_vector(x, y0_vec, max_iter, target):
    for n in range(2, max_iter + 1):  # minimo 2 iterazioni, fix gia' validato
        y = newton_raphson_isqrt(x, y0_vec, n)
        err = relative_error(y, x)
        if np.mean(err < target) >= 0.999:
            return n, np.percentile(err, 99.9)
    return None, None


def find_min_iter_scalar(x, y0, max_iter, target):
    for n in range(1, max_iter + 1):
        y = newton_raphson_isqrt(x, y0, n)
        err = relative_error(y, x)
        if np.mean(err < target) >= 0.999:
            return n, np.percentile(err, 99.9)
    return None, None


def calibrate_layer_chebyshev(x, extra_safety, cheb_degrees, max_iter, target_rel_error):
    """Prova Chebyshev monotono su tutti i gradi disponibili, ritorna
    la migliore configurazione o None se nessuna converge."""
    x_min_calib = np.percentile(x, 0.1)
    x_max_calib = np.percentile(x, 99.9) * 1.2
    x_clipped = np.clip(x, x_min_calib, x_max_calib)

    best_depth, best_entry = None, None
    for deg in cheb_degrees:
        poly = fit_chebyshev_isqrt(x_min_calib, x_max_calib, deg)
        y_true = 1.0 / np.sqrt(x_clipped)
        overshoot = np.max(poly(x_clipped) - y_true)
        shift = max(0.0, overshoot) * extra_safety

        coeffs = poly.coef.copy()
        coeffs[0] -= shift
        mono_poly = C.Chebyshev(coeffs, domain=poly.domain)
        y0_vec = np.maximum(mono_poly(x_clipped), 1e-6)

        n_iter, err = find_min_iter_vector(x, y0_vec, max_iter, target_rel_error)
        if n_iter is None:
            continue
        total_d = 3 * n_iter + deg
        if best_depth is None or total_d < best_depth:
            best_depth = total_d
            best_entry = {
                "schema": "cheb",
                "cheb_coeffs": mono_poly.coef.tolist(),
                "cheb_domain": [float(x_min_calib), float(x_max_calib)],
                "post_iter": n_iter,
                "poly_degree": deg,
                "total_depth": total_d,
            }
    return best_entry


def calibrate_layer_newton_fallback(x, max_iter, target_rel_error):
    """Fallback Newton puro (y0 scalare sul massimo) per layer con range
    troppo ampio per Chebyshev."""
    x_max_calib = np.percentile(x, 99.9) * 1.2
    y0 = 1.0 / np.sqrt(x_max_calib)
    n_iter, err = find_min_iter_scalar(x, y0, max_iter, target_rel_error)
    if n_iter is None:
        return None
    return {
        "schema": "newton",
        "y0": float(y0),
        "iterations_needed": n_iter,
        "total_depth": 3 * n_iter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--k', type=int, required=True, choices=[3, 4, 5])
    parser.add_argument('--filters', type=int, nargs='+', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--cheb_degrees', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    parser.add_argument('--max_iter', type=int, default=15)
    parser.add_argument('--target_rel_error', type=float, default=0.01)
    parser.add_argument('--extra_safety', type=float, default=1.2)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bypass_layers', type=str, default='',
                        help='Layer da escludere dalla calibrazione (gia\' bypassati nel checkpoint)')
    args = parser.parse_args()

    assert len(args.filters) == args.k

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    model = UNetKStage(args.k, args.filters, clamp_values=clamp_values).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    print(f'Checkpoint caricato: {args.checkpoint} (k={args.k}, filters={args.filters})')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    print('Raccolgo varianza su tutto il validation set...\n')
    raw = collect_raw_variance_samples(model, val_loader, device)
    bypass_set = set(l.strip() for l in args.bypass_layers.split(',') if l.strip())
    raw = {k: v for k, v in raw.items() if k not in bypass_set}
    print(f'Layer esclusi dalla calibrazione (gia\' bypassati): {sorted(bypass_set)}')
    print(f'Layer di normalizzazione trovati: {len(raw)}\n')

    print(f"{'Layer':16s} {'schema':>8s} {'depth':>7s}")
    print("-" * 35)

    result = {}
    total_depth = 0
    for name in sorted(raw.keys()):
        x = raw[name].astype(np.float64)
        x = x[x > 0]

        entry = calibrate_layer_chebyshev(x, args.extra_safety, args.cheb_degrees,
                                           args.max_iter, args.target_rel_error)
        if entry is None:
            entry = calibrate_layer_newton_fallback(x, args.max_iter, args.target_rel_error)
        if entry is None:
            print(f"{name:16s} \u26a0\ufe0f  NESSUNO schema converge -- richiede attenzione manuale")
            continue

        result[name] = entry
        total_depth += entry['total_depth']
        print(f"{name:16s} {entry['schema']:>8s} {entry['total_depth']:7d}")

    print("-" * 35)
    print(f"\nProfondita' totale di normalizzazione ({len(result)} layer): {total_depth}")

    for name in bypass_set:
        result[name] = {"schema": "bypass", "total_depth": 0}
        
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Calibrazione salvata in: {args.out}")


if __name__ == '__main__':
    main()