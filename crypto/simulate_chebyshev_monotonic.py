"""
crypto/simulate_chebyshev_monotonic.py

Verifica l'ipotesi che l'errore end-to-end elevato con l'inizializzazione
Chebyshev sia dovuto alla rottura della garanzia di convergenza monotona
di Newton-Raphson (y0 <= 1/sqrt(x) sempre, mai sovrastima) -- proprieta'
che lo schema originale (y0 scalare sul massimo) garantiva per
costruzione, ma che un polinomio di approssimazione libero non rispetta.

FIX TESTATO: dopo aver fittato il polinomio di Chebyshev normalmente,
si misura il MASSIMO errore di sovrastima (y0_cheb - vero_valore) sul
campione di calibrazione, e si trasla l'intero polinomio verso il basso
di quella quantita' (piu' un margine di sicurezza) -- cosi' y0_cheb <=
1/sqrt(x) e' garantito per costruzione, esattamente come nello schema
originale, ma il punto di partenza resta comunque MOLTO piu' vicino al
valore vero rispetto a un y0 scalare fisso (perche' e' comunque un
polinomio che segue la forma della funzione, solo spostato in basso di
una piccola costante).

Confronta tre schemi per ogni layer:
  (a) Newton puro (y0 scalare fisso) -- gia' calibrato, il riferimento
  (b) Chebyshev libero (nessuna garanzia di monotonia) -- quello gia'
      testato, che ha dato diff end-to-end troppo alto
  (c) Chebyshev traslato (garanzia di monotonia ripristinata) -- il fix
      proposto qui
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


def collect_raw_variance_samples(model, val_loader, device):
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


def newton_raphson_isqrt(x, y0, n_iter):
    y = np.asarray(y0, dtype=np.float64) * np.ones_like(x, dtype=np.float64)
    for _ in range(n_iter):
        y = y * (1.5 - 0.5 * x * y * y)
    return y


def relative_error(y, x):
    y_true = 1.0 / np.sqrt(x)
    return np.abs(y - y_true) / y_true


def fit_chebyshev_isqrt(x_min, x_max, degree):
    n_fit_points = max(degree * 4, 50)
    k = np.arange(n_fit_points)
    cheb_nodes = np.cos((2 * k + 1) * np.pi / (2 * n_fit_points))
    x_nodes = 0.5 * (x_max - x_min) * cheb_nodes + 0.5 * (x_max + x_min)
    y_nodes = 1.0 / np.sqrt(x_nodes)
    poly = C.Chebyshev.fit(x_nodes, y_nodes, deg=degree, domain=[x_min, x_max])
    return poly


def make_monotonic(poly, x, safety_margin=1.05):
    """
    Trasla il polinomio verso il basso in modo che y0_cheb <= 1/sqrt(x)
    SEMPRE (sui campioni forniti), ripristinando la garanzia di
    convergenza monotona di Newton-Raphson. safety_margin > 1 aggiunge un
    margine extra oltre il massimo overshoot osservato, per robustezza
    su campioni futuri non visti in calibrazione (stesso principio del
    margine 1.2x gia' usato per y0_max nello schema originale).
    """
    y_true = 1.0 / np.sqrt(x)
    y0_raw = poly(x)
    overshoot = y0_raw - y_true
    max_overshoot = np.max(overshoot)
    if max_overshoot <= 0:
        shift = 0.0  # gia' monotono, nessuna traslazione necessaria
    else:
        shift = max_overshoot * safety_margin
    new_coeffs = poly.coef.copy()
    new_coeffs[0] -= shift
    return C.Chebyshev(new_coeffs, domain=poly.domain), shift


def find_min_iter(x, y0_vec_or_scalar, max_iter, target, allow_zero=True):
    start = 0 if allow_zero else 1
    for n in range(start, max_iter + 1):
        y = newton_raphson_isqrt(x, y0_vec_or_scalar, n)
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
    print(f'Checkpoint caricato: {args.checkpoint}\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    print('Raccolgo campioni grezzi di varianza per layer...\n')
    raw = collect_raw_variance_samples(model, val_loader, device)

    print(f"{'Layer':16s} {'shift applicato':>16s} {'deg migliore':>12s} "
          f"{'iter (monotono)':>16s} {'profondita totale':>18s} {'vs cheb libero':>16s}")
    print("-" * 90)

    final_calibration = {}
    total_monotonic_depth = 0
    total_free_depth = 0  # dal file gia' generato, per confronto

    free_calib = {}
    if os.path.exists('crypto/chebyshev_calibration_final.json'):
        with open('crypto/chebyshev_calibration_final.json') as f:
            free_calib = json.load(f)

    for name in sorted(raw.keys()):
        x = raw[name].astype(np.float64)
        x = x[x > 0]
        x_min_calib = np.percentile(x, 0.1)
        x_max_calib = np.percentile(x, 99.9) * 1.2

        best_depth = None
        best_entry = None
        for deg in args.cheb_degrees:
            poly = fit_chebyshev_isqrt(x_min_calib, x_max_calib, deg)
            x_clipped = np.clip(x, x_min_calib, x_max_calib)

            # Rendi il polinomio monotono (mai sovrastima) su questo layer
            mono_poly, shift = make_monotonic(poly, x_clipped, safety_margin=1.4)

            y0_vec = np.maximum(mono_poly(x_clipped), 1e-6)
            n_iter, err = find_min_iter(x, y0_vec, args.max_iter, args.target_rel_error, allow_zero=False)
            if n_iter is None:
                continue
            total_depth = 3 * n_iter + deg
            if best_depth is None or total_depth < best_depth:
                best_depth = total_depth
                best_entry = {
                    "cheb_coeffs": mono_poly.coef.tolist(),
                    "cheb_domain": [float(x_min_calib), float(x_max_calib)],
                    "post_iter": n_iter,
                    "poly_degree": deg,
                    "total_depth": total_depth,
                    "shift_applied": float(shift),
                }

        if best_entry:
            final_calibration[name] = best_entry
            total_monotonic_depth += best_entry['total_depth']
            free_depth = free_calib.get(name, {}).get('total_depth', '--')
            if isinstance(free_depth, (int, float)):
                total_free_depth += free_depth
            print(f"{name:16s} {best_entry['shift_applied']:16.4f} {best_entry['poly_degree']:12d} "
                  f"{best_entry['post_iter']:16d} {best_entry['total_depth']:18d} {str(free_depth):>16s}")

    print("-" * 90)
    print(f"\nPROFONDITA' TOTALE Chebyshev MONOTONO (fix): {total_monotonic_depth}")
    print(f"PROFONDITA' TOTALE Chebyshev libero (gia' testato, instabile): {total_free_depth}")
    print(f"PROFONDITA' TOTALE Newton puro (riferimento): 363 (dalla calibrazione precedente)")

    with open('crypto/chebyshev_calibration_monotonic.json', 'w') as f:
        json.dump(final_calibration, f, indent=2)
    print("\nCalibrazione monotona salvata in: crypto/chebyshev_calibration_monotonic.json")


if __name__ == '__main__':
    main()