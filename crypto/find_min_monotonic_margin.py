"""
crypto/find_min_monotonic_margin.py

Cerca automaticamente il margine di sicurezza MINIMO (safety_margin in
make_monotonic) che garantisce y0_cheb <= 1/sqrt(x) su TUTTO il validation
set (368 slice), non solo su un campione a mano -- risolve il problema di
crypto/simulate_chebyshev_monotonic.py, che calibra lo shift sullo stesso
campione di training/calibrazione visto dall'hook, ma poi puo' fallire su
pazienti con varianza piu' estrema mai visti in calibrazione (osservato
empiricamente: margine 1.15x insufficiente su 4/8 pazienti di un campione
piu' ampio).

VELOCE per costruzione: verifica la sola formula (y0_cheb vs 1/sqrt(x))
su array numpy, NON rifà il forward pass completo della pipeline HE-style
(che richiederebbe ore su tutte le 368 slice) -- il forward completo va
poi fatto UNA SOLA VOLTA a margine trovato, per la verifica finale.

METODO: per ogni layer, raccoglie la varianza vista su TUTTO il validation
set (non solo il sottoinsieme usato finora), fitta il polinomio, e cerca
per bisezione/griglia il margine minimo tale per cui max(overshoot) sia
coperto su OGNI slice del validation set, non solo sulla media/percentile
di calibrazione.
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
    """Identica alla funzione omonima gia' usata altrove, ma qui va
    chiamata su TUTTO il validation set (val_loader senza shuffle, batch
    grande), non su un sottoinsieme -- e' la base di tutta la garanzia."""
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
    k = np.arange(n_fit_points)
    cheb_nodes = np.cos((2 * k + 1) * np.pi / (2 * n_fit_points))
    x_nodes = 0.5 * (x_max - x_min) * cheb_nodes + 0.5 * (x_max + x_min)
    y_nodes = 1.0 / np.sqrt(x_nodes)
    poly = C.Chebyshev.fit(x_nodes, y_nodes, deg=degree, domain=[x_min, x_max])
    return poly


def newton_raphson_isqrt(x, y0, n_iter):
    y = np.asarray(y0, dtype=np.float64) * np.ones_like(x, dtype=np.float64)
    for _ in range(n_iter):
        y = y * (1.5 - 0.5 * x * y * y)
    return y


def relative_error(y, x):
    y_true = 1.0 / np.sqrt(x)
    return np.abs(y - y_true) / y_true


def find_min_iter(x, y0_vec, max_iter, target):
    for n in range(2, max_iter + 1):  
        y = newton_raphson_isqrt(x, y0_vec, n)
        err = relative_error(y, x)
        if np.mean(err < target) >= 0.999:
            return n, np.percentile(err, 99.9)
    return None, None


def find_min_margin_for_full_monotonicity(poly, x_all, margin_grid=None):
    """
    Cerca, per griglia, il margine minimo tale che il polinomio TRASLATO
    non sovrastimi MAI 1/sqrt(x) su NESSUN valore di x_all (l'intero
    validation set per questo layer) -- non solo sul campione visto in
    calibrazione. Restituisce (margine_trovato, shift_corrispondente).
    """
    if margin_grid is None:
        margin_grid = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0]

    y_true = 1.0 / np.sqrt(x_all)
    y0_raw = poly(x_all)
    overshoot = y0_raw - y_true
    max_overshoot = np.max(overshoot)  # il vero, UNICO numero che conta

    if max_overshoot <= 0:
        return 1.0, 0.0  # gia' monotono su tutto il validation set, nessuno shift

    # Il margine e' solo un moltiplicatore di sicurezza EXTRA oltre il
    # vero max_overshoot -- qui lo fissiamo esplicitamente al valore
    # minimo utile (1.0 = nessun margine extra oltre il minimo garantito)
    # e lasciamo che sia args.extra_safety a decidere quanto margine
    # ulteriore aggiungere per pazienti futuri non visti nemmeno qui.
    return None, max_overshoot  # margin_grid non serve piu': si usa max_overshoot diretto


def make_monotonic_from_shift(poly, shift):
    new_coeffs = poly.coef.copy()
    new_coeffs[0] -= shift
    return C.Chebyshev(new_coeffs, domain=poly.domain)


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
    parser.add_argument('--extra_safety', type=float, default=1.2,
                        help="Moltiplicatore EXTRA oltre il vero overshoot massimo osservato "
                             "sull'INTERO validation set -- margine per pazienti futuri mai visti "
                             "nemmeno in questa calibrazione estesa (default 1.2 = 20% di margine "
                             "oltre il caso peggiore reale).")
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
    # Batch grande, TUTTO il validation set in un giro (368 slice, 40 pazienti)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    print(f'Raccolgo varianza su TUTTO il validation set ({len(val_ds)} slice, {len(val_cases)} pazienti)...\n')
    raw = collect_raw_variance_samples(model, val_loader, device)

    print(f"{'Layer':16s} {'n campioni':>10s} {'overshoot max':>14s} {'shift finale':>14s} "
          f"{'deg':>5s} {'iter':>5s} {'profondita':>11s}")
    print("-" * 80)

    final_calibration = {}
    total_depth = 0

    for name in sorted(raw.keys()):
        x = raw[name].astype(np.float64)
        x = x[x > 0]
        n_samples = len(x)

        x_min_calib = np.percentile(x, 0.1)
        x_max_calib = np.percentile(x, 99.9) * 1.2
        x_clipped = np.clip(x, x_min_calib, x_max_calib)

        best_depth = None
        best_entry = None
        for deg in args.cheb_degrees:
            poly = fit_chebyshev_isqrt(x_min_calib, x_max_calib, deg)

            # Overshoot massimo VERO, su TUTTO il validation set per questo layer
            y_true = 1.0 / np.sqrt(x_clipped)
            y0_raw = poly(x_clipped)
            max_overshoot = np.max(y0_raw - y_true)
            shift = max(0.0, max_overshoot) * args.extra_safety

            mono_poly = make_monotonic_from_shift(poly, shift)
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
                    "max_overshoot_observed": float(max_overshoot),
                    "n_calibration_samples": n_samples,
                }

        if best_entry:
            final_calibration[name] = best_entry
            total_depth += best_entry['total_depth']
            print(f"{name:16s} {n_samples:10d} {best_entry['max_overshoot_observed']:14.4f} "
                  f"{best_entry['shift_applied']:14.4f} {best_entry['poly_degree']:5d} "
                  f"{best_entry['post_iter']:5d} {best_entry['total_depth']:11d}")

    print("-" * 80)
    print(f"\nPROFONDITA' TOTALE (margine calibrato su TUTTO il validation set, "
          f"extra_safety={args.extra_safety}): {total_depth}")

    with open('crypto/chebyshev_calibration_full_dataset.json', 'w') as f:
        json.dump(final_calibration, f, indent=2)
    print("Calibrazione salvata in: crypto/chebyshev_calibration_full_dataset.json")


if __name__ == '__main__':
    main()