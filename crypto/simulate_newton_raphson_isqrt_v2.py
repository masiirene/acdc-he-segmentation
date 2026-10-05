"""
crypto/simulate_newton_raphson_isqrt_v2.py

v2: calibra y0 per layer sul MASSIMO osservato (non sulla media geometrica
di p1/p99, come in v1) -- garantisce convergenza monotona senza rischio di
divergenza, al costo di piu' iterazioni per i campioni con varianza molto
piu' piccola del massimo.

MOTIVO DELLA DIVERGENZA IN v1 (diagnosticato dai risultati): Newton-Raphson
per 1/sqrt(x) diverge (y diventa negativo/esplode) quando x*y0^2 > 3 --
cioe' quando capita un campione con x MOLTO PIU' GRANDE di quello su cui
y0 era calibrato. Il minimo osservato non causa mai divergenza (solo
convergenza piu' lenta, l'iterazione cresce piano verso il valore giusto).
La media geometrica di v1 lasciava scoperta la coda ALTA della
distribuzione -> alcuni campioni con x elevato mandavano l'iterazione
fuori scala. Qui invece si parte sempre "da sotto" (y0 = 1/sqrt(x_max)),
quindi x*y0^2 <= 1 per costruzione su OGNI campione osservato in
calibrazione: convergenza monotona garantita, mai overshoot.

MARGINE DI SICUREZZA: si usa il 99.9-esimo percentile (non il massimo
assoluto) come "x_max_calib", moltiplicato per un fattore di margine
(default 1.2x) per coprire con ragionevole sicurezza anche pazienti mai
visti in validazione -- il massimo assoluto su un campione finito di 368
slice non e' detto sia il vero massimo della distribuzione.
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
    y = np.full_like(x, y0, dtype=np.float64)
    diverged_at = np.full(x.shape, -1, dtype=int)
    for it in range(n_iter):
        y = y * (1.5 - 0.5 * x * y * y)
        bad = (~np.isfinite(y)) | (y <= 0)
        newly_bad = bad & (diverged_at == -1)
        diverged_at[newly_bad] = it
        y = np.where(bad, y0, y)
    return y, diverged_at


def relative_error(y, x):
    y_true = 1.0 / np.sqrt(x)
    return np.abs(y - y_true) / y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--filters', type=int, nargs=6, required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--weight_standardization', action='store_true')
    parser.add_argument('--max_iter', type=int, default=25)
    parser.add_argument('--target_rel_error', type=float, default=0.01)
    parser.add_argument('--margin_factor', type=float, default=1.2,
                        help="Moltiplica x_max_calib (p99.9) per questo fattore prima di "
                             "invertirlo in y0, per un margine di sicurezza extra.")
    parser.add_argument('--skip_mode', default='concat', choices=['concat', 'sum'])
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode=args.skip_mode,
        weight_standardization=args.weight_standardization, filters=args.filters,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    print(f'Checkpoint caricato: {args.checkpoint}')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    print('Raccolgo campioni grezzi di varianza per layer...\n')
    raw = collect_raw_variance_samples(model, val_loader, device)

    print(f"{'Layer':16s} {'x_max_calib':>12s} {'y0 (=1/sqrt)':>13s} {'iter min':>9s} "
          f"{'err@iter':>10s} {'divergenze':>11s}")
    print("-" * 78)

    summary = {}
    worst_iter = 0
    any_divergence = False
    unconverged_layers = []

    for name in sorted(raw.keys()):
        x = raw[name].astype(np.float64)
        x = x[x > 0]

        x_max_calib = np.percentile(x, 99.9) * args.margin_factor
        y0 = 1.0 / np.sqrt(x_max_calib)

        found_iter = None
        found_err = None
        n_diverged = 0
        for n_iter in range(1, args.max_iter + 1):
            y, diverged_at = newton_raphson_isqrt(x, y0, n_iter)
            n_diverged = np.sum(diverged_at >= 0)
            err = relative_error(y, x)
            ok_frac = np.mean(err < args.target_rel_error)
            if ok_frac >= 0.999 and found_iter is None:
                found_iter = n_iter
                found_err = np.percentile(err, 99.9)

        if n_diverged > 0:
            any_divergence = True

        if found_iter is None:
            print(f"{name:16s} {x_max_calib:12.4f} {y0:13.4f} {'>'+str(args.max_iter):>9s} "
                  f"{'--':>10s} {n_diverged:11d}")
            unconverged_layers.append(name)
            worst_iter = max(worst_iter, args.max_iter)
        else:
            print(f"{name:16s} {x_max_calib:12.4f} {y0:13.4f} {found_iter:9d} "
                  f"{found_err:10.4%} {n_diverged:11d}")
            worst_iter = max(worst_iter, found_iter)

        summary[name] = {"x_max_calib": float(x_max_calib), "y0": float(y0),
                          "iterations_needed": found_iter, "n_diverged": int(n_diverged),
                          "n_samples": int(len(x))}

    print("-" * 78)
    print(f"\nIterazioni MASSIME richieste su tutti i layer convergenti: {worst_iter}")
    if unconverged_layers:
        print(f"\u26a0\ufe0f  Layer che NON convergono entro {args.max_iter} iterazioni: {unconverged_layers}")
    print(f"Profondita' moltiplicativa stimata (3 mult/iter x {worst_iter} iter): "
          f"{3 * worst_iter} livelli, PER OGNI singola InstanceNorm attraversata.")
    if any_divergence:
        print("\n\u26a0\ufe0f  ATTENZIONE: divergenza rilevata anche con y0 calibrato sul massimo -- "
              "verificare che non ci siano bug nel calcolo (non atteso con questa strategia).")
    else:
        print("\nNessuna divergenza: convergenza monotona garantita su tutti i layer con questo schema.")

    with open('crypto/newton_raphson_simulation_v2.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nDettaglio per layer salvato in: crypto/newton_raphson_simulation_v2.json")


if __name__ == '__main__':
    main()