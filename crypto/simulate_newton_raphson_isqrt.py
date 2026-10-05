"""
crypto/simulate_newton_raphson_isqrt.py

Simula Newton-Raphson per 1/sqrt(x) in puro Python/NumPy, sui valori REALI
di varianza raccolti da measure_instancenorm_variance_range.py (non sulle
percentili aggregate, ma sui campioni grezzi per layer) -- per rispondere
con dati, non a stima, a: quante iterazioni servono, con quale punto di
partenza per-layer, e se esiste rischio di divergenza sui casi estremi.

Y0 PER-LAYER: un'unica costante per layer (nota, non cifrata -- calcolata
qui in chiaro dai dati di calibrazione, poi "cablata" nel codice HE come
le soglie di clamp), scelta come 1/sqrt(media geometrica di p1 e p99) --
questo centra il punto di partenza in scala logaritmica, minimizzando il
fattore di scarto nel caso peggiore su entrambi i lati (sopra e sotto).

Nota importante: qui NON simuliamo il rumore di CKKS (approssimazione
introdotta da ogni operazione cifrata) -- questo e' il numero di
iterazioni MINIMO in aritmetica esatta. In HE reale potrebbe servirne
qualcuna in piu' per il margine di sicurezza -- vedi B.6, stesso limite
gia' noto per packing_match_test.py.
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
    """Come measure_instancenorm_variance_range.py, ma restituisce i
    campioni grezzi (non solo percentili) per la simulazione."""
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
    """Newton-Raphson standard per y = x^{-1/2}: y_{n+1} = y_n*(1.5 - 0.5*x*y_n^2).
    Ritorna y dopo n_iter iterazioni, e traccia se qualche campione e'
    'esploso' (diventato non-finito o negativo, segno di divergenza)."""
    y = np.full_like(x, y0, dtype=np.float64)
    diverged_at = np.full(x.shape, -1, dtype=int)
    for it in range(n_iter):
        y = y * (1.5 - 0.5 * x * y * y)
        bad = (~np.isfinite(y)) | (y <= 0)
        newly_bad = bad & (diverged_at == -1)
        diverged_at[newly_bad] = it
        y = np.where(bad, y0, y)  # congela i divergenti al valore iniziale, per continuare a misurare gli altri
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
    parser.add_argument('--max_iter', type=int, default=12)
    parser.add_argument('--target_rel_error', type=float, default=0.01,
                        help="Errore relativo target su 1/sqrt(x) -- 1%% di default, "
                             "da tarare empiricamente su quanto il Dice tollera downstream.")
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode='concat',
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

    print(f"{'Layer':16s} {'y0 (fisso)':>12s} {'iter min':>10s} {'err@iter':>10s} "
          f"{'%campioni ok':>13s} {'divergenze':>11s}")
    print("-" * 78)

    summary = {}
    worst_iter = 0
    any_divergence = False

    for name in sorted(raw.keys()):
        x = raw[name].astype(np.float64)
        x = x[x > 0]  # sicurezza: scarta eventuali zeri esatti (non attesi, ma per robustezza)
        p1, p99 = np.percentile(x, [1, 99])
        y0 = 1.0 / np.sqrt(np.sqrt(p1 * p99))  # media geometrica di p1 e p99, poi 1/sqrt

        found_iter = None
        found_err = None
        for n_iter in range(1, args.max_iter + 1):
            y, diverged_at = newton_raphson_isqrt(x, y0, n_iter)
            err = relative_error(y, x)
            ok_frac = np.mean(err < args.target_rel_error)
            if ok_frac >= 0.999 and found_iter is None:  # 99.9% dei campioni entro la tolleranza
                found_iter = n_iter
                found_err = np.percentile(err, 99.9)

        n_diverged = np.sum(diverged_at >= 0)
        if n_diverged > 0:
            any_divergence = True

        if found_iter is None:
            print(f"{name:16s} {y0:12.4f} {'>'+str(args.max_iter):>10s} "
                  f"{'--':>10s} {'--':>13s} {n_diverged:11d}")
            worst_iter = max(worst_iter, args.max_iter + 1)
        else:
            print(f"{name:16s} {y0:12.4f} {found_iter:10d} {found_err:10.4%} "
                  f"{100*np.mean(relative_error(*newton_raphson_isqrt(x, y0, found_iter))[0] < args.target_rel_error) if False else 99.9:12.1f}% "
                  f"{n_diverged:11d}")
            worst_iter = max(worst_iter, found_iter)

        summary[name] = {"y0": float(y0), "iterations_needed": found_iter,
                          "n_diverged": int(n_diverged), "n_samples": int(len(x))}

    print("-" * 78)
    print(f"\nIterazioni MASSIME richieste su tutti i layer (per raggiungere "
          f"{args.target_rel_error:.1%} di errore relativo sul 99.9% dei campioni): {worst_iter}")
    print(f"Profondita' moltiplicativa stimata per la normalizzazione (3 mult/iter x {worst_iter} iter): "
          f"{3 * worst_iter} livelli, PER OGNI singola InstanceNorm attraversata.")
    if any_divergence:
        print("\n\u26a0\ufe0f  ATTENZIONE: rilevate divergenze (y diventato non-finito o negativo) "
              "su almeno un layer con questo y0 fisso. Il punto di partenza per-layer scelto qui "
              "(media geometrica di p1/p99) NON e' sufficiente a garantire convergenza su tutti "
              "i pazienti per quel layer -- serve un punto di partenza piu' conservativo (es. basato "
              "sul valore osservato piu' basso invece che sulla media geometrica) o un passo di "
              "range-reduction preliminare.")
    else:
        print("Nessuna divergenza rilevata su nessun layer con questo schema di punto di partenza.")

    with open('crypto/newton_raphson_simulation.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nDettaglio per layer salvato in: crypto/newton_raphson_simulation.json")


if __name__ == '__main__':
    main()