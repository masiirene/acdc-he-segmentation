"""
crypto/simulate_chebyshev_init_isqrt.py

Verifica se un'inizializzazione y0 basata su un'approssimazione di
Chebyshev di 1/sqrt(x) (calcolata SOLO per ottenere un buon punto di
partenza, non come sostituto di Newton-Raphson -- coerente con
l'indicazione di usare Newton-Raphson) riduce il numero di iterazioni
Newton-Raphson necessarie, rispetto alla calibrazione attuale (y0 fisso
per layer, basato sul massimo osservato).

IDEA (da CryptoInvSqrt / PP-STAT, letteratura HE su statistiche private):
invece di un y0 COSTANTE uguale per ogni campione di un layer, si calcola
un y0 che dipende da x stesso, tramite un polinomio di Chebyshev di
grado basso che approssima 1/sqrt(x) sul range calibrato per quel layer.
Essendo il punto di partenza gia' vicino al valore vero per OGNI
campione (non solo per il campione "medio"), servono meno iterazioni
Newton-Raphson per arrivare alla precisione target.

COSTO DA NON DIMENTICARE: valutare il polinomio di Chebyshev su dato
cifrato non e' gratis -- richiede a sua volta delle moltiplicazioni
(una per ogni grado del polinomio, nella forma piu' semplice/sequenziale
-- schemi piu' furbi a albero possono ridurre la profondita' a circa
log2(grado), ma questo e' un dettaglio di implementazione HE da
confermare con Aurora/il wrapper, non assunto qui). Il confronto onesto
e': (iterazioni Newton risparmiate x 3 mult/iterazione) VERSUS (costo di
valutare il polinomio).

Qui si simula con gradi bassi (2, 3, 4) -- adeguati dato che il dominio
per layer, dopo aver fissato un range di calibrazione, non e' enorme.
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
    """y0 puo' essere uno scalare (schema attuale) O un array della stessa
    shape di x (schema Chebyshev, y0 diverso per ogni campione)."""
    y = np.array(y0, dtype=np.float64) * np.ones_like(x)
    for _ in range(n_iter):
        y = y * (1.5 - 0.5 * x * y * y)
    return y


def relative_error(y, x):
    y_true = 1.0 / np.sqrt(x)
    return np.abs(y - y_true) / y_true


def fit_chebyshev_isqrt(x_min, x_max, degree):
    """Approssima f(x) = x^{-1/2} con un polinomio di Chebyshev di grado
    'degree' sull'intervallo [x_min, x_max]. Restituisce l'oggetto
    Chebyshev di numpy, valutabile poi su qualunque x in quel range."""
    # Punti di Chebyshev nell'intervallo (non equispaziati: piu' densi ai
    # bordi, dove l'approssimazione polinomiale tende a essere piu' debole
    # -- e' lo stesso motivo per cui Chebyshev batte un fit ai minimi
    # quadrati su punti equispaziati).
    n_fit_points = max(degree * 4, 50)
    k = np.arange(n_fit_points)
    cheb_nodes = np.cos((2 * k + 1) * np.pi / (2 * n_fit_points))  # in [-1, 1]
    x_nodes = 0.5 * (x_max - x_min) * cheb_nodes + 0.5 * (x_max + x_min)
    y_nodes = 1.0 / np.sqrt(x_nodes)
    # Fit nel dominio [-1,1] standard di Chebyshev, poi si rimappa x prima
    # di valutare (numpy gestisce la mappatura con 'domain=').
    poly = C.Chebyshev.fit(x_nodes, y_nodes, deg=degree, domain=[x_min, x_max])
    return poly


def find_min_iter_scalar_y0(x, y0, max_iter, target):
    for n in range(1, max_iter + 1):
        y = newton_raphson_isqrt(x, y0, n)
        err = relative_error(y, x)
        if np.mean(err < target) >= 0.999:
            return n, np.percentile(err, 99.9)
    return None, None


def find_min_iter_vector_y0(x, y0_vec, max_iter, target):
    for n in range(1, max_iter + 1):
        y = newton_raphson_isqrt(x, y0_vec, n)
        err = relative_error(y, x)
        if np.mean(err < target) >= 0.999:
            return n, np.percentile(err, 99.9)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--filters', type=int, nargs=6, required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--skip_mode', default='sum', choices=['concat', 'sum'])
    parser.add_argument('--weight_standardization', action='store_true')
    parser.add_argument('--cheb_degrees', type=int, nargs='+', default=[2, 3, 4],
                        help="Gradi del polinomio di Chebyshev da provare per layer.")
    parser.add_argument('--max_iter', type=int, default=15)
    parser.add_argument('--target_rel_error', type=float, default=0.01)
    parser.add_argument('--baseline_json', default='crypto/newton_raphson_simulation_v2.json',
                        help="Calibrazione attuale (y0 scalare sul massimo), per il confronto.")
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)
    with open(args.baseline_json) as f:
        baseline = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode=args.skip_mode,
        weight_standardization=args.weight_standardization, filters=args.filters,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    print(f'Checkpoint caricato: {args.checkpoint}\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    print('Raccolgo campioni grezzi di varianza per layer...\n')
    raw = collect_raw_variance_samples(model, val_loader, device)

    print(f"{'Layer':16s} {'iter (attuale)':>15s} | " +
          " | ".join(f"deg{d} iter+val_cost" for d in args.cheb_degrees))
    print("-" * (16 + 18 + 24 * len(args.cheb_degrees)))

    summary = {}
    for name in sorted(raw.keys()):
        x = raw[name].astype(np.float64)
        x = x[x > 0]

        baseline_iter = baseline.get(name, {}).get('iterations_needed')
        baseline_depth = 3 * baseline_iter if baseline_iter else None

        # Stesso range di calibrazione conservativo gia' usato per lo
        # schema attuale (margine di sicurezza sul massimo), per un
        # confronto ad armi pari -- MA per Chebyshev usiamo anche il
        # minimo osservato (non solo il massimo), perche' qui il
        # polinomio deve approssimare bene su TUTTO il range, non solo
        # aver un punto di partenza sicuro per il caso peggiore.
        x_min_calib = np.percentile(x, 0.1)
        x_max_calib = np.percentile(x, 99.9) * 1.2

        row = f"{name:16s} {str(baseline_iter):>15s} | "
        row_results = {}
        for deg in args.cheb_degrees:
            poly = fit_chebyshev_isqrt(x_min_calib, x_max_calib, deg)
            y0_vec = poly(np.clip(x, x_min_calib, x_max_calib))
            y0_vec = np.maximum(y0_vec, 1e-6)  # sicurezza: mai negativo o zero

            n_iter, err = find_min_iter_vector_y0(x, y0_vec, args.max_iter, args.target_rel_error)
            if n_iter is None:
                row += f"deg{deg}: >{args.max_iter} iter | "
                row_results[deg] = None
            else:
                # Costo totale onesto: iterazioni Newton (3 mult/iter) +
                # costo di valutare il polinomio di grado 'deg' (qui
                # assunto = deg moltiplicazioni, caso sequenziale
                # pessimistico -- da rivedere se il wrapper HE supporta
                # una valutazione ad albero piu' efficiente).
                total_depth = 3 * n_iter + deg
                row += f"deg{deg}: {n_iter}it (+{deg}=={total_depth} tot) | "
                row_results[deg] = {"iterations": n_iter, "poly_degree": deg,
                                     "total_depth_incl_poly": total_depth}
        print(row)
        summary[name] = {"baseline_iterations": baseline_iter, "baseline_depth": baseline_depth,
                          "chebyshev_options": row_results}

    print("\n" + "=" * 100)
    print("CONFRONTO PROFONDITA' TOTALE (schema attuale vs miglior grado Chebyshev per layer)")
    print("=" * 100)
    total_baseline_depth = 0
    total_best_cheb_depth = 0
    for name, data in summary.items():
        bd = data['baseline_depth'] or 0
        total_baseline_depth += bd
        valid_options = [v['total_depth_incl_poly'] for v in data['chebyshev_options'].values() if v]
        best_cheb = min(valid_options) if valid_options else bd
        total_best_cheb_depth += best_cheb
        marker = " <-- Chebyshev conviene" if best_cheb < bd else ""
        print(f"  {name:16s} attuale={bd:3d}  miglior_chebyshev={best_cheb:3d}{marker}")

    print(f"\nPROFONDITA' TOTALE attuale (solo normalizzazione, tutti i layer): {total_baseline_depth}")
    print(f"PROFONDITA' TOTALE con Chebyshev (miglior grado per layer, costo polinomio incluso): "
          f"{total_best_cheb_depth}")
    if total_baseline_depth > 0:
        print(f"Variazione: {100*(1 - total_best_cheb_depth/total_baseline_depth):.1f}%")

        # Salva anche i coefficienti REALI del miglior polinomio per layer,
    # pronti per l'uso in crypto/packing.py (instance_norm_per_instance,
    # parametri isqrt_cheb_*) -- il file sopra (chebyshev_init_simulation.json)
    # e' solo un report leggibile, questo e' il file di calibrazione da consumare.
    final_calibration = {}
    for name in sorted(raw.keys()):
        x = raw[name].astype(np.float64)
        x = x[x > 0]
        x_min_calib = np.percentile(x, 0.1)
        x_max_calib = np.percentile(x, 99.9) * 1.2

        best_depth = None
        best_entry = None
        for deg in args.cheb_degrees:
            poly = fit_chebyshev_isqrt(x_min_calib, x_max_calib, deg)
            y0_vec = np.maximum(poly(np.clip(x, x_min_calib, x_max_calib)), 1e-6)
            n_iter, err = find_min_iter_vector_y0(x, y0_vec, args.max_iter, args.target_rel_error)
            if n_iter is None:
                continue
            total_depth = 3 * n_iter + deg
            if best_depth is None or total_depth < best_depth:
                best_depth = total_depth
                best_entry = {
                    "cheb_coeffs": poly.coef.tolist(),
                    "cheb_domain": [float(x_min_calib), float(x_max_calib)],
                    "post_iter": n_iter,
                    "poly_degree": deg,
                    "total_depth": total_depth,
                }
        if best_entry:
            final_calibration[name] = best_entry

    with open('crypto/chebyshev_calibration_final.json', 'w') as f:
        json.dump(final_calibration, f, indent=2)
    print("Calibrazione finale (coefficienti pronti per packing.py) salvata in: "
          "crypto/chebyshev_calibration_final.json")

if __name__ == '__main__':
    main()