"""
crypto/simulate_range_reduction.py

Quantifica il guadagno TEORICO MASSIMO della range reduction per Newton-
Raphson (1/sqrt(x)): nell'ipotesi che l'esponente di scala k (tale che
x/4^k cada in una finestra fissa e stretta, uguale per tutti i layer) sia
noto GRATIS -- cioe' SENZA contare il costo di calcolarlo su dato cifrato,
che e' la parte davvero difficile da tradurre in HE (richiede un
equivalente di confronto/arrotondamento, non nativo in CKKS).

Questo script risponde a una domanda preliminare, prima di investire
tempo nell'implementazione HE reale: "quanto risparmieremmo in iterazioni
di Newton-Raphson, nel MIGLIOR caso possibile?" Se il guadagno fosse gia'
marginale qui, non varrebbe la pena costruire il meccanismo di estrazione
dell'esponente (che ha un costo proprio, potenzialmente maggiore del
risparmio -- vedi discussione in fondo all'output).

METODO: per ogni layer, calcola k = floor(log_4(x)) per ogni campione
(operazione fatta in chiaro qui, essendo una simulazione -- in HE andrebbe
approssimata, vedi sotto), riscala x' = x / 4^k (sempre in [1,4) per
costruzione), e misura quante iterazioni di Newton-Raphson servono con UN
SOLO y0 CONDIVISO tra TUTTI i layer (la media geometrica della finestra
[1,4)) per raggiungere lo stesso target di errore usato finora.

Confronta poi:
  - profondita' Newton-Raphson SENZA range reduction (dati gia' calibrati,
    da crypto/newton_raphson_simulation_v2.json)
  - profondita' Newton-Raphson CON range reduction (nell'ipotesi ottimistica
    che trovare k sia gratis)
  - una stima ONESTA del costo di trovare k in HE (vedi discussione finale),
    per capire se il guadagno netto sia realmente positivo
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
    for _ in range(n_iter):
        y = y * (1.5 - 0.5 * x * y * y)
    return y


def relative_error(y, x):
    y_true = 1.0 / np.sqrt(x)
    return np.abs(y - y_true) / y_true


def find_min_iter(x, y0, max_iter, target):
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
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--weight_standardization', action='store_true')
    parser.add_argument('--baseline_json', default='crypto/newton_raphson_simulation_v2.json',
                        help="Calibrazione SENZA range reduction, gia' salvata, per il confronto.")
    parser.add_argument('--base', type=float, default=4.0,
                        help="Base della riduzione di range: x viene riscalato a x/base^k, "
                             "con k intero, per farlo cadere sempre in [1, base). base=4 e' "
                             "comoda perche' 1/sqrt(4^k) = 2^(-k), una potenza di 2 esatta.")
    parser.add_argument('--max_iter', type=int, default=15)
    parser.add_argument('--target_rel_error', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)
    with open(args.baseline_json) as f:
        baseline = json.load(f)

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

    # y0 condiviso per TUTTI i layer: media geometrica della finestra [1, base)
    y0_shared = 1.0 / np.sqrt(np.sqrt(1.0 * args.base))

    print(f"Finestra fissa dopo riduzione: [1, {args.base:.0f}) -- y0 condiviso = {y0_shared:.4f}\n")
    print(f"{'Layer':16s} {'iter SENZA riduz.':>18s} {'iter CON riduz.':>17s} {'risparmio':>10s}")
    print("-" * 68)

    total_without = 0
    total_with = 0
    worst_with = 0
    for name in sorted(raw.keys()):
        x = raw[name].astype(np.float64)
        x = x[x > 0]

        k = np.floor(np.log(x) / np.log(args.base))
        x_scaled = x / (args.base ** k)  # sempre in [1, base) per costruzione

        n_iter_with, err_with = find_min_iter(x_scaled, y0_shared, args.max_iter, args.target_rel_error)
        n_iter_without = baseline.get(name, {}).get('iterations_needed')

        if n_iter_with is None:
            print(f"{name:16s} {'--':>18s} {'>'+str(args.max_iter):>17s} {'--':>10s}")
            continue

        risparmio = n_iter_without - n_iter_with if n_iter_without else None
        risparmio_str = f"{risparmio:+d}" if risparmio is not None else "--"
        print(f"{name:16s} {n_iter_without if n_iter_without else '--':>18} "
              f"{n_iter_with:>17d} {risparmio_str:>10s}")

        if n_iter_without:
            total_without += n_iter_without
        total_with += n_iter_with
        worst_with = max(worst_with, n_iter_with)

    print("-" * 68)
    print(f"\nIterazioni totali SENZA range reduction (calibrazione per-layer attuale): {total_without}")
    print(f"Iterazioni totali CON range reduction (y0 condiviso, k GRATIS):            {total_with}")
    print(f"Caso peggiore CON riduzione (iterazioni condivise, unico valore per tutta "
          f"la rete): {worst_with}")

    depth_without = 3 * total_without + 2 * len(raw)  # 2 = varsq + apply, per layer
    # Con riduzione: serve un'unica n_iter CONDIVISA tra tutti i layer, dato che k
    # varia per CAMPIONE (non per layer) -- non ha senso un n_iter diverso per layer
    # quando lo stesso schema di riduzione si applica a ogni singolo valore cifrato
    # indipendentemente da quale layer appartiene.
    depth_with_math_only = len(raw) * (2 + 3 * worst_with)

    print(f"\nProfondita' matematica pura (Newton-Raphson, SENZA il costo di trovare k):")
    print(f"  Senza range reduction: {depth_without} livelli")
    print(f"  Con range reduction:   {depth_with_math_only} livelli "
          f"({100*(1 - depth_with_math_only/depth_without):.0f}% di risparmio)")

    print(f"\n\u26a0\ufe0f  QUESTO CONFRONTO E' OTTIMISTICO: assume che calcolare k = floor(log_{args.base:.0f}(x))")
    print("   su dato CIFRATO non costi nulla. In realta' CKKS non ha logaritmo ne'")
    print("   arrotondamento nativi: k va approssimato con una tecnica di confronto")
    print("   (es. polinomi che approssimano una funzione a gradino, iterati piu' volte")
    print("   per renderli abbastanza 'affilati' da distinguere i bucket) -- tecnica")
    print("   simile, per costo, alla stessa famiglia di approssimazioni che stiamo gia'")
    print("   usando per il clamp. Il numero di bucket necessari dipende dal range")
    print(f"   osservato: con un rapporto max/min fino a ~100x (vedi enc0.block.1), servono")
    print(f"   circa log_{args.base:.0f}(100) ~= {np.log(100)/np.log(args.base):.1f} confronti di bucket, ciascuno")
    print("   verosimilmente costando qualche livello -- un costo che puo' facilmente")
    print("   avvicinarsi o superare il risparmio mostrato sopra. Prima di investire")
    print("   tempo nell'implementazione HE reale di questo meccanismo, raccomando di")
    print("   discutere con Aurora se FIDESlib offre una primitiva di confronto/")
    print("   arrotondamento a basso costo -- altrimenti il guadagno netto e' incerto.")


if __name__ == '__main__':
    main()