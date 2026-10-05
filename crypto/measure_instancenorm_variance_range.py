"""
crypto/measure_instancenorm_variance_range.py

Misura il range REALE (min, max, percentili) della varianza calcolata da
OGNI InstanceNorm2d della rete (norm_mode=per_instance), su tutto il
validation set -- dato mancante finora: i log di training riportano solo
la varianza MINIMA per epoca (per verificare che non collassi verso 0),
mai il massimo.

PERCHE' SERVE: per dimensionare Newton-Raphson per 1/sqrt(x) in HE, serve
sapere il range di x = varianza che la formula iterativa dovra' gestire.
Un range ampio (es. 3+ ordini di grandezza tra min e max, su TUTTI i 22
layer insieme) richiede o piu' iterazioni, o uno scaling per-layer diverso
per portare x in un intervallo dove Newton-Raphson converge in poche
iterazioni con un punto di partenza fisso.

METODO: hook forward-PRE su ogni nn.InstanceNorm2d del modello (cattura
l'input, cioe' l'attivazione prima della normalizzazione) -- calcoliamo
la varianza esattamente come farebbe InstanceNorm2d internamente (media
sui due assi spaziali, per ogni immagine e per ogni canale separatamente,
correction=0 cioe' varianza non corretta per Bessel, coerente con
torch.nn.InstanceNorm2d di default). Eseguito con track_running_stats
irrilevante qui: leggiamo l'input, non le statistiche interne del modulo.

Il minimo qui misurato deve essere confrontabile con "varianza minima
InstanceNorm in tutta l'epoca" gia' loggato in training -- se non lo e'
(discrepanza ampia), verificare che patch_size/batch_size/dati di
validazione coincidano con quelli usati in training.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits


def collect_variance_stats(model, val_loader, device):
    """Registra un forward-pre-hook su ogni InstanceNorm2d, che calcola
    la varianza (per immagine, per canale) dell'input a quel layer --
    esattamente il valore che 1/sqrt() dovra' approssimare in HE."""
    per_layer_values = {}

    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0]  # (B, C, H, W)
            var = x.var(dim=[2, 3], unbiased=False)  # (B, C) -- una varianza per immagine e canale
            per_layer_values.setdefault(name, []).append(var.detach().flatten().cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.InstanceNorm2d):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    n_batches = 0
    n_exploded = 0
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            n_batches += 1
            if not torch.isfinite(logits).all():
                n_exploded += 1

    for h in handles:
        h.remove()

    if n_exploded:
        print(f"\u26a0\ufe0f  {n_exploded}/{n_batches} batch con NaN/Inf nei logit finali -- "
              f"le statistiche di varianza restano valide per i layer PRIMA del punto di esplosione, "
              f"ma vanno lette con cautela.")

    return {name: torch.cat(vals) for name, vals in per_layer_values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--filters', type=int, nargs=6, required=True,
                        help="6 interi [enc0..enc5], DEVE corrispondere al checkpoint")
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--weight_standardization', action='store_true')
    parser.add_argument('--out_json', default='crypto/instancenorm_variance_range.json',
                        help="Salva qui min/max/percentili per layer, per riuso da altri script "
                             "(es. dimensionamento di Newton-Raphson per layer)")
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode='concat',
        weight_standardization=args.weight_standardization, filters=args.filters,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'\u26a0\ufe0f  missing={len(missing)} unexpected={len(unexpected)}')
    print(f'Checkpoint caricato: {args.checkpoint}')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    print(f'Raccolgo la varianza di input per ogni InstanceNorm2d su {len(val_ds)} slice di validazione...\n')
    stats = collect_variance_stats(model, val_loader, device)

    results = {}
    print(f"{'Layer':16s} {'min':>12s} {'p1':>12s} {'p50':>12s} {'p99':>12s} {'max':>12s} {'max/min':>12s}")
    print("-" * 90)
    global_min, global_max = float('inf'), 0.0
    for name in sorted(stats.keys()):
        v = stats[name]
        v_min = v.min().item()
        v_max = v.max().item()
        v_p1 = torch.quantile(v, 0.01).item()
        v_p50 = torch.quantile(v, 0.50).item()
        v_p99 = torch.quantile(v, 0.99).item()
        ratio = v_max / max(v_min, 1e-12)
        print(f"{name:16s} {v_min:12.3e} {v_p1:12.3e} {v_p50:12.3e} {v_p99:12.3e} {v_max:12.3e} {ratio:12.1f}")
        results[name] = {"min": v_min, "p1": v_p1, "p50": v_p50, "p99": v_p99, "max": v_max}
        global_min = min(global_min, v_min)
        global_max = max(global_max, v_max)

    global_ratio = global_max / max(global_min, 1e-12)
    print("-" * 90)
    print(f"{'GLOBALE (tutti i layer)':16s} {global_min:12.3e} {'':>12s} {'':>12s} {'':>12s} "
          f"{global_max:12.3e} {global_ratio:12.1f}")

    print(f"\nRange globale di varianza osservato: [{global_min:.3e}, {global_max:.3e}] "
          f"-- rapporto max/min = {global_ratio:.1f}x")
    if global_ratio > 1000:
        print("\u26a0\ufe0f  Range ampio (>1000x): un singolo Newton-Raphson con punto di partenza fisso "
              "per TUTTI i layer rischia di convergere lentamente o male sui casi estremi. "
              "Valutare uno scaling per-layer (normalizzare x per un fattore noto per layer prima "
              "dell'iterazione) o un punto di partenza calibrato per-layer, non uno globale.")
    else:
        print("Range contenuto: un singolo schema di inizializzazione Newton-Raphson, "
              "eventualmente con un solo fattore di scala globale, dovrebbe bastare per tutti i layer.")

    results["_global"] = {"min": global_min, "max": global_max, "ratio": global_ratio}
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nStatistiche salvate in: {args.out_json}")


if __name__ == '__main__':
    main()