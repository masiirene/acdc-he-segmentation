"""
crypto/check_instancenorm_drift.py

Verifica un'ipotesi emersa osservando il training con la penalita' sulle
attivazioni: la penalita' scende in training (segno che i pesi si stanno
"comportando bene"), ma il clamp esplode in validazione -- un comportamento
opposto e sospetto.

Ipotesi: durante la Fase III si allenano i pesi convoluzionali (sblocati)
mentre InstanceNorm resta CONGELATA (running_mean/running_var accumulate
durante la Fase II, con pesi diversi da quelli attuali). Man mano che i
pesi conv si allontanano da quelli originali, la distribuzione del loro
output (l'input a InstanceNorm) puo' disallinearsi dalle statistiche
congelate -- che quindi non normalizzano piu' correttamente, producendo
input sballati a PolyAct SOLO in eval mode (dove le statistiche fisse
vengono usate), non in training (dove IN calcola comunque al volo, anche
se non le stiamo aggiornando esplicitamente).

Questo script confronta, per ogni InstanceNorm2d del modello:
  - le statistiche CONGELATE (running_mean/running_var, quelle usate in eval)
  - le statistiche "LIVE" (calcolate al volo sui dati di validazione reali,
    con i pesi conv ATTUALI del checkpoint)
Una grande differenza tra le due conferma il disallineamento.

USO:
    python3 crypto/check_instancenorm_drift.py --checkpoint <path>
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet
from training.dataset import ACDCDataset, load_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly', norm_type='instance').to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()  # le InstanceNorm useranno le statistiche CONGELATE nel forward
    print(f'Checkpoint: {args.checkpoint}\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f'Validation: {len(val_cases)} pazienti, {len(val_ds)} slice\n')

    # Hook pre-forward su ogni InstanceNorm2d: cattura l'input (l'output
    # delle convoluzioni ATTUALI, prima della normalizzazione), calcola
    # le statistiche "live" e le confronta con quelle congelate nel modulo.
    results = {}

    def make_hook(name, module):
        def hook(module, inputs):
            x = inputs[0]
            with torch.no_grad():
                live_mean = x.mean(dim=[0, 2, 3])
                live_var = x.var(dim=[0, 2, 3], unbiased=False)
                frozen_mean = module.running_mean
                frozen_var = module.running_var

                rel_diff_mean = ((live_mean - frozen_mean).abs() / (frozen_mean.abs() + 1e-3)).mean().item()
                rel_diff_var = ((live_var - frozen_var).abs() / (frozen_var.abs() + 1e-3)).mean().item()
                results.setdefault(name, []).append((rel_diff_mean, rel_diff_var))
        return hook

    handles = [m.register_forward_pre_hook(make_hook(name, m))
               for name, m in model.named_modules() if isinstance(m, nn.InstanceNorm2d)]

    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs = imgs.to(device)
            _ = model(imgs)  # eval mode: usa le statistiche congelate nel forward reale

    for h in handles:
        h.remove()

    print(f'{"Layer InstanceNorm":20s} {"diff. media mean":>18s} {"diff. media var":>18s}')
    print('-' * 60)
    max_diff = 0.0
    for name, vals in results.items():
        mean_diffs = [v[0] for v in vals]
        var_diffs = [v[1] for v in vals]
        avg_mean_diff = sum(mean_diffs) / len(mean_diffs) * 100
        avg_var_diff = sum(var_diffs) / len(var_diffs) * 100
        max_diff = max(max_diff, avg_mean_diff, avg_var_diff)
        flag = "  <-- ALTA" if max(avg_mean_diff, avg_var_diff) > 50 else ""
        print(f'{name:20s} {avg_mean_diff:17.1f}% {avg_var_diff:17.1f}%{flag}')

    print(f'\n=== Interpretazione ===')
    if max_diff < 20:
        print(f'Differenza massima osservata: {max_diff:.1f}% -- BASSA.')
        print('Le statistiche congelate sembrano ancora ben allineate ai pesi attuali.')
        print('Il disallineamento IN/pesi probabilmente NON e\' la causa principale.')
    elif max_diff < 100:
        print(f'Differenza massima osservata: {max_diff:.1f}% -- MODERATA.')
        print('Le statistiche congelate si sono disallineate in modo non trascurabile.')
        print('Probabile contributo al problema, ma forse non l\'unica causa.')
    else:
        print(f'Differenza massima osservata: {max_diff:.1f}% -- ALTA.')
        print('Le statistiche congelate sono SIGNIFICATIVAMENTE disallineate rispetto')
        print('ai pesi conv attuali. Questo conferma l\'ipotesi: allenare i pesi conv')
        print('mentre IN resta fissa (come richiesto dal metodo a fasi) puo\' causare')
        print('un disallineamento che spiega perche\' il problema si vede SOLO in eval')
        print('mode (validazione), non in training (dove IN si adatta comunque al volo).')


if __name__ == '__main__':
    main()