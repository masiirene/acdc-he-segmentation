"""
crypto/calibrate_clamp_mean_only.py

Ricalibra le soglie di clamp per il modello con MeanOnlyNorm2d al posto
di InstanceNorm2d -- le vecchie soglie (calibrate su un regime a
varianza unitaria) sono troppo basse per la scala reale delle
attivazioni senza divisione per la deviazione standard (vedi
diagnose_mean_only_scale_v2.py: fino a 12x oltre soglia).

Stessa metodologia di calibrate_clamp_threshold.py: percentili della
distribuzione REALE osservata sul validation set, con un margine di
sicurezza, sul checkpoint SORGENTE (prima di qualunque fine-tuning con
mean-only) -- cosi' la nuova soglia riflette la vera scala del problema,
non un tentativo di training gia' parzialmente divergente.
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from crypto.replace_with_mean_only_norm import replace_instancenorm_with_meanonly
from training.dataset import ACDCDataset, load_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--filters', type=int, nargs=6, required=True)
    parser.add_argument('--skip_mode', default='sum')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--old_clamp_json', default='crypto/calibrated_clamp_values.json',
                        help='Serve solo per costruire il modello con la forma giusta, i valori verranno sovrascritti')
    parser.add_argument('--percentile', type=float, default=99.9)
    parser.add_argument('--margin', type=float, default=1.5,
                        help='Moltiplicatore di sicurezza sopra il percentile osservato')
    parser.add_argument('--out', default='crypto/calibrated_clamp_values_mean_only.json')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    with open(args.old_clamp_json) as f:
        old_clamp = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values={k: float('inf') for k in old_clamp},  # clamp disattivato per raccogliere i veri valori
        norm_mode='per_instance', skip_mode=args.skip_mode,
        weight_standardization=False, filters=args.filters,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    replace_instancenorm_with_meanonly(model)
    model.eval()
    print(f'Checkpoint caricato: {args.checkpoint}, MeanOnlyNorm2d attiva, clamp DISATTIVATO per la raccolta')

    raw_values = {}
    def make_hook(name):
        def hook(mod, inp, out):
            raw_values.setdefault(name, []).append(mod.last_raw.detach().abs().flatten().cpu())
        return hook

    for name, m in model.named_modules():
        if type(m).__name__ == 'PolyAct':
            m.register_forward_hook(make_hook(name))

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    print('Raccolgo la distribuzione reale (clamp disattivato) su tutto il validation set...\n')
    with torch.no_grad():
        for imgs, _ in val_loader:
            model(imgs.to(device))

    print(f"{'Layer':16s} {'vecchia soglia':>14s} {f'p{args.percentile}':>12s} {'nuova soglia':>14s}")
    new_clamp = {}
    for name in sorted(raw_values.keys()):
        vals = torch.cat(raw_values[name])
        import numpy as np
        p = float(np.percentile(vals.numpy(), args.percentile))
        new_threshold = p * args.margin
        new_clamp[name] = new_threshold
        old = old_clamp.get(name, float('nan'))
        print(f"{name:16s} {old:14.2f} {p:12.2f} {new_threshold:14.2f}")

    with open(args.out, 'w') as f:
        json.dump(new_clamp, f, indent=2)
    print(f"\nSalvato in: {args.out}")


if __name__ == '__main__':
    main()