"""
crypto/ablate_decoder_stages.py (aggiornato)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits
from training.train import dice_score


class LegacyWSConv2d(nn.Conv2d):
    """
    Ricostruisce ESATTAMENTE il comportamento di WSConv2d PRIMA del fix
    gain_floor (gain libero, nessun pavimento) -- necessario per valutare
    correttamente checkpoint allenati prima del fix (es. test_ws_trial50_v2),
    la cui chiave di stato si chiama 'gain', non 'raw_gain'. Caricare quei
    pesi nella WSConv2d attuale li scarterebbe silenziosamente (strict=False),
    lasciando il gain a zero per ogni canale e falsando qualunque valutazione.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = nn.Parameter(torch.ones(self.out_channels))

    def forward(self, x):
        w = self.weight
        out_ch = w.shape[0]
        w_flat = w.reshape(out_ch, -1)
        mean = w_flat.mean(dim=1, keepdim=True)
        std = w_flat.std(dim=1, keepdim=True, unbiased=False)
        w_std = (w_flat - mean) / (std + 1e-5)
        w_std = w_std * self.gain.view(-1, 1)
        w_std = w_std.reshape(w.shape)
        return nn.functional.conv2d(x, w_std, self.bias, self.stride,
                                     self.padding, self.dilation, self.groups)


def evaluate(model, val_loader, device, bypass_set):
    model.decoder_bypass = bypass_set
    model.eval()
    dice_rv, dice_myo, dice_lv = [], [], []
    n_nan_batches = 0
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            logits = model(imgs)
            if not torch.isfinite(logits).all():
                n_nan_batches += 1
                continue
            preds = logits.argmax(dim=1)
            scores = dice_score(preds, segs)
            dice_rv.append(scores[1]); dice_myo.append(scores[2]); dice_lv.append(scores[3])
    model.decoder_bypass = set()
    if not dice_rv:
        return None
    rv, myo, lv = sum(dice_rv)/len(dice_rv), sum(dice_myo)/len(dice_myo), sum(dice_lv)/len(dice_lv)
    return {'rv': rv, 'myo': myo, 'lv': lv, 'mean': (rv+myo+lv)/3, 'n_nan_batches': n_nan_batches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--norm_mode', default='per_instance', choices=['population', 'per_instance'])
    parser.add_argument('--skip_mode', default='concat', choices=['concat', 'sum'])
    parser.add_argument('--weight_standardization', action='store_true')
    parser.add_argument('--legacy_ws', action='store_true',
                        help='Usa la vecchia WSConv2d (gain libero, pre-fix gain_floor) -- '
                             'necessario per checkpoint allenati PRIMA del fix, come '
                             'test_ws_trial50_v2. Senza questo flag, la chiave "gain" del '
                             'checkpoint viene scartata silenziosamente e il modello gira '
                             'con gain sbagliato.')
    parser.add_argument('--clamp_values_json', default=None,
                        help='Path alle soglie di clamp calibrate per layer, LO STESSO file '
                         'usato in training (es. crypto/calibrated_clamp_values.json). '
                         'Se omesso, ogni PolyAct usa il default (50.0), che NON coincide '
                         'con come il checkpoint e\' stato allenato -- falsa la valutazione.')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    if args.legacy_ws:
        hf.WSConv2d = LegacyWSConv2d  # monkeypatch: ConvBlock la risolve a runtime

    clamp_values = None
    if args.clamp_values_json:
        import json
        with open(args.clamp_values_json) as f:
            clamp_values = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values,
        norm_mode=args.norm_mode, skip_mode=args.skip_mode,
        weight_standardization=args.weight_standardization,
    )

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'\u26a0\ufe0f  missing={len(missing)} unexpected={len(unexpected)} '
              f'(dovrebbero essere 0 se --legacy_ws e' 'norm_mode sono corretti)')
        if unexpected:
            print(f'   Chiavi inattese (esempio): {unexpected[:5]}')
    model.eval()

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    stage_order = ['dec0', 'dec1', 'dec2', 'dec3', 'dec4']
    configs = [set()] + [set(stage_order[:k]) for k in range(1, 6)]

    print(f'{"Bypass":40s} {"Dice mean":>10s} {"RV":>7s} {"MYO":>7s} {"LV":>7s} {"NaN batch":>10s}')
    baseline_mean = None
    for cfg in configs:
        result = evaluate(model, val_loader, device, cfg)
        label = 'nessuno (baseline)' if not cfg else '+'.join(sorted(cfg, key=stage_order.index))
        if result is None:
            print(f'{label:40s} TUTTI I BATCH ESPLOSI IN NaN/Inf')
        else:
            print(f'{label:40s} {result["mean"]:10.3f} {result["rv"]:7.3f} '
                  f'{result["myo"]:7.3f} {result["lv"]:7.3f} {result["n_nan_batches"]:10d}')
            if baseline_mean is None:
                baseline_mean = result['mean']
            elif abs(result['mean'] - baseline_mean) < 1e-6 and cfg == set(stage_order):
                print('\n\u26a0\ufe0f  ATTENZIONE: il bypass completo da\' lo STESSO identico Dice del '
                      'baseline -- probabile che models/he_friendly.py NON contenga ancora la '
                      'logica di bypass nel forward(). Verifica prima di fidarti dei numeri sopra.')


if __name__ == '__main__':
    main()