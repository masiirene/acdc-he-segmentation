"""
crypto/ablate_dec0_width.py

Domanda: quanto si puo' restringere la LARGHEZZA (numero di canali) di uno
stage del decoder senza perdere troppo Dice? A differenza di
ablate_decoder_stages.py (bypass completo -> Dice crolla a ~0.01, lo
stage NON e' removibile), qui testiamo una riduzione GRADUALE, non un
azzeramento totale.

Nessun retraining: non possiamo cambiare la shape dei pesi senza riallenare,
ma possiamo simulare una rete piu' stretta SPEGNENDO (mascherando a zero)
i canali meno importanti dell'output di uno stage, prima che raggiungano
out_conv. E' un proxy di magnitude/activation pruning, non un vero
retraining a larghezza ridotta -- ma da' un segnale rapido e informativo
su quali canali contano davvero, per decidere se vale la pena investire
in un retraining vero a larghezza ridotta.

Importanza di un canale c dello stage scelto, stimata come:
    score_c = ||out_conv.weight[:, c]||_2 * mean_abs_activation_c
dove mean_abs_activation_c e' calcolata su un giro di calibrazione sul
validation set. Combina quanto quel canale PESA sull'output (struttura)
con quanto si ATTIVA realmente sui dati (uso effettivo) -- un canale con
peso alto ma sempre vicino a zero, o viceversa, conta comunque poco.

NOTA: questa stima e' valida solo per lo stage IMMEDIATAMENTE PRIMA di
out_conv (cioe' dec0, l'unico il cui output alimenta out_conv senza altri
layer intermedi). Per stage piu' a monte (dec1..dec4) l'importanza
strutturale andrebbe propagata attraverso i layer successivi -- non
implementato qui, ma lo stesso principio si estenderebbe con la catena
delle jacobiane, se servisse in futuro.

USO:
    python3 crypto/ablate_dec0_width.py \
        --checkpoint results/.../best_model.pth \
        --norm_mode per_instance --weight_standardization --legacy_ws
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
    """Vedi crypto/ablate_decoder_stages.py per la motivazione completa:
    ricostruisce la vecchia WSConv2d (gain libero, pre-fix gain_floor)
    per checkpoint allenati prima del fix."""
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


def compute_channel_importance(model, stage_module, val_loader, device, n_calib_batches=10):
    """Stima l'importanza di ogni canale dell'output di stage_module,
    combinando la norma dei pesi di out_conv per quel canale con
    l'attivazione media assoluta osservata su un giro di calibrazione."""
    activations = []

    def hook(module, inputs, output):
        activations.append(output.detach().abs().mean(dim=[0, 2, 3]))  # shape (C,)

    handle = stage_module.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for i, (imgs, segs) in enumerate(val_loader):
            if i >= n_calib_batches:
                break
            imgs = imgs.to(device)
            model(imgs)
    handle.remove()

    mean_abs_activation = torch.stack(activations).mean(dim=0)  # (C,)

    # Peso strutturale: norma L2 dei pesi di out_conv per canale in ingresso.
    # out_conv.weight ha forma (num_classes, C, kH, kW) -- per dec0, kH=kW=1.
    out_conv_weight = model.out_conv.weight.detach()  # (num_classes, C, kH, kW)
    weight_norm_per_channel = out_conv_weight.reshape(out_conv_weight.shape[0], out_conv_weight.shape[1], -1) \
        .norm(dim=(0, 2))  # (C,)

    score = weight_norm_per_channel.cpu() * mean_abs_activation.cpu()
    return score


def evaluate_with_mask(model, stage_module, mask, val_loader, device):
    """Valuta il modello mascherando a zero i canali NON inclusi in mask
    (tensore booleano di lunghezza C) all'uscita di stage_module."""
    def hook(module, inputs, output):
        return output * mask.to(output.device).view(1, -1, 1, 1)

    handle = stage_module.register_forward_hook(hook)
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
    handle.remove()
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
    parser.add_argument('--legacy_ws', action='store_true')
    parser.add_argument('--fractions', type=float, nargs='+',
                        default=[1.0, 0.75, 0.5, 0.25, 0.125])
    parser.add_argument('--clamp_values_json', default=None,
                        help='Path alle soglie di clamp calibrate per layer, LO STESSO file '
                         'usato in training (es. crypto/calibrated_clamp_values.json). '
                         'Se omesso, ogni PolyAct usa il default (50.0), che NON coincide '
                         'con come il checkpoint e\' stato allenato -- falsa la valutazione.')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    if args.legacy_ws:
        hf.WSConv2d = LegacyWSConv2d

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
    ).to(device)
    

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f'\u26a0\ufe0f  {len(unexpected)} chiavi inattese (verifica --legacy_ws/--weight_standardization): '
              f'{unexpected[:3]}')
    model.eval()

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    stage = model.dec0
    n_channels = model.out_conv.weight.shape[1]

    print('Calcolo importanza per canale (calibrazione su 10 batch di validazione)...')
    score = compute_channel_importance(model, stage, val_loader, device)
    ranking = torch.argsort(score, descending=True)

    print(f'\nStage: dec0 ({n_channels} canali totali)')
    print(f'{"Frazione canali":>16s} {"# canali attivi":>16s} {"Dice mean":>10s} '
          f'{"RV":>7s} {"MYO":>7s} {"LV":>7s} {"NaN batch":>10s}')

    for frac in sorted(args.fractions, reverse=True):
        k = max(1, round(n_channels * frac))
        mask = torch.zeros(n_channels)
        mask[ranking[:k]] = 1.0
        result = evaluate_with_mask(model, stage, mask, val_loader, device)
        if result is None:
            print(f'{frac:16.3f} {k:16d}  TUTTI I BATCH ESPLOSI IN NaN/Inf')
        else:
            print(f'{frac:16.3f} {k:16d} {result["mean"]:10.3f} {result["rv"]:7.3f} '
                  f'{result["myo"]:7.3f} {result["lv"]:7.3f} {result["n_nan_batches"]:10d}')


if __name__ == '__main__':
    main()