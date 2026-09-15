"""
crypto/compute_dice_by_config.py

Completa la tabella richiesta da Aurora: Dice della rete NON approssimata
(quella da cui si calcola la loss) incrociato con statistiche live/statiche
e clamp attivo/disattivato.

Il dato mancante finora: sappiamo che con statistiche per-istanza e clamp
disattivato il modello non produce NaN/Inf (0/46 batch), ma non avevamo
mai calcolato il Dice effettivo in quella configurazione -- gli script
precedenti (check_inference_stability.py) verificavano solo l'assenza di
esplosioni, non la qualita' della segmentazione risultante.

USO:
    python3 -m crypto.compute_dice_by_config \\
        --checkpoint <path> \\
        --norm_mode per_instance \\
        --clamp_mode disabled
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet, PolyAct
from training.dataset import ACDCDataset, load_splits


def dice_score(pred, target, num_classes=4):
    scores = {}
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        score = (2 * intersection + 1e-5) / (p.sum() + t.sum() + 1e-5)
        scores[c] = score.item()
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--norm_mode', default='population', choices=['population', 'per_instance'],
                        help='DEVE corrispondere al norm_mode con cui e\' stato allenato il checkpoint.')
    parser.add_argument('--clamp_mode', default='calibrated', choices=['calibrated', 'disabled', 'fixed50'],
                        help="'calibrated': usa --clamp_values_json (le soglie per-layer usate in "
                             "training, se fornito). 'disabled': clamp_value=inf su ogni PolyAct -- "
                             "la rete NON approssimata, quella da cui si calcolerebbe la loss senza "
                             "alcun taglio. 'fixed50': clamp_value=50.0 uniforme su tutti i layer "
                             "(il comportamento originale, prima di qualunque calibrazione).")
    parser.add_argument('--clamp_values_json', default=None,
                        help='Richiesto se --clamp_mode calibrated. Path al JSON con le soglie '
                             'per-layer usate in training (es. crypto/calibrated_clamp_values.json).')
    args = parser.parse_args()

    if args.clamp_mode == 'calibrated' and not args.clamp_values_json:
        raise ValueError('--clamp_mode calibrated richiede --clamp_values_json')

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')
    print(f'norm_mode: {args.norm_mode}   clamp_mode: {args.clamp_mode}\n')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly',
                           norm_type='instance', norm_mode=args.norm_mode).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  ({len(unexpected)} chiavi del checkpoint ignorate: buffer di popolazione o '
          f'incompatibilita\' di norm_mode, atteso se checkpoint e modello non coincidono)')
    model.eval()

    # Applica la configurazione di clamp richiesta
    if args.clamp_mode == 'disabled':
        for name, m in model.named_modules():
            if isinstance(m, PolyAct):
                m.clamp_value = float('inf')
        print('Clamp DISATTIVATO su ogni PolyAct (rete non approssimata).\n')
    elif args.clamp_mode == 'fixed50':
        for name, m in model.named_modules():
            if isinstance(m, PolyAct):
                m.clamp_value = 50.0
        print('Clamp fissato uniformemente a 50.0 su ogni PolyAct.\n')
    else:  # calibrated
        import json
        with open(args.clamp_values_json) as f:
            clamp_values = json.load(f)
        n_set = 0
        for name, m in model.named_modules():
            if isinstance(m, PolyAct) and name in clamp_values:
                m.clamp_value = clamp_values[name]
                n_set += 1
        print(f'Soglie calibrate applicate: {n_set} layer.\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'Validation: {len(val_cases)} pazienti, {len(val_ds)} slice\n')

    dice_rv, dice_myo, dice_lv = [], [], []
    n_batches_total = 0
    n_batches_nan_inf = 0

    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs = imgs.to(device)
            segs = segs.to(device)
            logits = model(imgs)
            n_batches_total += 1

            if not torch.isfinite(logits).all():
                n_batches_nan_inf += 1
                # Sostituiamo NaN/Inf per poter comunque calcolare un Dice
                # (altrimenti argmax su NaN da' risultati indefiniti) --
                # segnaliamo comunque il conteggio separatamente.
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

            preds = logits.argmax(dim=1)
            scores = dice_score(preds, segs)
            dice_rv.append(scores[1])
            dice_myo.append(scores[2])
            dice_lv.append(scores[3])

    rv = sum(dice_rv) / len(dice_rv)
    myo = sum(dice_myo) / len(dice_myo)
    lv = sum(dice_lv) / len(dice_lv)
    mean_dice = (rv + myo + lv) / 3

    print('=== RISULTATO ===')
    print(f'RV={rv:.3f}  MYO={myo:.3f}  LV={lv:.3f}  MEAN={mean_dice:.3f}')
    print(f'Batch con NaN/Inf nei logits: {n_batches_nan_inf}/{n_batches_total}')
    if n_batches_nan_inf > 0:
        print('ATTENZIONE: il Dice sopra include batch dove i NaN/Inf sono stati sostituiti')
        print('con un valore grande arbitrario per poter comunque calcolare un argmax -- non')
        print('e\' un Dice "pulito" in quei casi, il modello sta comunque fallendo li\'.')


if __name__ == '__main__':
    main()