"""
crypto/evaluate_on_test_set.py

Valuta un checkpoint sul vero test set ufficiale ACDC (patient101-150),
COMPLETAMENTE separato dai 5 fold di train/val usati per tutto lo
sviluppo/tuning fatto finora.

REGOLA D'USO, decisa esplicitamente per questo progetto:
- Va bene controllarlo ORA, una volta, su un checkpoint gia' "chiuso"
  (una configurazione che non verra' piu' modificata sulla base di questo
  risultato) -- solo per verifica TECNICA (nessun NaN, pipeline funziona,
  numeri nell'ordine di grandezza plausibile).
- NON va usato per scegliere tra configurazioni diverse (es. quale
  normalizzazione, quale dimensione di rete) -- quello resta compito del
  validation set (fold 0), come fatto finora.
- Il numero VERO da riportare in tesi come valutazione finale va
  ricalcolato una volta sola, a fine progetto, sulla configurazione
  DEFINITIVA -- non su questo primo controllo.

Ogni esecuzione stampa un avviso esplicito e chiede conferma prima di
procedere, per evitare che diventi un'abitudine invece di un'eccezione
documentata.

USO:
    python3 -m crypto.evaluate_on_test_set \\
        --checkpoint <path> \\
        --norm_mode per_instance \\
        --test_cases_json crypto/test_cases.json \\
        --testing_dir ~/Desktop/tesi_acdc/testing \\
        --note "verifica tecnica sul checkpoint vincente 0.865, non usato per decisioni"
"""

import os
import sys
import json
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet
from training.dataset import ACDCDataset


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
    parser.add_argument('--testing_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/testing'))
    parser.add_argument('--test_cases_json', default='crypto/test_cases.json')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--act', default='poly', choices=['identity', 'linear', 'squared', 'poly'])
    parser.add_argument('--norm', default='instance', choices=['none', 'batch', 'instance', 'group', 'poly'])
    parser.add_argument('--norm_mode', default='population', choices=['population', 'per_instance'])
    parser.add_argument('--note', required=True,
                        help='Nota obbligatoria: perche\' stai facendo questa valutazione ORA. '
                             'Verra\' salvata nel log insieme al risultato, per tracciabilita\'.')
    parser.add_argument('--log_file', default='crypto/test_set_evaluation_log.json',
                        help='Ogni esecuzione viene appesa qui, cosi\' resta uno storico di quando '
                             'e perche\' il test set e\' stato controllato.')
    parser.add_argument('--skip_confirmation', action='store_true',
                        help='Salta la richiesta di conferma interattiva (utile in script automatici, '
                             'ma usare con cautela).')
    args = parser.parse_args()

    print('=' * 70)
    print('ATTENZIONE: stai per valutare sul VERO TEST SET (patient101-150)')
    print('=' * 70)
    print('Regola concordata: questo NON deve essere usato per scegliere tra')
    print('configurazioni diverse. Va bene solo per una verifica tecnica una')
    print('tantum, o per la valutazione FINALE a fine progetto.')
    print(f'\nNota fornita: "{args.note}"')
    print()

    if not args.skip_confirmation:
        risposta = input('Confermi di voler procedere? Scrivi "si" per continuare: ')
        if risposta.strip().lower() not in ('si', 'sì', 'yes', 'y'):
            print('Annullato.')
            return

    with open(args.test_cases_json) as f:
        test_cases = json.load(f)
    print(f'\nCasi di test caricati: {len(test_cases)}')

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type=args.act,
                           norm_type=args.norm, norm_mode=args.norm_mode).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  ({len(unexpected)} chiavi ignorate, atteso se norm_mode differisce dal checkpoint)')
    model.eval()

    test_ds = ACDCDataset(args.testing_dir, test_cases, patch_size=(256, 224), augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'Slice di test: {len(test_ds)}\n')

    dice_rv, dice_myo, dice_lv = [], [], []
    n_batches_total = 0
    n_batches_nan_inf = 0

    with torch.no_grad():
        for imgs, segs in test_loader:
            imgs = imgs.to(device)
            segs = segs.to(device)
            logits = model(imgs)
            n_batches_total += 1

            if not torch.isfinite(logits).all():
                n_batches_nan_inf += 1
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

    print('=== RISULTATO SUL TEST SET UFFICIALE ===')
    print(f'RV={rv:.3f}  MYO={myo:.3f}  LV={lv:.3f}  MEAN={mean_dice:.3f}')
    print(f'Batch con NaN/Inf: {n_batches_nan_inf}/{n_batches_total}')

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'config': {'act': args.act, 'norm': args.norm, 'norm_mode': args.norm_mode},
        'note': args.note,
        'n_test_cases': len(test_cases),
        'dice_rv': rv, 'dice_myo': myo, 'dice_lv': lv, 'mean_dice': mean_dice,
        'n_batches_nan_inf': n_batches_nan_inf,
        'n_batches_total': n_batches_total,
    }

    log_history = []
    if os.path.exists(args.log_file):
        with open(args.log_file) as f:
            log_history = json.load(f)
    log_history.append(log_entry)
    with open(args.log_file, 'w') as f:
        json.dump(log_history, f, indent=2)

    print(f'\nQuesta valutazione e\' stata registrata in: {args.log_file}')
    print(f'(Valutazioni totali del test set finora: {len(log_history)} -- se questo numero')
    print(f' cresce troppo velocemente, e\' un segnale che il test set viene usato troppo spesso)')


if __name__ == '__main__':
    main()