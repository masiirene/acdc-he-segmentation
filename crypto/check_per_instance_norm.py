"""
crypto/check_per_instance_norm.py

Testa una nuova ipotesi: il problema del clamp non e' (solo) il
disallineamento delle statistiche IN nel tempo, ma il fatto che in
inferenza usiamo statistiche di POPOLAZIONE fisse (running_mean/var),
mentre InstanceNorm dovrebbe -- per design -- normalizzare ogni immagine
con le SUE proprie statistiche (quello che fa gia' in training).

Se un paziente ha caratteristiche diverse dalla media di popolazione, la
normalizzazione con statistiche fisse lo "tratta male" indipendentemente
da quanto siano aggiornate -- spiegherebbe perche' la ricalibrazione
periodica non ha risolto il problema.

Questo script NON richiede retraining: prende un checkpoint esistente e,
solo per il forward di validazione, forza ogni InstanceNorm2d a calcolare
le statistiche al volo per-istanza (esattamente il comportamento di
default di nn.InstanceNorm2d quando track_running_stats=False) invece di
usare quelle congelate nel checkpoint.

Nota su HE: questo e' naturalmente compatibile -- in inferenza reale il
server riceve un paziente alla volta (un ciphertext), quindi calcolare
media/varianza SU QUELLA SPECIFICA immagine cifrata e' un'operazione
lineare (somma + una moltiplicazione per il quadrato), fattibile con
EvalSumRows/Cols come gia' menzionato da Aurora. Non serve alcuna media
di popolazione precalcolata.

USO:
    python3 crypto/check_per_instance_norm.py --checkpoint <path>
"""

import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
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
        scores[c] = ((2 * intersection + 1e-5) / (p.sum() + t.sum() + 1e-5)).item()
    return scores


def force_per_instance_norm(model, device):
    """
    Ricostruisce ogni InstanceNorm2d del modello con track_running_stats=False,
    copiando i coefficienti gamma/beta appresi ma SCARTANDO le statistiche
    congelate. Con track_running_stats=False, PyTorch calcola SEMPRE le
    statistiche al volo per ogni singola istanza (paziente/slice), sia in
    train che in eval mode -- comportamento deterministico, indipendente
    dalla composizione del batch (ogni sample e' normalizzato solo con le
    proprie statistiche spaziali, mai mescolato con altri sample).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.InstanceNorm2d):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            new_norm = nn.InstanceNorm2d(
                module.num_features, affine=True, track_running_stats=False
            ).to(device)
            with torch.no_grad():
                new_norm.weight.copy_(module.weight)
                new_norm.bias.copy_(module.bias)
            setattr(parent, child_name, new_norm)
    return model


def evaluate(model, loader, device, disable_clamp=False):
    model.eval()

    original_clamps = {}
    if disable_clamp:
        for name, m in model.named_modules():
            if isinstance(m, PolyAct):
                original_clamps[name] = m.clamp_value
                m.clamp_value = float('inf')

    dice_rv, dice_myo, dice_lv = [], [], []
    n_nan_inf_batches = 0
    n_total = 0

    with torch.no_grad():
        for imgs, segs in loader:
            imgs = imgs.to(device)
            segs = segs.to(device)
            logits = model(imgs)
            n_total += 1
            if not torch.isfinite(logits).all():
                n_nan_inf_batches += 1
                continue
            preds = logits.argmax(dim=1)
            scores = dice_score(preds, segs)
            dice_rv.append(scores[1])
            dice_myo.append(scores[2])
            dice_lv.append(scores[3])

    if disable_clamp:
        for name, m in model.named_modules():
            if isinstance(m, PolyAct):
                m.clamp_value = original_clamps[name]

    if len(dice_rv) == 0:
        return None, n_nan_inf_batches, n_total

    rv = sum(dice_rv) / len(dice_rv)
    myo = sum(dice_myo) / len(dice_myo)
    lv = sum(dice_lv) / len(dice_lv)
    mean_dice = (rv + myo + lv) / 3
    return {'rv': rv, 'myo': myo, 'lv': lv, 'mean': mean_dice}, n_nan_inf_batches, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}\n')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly', norm_type='instance').to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    print(f'Checkpoint: {args.checkpoint}\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f'Validation: {len(val_cases)} pazienti, {len(val_ds)} slice\n')

    print('=' * 70)
    print('CONFRONTO: statistiche di popolazione (originale) vs per-istanza')
    print('=' * 70)

    # --- Caso 1: modello originale, statistiche di popolazione congelate ---
    model_pop = copy.deepcopy(model)
    res_pop, nan_pop, total_pop = evaluate(model_pop, val_loader, device, disable_clamp=False)
    print(f'\n[POPOLAZIONE] (comportamento attuale)')
    if res_pop:
        print(f'  Dice: RV={res_pop["rv"]:.3f} MYO={res_pop["myo"]:.3f} LV={res_pop["lv"]:.3f} Mean={res_pop["mean"]:.3f}')
    print(f'  Batch con NaN/Inf: {nan_pop}/{total_pop}')

    # Stesso modello, ma con clamp disattivato -- vediamo se esplode (come gia' sappiamo)
    model_pop2 = copy.deepcopy(model)
    res_pop2, nan_pop2, total_pop2 = evaluate(model_pop2, val_loader, device, disable_clamp=True)
    print(f'  [senza clamp] Batch con NaN/Inf: {nan_pop2}/{total_pop2}')

    # --- Caso 2: statistiche PER-ISTANZA (nuova ipotesi) ---
    model_inst = copy.deepcopy(model)
    model_inst = force_per_instance_norm(model_inst, device)
    res_inst, nan_inst, total_inst = evaluate(model_inst, val_loader, device, disable_clamp=False)
    print(f'\n[PER-ISTANZA] (nuova ipotesi, con clamp comunque attivo)')
    if res_inst:
        print(f'  Dice: RV={res_inst["rv"]:.3f} MYO={res_inst["myo"]:.3f} LV={res_inst["lv"]:.3f} Mean={res_inst["mean"]:.3f}')
    print(f'  Batch con NaN/Inf: {nan_inst}/{total_inst}')

    # Il test decisivo: per-istanza E clamp disattivato
    model_inst2 = copy.deepcopy(model)
    model_inst2 = force_per_instance_norm(model_inst2, device)
    res_inst2, nan_inst2, total_inst2 = evaluate(model_inst2, val_loader, device, disable_clamp=True)
    print(f'\n[PER-ISTANZA + clamp DISATTIVATO] -- il test decisivo')
    if res_inst2:
        print(f'  Dice: RV={res_inst2["rv"]:.3f} MYO={res_inst2["myo"]:.3f} LV={res_inst2["lv"]:.3f} Mean={res_inst2["mean"]:.3f}')
    print(f'  Batch con NaN/Inf: {nan_inst2}/{total_inst2}')

    print('\n' + '=' * 70)
    print('INTERPRETAZIONE')
    print('=' * 70)
    if nan_inst2 == 0:
        print('Con statistiche PER-ISTANZA, il modello NON esplode nemmeno senza clamp!')
        print('-> Ipotesi CONFERMATA: il problema era usare una media di popolazione')
        print('   fissa invece delle statistiche naturali per-istanza di InstanceNorm.')
        print('-> Il clamp su PolyAct potrebbe non servire piu\', o servire molto meno,')
        print('   se si passa a normalizzazione per-istanza anche in inferenza HE.')
    elif nan_inst2 < nan_pop2:
        print(f'Miglioramento parziale: {nan_pop2}->{nan_inst2} batch con NaN/Inf.')
        print('La normalizzazione per-istanza aiuta ma non risolve del tutto.')
    else:
        print('Nessun miglioramento significativo. L\'ipotesi non e\' confermata,')
        print('la causa del problema e\' probabilmente altrove.')


if __name__ == '__main__':
    main()