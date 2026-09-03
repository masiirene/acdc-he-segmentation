"""
crypto/check_inference_stability.py

Domanda: il modello GIA' ALLENATO (con clamp usato solo durante il
training) ha ancora bisogno del clamp in INFERENZA pura (eval mode,
pesi fissi, singolo forward pass -- esattamente lo scenario HE)?

Se la risposta e' "quasi mai", il clamp e' stato un attrezzo necessario
SOLO per ottenere pesi stabili in training, e in inferenza il modello si
comporta gia' bene da solo -- nessun bisogno di implementare un clamp in
CKKS. Se invece interviene spesso anche in eval, e' un problema reale da
affrontare (vedi note in fondo).

Nessun retraining richiesto: solo forward pass su checkpoint esistente.

USO:
    python3 crypto/check_inference_stability.py --checkpoint <path>
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet, PolyAct
from training.dataset import ACDCDataset, load_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_value', type=float, default=50.0,
                        help='soglia di clamp usata in training, per confronto')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly', norm_type='instance').to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()  # <-- cruciale: esattamente lo scenario HE (statistiche IN fisse, no grad)
    print(f'Checkpoint: {args.checkpoint}')

    # --- Disattiva TEMPORANEAMENTE il clamp interno di PolyAct ---
    # Il clamp e' sempre attivo nel forward normale (anche in eval), quindi
    # senza questo passaggio misureremmo sempre valori <= clamp_value per
    # costruzione. Alziamo la soglia a un valore enorme per vedere il vero
    # comportamento "naturale" dei pesi allenati, poi la ripristiniamo.
    original_clamp_values = {}
    for name, m in model.named_modules():
        if isinstance(m, PolyAct):
            original_clamp_values[name] = m.clamp_value
            m.clamp_value = float('inf')  # nessun clamp: vediamo il comportamento vero
    print('Clamp interno DISATTIVATO per la misura (nessun limite)\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f'Validation: {len(val_cases)} pazienti, {len(val_ds)} slice\n')

    # Hook su ogni PolyAct: registra il valore massimo assoluto PRIMA di
    # un eventuale clamp (che qui non applichiamo -- vogliamo vedere il
    # comportamento "naturale" del modello allenato).
    layer_max_values = {name: [] for name, m in model.named_modules() if isinstance(m, PolyAct)}

    def make_hook(name):
        def hook(module, inp, out):
            layer_max_values[name].append(out.detach().abs().max().item())
        return hook

    handles = [m.register_forward_hook(make_hook(name))
               for name, m in model.named_modules() if isinstance(m, PolyAct)]

    n_batches_over_threshold = 0
    n_batches_total = 0
    global_max = 0.0

    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            n_batches_total += 1
            batch_max = max(v[-1] for v in layer_max_values.values())
            global_max = max(global_max, batch_max)
            if batch_max > args.clamp_value:
                n_batches_over_threshold += 1

    for h in handles:
        h.remove()

    # Ripristina il clamp originale
    for name, m in model.named_modules():
        if isinstance(m, PolyAct):
            m.clamp_value = original_clamp_values[name]

    print('=== Valore massimo assoluto per layer PolyAct (su tutto il validation set) ===')
    for name, values in layer_max_values.items():
        print(f'  {name:20s}  max={max(values):10.3f}  mean_per_batch_max={sum(values)/len(values):8.3f}')

    print(f'\nValore massimo assoluto GLOBALE osservato: {global_max:.3f}')
    print(f'Soglia di clamp usata in training: {args.clamp_value}')
    print(f'Batch che avrebbero attivato il clamp (>{args.clamp_value}): '
          f'{n_batches_over_threshold}/{n_batches_total} '
          f'({100*n_batches_over_threshold/n_batches_total:.1f}%)')

    print('\n=== Interpretazione ===')
    if n_batches_over_threshold == 0:
        print('Il modello NON supera mai la soglia di clamp in eval mode pura.')
        print('-> In inferenza (scenario HE) il clamp non sembra necessario: i pesi')
        print('   allenati si comportano gia in modo stabile da soli su dati reali.')
        print('-> Non serve implementare un\'operazione di clamp in CKKS per QUESTO')
        print('   checkpoint, sui dati di validazione testati.')
    elif n_batches_over_threshold / n_batches_total < 0.05:
        print('Il clamp interviene raramente (<5% dei batch) in eval mode.')
        print('-> Rischio residuo basso ma non nullo: da monitorare, non bloccante.')
    else:
        print('Il clamp interviene frequentemente anche in eval mode pura.')
        print('-> Problema reale per HE: servira\' una strategia (regolarizzazione')
        print('   piu\' forte in training, o approssimazione polinomiale del clamp')
        print('   in CKKS) -- da discutere con Aurora, ma NON blocca il lavoro sul')
        print('   packing (crypto/packing.py), che resta valido indipendentemente.')


if __name__ == '__main__':
    main()