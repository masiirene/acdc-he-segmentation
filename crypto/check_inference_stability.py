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

CORREZIONE IMPORTANTE (stesso errore gia' visto in calibrate_clamp_
threshold.py): il modello va costruito con lo STESSO norm_mode del
checkpoint che si sta testando, altrimenti si valuta un modello diverso
da quello allenato. --norm_mode ora e' un argomento esplicito.

AGGIUNTA: oltre al confronto con clamp_value (soglia usata in training),
questo script ora controlla esplicitamente la presenza di NaN/Inf nei
LOGITS FINALI -- il vero criterio di fallimento usato fin dall'inizio
del progetto (vedi CONTESTO_PROGETTO_TESI: "46/46 batch esplosi in
NaN/Inf" con normalizzazione di popolazione). Superare una soglia
arbitraria non e' di per se' un fallimento; produrre NaN/Inf lo e'.

USO (esempio, per il checkpoint allenato con norm_mode=per_instance):
    python3 crypto/check_inference_stability.py \\
        --checkpoint results/.../best_model.pth \\
        --norm_mode per_instance
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
                        help='soglia di clamp usata in training, per confronto (solo informativo, '
                             'non e\' il vero criterio di fallimento -- vedi controllo NaN/Inf)')
    parser.add_argument('--norm_mode', default='population', choices=['population', 'per_instance'],
                        help="DEVE corrispondere al norm_mode con cui e\' stato allenato il "
                             "checkpoint. Default 'population' per compatibilita\' con l\'uso "
                             "originale. Se il checkpoint usa per_instance (es. tutti i run di "
                             "Fase III di oggi), va impostato di conseguenza -- altrimenti si sta "
                             "valutando un modello diverso da quello allenato.")
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')
    print(f'norm_mode: {args.norm_mode}')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly',
                           norm_type='instance', norm_mode=args.norm_mode).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # strict=False: necessario se il checkpoint contiene buffer di popolazione
    # (running_mean/var/num_batches_tracked) non presenti in un modello
    # costruito con norm_mode='per_instance', o viceversa.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  \u26a0\ufe0f  Chiavi mancanti nel checkpoint (verificare): {missing}')
    print(f'  ({len(unexpected)} chiavi del checkpoint ignorate: buffer di popolazione o '
          f'incompatibilita\' di norm_mode, atteso se i due non coincidono)')
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
    n_batches_nan_inf = 0
    global_max = 0.0

    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            n_batches_total += 1

            # Criterio di fallimento VERO: NaN/Inf nei logits finali, non
            # solo "supera una soglia arbitraria" -- coerente col criterio
            # usato fin dall'inizio del progetto.
            if not torch.isfinite(logits).all():
                n_batches_nan_inf += 1

            batch_max = max(v[-1] for v in layer_max_values.values())
            global_max = max(global_max, batch_max) if torch.isfinite(torch.tensor(batch_max)) else global_max
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
        finite_values = [v for v in values if v == v and v != float('inf')]  # esclude NaN/Inf dal max
        vmax = max(finite_values) if finite_values else float('nan')
        vmean = sum(finite_values)/len(finite_values) if finite_values else float('nan')
        print(f'  {name:20s}  max={vmax:10.3f}  mean_per_batch_max={vmean:8.3f}')

    print(f'\nValore massimo assoluto GLOBALE osservato (batch finiti): {global_max:.3f}')
    print(f'Soglia di clamp usata in training: {args.clamp_value}')
    print(f'Batch che avrebbero attivato il clamp (>{args.clamp_value}): '
          f'{n_batches_over_threshold}/{n_batches_total} '
          f'({100*n_batches_over_threshold/n_batches_total:.1f}%)')
    print(f'\nBatch con NaN/Inf nei LOGITS FINALI (vero criterio di fallimento): '
          f'{n_batches_nan_inf}/{n_batches_total} '
          f'({100*n_batches_nan_inf/n_batches_total:.1f}%)')

    print('\n=== Interpretazione ===')
    if n_batches_nan_inf == 0:
        print('NESSUNA esplosione NaN/Inf su tutto il validation set, clamp completamente')
        print('disattivato -- questo e\' il vero criterio HE (CKKS non gestisce NaN/Inf).')
        if n_batches_over_threshold == 0:
            print('Il modello NON supera mai nemmeno la soglia di clamp usata in training.')
            print('-> In inferenza (scenario HE) il clamp non sembra necessario: i pesi')
            print('   allenati si comportano gia in modo stabile da soli su dati reali.')
        else:
            print(f'Il modello supera la soglia {args.clamp_value} in alcuni batch '
                  f'({n_batches_over_threshold}/{n_batches_total}) ma senza mai produrre NaN/Inf.')
            print('-> Valori grandi ma finiti: gestibili con un clamp HE-friendly a basso costo')
            print('   se necessario, ma NON e\' un\'esplosione numerica reale.')
    else:
        print('ATTENZIONE: il modello produce NaN/Inf nei logits finali anche senza alcun')
        print('clamp interno -- questo E\' un\'esplosione numerica reale, non solo un valore')
        print('grande. Per HE (CKKS non gestisce NaN/Inf) questo e\' un problema bloccante:')
        print('il clamp resta necessario in produzione per QUESTO checkpoint, andra\'')
        print('discusso con Aurora come limitarlo o approssimarlo in modo HE-friendly.')


if __name__ == '__main__':
    main()