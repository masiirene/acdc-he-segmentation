"""
crypto/calibrate_clamp_threshold.py

Implementa la seconda parte del suggerimento di Aurora: calcolare il range
di valori atteso per ogni PolyAct osservando un modello che funziona bene,
e usare un percentile stringente di quella distribuzione come nuova soglia
di clamp -- invece del valore "indovinato" (50) usato finora.

IMPORTANTE: qui il clamp ORIGINALE resta attivo durante il forward (non lo
disattiviamo come in check_inference_stability.py). Vogliamo il valore
GREZZO (pre-clamp) di ogni PolyAct, ma calcolato in condizioni realistiche
in cui il resto della rete è comunque stabilizzata dal clamp -- altrimenti,
disattivando il clamp ovunque, i layer profondi esploderebbero e i dati
raccolti sarebbero inutilizzabili (l'abbiamo gia' visto).

CORREZIONE IMPORTANTE (dopo un errore di calibrazione incrociata): il
modello usato per la calibrazione DEVE essere costruito con lo STESSO
norm_mode (population/per_instance) e lo stesso stato di 'a' (libero vs
resettato) del training reale che si intende poi lanciare con le soglie
calibrate. In caso contrario si calibra la distribuzione di un modello e
la si applica a un modello diverso -- le soglie risultanti non hanno
alcun significato per il secondo. Per questo --norm_mode e --reset_poly_a
ora sono argomenti espliciti, di default coerenti col comportamento
originale (population, nessun reset) per non rompere l'uso precedente.

USO (esempio, per calibrare esattamente come nel training reale di Fase III
con normalizzazione per-istanza e 'a' resettato):
    python3 crypto/calibrate_clamp_threshold.py \\
        --checkpoint ~/Desktop/tesi_acdc/best_model_phase2.pth \\
        --norm_mode per_instance --reset_poly_a
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
    parser.add_argument('--checkpoint', required=True,
                        help='Un checkpoint da cui partire per osservare la distribuzione di PolyAct')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--percentile', type=float, default=99.9,
                        help='Percentile da usare per la nuova soglia. Con 95 (primo tentativo) '
                             'le soglie sono risultate troppo strette, soffocando anche segnale '
                             'legittimo (Dice sceso da 0.70 a 0.33 in 10 epoche). Default ora 99.9: '
                             'tagliamo solo gli outlier davvero estremi, non la coda normale.')
    parser.add_argument('--margin', type=float, default=1.5,
                        help='Fattore moltiplicativo di sicurezza sul percentile scelto '
                             '(default 1.5 = +50%%, piu\' permissivo del +20%% iniziale)')
    parser.add_argument('--current_threshold', type=float, default=50.0,
                        help='Soglia attuale, usata come TETTO: la nuova soglia non la supera mai')
    parser.add_argument('--output_json', default='crypto/calibrated_clamp_values.json',
                        help='Path dove salvare il dizionario calibrato')
    parser.add_argument('--norm_mode', default='population', choices=['population', 'per_instance'],
                        help="DEVE corrispondere al norm_mode del training reale che userai con le "
                             "soglie calibrate qui prodotte (vedi models/he_friendly.py get_norm()). "
                             "Default 'population' per compatibilita\' con l\'uso originale dello "
                             "script, MA se il training reale usa --norm_mode per_instance, questo "
                             "flag va impostato di conseguenza, altrimenti le soglie calibrate "
                             "riflettono la distribuzione di un modello diverso da quello allenato.")
    parser.add_argument('--reset_poly_a', action='store_true',
                        help="Se presente, forza 'a'=0.1 su ogni PolyAct dopo il caricamento del "
                             "checkpoint -- DEVE essere usato se il training reale usera\' anche "
                             "--reset_poly_a, per calibrare sulla stessa condizione di partenza "
                             "(altrimenti si calibra su 'a' gia\' derivato dal checkpoint, diverso "
                             "dal valore iniziale 0.1 con cui partira\' davvero il training).")
    parser.add_argument('--split', default='val', choices=['val', 'train'],
                        help="'val' (default): usa il validation set, senza augmentation -- stima "
                             "piu\' stabile/riproducibile. 'train': usa il training set CON "
                             "augmentation (rotazioni, flip, variazioni di intensita\'), la stessa "
                             "che vede il vero training -- cattura code della distribuzione un po\' "
                             "piu\' larghe, piu\' fedele al comportamento reale ma meno riproducibile "
                             "run-to-run (l\'augmentation e\' casuale).")
    parser.add_argument('--max_batches', type=int, default=None,
                        help="Limita il numero di batch processati (utile con --split train, che ha "
                             "~4x piu\' slice del validation set). None (default) = tutti i batch.")
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')
    print(f'norm_mode: {args.norm_mode}   reset_poly_a: {args.reset_poly_a}')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly',
                           norm_type='instance', norm_mode=args.norm_mode).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # strict=False: necessario quando norm_mode del checkpoint differisce da
    # quello del modello appena costruito (es. checkpoint population, modello
    # per_instance -> i buffer running_mean/var/num_batches_tracked non hanno
    # posto nel modello per_instance, vanno ignorati senza errore).
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  \u26a0\ufe0f  Chiavi mancanti nel checkpoint (verificare): {missing}')
    print(f'  ({len(unexpected)} chiavi del checkpoint ignorate: buffer di popolazione o '
          f'incompatibilita\' di norm_mode, atteso se i due non coincidono)')

    if args.reset_poly_a:
        n_reset = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, PolyAct) and m.max_a is None:
                    m.a.fill_(0.1)
                    n_reset += 1
        print(f'  Reset esplicito di \'a\'=0.1 su {n_reset} layer PolyAct')

    model.eval()
    print(f'Checkpoint: {args.checkpoint}  (clamp ORIGINALE lasciato attivo)\n')

    train_cases, val_cases = load_splits(args.splits_path, fold=args.fold)
    if args.split == 'train':
        # Usa il TRAINING set CON augmentation (rotazioni, flip, variazioni di
        # intensita') -- riflette meglio la vera distribuzione di input che
        # PolyAct vedra' durante il training reale, che usa sempre augment=True
        # (vedi training/train.py). Il validation set (default, augment=False)
        # da' una stima piu' stabile/riproducibile ma leggermente piu' stretta
        # delle code della distribuzione rispetto a quanto accade in training.
        cases = train_cases
        ds = ACDCDataset(args.data_dir, cases, patch_size=(256, 224), augment=True)
        print(f'Calibrazione su TRAINING set (con augmentation): {len(cases)} pazienti, {len(ds)} slice\n')
    else:
        cases = val_cases
        ds = ACDCDataset(args.data_dir, cases, patch_size=(256, 224), augment=False)
        print(f'Calibrazione su VALIDATION set (senza augmentation): {len(cases)} pazienti, {len(ds)} slice\n')
    val_loader = DataLoader(ds, batch_size=8, shuffle=(args.split == 'train'), num_workers=0)

    # Hook che intercetta il valore GREZZO (pre-clamp) ricostruendolo dai
    # parametri a,b,c del modulo e dall'input x, senza modificare il
    # comportamento reale del forward (che resta clampato normalmente).
    raw_values_by_layer = {name: [] for name, m in model.named_modules() if isinstance(m, PolyAct)}

    def make_pre_hook(name, module):
        def hook(module, inputs):
            x = inputs[0]
            with torch.no_grad():
                a_val = module.current_a()
                raw = a_val * x * x + module.b * x + module.c
                # Campioniamo un sottoinsieme di valori per non esplodere in RAM
                sample = raw.flatten()
                if sample.numel() > 5000:
                    idx = torch.randperm(sample.numel())[:5000]
                    sample = sample[idx]
                raw_values_by_layer[name].append(sample.abs().cpu())
        return hook

    handles = [m.register_forward_pre_hook(make_pre_hook(name, m))
               for name, m in model.named_modules() if isinstance(m, PolyAct)]

    with torch.no_grad():
        for i, (imgs, segs) in enumerate(val_loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            imgs = imgs.to(device)
            _ = model(imgs)  # il clamp originale e' attivo, la rete resta stabile

    for h in handles:
        h.remove()

    print(f'{"Layer":20s} {"max":>10s} {"p50":>10s} {"p95":>10s} {"p99":>10s} {"p99.9":>10s} '
          f'{"soglia attuale":>15s} {"NUOVA soglia":>14s}')
    print('-' * 110)

    suggested = {}
    for name, chunks in raw_values_by_layer.items():
        all_vals = torch.cat(chunks)
        p50 = torch.quantile(all_vals, 0.50).item()
        p95 = torch.quantile(all_vals, 0.95).item()
        p99 = torch.quantile(all_vals, 0.99).item()
        p999 = torch.quantile(all_vals, 0.999).item()
        vmax = all_vals.max().item()

        pct = torch.quantile(all_vals, args.percentile / 100.0).item()
        # Margine di sicurezza configurabile, ma MAI oltre la soglia attuale: in
        # alcuni layer la coda della distribuzione e' cosi' pesante che
        # anche un percentile alto supererebbe 50 -- vogliamo solo
        # STRINGERE la soglia dove i dati lo permettono, mai allargarla.
        new_threshold = min(round(pct * args.margin, 1), args.current_threshold)
        suggested[name] = new_threshold

        print(f'{name:20s} {vmax:10.2f} {p50:10.2f} {p95:10.2f} {p99:10.2f} {p999:10.2f} '
              f'{args.current_threshold:15.1f} {new_threshold:14.1f}')

    print('\n=== Dizionario pronto da usare (clamp_values per layer) ===')
    print(suggested)

    import json
    out_path = args.output_json
    with open(out_path, 'w') as f:
        json.dump(suggested, f, indent=2)
    print(f'\nSalvato in: {out_path}')
    print(f'Usalo con: --clamp_values_json {out_path}')


if __name__ == '__main__':
    main()