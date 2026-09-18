"""
crypto/inspect_splits.py

Non supponiamo piu' nulla sulla struttura di train/validation/test --
questo script ispeziona direttamente splits_final.json e i dati reali,
e risponde a domande concrete:

1. Che formato ha splits_final.json? (nnU-Net standard: lista di dict,
   uno per fold, con chiavi 'train' e 'val' -- oppure un formato diverso?)
2. Quanti fold ci sono, quanti pazienti in train/val per ciascuno?
3. C'e' overlap tra train e val nello STESSO fold? (bug grave se si')
4. I val set di fold DIVERSI si sovrappongono? (normale in k-fold: NO,
   ogni paziente deve stare nel val set di esattamente un fold)
5. L'unione di tutti i val set copre tutti i pazienti? (dovrebbe, se e'
   vera k-fold cross-validation)
6. C'e' un vero TEST SET separato da qualche parte -- un'altra chiave nel
   json, o una cartella dati diversa da quella usata per train/val?
7. Quanti pazienti fisici totali ci sono nella cartella dati?

USO:
    python3 -m crypto.inspect_splits
"""

import os
import sys
import json
import argparse

sys.path.insert(0, '.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    args = parser.parse_args()

    print('=' * 70)
    print('1. STRUTTURA GREZZA DI splits_final.json')
    print('=' * 70)
    with open(args.splits_path) as f:
        raw = json.load(f)

    print(f'Tipo Python: {type(raw).__name__}')
    if isinstance(raw, list):
        print(f'Lista di {len(raw)} elementi (probabile: uno per fold)')
        for i, fold_data in enumerate(raw):
            if isinstance(fold_data, dict):
                keys = list(fold_data.keys())
                print(f'  Fold {i}: chiavi = {keys}')
                for k in keys:
                    v = fold_data[k]
                    if isinstance(v, list):
                        print(f'    "{k}": lista di {len(v)} elementi, es. {v[:2]}')
            else:
                print(f'  Fold {i}: tipo inatteso {type(fold_data)}')
    elif isinstance(raw, dict):
        print(f'Dizionario con chiavi: {list(raw.keys())}')
        for k, v in raw.items():
            if isinstance(v, list):
                print(f'  "{k}": lista di {len(v)} elementi, es. {v[:2]}')
    else:
        print('Formato NON riconosciuto -- ispeziona manualmente il file')
        return

    print()
    print('=' * 70)
    print('2. C\'E\' UNA CHIAVE "test" DA QUALCHE PARTE?')
    print('=' * 70)
    found_test_key = False
    if isinstance(raw, list):
        for i, fold_data in enumerate(raw):
            if isinstance(fold_data, dict) and 'test' in fold_data:
                found_test_key = True
                print(f'  TROVATA chiave "test" nel fold {i}: {len(fold_data["test"])} elementi')
    elif isinstance(raw, dict) and 'test' in raw:
        found_test_key = True
        print(f'  TROVATA chiave "test" di primo livello: {len(raw["test"])} elementi')
    if not found_test_key:
        print('  NESSUNA chiave "test" trovata in splits_final.json.')
        print('  -> Il file definisce solo train/val per k-fold cross-validation,')
        print('     NON un test set separato mai usato durante lo sviluppo.')

    if isinstance(raw, list) and all(isinstance(f, dict) and 'train' in f and 'val' in f for f in raw):
        print()
        print('=' * 70)
        print('3. OVERLAP TRAIN/VAL NELLO STESSO FOLD (deve essere SEMPRE vuoto)')
        print('=' * 70)
        for i, fold_data in enumerate(raw):
            train_set = set(fold_data['train'])
            val_set = set(fold_data['val'])
            overlap = train_set & val_set
            status = 'OK (nessun overlap)' if not overlap else f'\u26a0\ufe0f  BUG: {len(overlap)} casi in comune!'
            print(f'  Fold {i}: train={len(train_set)}, val={len(val_set)} -> {status}')
            if overlap:
                print(f'    Casi duplicati: {list(overlap)[:5]}')

        print()
        print('=' * 70)
        print('4. OVERLAP TRA VAL SET DI FOLD DIVERSI (deve essere SEMPRE vuoto)')
        print('=' * 70)
        all_val_sets = [set(f['val']) for f in raw]
        any_cross_overlap = False
        for i in range(len(all_val_sets)):
            for j in range(i + 1, len(all_val_sets)):
                cross_overlap = all_val_sets[i] & all_val_sets[j]
                if cross_overlap:
                    any_cross_overlap = True
                    print(f'  \u26a0\ufe0f  Fold {i} e Fold {j} condividono {len(cross_overlap)} casi nel val set!')
        if not any_cross_overlap:
            print('  OK: nessun paziente compare nel val set di piu\' di un fold.')

        print()
        print('=' * 70)
        print('5. COPERTURA: l\'unione di tutti i val set copre tutti i pazienti?')
        print('=' * 70)
        union_val = set()
        for vs in all_val_sets:
            union_val |= vs
        union_train_fold0 = set(raw[0]['train']) | set(raw[0]['val'])
        print(f'  Pazienti totali visti nel Fold 0 (train+val): {len(union_train_fold0)}')
        print(f'  Pazienti totali coperti dall\'unione di TUTTI i val set: {len(union_val)}')
        if union_val == union_train_fold0:
            print('  OK: ogni paziente finisce nel val set di esattamente un fold '
                  '(vera k-fold cross-validation).')
        else:
            missing = union_train_fold0 - union_val
            print(f'  \u26a0\ufe0f  {len(missing)} pazienti non compaiono MAI in nessun val set: {list(missing)[:5]}')

    print()
    print('=' * 70)
    print('6. PAZIENTI FISICI NELLA CARTELLA DATI')
    print('=' * 70)
    if os.path.isdir(args.data_dir):
        entries = sorted(os.listdir(args.data_dir))
        dirs = [e for e in entries if os.path.isdir(os.path.join(args.data_dir, e))]
        non_dirs = [e for e in entries if not os.path.isdir(os.path.join(args.data_dir, e))]
        print(f'  Cartelle (probabili pazienti): {len(dirs)}')
        if non_dirs:
            print(f'  File NON cartella nella stessa directory (es. .DS_Store): {non_dirs}')
        print(f'  Primi 5 nomi: {dirs[:5]}')
    else:
        print(f'  Cartella {args.data_dir} non trovata su questa macchina.')

    print()
    print('=' * 70)
    print('RIEPILOGO')
    print('=' * 70)
    print('Se sopra NON e\' comparsa nessuna chiave "test": questo progetto usa SOLO')
    print('train/val in k-fold cross-validation. Il validation set di ogni fold viene')
    print('usato ripetutamente per scegliere tra configurazioni diverse (come fatto')
    print('questa settimana) -- utile saperlo se si vuole poi riservare un vero test')
    print('set finale, mai toccato durante lo sviluppo, per una stima onesta.')


if __name__ == '__main__':
    main()