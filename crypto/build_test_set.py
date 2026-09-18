"""
crypto/build_test_set.py

Costruisce la lista dei casi del vero test set ufficiale ACDC
(patient101-150, 50 pazienti con etichette manuali fornite -- vedi
https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTesting.html).

Questo test set e' COMPLETAMENTE SEPARATO da splits_final.json (che
definisce solo train/val sui 100 pazienti 1-100, in 5 fold di k-fold
cross-validation, senza alcun test set integrato). Va tenuto separato
e usato con parsimonia -- vedi crypto/evaluate_on_test_set.py per le
regole d'uso.

I numeri di frame variano da paziente a paziente (es. patient101 ha
frame01+frame14, altri pazienti potrebbero avere frame01+frame12 o altre
combinazioni) -- questo script scopre dinamicamente quali file esistono,
non assume numeri fissi.

USO:
    python3 -m crypto.build_test_set \\
        --testing_dir ~/Desktop/tesi_acdc/testing \\
        --output crypto/test_cases.json
"""

import os
import re
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/testing'))
    parser.add_argument('--output', default='crypto/test_cases.json')
    args = parser.parse_args()

    if not os.path.isdir(args.testing_dir):
        print(f'ERRORE: cartella non trovata: {args.testing_dir}')
        return

    patient_dirs = sorted([
        d for d in os.listdir(args.testing_dir)
        if os.path.isdir(os.path.join(args.testing_dir, d)) and d.startswith('patient')
    ])

    print(f'Pazienti trovati in {args.testing_dir}: {len(patient_dirs)}')

    test_cases = []
    frame_pattern = re.compile(r'^(patient\d+)_frame(\d+)\.nii\.gz$')

    for patient_id in patient_dirs:
        patient_path = os.path.join(args.testing_dir, patient_id)
        files = os.listdir(patient_path)

        for f in files:
            m = frame_pattern.match(f)
            if not m:
                continue
            p_id, frame_num = m.group(1), m.group(2)
            gt_file = f'{p_id}_frame{frame_num}_gt.nii.gz'
            if gt_file in files:
                test_cases.append(f'{p_id}_frame{frame_num}')
            else:
                print(f'  \u26a0\ufe0f  {p_id}_frame{frame_num}: immagine trovata ma NESSUNA _gt.nii.gz corrispondente -- escluso')

    test_cases = sorted(test_cases)

    print(f'\nCasi totali costruiti (immagine + etichetta entrambe presenti): {len(test_cases)}')
    print(f'Attesi: 100 (50 pazienti x 2 frame annotati ciascuno)')
    if len(test_cases) != 100:
        print(f'\u26a0\ufe0f  Il numero non coincide con l\'atteso -- controllare sopra quali file sono stati esclusi.')

    print(f'\nPrimi 5 casi: {test_cases[:5]}')
    print(f'Ultimi 5 casi: {test_cases[-5:]}')

    with open(args.output, 'w') as f:
        json.dump(test_cases, f, indent=2)
    print(f'\nSalvato in: {args.output}')
    print('\nQuesto file NON va mai fuso con splits_final.json -- tenerlo separato.')


if __name__ == '__main__':
    main()