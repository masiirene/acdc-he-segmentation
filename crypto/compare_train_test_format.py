"""
crypto/compare_train_test_format.py

Prima di integrare il vero test set ACDC (patient101-150) nel flusso di
lavoro, verifica che i dati abbiano lo STESSO formato dei 100 pazienti di
training gia' usati -- altrimenti crop_to_nonzero() e il resto della
pipeline di preprocessing potrebbero comportarsi in modo diverso o rompersi.

Confronta, per un paziente di training e uno di test:
  - shape del volume immagine e della segmentazione
  - dtype (tipo di dato: int16, float32, ecc.)
  - spacing voxel (dimensione fisica di ogni voxel, dai metadati NIfTI)
  - valori unici nella segmentazione (deve essere {0,1,2,3}: sfondo, RV,
    miocardio, LV -- se il test set avesse etichette diverse sarebbe un
    problema serio)

USO:
    python3 -m crypto.compare_train_test_format \\
        --train_patient ~/Desktop/tesi_acdc/training/patient001 \\
        --test_patient ~/Desktop/tesi_acdc/testing/patient101
"""

import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib


def inspect_patient(patient_dir, label):
    print(f'=== {label}: {patient_dir} ===')

    gt_files = sorted(glob.glob(os.path.join(patient_dir, '*_frame*_gt.nii.gz')))
    if not gt_files:
        print('  \u26a0\ufe0f  Nessun file _gt.nii.gz trovato in questa cartella!')
        return None

    gt_path = gt_files[0]
    img_path = gt_path.replace('_gt.nii.gz', '.nii.gz')

    if not os.path.exists(img_path):
        print(f'  \u26a0\ufe0f  Immagine corrispondente non trovata: {img_path}')
        return None

    img = nib.load(img_path)
    gt = nib.load(gt_path)

    img_data = img.get_fdata()
    gt_data = gt.get_fdata()

    info = {
        'img_shape': img_data.shape,
        'gt_shape': gt_data.shape,
        'img_dtype': img_data.dtype,
        'gt_dtype': gt_data.dtype,
        'spacing': img.header.get_zooms(),
        'gt_unique_values': sorted(np.unique(gt_data).tolist()),
        'img_value_range': (float(img_data.min()), float(img_data.max())),
    }

    print(f'  File immagine: {os.path.basename(img_path)}')
    print(f'  File segmentazione: {os.path.basename(gt_path)}')
    print(f'  Shape immagine: {info["img_shape"]}')
    print(f'  Shape segmentazione: {info["gt_shape"]}')
    print(f'  Dtype immagine: {info["img_dtype"]}  (originale, prima di eventuale cast)')
    print(f'  Dtype segmentazione: {info["gt_dtype"]}')
    print(f'  Spacing voxel (mm): {info["spacing"]}')
    print(f'  Valori unici nella segmentazione: {info["gt_unique_values"]}')
    print(f'  Range valori immagine: {info["img_value_range"][0]:.1f} .. {info["img_value_range"][1]:.1f}')
    print()

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_patient', required=True,
                        help='Path alla cartella di un paziente di TRAINING, es. .../training/patient001')
    parser.add_argument('--test_patient', required=True,
                        help='Path alla cartella di un paziente di TEST, es. .../testing/patient101')
    args = parser.parse_args()

    train_info = inspect_patient(args.train_patient, 'TRAINING (patient noto)')
    test_info = inspect_patient(args.test_patient, 'TEST (patient101+)')

    if train_info is None or test_info is None:
        print('Impossibile completare il confronto -- vedi avvisi sopra.')
        return

    print('=' * 70)
    print('CONFRONTO')
    print('=' * 70)

    checks = [
        ('Numero di dimensioni (2D+tempo o 3D)', len(train_info['img_shape']) == len(test_info['img_shape'])),
        ('Dtype immagine', train_info['img_dtype'] == test_info['img_dtype']),
        ('Dtype segmentazione', train_info['gt_dtype'] == test_info['gt_dtype']),
        ('Valori unici in segmentazione uguali',
         train_info['gt_unique_values'] == test_info['gt_unique_values']),
    ]

    all_ok = True
    for check_name, passed in checks:
        status = 'OK' if passed else '\u26a0\ufe0f  DIVERSO'
        print(f'  {check_name}: {status}')
        if not passed:
            all_ok = False

    print(f'  Shape (puo\' variare normalmente tra pazienti): '
          f'train={train_info["img_shape"]}  test={test_info["img_shape"]}')
    print(f'  Spacing (puo\' variare normalmente tra pazienti): '
          f'train={train_info["spacing"]}  test={test_info["spacing"]}')

    print()
    if all_ok:
        print('=> Formato compatibile: stesso dtype, stessi valori di etichetta.')
        print('   Il preprocessing esistente (crop_to_nonzero) dovrebbe funzionare')
        print('   sui dati di test senza modifiche.')
    else:
        print('=> ATTENZIONE: differenze trovate sopra -- da investigare prima di')
        print('   integrare il test set nel training pipeline esistente.')


if __name__ == '__main__':
    main()