"""
Ispeziona i dati grezzi (immagine + segmentazione) di un paziente specifico
per capire perché produce NaN/Inf nei logits durante la validazione.

Replica ESATTAMENTE la pipeline di training/dataset.py (zscore_normalize +
pad_or_crop) per vedere se qualche slice produce valori normalizzati estremi.

USO:
    python3 inspect_patient.py patient006
"""

import sys
import os
import numpy as np
import nibabel as nib

DATA_DIR = os.path.expanduser("~/Desktop/tesi_acdc/training")
PATCH_SIZE = (256, 224)  # deve combaciare con train.py


def zscore_normalize(img: np.ndarray) -> np.ndarray:
    """Identica a quella in training/dataset.py"""
    mask = img != 0
    if mask.sum() == 0:
        return img
    mean = img[mask].mean()
    std  = img[mask].std()
    return (img - mean) / (std + 1e-8)


def pad_or_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Identica a quella in training/dataset.py"""
    h, w = img.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2),
                           (pad_w//2, pad_w - pad_w//2)), mode='constant')
    h, w = img.shape
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return img[start_h:start_h+target_h, start_w:start_w+target_w]


def inspect_volume(path: str, label: str):
    if not os.path.exists(path):
        print(f"  [{label}] FILE NON TROVATO: {path}")
        return
    data = nib.load(path).get_fdata()

    n_nan = np.isnan(data).sum()
    n_inf = np.isinf(data).sum()

    print(f"  [{label}] shape={data.shape}  dtype={data.dtype}")
    print(f"    min={np.nanmin(data):.4f}  max={np.nanmax(data):.4f}  "
          f"mean={np.nanmean(data):.4f}  std={np.nanstd(data):.4f}")
    print(f"    NaN nel file grezzo: {n_nan}   Inf nel file grezzo: {n_inf}")

    if label == "IMG":
        print(f"\n    --- Replica pipeline (zscore + pad/crop a {PATCH_SIZE}) ---")
        for s in range(data.shape[2]):
            sl_raw = data[:, :, s].astype(np.float32)
            nonzero = (sl_raw != 0).sum()
            total = sl_raw.size

            sl_norm = zscore_normalize(sl_raw)
            sl_final = pad_or_crop(sl_norm, *PATCH_SIZE)

            z_min, z_max = sl_final.min(), sl_final.max()
            z_absmax = max(abs(z_min), abs(z_max))
            flag = "  <-- VALORE Z-SCORE ESTREMO!" if z_absmax > 10 else ""
            print(f"    slice {s}: nonzero={nonzero}/{total} ({nonzero/total*100:.1f}%)  "
                  f"z-score range=[{z_min:.2f}, {z_max:.2f}]{flag}")

    if label == "SEG":
        unique_vals = np.unique(data)
        print(f"    valori unici nella segmentazione: {unique_vals}")
        if not set(unique_vals).issubset({0, 1, 2, 3}):
            print(f"    ATTENZIONE: valori fuori range [0,1,2,3] trovati!")


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 inspect_patient.py patientXXX")
        sys.exit(1)

    patient_id = sys.argv[1]
    patient_dir = os.path.join(DATA_DIR, patient_id)

    if not os.path.isdir(patient_dir):
        print(f"Cartella non trovata: {patient_dir}")
        sys.exit(1)

    files = sorted(f for f in os.listdir(patient_dir) if f.endswith(".nii.gz"))
    frames = sorted(set(
        f.split("_frame")[1].split(".")[0].replace("_gt", "")
        for f in files if "_frame" in f and "_4d" not in f
    ))

    print(f"Paziente: {patient_id}")
    print(f"Frame trovati: {frames}")
    print("=" * 70)

    for frame in frames:
        img_path = os.path.join(patient_dir, f"{patient_id}_frame{frame}.nii.gz")
        seg_path = os.path.join(patient_dir, f"{patient_id}_frame{frame}_gt.nii.gz")

        print(f"\nFrame {frame}")
        inspect_volume(img_path, "IMG")
        inspect_volume(seg_path, "SEG")


if __name__ == "__main__":
    main()
