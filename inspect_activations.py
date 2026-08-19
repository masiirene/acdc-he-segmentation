"""
Diagnostica layer-per-layer: registra un forward hook su ogni istanza di
PolyAct nel modello, per vedere in quale layer partono i valori estremi
(che poi esplodono a Inf/NaN nei logits finali).

Confronta una slice "problematica" (compare spesso nei warning NaN/Inf)
con una slice "pulita" (mai comparsa) dello stesso fold, per isolare se
il problema e' specifico del contenuto dell'immagine o generale.

USO:
    python3 inspect_activations.py
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet, PolyAct
from training.dataset import zscore_normalize, pad_or_crop

DATA_DIR = os.path.expanduser("~/Desktop/tesi_acdc/training")
PATCH_SIZE = (256, 224)

# Checkpoint dal run di debug (fold 2, con clamp e clipping)
CKPT_PATH = "results/debug_nan_fold2_clamped/act=poly_norm=instance_bs16_lr0.0001/best_model.pth"

# Casi da confrontare: (patient_frame, slice_idx, label)
CASES = [
    ("patient002_frame01", 2, "PROBLEMATICO (18/20 epoche)"),
    ("patient006_frame01", 5, "PROBLEMATICO (16/20 epoche)"),
    ("patient015_frame01", 5, "PULITO (mai comparso)"),
]


def load_slice(case: str, slice_idx: int):
    patient_id = case[:10]
    frame_num = case[11:].replace("frame", "")
    img_path = os.path.join(DATA_DIR, patient_id, f"{patient_id}_frame{frame_num}.nii.gz")
    if not os.path.exists(img_path):
        return None
    img = nib.load(img_path).get_fdata()[:, :, slice_idx].astype(np.float32)
    img = zscore_normalize(img, clip_range=5.0)
    img = pad_or_crop(img, *PATCH_SIZE)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    model = HEFriendlyUNet(in_channels=1, num_classes=4,
                           act_type='poly', norm_type='instance').to(device)

    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state)
        print(f"Checkpoint caricato da: {CKPT_PATH}\n")
    else:
        print(f"ATTENZIONE: checkpoint non trovato ({CKPT_PATH}), uso pesi random.\n")

    model.eval()

    # Registra un hook su ogni PolyAct, con il suo nome di modulo
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, PolyAct):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    print(f"Hook registrati su {len(handles)} istanze di PolyAct.\n")
    print("=" * 90)

    for case, slice_idx, label in CASES:
        x = load_slice(case, slice_idx)
        if x is None:
            print(f"[{case} slice {slice_idx}] file non trovato, skip.")
            continue
        x = x.to(device)

        activations.clear()
        with torch.no_grad():
            logits = model(x)

        n_nan_inf = (~torch.isfinite(logits)).sum().item()
        logit_max = logits.abs().max().item() if torch.isfinite(logits).all() else float('nan')

        print(f"\n{case} slice {slice_idx}  [{label}]")
        print(f"  Output finale: NaN/Inf={n_nan_inf}  max|logit|={logit_max}")
        print(f"  {'Layer':30s} {'max|attivazione|':>18s}")
        for name, act in activations.items():
            act_max = act.abs().max().item()
            finite = torch.isfinite(act).all().item()
            flag = "" if finite else "  <-- NaN/Inf QUI"
            print(f"  {name:30s} {act_max:18.4f}{flag}")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
