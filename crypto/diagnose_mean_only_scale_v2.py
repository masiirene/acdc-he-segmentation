"""
crypto/diagnose_mean_only_scale_v2.py

Versione corretta: legge m.last_raw (il valore PRIMA del clamp, salvato
internamente da PolyAct ad ogni forward) invece dell'output del modulo
(che e' gia' stato tagliato dal clamp, e quindi nasconde l'esplosione
reale -- vedi i valori identici alla soglia nel test precedente).
"""

import sys
import json
import os
import torch

sys.path.insert(0, '.')
import models.he_friendly as hf
from crypto.replace_with_mean_only_norm import replace_instancenorm_with_meanonly
from training.dataset import ACDCDataset, load_splits

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

with open('crypto/calibrated_clamp_values.json') as f:
    clamp_values = json.load(f)

model = hf.HEFriendlyUNet(
    in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
    clamp_values=clamp_values, norm_mode='per_instance', skip_mode='sum',
    weight_standardization=False, filters=[32, 64, 128, 256, 128, 64],
).to(device)
state = torch.load(
    "results/test_sum_aggressive_pruned_warmstart/act=poly_norm=instance_mode=per_instance_skip-sum_bs16_lr1e-05/best_model.pth",
    map_location=device, weights_only=False)
model.load_state_dict(state, strict=False)
replace_instancenorm_with_meanonly(model)
model.eval()

splits_path = os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json')
data_dir = os.path.expanduser('~/Desktop/tesi_acdc/training')
_, val_cases = load_splits(splits_path, fold=0)
val_ds = ACDCDataset(data_dir, val_cases[:2], patch_size=(256, 224), augment=False)
img, _ = val_ds[0]

with torch.no_grad():
    model(img.unsqueeze(0).to(device))

print(f"{'Layer':16s} {'clamp calibrato':>16s} {'RAW pre-clamp max':>18s} {'rapporto':>10s}")
for name, m in model.named_modules():
    if hasattr(m, 'last_raw') and m.last_raw is not None:
        real_max = m.last_raw.abs().max().item()
        calib = clamp_values.get(name, None)
        if calib:
            flag = "  <-- SATURA" if real_max > calib else ""
            print(f"{name:16s} {calib:16.2f} {real_max:18.2f} {real_max/calib:10.1f}x{flag}")