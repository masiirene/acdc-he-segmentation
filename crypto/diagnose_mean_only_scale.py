"""
crypto/diagnose_mean_only_scale.py

Verifica se le attivazioni con MeanOnlyNorm2d hanno una scala molto
diversa da quella prevista dalle soglie di clamp calibrate (che
assumono InstanceNorm completa, varianza unitaria).
"""

import sys
import json
import torch
import torch.nn as nn

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

activations = {}
def make_hook(name):
    def hook(mod, inp, out):
        activations[name] = out.detach()
    return hook

for name, m in model.named_modules():
    if isinstance(m, type(model.enc0.block[2])):  # PolyAct
        m.register_forward_hook(make_hook(name))

_, val_cases = load_splits('~/Desktop/tesi_acdc/splits_final.json'.replace('~', __import__('os').path.expanduser('~')), fold=0)
val_ds = ACDCDataset(__import__('os').path.expanduser('~/Desktop/tesi_acdc/training'), val_cases[:2], patch_size=(256,224), augment=False)
img, _ = val_ds[0]

with torch.no_grad():
    model(img.unsqueeze(0).to(device))

print(f"{'Layer':16s} {'clamp calibrato':>16s} {'valore reale max':>18s} {'rapporto':>10s}")
for name, act in activations.items():
    real_max = act.abs().max().item()
    calib = clamp_values.get(name, None)
    if calib:
        print(f"{name:16s} {calib:16.2f} {real_max:18.2f} {real_max/calib:10.1f}x")