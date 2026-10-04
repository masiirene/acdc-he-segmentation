"""
crypto/check_refs_against_torch.py

Confronta i riferimenti numpy di prototype_lattice_resample.py con PyTorch,
per confermare che codificano DAVVERO la semantica di nn.Conv2d(k=3,
padding=1, stride 1 o 2) e nn.ConvTranspose2d(k=2, stride=2).
Da lanciare sul Mac (dove torch c'e'), dalla cartella del progetto:

    python3 crypto/check_refs_against_torch.py
"""
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype_lattice_resample import conv_ref, upconv_ref

rng = np.random.default_rng(0)
ok = True

x = rng.normal(size=(3, 16, 12))
w = rng.normal(size=(5, 3, 3, 3))
b = rng.normal(size=5)
for stride in (1, 2):
    mine = conv_ref(x, w, b, stride)
    ref = F.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b),
                   stride=stride, padding=1)[0].numpy()
    err = np.abs(mine - ref).max()
    ok &= (mine.shape == ref.shape) and err < 1e-12
    print(f"Conv2d k=3 pad=1 stride={stride}: forma {mine.shape} vs {ref.shape}, errore {err:.2e}")

x2 = rng.normal(size=(4, 8, 6))
w2 = rng.normal(size=(4, 3, 2, 2))      # (in, out, kH, kW) come ConvTranspose2d
b2 = rng.normal(size=3)
mine = upconv_ref(x2, w2, b2)
ref = F.conv_transpose2d(torch.tensor(x2)[None], torch.tensor(w2), torch.tensor(b2), stride=2)[0].numpy()
err = np.abs(mine - ref).max()
ok &= (mine.shape == ref.shape) and err < 1e-12
print(f"ConvTranspose2d k=2 stride=2: forma {mine.shape} vs {ref.shape}, errore {err:.2e}")

print("\n" + ("OK: i riferimenti coincidono con PyTorch." if ok else "ATTENZIONE: i riferimenti NON coincidono con PyTorch."))