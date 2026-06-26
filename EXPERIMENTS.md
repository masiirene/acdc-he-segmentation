# Phase 2 Experiments — HE-Friendly U-Net

## Baseline (Phase 1 — nnU-Net original, 150 epochs on Colab T4)
| Structure | Dice | HD95 |
|-----------|------|------|
| RV  | 0.912 | 2.99 mm |
| MYO | 0.901 | 2.79 mm |
| LV  | 0.940 | 2.48 mm |

Architecture: Conv3×3 → InstanceNorm → LeakyReLU × 2 per stage
6 encoder stages, 5 decoder stages, patch 256×224, batch_size=56, 150 epochs

---

## Experiment 1 — act=identity, norm=none
**Status**: done (2 epochs, sanity check only)
**Notes**: mean Dice ~0.038 — expected, identity makes the network fully linear, lower bound

---

## Experiment 2 — act=squared, norm=none
**Status**: failed — NaN loss from epoch 1
**Notes**: x² without normalization causes immediate numerical explosion in a deep network.
Confirms ULD-Net finding: polynomial activations require normalization to stay stable.

---

## Experiment 3 — act=squared, norm=batch
**Status**: failed — val_loss NaN, Dice stuck at 0
**Notes**: BatchNorm helps but not enough — x² is too unstable without pretrained initialization.

---

## Experiment 4 — act=poly, norm=batch ← RUNNING NOW
**Status**: running (50 epochs, Mac M4 MPS, lr=1e-4, batch_size=4)
**HE cost**: 1 multiplicative level per activation (ax²+bx+c), BatchNorm free at inference
**Notes**: much more stable than x² — PolyAct initialized with c0=0.5, c1=1.0, c2=0.1
           behaves close to linear at initialization, avoids explosion.
           Epoch 5 already reached mean Dice 0.710 — promising.
           Some instability (val_loss NaN on some epochs) still present.

| Structure | Dice | HD95 |
|-----------|------|------|
| RV  | TBD | TBD |
| MYO | TBD | TBD |
| LV  | TBD | TBD |

---

## Planned experiments
- act=poly, norm=none
- act=squared, norm=batch (with pretrained init — future)

---

## Setup
- Training: Mac M4, MPS, Adam optimizer, DiceCE loss, gradient clipping max_norm=1.0
- Data: ACDC 100 patients, 200 cases, fold 0 (160 train / 40 val), fallback 80/20 split
- Patch size: 256×224, Z-score normalization on nonzero voxels
- All results comparable with baseline (same fold, same metrics: Dice + HD95)

---

## Key findings so far
- x² without norm → immediate NaN (numerical explosion)
- x² with BatchNorm → still unstable, Dice stuck at 0
- ax²+bx+c (PolyAct) with BatchNorm → stable, learning well
- This confirms ULD-Net (Xie et al., ICLR 2026): normalization is essential
  for stable training of polynomial networks