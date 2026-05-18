# ACDC HE Segmentation

Thesis project: cardiac MRI segmentation on the ACDC dataset with Homomorphic Encryption (HE) compatibility.

The goal is to adapt nnU-Net — the state-of-the-art self-configuring segmentation framework — for encrypted inference using the CKKS scheme (OpenFHE). This allows a client to send encrypted medical images to a server, obtain a segmentation result, and decrypt it locally, without the server ever seeing the data.

---

## Phases

**Phase 1 — Baseline** (complete)
Train the original nnU-Net 2D on ACDC and obtain reference results.

| Structure | Dice | HD95 |
|-----------|------|------|
| RV | 0.912 | 2.99 mm |
| MYO | 0.901 | 2.79 mm |
| LV | 0.940 | 2.48 mm |

**Phase 2 — HE-friendly network** (in progress)
Replace LeakyReLU with polynomial activations (x², then ax²+bx+c) and remove InstanceNorm. Measure accuracy degradation vs baseline.

**Phase 3 — Encrypted inference** (upcoming)
Implement inference on encrypted images using OpenFHE/CKKS.

---

## Dataset

ACDC (Automated Cardiac Diagnosis Challenge) — 100 training patients, 3 structures (LV, MYO, RV), 2 frames per patient (ED and ES).

Bernard et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis", IEEE TMI, 2018.

---

## Installation

```bash
git clone https://github.com/masiirene/acdc-he-segmentation.git
cd acdc-he-segmentation
pip install -r requirements.txt
```

## Testing

```bash
pytest tests/
```

---

## References

- Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods, 2021.
- Cheon et al., "Homomorphic Encryption for Arithmetic of Approximate Numbers" (CKKS), Asiacrypt, 2017.