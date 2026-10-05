import torch

ckpt = torch.load("results/test_5stage_no4norm_finetuned/best_model.pth", map_location="cpu", weights_only=False)

stages = ["enc0", "enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1", "dec0"]

print(f"{'Layer':<18} {'weight mean':>12} {'weight std':>12} {'bias mean':>12} {'bias std':>12}  {'SOSPETTO'}")
print("-" * 85)

for stage in stages:
    for block_idx, label in [(1, "norm1"), (4, "norm2")]:
        w_key = f"{stage}.block.{block_idx}.weight"
        b_key = f"{stage}.block.{block_idx}.bias"
        if w_key not in ckpt:
            print(f"{stage}.{label:<10} -- chiave non trovata ({w_key})")
            continue
        w = ckpt[w_key]
        b = ckpt[b_key]
        w_mean, w_std = w.mean().item(), w.std().item()
        b_mean, b_std = b.mean().item(), b.std().item()
        suspect = (abs(w_mean - 1.0) < 1e-4 and w_std < 1e-4 and
                   abs(b_mean) < 1e-4 and b_std < 1e-4)
        flag = "<<<< MAI ALLENATO (probabilmente disattivato)" if suspect else ""
        print(f"{stage}.{label:<10} {w_mean:12.6f} {w_std:12.6f} {b_mean:12.6f} {b_std:12.6f}  {flag}")
