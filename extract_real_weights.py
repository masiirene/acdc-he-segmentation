import torch
import numpy as np

CKPT_PATH = "results/test_5stage_no4norm_finetuned/best_model.pth"
OUT_PATH = "crypto/real_weights_5stage_no4norm.npz"
BYPASSED_NORMS = {"dec0.block.1", "enc4.block.1", "dec2.block.4", "enc3.block.1"}
STAGES = ["enc0", "enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1", "dec0"]

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

def t2n(x):
    return x.detach().cpu().numpy()

out = {}
active_norms = []
bypassed_found = []

for stage in STAGES:
    out[f"{stage}_conv1_w"] = t2n(ckpt[f"{stage}.block.0.weight"])
    out[f"{stage}_conv1_b"] = t2n(ckpt[f"{stage}.block.0.bias"])
    out[f"{stage}_conv2_w"] = t2n(ckpt[f"{stage}.block.3.weight"])
    out[f"{stage}_conv2_b"] = t2n(ckpt[f"{stage}.block.3.bias"])

    out[f"{stage}_poly1_a"] = t2n(ckpt[f"{stage}.block.2.a"])
    out[f"{stage}_poly1_b"] = t2n(ckpt[f"{stage}.block.2.b"])
    out[f"{stage}_poly1_c"] = t2n(ckpt[f"{stage}.block.2.c"])
    out[f"{stage}_poly2_a"] = t2n(ckpt[f"{stage}.block.5.a"])
    out[f"{stage}_poly2_b"] = t2n(ckpt[f"{stage}.block.5.b"])
    out[f"{stage}_poly2_c"] = t2n(ckpt[f"{stage}.block.5.c"])

    norm1_name = f"{stage}.block.1"
    if norm1_name in BYPASSED_NORMS:
        bypassed_found.append(norm1_name)
    else:
        out[f"{stage}_norm1_gamma"] = t2n(ckpt[f"{norm1_name}.weight"])
        out[f"{stage}_norm1_beta"] = t2n(ckpt[f"{norm1_name}.bias"])
        active_norms.append(norm1_name)

    norm2_name = f"{stage}.block.4"
    if norm2_name in BYPASSED_NORMS:
        bypassed_found.append(norm2_name)
    else:
        out[f"{stage}_norm2_gamma"] = t2n(ckpt[f"{norm2_name}.weight"])
        out[f"{stage}_norm2_beta"] = t2n(ckpt[f"{norm2_name}.bias"])
        active_norms.append(norm2_name)

for up_name in ["up3", "up2", "up1", "up0"]:
    out[f"{up_name}_w"] = t2n(ckpt[f"{up_name}.weight"])
    out[f"{up_name}_b"] = t2n(ckpt[f"{up_name}.bias"])

out["out_conv_w"] = t2n(ckpt["out_conv.weight"])
out["out_conv_b"] = t2n(ckpt["out_conv.bias"])

np.savez(OUT_PATH, **out)

print(f"Salvato: {OUT_PATH}")
print(f"Array totali: {len(out)}")
print(f"\nLayer di norm ATTIVI salvati ({len(active_norms)}): {active_norms}")
print(f"Layer di norm BYPASSATI, correttamente esclusi ({len(bypassed_found)}): {bypassed_found}")
assert len(bypassed_found) == 4
assert len(active_norms) == 14
print("\nControllo superato: 14 attivi + 4 bypassati = 18 totali.")
