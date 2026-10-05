import torch

ckpt = torch.load("results/test_5stage_no4norm_finetuned/best_model.pth", map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

norm_keys = sorted(k for k in state_dict.keys() if "norm" in k.lower() or "instancenorm" in k.lower())
for k in norm_keys:
    print(k)

print(f"\nTotale layer di normalizzazione trovati: {len(set(k.rsplit('.', 1)[0] for k in norm_keys))}")
