import torch

ckpt = torch.load("results/test_5stage_no4norm_finetuned/best_model.pth", map_location="cpu", weights_only=False)

print(f"Tipo di ckpt: {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"Chiavi di primo livello: {list(ckpt.keys())}")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"  '{k}' e' un dict con {len(v)} chiavi, prime 5: {list(v.keys())[:5]}")
        else:
            print(f"  '{k}': {type(v)}")
