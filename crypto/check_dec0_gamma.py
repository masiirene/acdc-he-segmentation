"""
crypto/check_dec0_normalized_value.py

Segue crypto/investigate_dec0_block5.py e crypto/check_dec0_gamma.py: gamma
NON e' anomalo (varia solo 1x-3x in tutta la rete), quindi il salto enorme
norm2->act2 (es. 300 -> 5491) non puo' venire da li'. Questo script
verifica l'ipotesi successiva: il valore NORMALIZZATO PRIMA di gamma/beta,
cioe' (x-mean)/sqrt(var+eps), e' gia' enorme per conto suo -- il che
punterebbe a una varianza troppo piccola in quello specifico canale, per
quella specifica istanza (non nel minimo aggregato su tutta la rete, che
avevamo gia' escluso, ma nel caso particolare canale+istanza).

Per ogni caso estremo, ricalcola manualmente (x-mean)/sqrt(var+eps) usando
la STESSA formula di InstanceNorm2d (per-istanza, per-canale, su tutti i
pixel spaziali del canale in quell'istanza), a partire dall'input reale
catturato (l'output di conv2, prima della normalizzazione).

USO:
    python3 -m crypto.check_dec0_normalized_value --checkpoint <path>
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet, PolyAct
from training.dataset import ACDCDataset, load_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}\n')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly',
                           norm_type='instance', norm_mode='per_instance').to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    for name, m in model.named_modules():
        if isinstance(m, PolyAct):
            m.clamp_value = float('inf')

    gamma = model.dec0.block[4].weight.detach().cpu().numpy()
    beta = model.dec0.block[4].bias.detach().cpu().numpy()
    eps = model.dec0.block[4].eps

    conv2_out = {}
    act2_out = {}

    def make_hook(storage):
        def hook(module, inp, out):
            storage['last'] = out.detach().cpu()
        return hook

    h1 = model.dec0.block[3].register_forward_hook(make_hook(conv2_out))
    h3 = model.dec0.block[5].register_forward_hook(make_hook(act2_out))

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f'Validation: {len(val_cases)} pazienti, {len(val_ds)} slice\n')

    records = []
    with torch.no_grad():
        for batch_idx, (imgs, segs) in enumerate(val_loader):
            imgs = imgs.to(device)
            _ = model(imgs)

            act2 = act2_out['last']    # (B, C, H, W)
            conv2 = conv2_out['last']  # (B, C, H, W), input a norm2

            B = act2.shape[0]
            for b in range(B):
                global_idx = batch_idx * val_loader.batch_size + b
                if global_idx >= len(val_ds.slices):
                    continue
                img_path, _, slice_idx = val_ds.slices[global_idx]
                patient_name = os.path.basename(img_path)

                sample_act2 = act2[b]
                flat_abs = sample_act2.abs().flatten()
                max_val = flat_abs.max().item()
                max_pos = flat_abs.argmax().item()
                c, h, w = np.unravel_index(max_pos, sample_act2.shape)

                channel_data = conv2[b, c].numpy()  # (H, W)
                mean_c = channel_data.mean()
                var_c = channel_data.var()  # biased, come PyTorch
                x_val = conv2[b, c, h, w].item()
                normalized_val = (x_val - mean_c) / np.sqrt(var_c + eps)

                reconstructed = gamma[c] * normalized_val + beta[c]

                records.append({
                    'patient': patient_name,
                    'slice_idx': slice_idx,
                    'channel': int(c),
                    'act2_val': max_val,
                    'x_val': x_val,
                    'mean_c': mean_c,
                    'var_c': var_c,
                    'std_c': np.sqrt(var_c + eps),
                    'normalized_val': normalized_val,
                    'reconstructed_norm2': reconstructed,
                })

    h1.remove(); h3.remove()

    records_sorted = sorted(records, key=lambda r: -abs(r['act2_val']))

    print(f'=== I {args.top_k} casi piu\' estremi: da dove viene il valore normalizzato? ===\n')
    print(f'{"Paziente":28s} {"canale":>6s} {"x (conv2)":>10s} {"media_c":>9s} {"std_c":>9s} '
          f'{"normalizzato":>13s} {"act2 finale":>12s}')
    print('-' * 100)
    for r in records_sorted[:args.top_k]:
        print(f'{r["patient"]:28s} {r["channel"]:6d} {r["x_val"]:10.2f} {r["mean_c"]:9.3f} '
              f'{r["std_c"]:9.5f} {r["normalized_val"]:13.2f} {r["act2_val"]:12.2f}')

    print()
    all_norm_vals = np.array([r['normalized_val'] for r in records])
    all_std_vals = np.array([r['std_c'] for r in records])
    print('=== Statistiche generali sul valore normalizzato (prima di gamma/beta) ===')
    print(f'  |normalizzato| -- p50={np.percentile(np.abs(all_norm_vals),50):.2f}  '
          f'p90={np.percentile(np.abs(all_norm_vals),90):.2f}  '
          f'p99={np.percentile(np.abs(all_norm_vals),99):.2f}  '
          f'max={np.abs(all_norm_vals).max():.2f}')
    print(f'  std_c (deviazione standard del canale in quell\'istanza) -- '
          f'p50={np.percentile(all_std_vals,50):.4f}  p10={np.percentile(all_std_vals,10):.4f}  '
          f'min={all_std_vals.min():.4f}')
    print()
    print('Interpretazione: se "normalizzato" e\' gia\' enorme (decine-centinaia) PRIMA di')
    print('gamma/beta, il problema e\' a monte -- probabile std_c molto piccola (varianza')
    print('quasi nulla) in quel canale specifico per quell\'istanza specifica, anche se il')
    print('minimo aggregato su TUTTA la rete (controllato in precedenza) restava normale.')


if __name__ == '__main__':
    main()