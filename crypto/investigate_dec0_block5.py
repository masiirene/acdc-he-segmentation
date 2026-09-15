"""
crypto/investigate_dec0_block5.py

Indagine mirata sul layer identificato come sistematicamente il piu'
instabile della rete (dec0.block.5, l'ultima PolyAct prima dell'output).
Non ci basta piu' sapere "produce valori grandi" -- vogliamo sapere:

1. DOVE (quale pixel spaziale, quale canale) si verificano i valori estremi
   -- sono concentrati ai bordi dell'immagine (sfondo), al centro (cuore),
   o distribuiti senza pattern?
2. QUANDO (quali pazienti/slice) -- e' un fenomeno raro concentrato su
   pochi casi "difficili", o sistematico su (quasi) ogni immagine?
3. COME si forma -- guardando i due passaggi immediatamente precedenti
   (conv2 e norm2 dentro lo stesso blocco dec0), il salto di magnitudine
   avviene gia' nella convoluzione, o e' la normalizzazione/il quadrato
   di PolyAct ad amplificarlo?

USO:
    python3 -m crypto.investigate_dec0_block5 --checkpoint <path>
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet, PolyAct
from training.dataset import ACDCDataset, load_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=10,
                        help='Quanti casi (slice) con il valore piu\' estremo mostrare in dettaglio')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}\n')

    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly',
                           norm_type='instance', norm_mode='per_instance').to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Disattiva il clamp interno per vedere il comportamento NATURALE
    # (senza il clamp a "nascondere" dove il polinomio vorrebbe davvero
    # andare) -- stessa tecnica di check_inference_stability.py.
    for name, m in model.named_modules():
        if isinstance(m, PolyAct):
            m.clamp_value = float('inf')
    print('Clamp interno disattivato per osservare il comportamento naturale.\n')

    # --- Hook sui 3 stadi rilevanti dentro dec0: conv2 (block.3), ---
    # --- norm2 (block.4), act2 = dec0.block.5 (block.5)            ---
    conv2_out = {}
    norm2_out = {}
    act2_out = {}

    def make_hook(storage, key='last'):
        def hook(module, inp, out):
            storage[key] = out.detach().cpu()
        return hook

    h1 = model.dec0.block[3].register_forward_hook(make_hook(conv2_out))
    h2 = model.dec0.block[4].register_forward_hook(make_hook(norm2_out))
    h3 = model.dec0.block[5].register_forward_hook(make_hook(act2_out))

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f'Validation: {len(val_cases)} pazienti, {len(val_ds)} slice\n')

    # Per ogni slice individuale: valore massimo assoluto di dec0.block.5,
    # posizione (canale, riga, colonna), e i valori corrispondenti nei due
    # passaggi precedenti nella STESSA posizione.
    records = []

    with torch.no_grad():
        for batch_idx, (imgs, segs) in enumerate(val_loader):
            imgs = imgs.to(device)
            _ = model(imgs)

            act2 = act2_out['last']       # (B, C, H, W)
            norm2 = norm2_out['last']
            conv2 = conv2_out['last']

            B = act2.shape[0]
            for b in range(B):
                global_idx = batch_idx * val_loader.batch_size + b
                if global_idx >= len(val_ds.slices):
                    continue
                img_path, _, slice_idx = val_ds.slices[global_idx]
                patient_name = os.path.basename(img_path)

                sample = act2[b]  # (C, H, W)
                flat_abs = sample.abs().flatten()
                max_val = flat_abs.max().item()
                max_pos = flat_abs.argmax().item()
                c, h, w = np.unravel_index(max_pos, sample.shape)

                conv2_val = conv2[b, c, h, w].item()
                norm2_val = norm2[b, c, h, w].item()
                act2_val = act2[b, c, h, w].item()

                # Posizione relativa nell'immagine: bordo o centro?
                H, W = sample.shape[1], sample.shape[2]
                edge_dist = min(h, H - 1 - h, w, W - 1 - w)  # distanza dal bordo piu' vicino
                is_border = edge_dist < 10  # entro 10 pixel dal bordo

                records.append({
                    'patient': patient_name,
                    'slice_idx': slice_idx,
                    'max_abs': max_val,
                    'channel': int(c),
                    'row': int(h),
                    'col': int(w),
                    'edge_dist': int(edge_dist),
                    'is_border': is_border,
                    'conv2_val': conv2_val,
                    'norm2_val': norm2_val,
                    'act2_val': act2_val,
                })

    h1.remove(); h2.remove(); h3.remove()

    # ------------------------------------------------------------------
    # ANALISI 1: distribuzione spaziale -- bordo vs centro
    # ------------------------------------------------------------------
    n_border = sum(1 for r in records if r['is_border'])
    n_total = len(records)
    print('=== ANALISI 1: il valore massimo e\' vicino al bordo dell\'immagine? ===')
    print(f'{n_border}/{n_total} slice ({100*n_border/n_total:.1f}%) hanno il valore massimo '
          f'entro 10 pixel dal bordo (su un\'immagine 256x224)')
    print(f'  -- se questa percentuale e\' molto piu\' alta di quanto ci si aspetterebbe per caso')
    print(f'     (l\'area vicino al bordo e\' una piccola frazione dell\'area totale), il fenomeno')
    print(f'     e\' legato allo sfondo/bordi dell\'immagine, non alla struttura cardiaca.\n')

    # Frazione di area "vicino al bordo" per confronto
    H, W = 256, 224
    border_area = H * W - (H - 20) * (W - 20)  # area entro 10px dal bordo su tutti i lati
    print(f'  (per confronto: l\'area entro 10px dal bordo e\' {100*border_area/(H*W):.1f}% '
          f'dell\'area totale -- se la percentuale sopra e\' molto piu\' alta, non e\' casuale)\n')

    # ------------------------------------------------------------------
    # ANALISI 2: distribuzione sui canali -- sempre lo stesso canale?
    # ------------------------------------------------------------------
    channel_counts = defaultdict(int)
    for r in records:
        channel_counts[r['channel']] += 1
    top_channels = sorted(channel_counts.items(), key=lambda kv: -kv[1])[:5]
    print('=== ANALISI 2: il valore massimo e\' sempre sullo stesso canale? ===')
    print(f'Numero di canali diversi coinvolti: {len(channel_counts)} (su 32 canali di dec0)')
    print('Top 5 canali piu\' frequenti come "colpevoli":')
    for ch, count in top_channels:
        print(f'  canale {ch}: {count}/{n_total} slice ({100*count/n_total:.1f}%)')
    print()

    # ------------------------------------------------------------------
    # ANALISI 3: e' un fenomeno raro o sistematico? distribuzione dei max
    # ------------------------------------------------------------------
    max_vals = np.array([r['max_abs'] for r in records])
    print('=== ANALISI 3: quanto e\' diffuso il fenomeno tra le slice? ===')
    print(f'  Percentili del valore massimo per slice: '
          f'p50={np.percentile(max_vals,50):.1f}  p90={np.percentile(max_vals,90):.1f}  '
          f'p99={np.percentile(max_vals,99):.1f}  max={max_vals.max():.1f}')
    n_over_100 = (max_vals > 100).sum()
    n_over_1000 = (max_vals > 1000).sum()
    print(f'  Slice con valore massimo > 100: {n_over_100}/{n_total} ({100*n_over_100/n_total:.1f}%)')
    print(f'  Slice con valore massimo > 1000: {n_over_1000}/{n_total} ({100*n_over_1000/n_total:.1f}%)\n')

    # ------------------------------------------------------------------
    # ANALISI 4: dove nasce l'amplificazione? conv2 -> norm2 -> act2
    # ------------------------------------------------------------------
    records_sorted = sorted(records, key=lambda r: -r['max_abs'])
    print(f'=== ANALISI 4: i {args.top_k} casi piu\' estremi, passo per passo ===')
    print(f'{"Paziente":30s} {"slice":>5s} {"canale":>6s} {"conv2":>10s} {"norm2":>12s} {"act2 (out)":>12s}')
    print('-' * 90)
    for r in records_sorted[:args.top_k]:
        print(f'{r["patient"]:30s} {r["slice_idx"]:5d} {r["channel"]:6d} '
              f'{r["conv2_val"]:10.2f} {r["norm2_val"]:12.2f} {r["act2_val"]:12.2f}')

    print()
    print('Interpretazione: confronta la crescita conv2 -> norm2 -> act2 in questi casi.')
    print('Se il salto grande avviene gia\' conv2->norm2, la normalizzazione per-istanza sta')
    print('amplificando (probabile bassa varianza locale in quel canale/istanza). Se invece')
    print('norm2->act2 e\' il salto principale, e\' il quadrato di PolyAct a fare il grosso')
    print('del lavoro su un input gia\' moderatamente elevato.')


if __name__ == '__main__':
    main()