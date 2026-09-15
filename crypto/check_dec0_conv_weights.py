"""
crypto/check_dec0_conv_weights.py

Segue la catena di indagine su dec0.block.5: normalizzazione esclusa
(std_c sempre normale, mai vicina a zero), gamma/beta esclusi (variano
poco). La causa e' un valore GREZZO enorme gia' in uscita da dec0.block.3
(la seconda convoluzione del blocco) su pochi pixel isolati.

Verifica diretta: i pesi di quella convoluzione, per i canali di OUTPUT
identificati come sospetti (15, 4, 29, 16, 8, 31 -- vedi
check_dec0_normalized_value.py), hanno un valore anomalo (un singolo peso
del kernel molto piu' grande degli altri, o l'intero canale con pesi
sistematicamente piu' grandi) rispetto agli altri canali di output dello
stesso layer?

USO:
    python3 -m crypto.check_dec0_conv_weights --checkpoint <path>
"""

import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--suspect_out_channels', default='15,4,29,16,8,31',
                        help='Canali di OUTPUT identificati come sospetti in '
                             'check_dec0_normalized_value.py')
    args = parser.parse_args()

    device = torch.device('cpu')
    model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly',
                           norm_type='instance', norm_mode='per_instance').to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)

    # dec0.block[3] e' la seconda Conv2d del blocco dec0 (conv2, quella
    # che alimenta norm2 -> act2 = dec0.block.5)
    weight = model.dec0.block[3].weight.detach().numpy()  # (Cout=32, Cin=64, K=3, K=3)
    bias = model.dec0.block[3].bias.detach().numpy()      # (Cout=32,)

    suspects = [int(c) for c in args.suspect_out_channels.split(',')]

    print(f'Pesi di dec0.block.3 (seconda conv del blocco dec0) -- shape {weight.shape}\n')

    # Statistiche per canale di OUTPUT: norma L2 di tutti i pesi che
    # contribuiscono a quel canale, e valore massimo assoluto di un
    # singolo peso in quel canale.
    n_out = weight.shape[0]
    per_channel_norm = np.zeros(n_out)
    per_channel_max = np.zeros(n_out)
    for co in range(n_out):
        per_channel_norm[co] = np.linalg.norm(weight[co])
        per_channel_max[co] = np.abs(weight[co]).max()

    print('Statistiche su TUTTI i 32 canali di output:')
    print(f'  norma pesi   -- media={per_channel_norm.mean():.3f}  mediana={np.median(per_channel_norm):.3f}  '
          f'p90={np.percentile(per_channel_norm,90):.3f}  max={per_channel_norm.max():.3f}')
    print(f'  peso singolo -- media={per_channel_max.mean():.3f}  mediana={np.median(per_channel_max):.3f}  '
          f'p90={np.percentile(per_channel_max,90):.3f}  max={per_channel_max.max():.3f}')
    print(f'  bias         -- media={np.abs(bias).mean():.3f}  max={np.abs(bias).max():.3f}\n')

    norm_sorted = np.sort(per_channel_norm)
    max_sorted = np.sort(per_channel_max)

    print(f'{"Canale":>8s} {"norma pesi":>12s} {"pctile":>8s} {"peso max":>10s} {"pctile":>8s} {"bias":>8s}  sospetto?')
    print('-' * 80)
    for co in range(n_out):
        if co not in suspects and per_channel_norm[co] < np.percentile(per_channel_norm, 85):
            continue
        pct_norm = 100 * np.searchsorted(norm_sorted, per_channel_norm[co]) / n_out
        pct_max = 100 * np.searchsorted(max_sorted, per_channel_max[co]) / n_out
        marker = '  <-- SOSPETTO' if co in suspects else ''
        print(f'{co:8d} {per_channel_norm[co]:12.3f} {pct_norm:7.1f}% {per_channel_max[co]:10.3f} '
              f'{pct_max:7.1f}% {bias[co]:8.3f}{marker}')

    print()
    print('Interpretazione: se i canali sospetti hanno norma/peso massimo molto sopra il')
    print('90 percentile, i pesi di QUEI canali sono davvero anomali (spiegherebbe i pixel')
    print('isolati enormi). Se invece sono nella norma, il problema non e\' nei pesi in se\'')
    print('ma in un input particolare che alimenta la convoluzione (dec0.block.2, la prima')
    print('PolyAct del blocco) -- da verificare come prossimo passo.')


if __name__ == '__main__':
    main()