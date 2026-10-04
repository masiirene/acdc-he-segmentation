"""
crypto/remove_deep_stages.py

Versione corretta: usa attributi nominati esplicitamente (enc0, enc1,
dec2, up2, ecc. -- stessi nomi ESATTI del checkpoint a 6 stage) invece
di nn.ModuleList, che genera nomi automatici (encoders.0, ups.0) senza
corrispondenza con lo state_dict sorgente. Questo e' l'errore che ha
causato missing=104/unexpected=164 nel primo tentativo -- va rifatto
il salvataggio, il file gia' prodotto NON e' valido.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, '.')
import models.he_friendly as hf

ORIGINAL_FILTERS = [32, 64, 128, 256, 128, 64]  # enc0..enc5, checkpoint aggressivo+sum


class UNetKStage(nn.Module):
    def __init__(self, k, filters, clamp_values=None, in_channels=1, num_classes=4):
        super().__init__()
        assert len(filters) == k
        self.k = k
        common = dict(norm_type='instance', act_type='poly', norm_mode='per_instance')

        def cv(name):
            if clamp_values is None:
                return None
            k1, k2 = f'{name}.block.2', f'{name}.block.5'
            return (clamp_values[k1], clamp_values[k2]) if k1 in clamp_values else None

        # Encoder: nomi enc0, enc1, ..., enc(k-1) -- IDENTICI al checkpoint sorgente
        self.enc_names = [f'enc{i}' for i in range(k)]
        in_ch = in_channels
        for i, name in enumerate(self.enc_names):
            stride = 1 if i == 0 else 2
            setattr(self, name, hf.ConvBlock(in_ch, filters[i], stride=stride,
                                              clamp_values=cv(name), **common))
            in_ch = filters[i]

        # Decoder + upsampling: nomi dec(k-2)...dec0, up(k-2)...up0 -- IDENTICI al sorgente
        self.dec_names = [f'dec{i}' for i in range(k - 2, -1, -1)]
        self.up_names = [f'up{i}' for i in range(k - 2, -1, -1)]
        for i, (dec_name, up_name) in enumerate(zip(self.dec_names, self.up_names)):
            stage_idx = k - 2 - i
            setattr(self, up_name, nn.ConvTranspose2d(filters[stage_idx + 1], filters[stage_idx], 2, stride=2))
            setattr(self, dec_name, hf.ConvBlock(filters[stage_idx], filters[stage_idx],
                                                  clamp_values=cv(dec_name), **common))

        self.out_conv = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for name in self.enc_names:
            x = getattr(self, name)(x)
            skips.append(x)
        x = skips[-1]
        for i, (up_name, dec_name) in enumerate(zip(self.up_names, self.dec_names)):
            stage_idx = self.k - 2 - i
            up_out = getattr(self, up_name)(x)
            x = up_out + skips[stage_idx]
            x = getattr(self, dec_name)(x)
        return self.out_conv(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint ORIGINALE a 6 stage, skip_mode=sum')
    parser.add_argument('--k', type=int, required=True, choices=[3, 4, 5])
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    source_state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    filters_k = ORIGINAL_FILTERS[:args.k]
    print(f'k={args.k}, filters mantenuti: {filters_k}')

    target = UNetKStage(args.k, filters_k, clamp_values=clamp_values).to(device)
    n_params = sum(p.numel() for p in target.parameters())
    print(f'Modello costruito: {n_params:,} parametri')

    missing, unexpected = target.load_state_dict(source_state, strict=False)
    print(f'missing={len(missing)} (atteso 0)')
    if missing:
        print(f'  Prime chiavi mancanti: {missing[:10]}')
    print(f'unexpected={len(unexpected)} (atteso: solo le chiavi degli stage rimossi)')
    if unexpected:
        removed_stage_names = ([f'enc{i}.' for i in range(args.k, 6)] +
                               [f'dec{i}.' for i in range(args.k - 1, 5)] +
                               [f'up{i}.' for i in range(args.k - 1, 5)])
        bad_unexpected = [u for u in unexpected if not any(u.startswith(p) for p in removed_stage_names)]
        if bad_unexpected:
            print(f'  \u26a0\ufe0f  Chiavi impreviste NON relative agli stage rimossi: {bad_unexpected[:10]}')
        else:
            print(f'  OK: tutte le chiavi impreviste sono relative agli stage rimossi, come atteso.')

    torch.save(target.state_dict(), args.out)
    print(f'Salvato: {args.out}')


if __name__ == '__main__':
    main()