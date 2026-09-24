"""
models/nstage_unet.py

U-Net generico a N stage encoder/decoder (N=3,4,5,6), skip_mode='sum',
per il test richiesto da Aurora: allenare reti via via piu' corte DA
CAPO, non solo simulare la rimozione su una rete gia' allenata a 6 stage.

Riusa ConvBlock/get_norm da models.he_friendly per comportamento
numerico identico (stessa InstanceNorm, PolyAct, clamp).
"""

import torch
import torch.nn as nn
import models.he_friendly as hf


class NStageUNet(nn.Module):
    def __init__(self, n_stages, filters, clamp_values=None, in_channels=1, num_classes=4):
        """
        n_stages: 3, 4, 5, o 6 -- numero di stage encoder (e decoder,
        simmetrico, n_stages-1 blocchi decoder + il bottleneck).
        filters: lista di n_stages interi, canali per stage encoder
        (l'ultimo e' il bottleneck).
        """
        super().__init__()
        assert len(filters) == n_stages
        self.n_stages = n_stages
        common = dict(norm_type='instance', act_type='poly', norm_mode='per_instance')

        def cv(name):
            if clamp_values is None:
                return None
            k1, k2 = f'{name}.block.2', f'{name}.block.5'
            return (clamp_values[k1], clamp_values[k2]) if k1 in clamp_values else None

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(n_stages):
            stride = 1 if i == 0 else 2
            self.encoders.append(hf.ConvBlock(in_ch, filters[i], stride=stride,
                                               clamp_values=cv(f'enc{i}'), **common))
            in_ch = filters[i]

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n_stages - 2, -1, -1):
            self.ups.append(nn.ConvTranspose2d(filters[i+1], filters[i], 2, stride=2))
            self.decoders.append(hf.ConvBlock(filters[i], filters[i], clamp_values=cv(f'dec{i}'), **common))

        self.out_conv = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        x = skips[-1]
        for j, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            skip_idx = self.n_stages - 2 - j
            x = up(x) + skips[skip_idx]
            x = dec(x)
        return self.out_conv(x)