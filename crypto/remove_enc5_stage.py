"""
crypto/remove_enc5_stage.py

Rimozione STRUTTURALE dello stage enc5 (bottleneck) dal checkpoint
aggressivo+sum, con warm-start DIRETTO (un singolo load_state_dict,
nessuna selezione di canali) e fine-tuning breve.

MOTIVAZIONE: crypto/ablate_full_stage.py ha mostrato che azzerare enc5
costa solo -0.0085 di Dice, il piu' basso di tutti gli 11 stage testati.
Qui si verifica se la rimozione VERA (risparmio reale del costo HE di
quello stage, non solo simulato) regge altrettanto bene dopo fine-tuning.

PERCHE' NON SERVE PRUNING/SELEZIONE DI CANALI: in HEFriendlyUNet, up3 e'
definito come ConvTranspose2d(filters[4], filters[3], ...) -- prende in
ingresso l'output di dec4, che ha filters[4] canali. Ma e4 (l'uscita di
enc4, PRIMA di passare per enc5->up4->dec4) ha GIA' esattamente filters[4]
canali. Collegando up3 direttamente a e4 invece che a d4, tutti gli stage
a valle (up3, dec3, up2, dec2, up1, dec1, up0, dec0, out_conv) hanno
shape e nomi IDENTICI al checkpoint sorgente -- un singolo
load_state_dict(strict=False) li trapianta correttamente, ignorando solo
le chiavi di enc5/up4/dec4 (che nel modello a 5 stage non esistono piu').

ARCHITETTURA RISULTANTE: 5 stage encoder (enc0..enc4, enc4 e' il nuovo
bottleneck) + 4 stage decoder (dec3..dec0) + out_conv. Costruita qui
riusando le classi VERE del modello (ConvBlock, get_norm) importate da
models.he_friendly, non reimplementate -- garantisce comportamento
numerico identico (stessa InstanceNorm, stesso PolyAct, stesso clamp).

LIMITE NOTO: implementato solo per skip_mode='sum', senza Weight
Standardization -- coerente con il checkpoint aggressivo+sum attuale.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits
from training.train import DiceCELoss, dice_score


class UNet5Stage(nn.Module):
    """
    Architettura a 5 stage encoder (enc0..enc4, enc4 e' il nuovo
    bottleneck) + 4 stage decoder (dec3..dec0), skip_mode='sum'.
    Riusa ConvBlock da models.he_friendly -- stessa InstanceNorm
    (per_instance), stesso PolyAct, stesso meccanismo di clamp.

    filters: lista di 5 interi [enc0, enc1, enc2, enc3, enc4] -- NON 6,
    niente enc5. Coerente con filters[:5] del checkpoint sorgente a 6 stage.
    """
    def __init__(self, filters, clamp_values=None, in_channels=1, num_classes=4,
                 skip_mode='sum'):
        super().__init__()
        assert skip_mode == 'sum', "Implementato solo per skip_mode='sum' (vedi docstring)"
        self.skip_mode = skip_mode
        f = filters

        def cv(block_name):
            if clamp_values is None:
                return None
            k1, k2 = f'{block_name}.block.2', f'{block_name}.block.5'
            if k1 in clamp_values and k2 in clamp_values:
                return (clamp_values[k1], clamp_values[k2])
            return None

        common = dict(norm_type='instance', act_type='poly', norm_mode='per_instance')

        self.enc0 = hf.ConvBlock(in_channels, f[0], stride=1, clamp_values=cv('enc0'), **common)
        self.enc1 = hf.ConvBlock(f[0], f[1], stride=2, clamp_values=cv('enc1'), **common)
        self.enc2 = hf.ConvBlock(f[1], f[2], stride=2, clamp_values=cv('enc2'), **common)
        self.enc3 = hf.ConvBlock(f[2], f[3], stride=2, clamp_values=cv('enc3'), **common)
        self.enc4 = hf.ConvBlock(f[3], f[4], stride=2, clamp_values=cv('enc4'), **common)

        # dec_in_mult=1 con skip_mode='sum' (coerente con HEFriendlyUNet)
        self.up3 = nn.ConvTranspose2d(f[4], f[3], 2, stride=2)
        self.dec3 = hf.ConvBlock(f[3], f[3], clamp_values=cv('dec3'), **common)
        self.up2 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec2 = hf.ConvBlock(f[2], f[2], clamp_values=cv('dec2'), **common)
        self.up1 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec1 = hf.ConvBlock(f[1], f[1], clamp_values=cv('dec1'), **common)
        self.up0 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec0 = hf.ConvBlock(f[0], f[0], clamp_values=cv('dec0'), **common)

        self.out_conv = nn.Conv2d(f[0], num_classes, 1)

    def _combine_skip(self, upsampled, skip):
        return upsampled + skip  # skip_mode='sum', vedi assert nel costruttore

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)  # nuovo bottleneck -- niente enc5

        # up3 collegato DIRETTAMENTE a e4 (invece che a d4=dec4(...)) --
        # e4 ha gia' filters[4] canali, la stessa shape che up3 si aspetta.
        d3 = self.dec3(self._combine_skip(self.up3(e4), e3))
        d2 = self.dec2(self._combine_skip(self.up2(d3), e2))
        d1 = self.dec1(self._combine_skip(self.up1(d2), e1))
        d0 = self.dec0(self._combine_skip(self.up0(d1), e0))
        return self.out_conv(d0)


def evaluate_dice(model, val_loader, device):
    model.eval()
    dice_rv, dice_myo, dice_lv = [], [], []
    n_exploded = 0
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            logits = model(imgs)
            if not torch.isfinite(logits).all():
                n_exploded += 1
                continue
            preds = logits.argmax(dim=1)
            scores = dice_score(preds, segs)
            dice_rv.append(scores[1]); dice_myo.append(scores[2]); dice_lv.append(scores[3])
    if not dice_rv:
        return None, n_exploded
    mean_dice = (sum(dice_rv)/len(dice_rv) + sum(dice_myo)/len(dice_myo) + sum(dice_lv)/len(dice_lv)) / 3
    return mean_dice, n_exploded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint sorgente a 6 stage, skip_mode=sum (es. aggressivo+sum, Dice 0.863)')
    parser.add_argument('--filters', type=int, nargs=6, required=True,
                        help='I 6 filters del checkpoint SORGENTE (usati solo per costruire e '
                             'caricare il modello sorgente a 6 stage, servono i suoi pesi grezzi)')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--out', default='crypto/removed_enc5_warmstart.pth')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    # Carichiamo il checkpoint sorgente solo per avere il suo state_dict
    # in memoria -- non serve nemmeno costruire il modello completo,
    # basta lo state_dict grezzo.
    source_state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f'Checkpoint sorgente caricato: {args.checkpoint}')

    filters_5stage = args.filters[:5]  # scarta filters[5] (enc5, non piu' usato)
    print(f'Filters modello a 5 stage: {filters_5stage} (enc5 rimosso)')

    target = UNet5Stage(filters_5stage, clamp_values=clamp_values, skip_mode='sum').to(device)
    n_params = sum(p.numel() for p in target.parameters())
    print(f'Modello a 5 stage costruito: {n_params:,} parametri')

    missing, unexpected = target.load_state_dict(source_state, strict=False)
    print(f'\nCaricamento diretto (nessuna selezione di canali):')
    print(f'  Chiavi mancanti nel target (dovrebbero essere 0): {len(missing)}')
    if missing:
        print(f'    {missing}')
    print(f'  Chiavi ignorate dalla sorgente (attese: tutte enc5.*/up4.*/dec4.*): {len(unexpected)}')
    unexpected_ok = all(k.startswith(('enc5.', 'up4.', 'dec4.')) for k in unexpected)
    if not unexpected_ok:
        print(f'  \u26a0\ufe0f  ATTENZIONE: alcune chiavi ignorate non riguardano enc5/up4/dec4 -- '
              f'controllare: {[k for k in unexpected if not k.startswith(("enc5.", "up4.", "dec4."))]}')
    else:
        print(f'  OK: tutte le chiavi ignorate riguardano solo enc5/up4/dec4, come atteso.')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    print('\nVerifica rapida su validation set (nessun fine-tuning ancora)...')
    dice, n_exploded = evaluate_dice(target, val_loader, device)
    if dice is not None:
        print(f'Dice del modello a 5 stage (prima del fine-tuning): {dice:.4f}')
    else:
        print(f'\u26a0\ufe0f  Tutti i batch sono esplosi in NaN/Inf ({n_exploded}) -- qualcosa non torna.')

    torch.save(target.state_dict(), args.out)
    print(f'\nCheckpoint salvato in: {args.out}')
    print(f'Pronto per il fine-tuning con: --init_from {args.out}')


if __name__ == '__main__':
    main()