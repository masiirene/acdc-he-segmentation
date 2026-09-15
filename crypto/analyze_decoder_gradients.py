"""
crypto/analyze_decoder_gradients.py

Risposta alla richiesta di Aurora: analizzare i GRADIENTI (non solo i
valori nel forward, come fatto finora) per capire se il contributo
dell'upsampling e quello della skip connection, in ogni stage del
decoder, hanno ordini di grandezza molto diversi -- il che squilibrerebbe
quanto l'output finale "ascolta" i due percorsi.

PRECISAZIONE IMPORTANTE: in una U-Net, isolare perfettamente il
contributo della sola skip connection da quello del resto della catena
encoder non è possibile in senso stretto -- l'encoder alimenta anche il
bottleneck, che alimenta tutto il decoder. Quello che misuriamo qui è
concreto e ben definito: il gradiente che arriva a) all'output del ramo
di upsampling (up_i) e b) all'output dell'encoder allo stesso stage (e_i,
il tensore che viene concatenato), nel punto ESATTO in cui i due vengono
uniti (torch.cat). Non è "tutto il contributo storico dell'encoder", è
il gradiente locale in quel punto preciso del grafo -- la domanda che
serve per capire lo squilibrio nella concatenazione.

LOSS: DiceCELoss (DiceLoss + CrossEntropyLoss), copiata identica da
training/train.py -- non un'approssimazione, è la vera loss usata in
tutti i run di questa settimana. Nota: DiceLoss media su TUTTE le 4
classi incluso lo sfondo (classe 0), non solo le 3 anatomiche.

USO:
    python3 -m crypto.analyze_decoder_gradients --checkpoint <path> \\
        --clamp_values_json crypto/calibrated_clamp_values.json
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models.he_friendly import HEFriendlyUNet, PolyAct
from training.dataset import ACDCDataset, load_splits


class DiceLoss(torch.nn.Module):
    """Copiata identica da training/train.py, per usare la VERA loss del
    progetto invece di un'approssimazione. Nota: media su TUTTE le 4
    classi (incluso lo sfondo, classe 0), non solo le 3 anatomiche."""
    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(self.num_classes):
            p = probs[:, c]
            t = (targets == c).float()
            intersection = (p * t).sum()
            loss = loss + 1 - (2 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth)
        return loss / self.num_classes


class DiceCELoss(torch.nn.Module):
    """Copiata identica da training/train.py -- la loss REALMENTE usata
    in tutti i run di questa settimana."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.ce(logits, targets)


def run_gradient_analysis(model, val_loader, device, criterion, n_batches=5):
    """Ritorna, per ogni stage del decoder, la norma L2 media del
    gradiente su up_i (upsampling) e su e_i (skip, encoder allo stesso
    stage), calcolate al punto di concatenazione."""

    up_names = ['up4', 'up3', 'up2', 'up1', 'up0']
    enc_names = ['enc4', 'enc3', 'enc2', 'enc1', 'enc0']  # skip corrispondenti

    captured = {}

    def make_hook(key):
        def hook(module, inp, out):
            out.retain_grad()
            captured[key] = out
        return hook

    handles = []
    for name in up_names:
        handles.append(getattr(model, name).register_forward_hook(make_hook(name)))
    for name in enc_names:
        # l'output dell'encoder e' l'output dell'intero ConvBlock (il modulo stesso)
        handles.append(getattr(model, name).register_forward_hook(make_hook(name)))

    grad_norms = {k: [] for k in up_names + enc_names}

    model.train()  # servono i gradienti; per_instance norm si comporta uguale in train/eval
    it = iter(val_loader)
    for _ in range(n_batches):
        try:
            imgs, segs = next(it)
        except StopIteration:
            break
        imgs, segs = imgs.to(device), segs.to(device)

        model.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, segs)
        loss.backward()

        for name in up_names + enc_names:
            t = captured[name]
            if t.grad is not None:
                grad_norms[name].append(t.grad.norm().item())

    for h in handles:
        h.remove()

    # Media sui batch testati
    result = {}
    for name in up_names + enc_names:
        vals = grad_norms[name]
        result[name] = sum(vals) / len(vals) if vals else float('nan')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_batches', type=int, default=5,
                        help='Numero di batch su cui mediare i gradienti (piu\' alto = stima piu\' stabile)')
    parser.add_argument('--clamp_values_json', default=None,
                        help='Soglie calibrate per la configurazione "rete reale/approssimata". '
                             'Se omesso, la configurazione "reale" usa clamp fisso a 50.')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}\n')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    def load_fresh_model():
        model = HEFriendlyUNet(in_channels=1, num_classes=4, act_type='poly',
                               norm_type='instance', norm_mode='per_instance').to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        return model

    configs = {}

    # --- Configurazione 1: rete "approssimata" (clamp reale/calibrato) ---
    model_clamped = load_fresh_model()
    if args.clamp_values_json:
        with open(args.clamp_values_json) as f:
            clamp_values = json.load(f)
        for name, m in model_clamped.named_modules():
            if isinstance(m, PolyAct) and name in clamp_values:
                m.clamp_value = clamp_values[name]
    print('Configurazione "approssimata" (clamp reale/calibrato): calcolo gradienti...')
    criterion = DiceCELoss(num_classes=4)
    configs['clamp_on'] = run_gradient_analysis(model_clamped, val_loader, device, criterion, args.n_batches)

    # --- Configurazione 2: rete "non approssimata" (clamp disattivato) ---
    model_unclamped = load_fresh_model()
    for name, m in model_unclamped.named_modules():
        if isinstance(m, PolyAct):
            m.clamp_value = float('inf')
    print('Configurazione "non approssimata" (clamp disattivato): calcolo gradienti...\n')
    configs['clamp_off'] = run_gradient_analysis(model_unclamped, val_loader, device, criterion, args.n_batches)

    # --- Tabella riassuntiva ---
    stages = [
        ('dec4 (bottleneck->16x14)', 'up4', 'enc4'),
        ('dec3', 'up3', 'enc3'),
        ('dec2', 'up2', 'enc2'),
        ('dec1', 'up1', 'enc1'),
        ('dec0 (output finale)', 'up0', 'enc0'),
    ]

    for label, cfg in [('CLAMP ATTIVO (rete approssimata/reale)', 'clamp_on'),
                        ('CLAMP DISATTIVATO (rete non approssimata)', 'clamp_off')]:
        print(f'=== {label} ===')
        print(f'{"Stage":30s} {"|grad upsampling|":>18s} {"|grad skip|":>14s} {"rapporto up/skip":>18s}')
        print('-' * 85)
        for stage_label, up_key, enc_key in stages:
            g_up = configs[cfg][up_key]
            g_skip = configs[cfg][enc_key]
            ratio = g_up / g_skip if g_skip > 0 else float('nan')
            print(f'{stage_label:30s} {g_up:18.4e} {g_skip:14.4e} {ratio:18.2f}')
        print()

    print('=== Confronto clamp ON vs OFF (quanto si discosta la rete approssimata) ===')
    print(f'{"Stage":30s} {"up: on/off":>14s} {"skip: on/off":>14s}')
    print('-' * 65)
    for stage_label, up_key, enc_key in stages:
        up_ratio = configs['clamp_on'][up_key] / configs['clamp_off'][up_key] \
            if configs['clamp_off'][up_key] > 0 else float('nan')
        skip_ratio = configs['clamp_on'][enc_key] / configs['clamp_off'][enc_key] \
            if configs['clamp_off'][enc_key] > 0 else float('nan')
        print(f'{stage_label:30s} {up_ratio:14.2f} {skip_ratio:14.2f}')

    print()
    print('Interpretazione:')
    print('- "rapporto up/skip" molto diverso da 1 in uno stage = squilibrio in quel punto')
    print('  specifico -- l\'output "ascolta" un percorso molto piu\' dell\'altro.')
    print('- Se il rapporto up/skip cambia MOLTO tra clamp ON e OFF nello stesso stage,')
    print('  il clamp sta alterando l\'equilibrio tra i due percorsi, non solo il valore')
    print('  assoluto -- rilevante per capire se il clamp e\' parte del problema o della')
    print('  soluzione qui.')


if __name__ == '__main__':
    main()