"""
crypto/ablate_decoder_width.py

Estende ablate_dec0_width.py a TUTTI gli stage encoder+decoder, usando un
criterio di importanza per canale che si generalizza correttamente a
qualunque profondita' nella rete (non solo allo stage immediatamente
prima di out_conv, a differenza della stima usata per dec0 in precedenza).

Criterio: Taylor del primo ordine (Molchanov et al., "Pruning Convolutional
Neural Networks for Resource Efficient Inference"), score_c = media su
batch di calibrazione di |attivazione_c * gradiente_loss_rispetto_a_c|,
sommato su (N,H,W). Il gradiente della loss rispetto all'uscita di uno
stage tiene conto AUTOMATICAMENTE di tutto quello che succede a valle
(altri layer, skip connection, out_conv) via backpropagation -- risolve
il limite esplicito lasciato in ablate_dec0_width.py, dove l'importanza
era stimata solo dal peso di out_conv e valida solo per dec0.

Per ogni stage (enc0..enc5, dec0..dec4): calcola l'importanza per canale
su un giro di calibrazione, poi valuta il Dice mascherando a zero i
canali meno importanti a varie frazioni (1.0, 0.75, 0.5, 0.25, 0.125),
UN STAGE ALLA VOLTA (tutti gli altri restano a piena larghezza) -- stessa
logica di ablate_dec0_width.py, generalizzata.

USO:
    python3 crypto/ablate_decoder_width.py \
        --checkpoint results/.../best_model.pth \
        --norm_mode per_instance --weight_standardization --legacy_ws \
        --clamp_values_json crypto/calibrated_clamp_values.json
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from training.dataset import ACDCDataset, load_splits
from training.train import DiceCELoss, dice_score


class LegacyWSConv2d(nn.Conv2d):
    """Vedi crypto/ablate_decoder_stages.py: ricostruisce la vecchia
    WSConv2d (gain libero, pre-fix gain_floor) per checkpoint allenati
    prima del fix."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = nn.Parameter(torch.ones(self.out_channels))

    def forward(self, x):
        w = self.weight
        out_ch = w.shape[0]
        w_flat = w.reshape(out_ch, -1)
        mean = w_flat.mean(dim=1, keepdim=True)
        std = w_flat.std(dim=1, keepdim=True, unbiased=False)
        w_std = (w_flat - mean) / (std + 1e-5)
        w_std = w_std * self.gain.view(-1, 1)
        w_std = w_std.reshape(w.shape)
        return nn.functional.conv2d(x, w_std, self.bias, self.stride,
                                     self.padding, self.dilation, self.groups)


def compute_taylor_importance(model, stage_modules, val_loader, device,
                              criterion, n_calib_batches=10):
    """
    Calcola l'importanza per canale di OGNI stage in stage_modules in un
    solo giro di calibrazione (un forward+backward per batch, non uno per
    stage -- i gradienti di tutti gli stage sono disponibili nella stessa
    backward pass).

    Ritorna {nome_stage: tensore importanza per canale}.
    """
    activations = {name: [] for name in stage_modules}
    grads = {name: [] for name in stage_modules}

    def make_fwd_hook(name):
        def hook(module, inputs, output):
            output.retain_grad()
            activations[name].append(output)  # NON .detach() -- serve il grafo per .grad
        return hook

    # NB: non possiamo catturare .grad su un tensore intermedio non-foglia
    # senza retain_grad(); lo facciamo con un forward hook che marca il
    # tensore, poi leggiamo .grad dopo backward(). Piu' robusto di un
    # backward hook per moduli con piu' input/output.
    handles = [m.register_forward_hook(make_fwd_hook(name))
               for name, m in stage_modules.items()]

    model.train()  # servono i gradienti; niente optimizer.step(), solo backward
    importance_sums = {name: None for name in stage_modules}
    n_batches_done = 0

    for i, (imgs, segs) in enumerate(val_loader):
        if i >= n_calib_batches:
            break
        for name in activations:
            activations[name].clear()
        model.zero_grad()
        imgs, segs = imgs.to(device), segs.to(device)
        logits = model(imgs)
        if not torch.isfinite(logits).all():
            continue  # salta batch che esplodono, non contribuiscono alla stima
        loss = criterion(logits, segs)
        loss.backward()

        for name, m in stage_modules.items():
            act = activations[name][-1]
            grad = act.grad
            if grad is None:
                continue
            score = (act.detach() * grad).abs().sum(dim=[0, 2, 3])
            if importance_sums[name] is None:
                importance_sums[name] = score
            else:
                importance_sums[name] += score

    for h in handles:
        h.remove()
    model.eval()

    return {name: (s / max(1, n_batches_done)).cpu() for name, s in importance_sums.items()}


def evaluate_with_mask(model, stage_module, mask, val_loader, device):
    def hook(module, inputs, output):
        return output * mask.to(output.device).view(1, -1, 1, 1)

    handle = stage_module.register_forward_hook(hook)
    model.eval()
    dice_rv, dice_myo, dice_lv = [], [], []
    n_nan_batches = 0
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            logits = model(imgs)
            if not torch.isfinite(logits).all():
                n_nan_batches += 1
                continue
            preds = logits.argmax(dim=1)
            scores = dice_score(preds, segs)
            dice_rv.append(scores[1]); dice_myo.append(scores[2]); dice_lv.append(scores[3])
    handle.remove()
    if not dice_rv:
        return None
    rv, myo, lv = sum(dice_rv)/len(dice_rv), sum(dice_myo)/len(dice_myo), sum(dice_lv)/len(dice_lv)
    return {'rv': rv, 'myo': myo, 'lv': lv, 'mean': (rv+myo+lv)/3, 'n_nan_batches': n_nan_batches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--norm_mode', default='per_instance', choices=['population', 'per_instance'])
    parser.add_argument('--skip_mode', default='concat', choices=['concat', 'sum'])
    parser.add_argument('--weight_standardization', action='store_true')
    parser.add_argument('--legacy_ws', action='store_true')
    parser.add_argument('--clamp_values_json', default=None)
    parser.add_argument('--fractions', type=float, nargs='+',
                        default=[1.0, 0.75, 0.5, 0.25, 0.125])
    parser.add_argument('--stages', nargs='+',
                        default=['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
                                'dec4', 'dec3', 'dec2', 'dec1', 'dec0'],
                        help='Stage da testare, in ordine di stampa. Default: tutti.')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    if args.legacy_ws:
        hf.WSConv2d = LegacyWSConv2d

    clamp_values = None
    if args.clamp_values_json:
        with open(args.clamp_values_json) as f:
            clamp_values = json.load(f)

    model = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode=args.norm_mode, skip_mode=args.skip_mode,
        weight_standardization=args.weight_standardization,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f'\u26a0\ufe0f  {len(unexpected)} chiavi inattese (verifica --legacy_ws/--weight_standardization/'
              f'--clamp_values_json): {unexpected[:3]}')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    criterion = DiceCELoss(num_classes=4)

    stage_modules = {name: getattr(model, name) for name in args.stages}

    print('Calcolo importanza Taylor per canale su tutti gli stage (10 batch di calibrazione)...')
    importance = compute_taylor_importance(model, stage_modules, val_loader, device, criterion)

    for name in args.stages:
        stage = stage_modules[name]
        score = importance[name]
        if score is None:
            print(f'\n\u26a0\ufe0f  Stage {name}: nessun gradiente valido raccolto (tutti i batch di calibrazione '
                  f'sono esplosi in NaN/Inf?) -- salto.')
            continue
        n_channels = score.shape[0]
        ranking = torch.argsort(score, descending=True)

        print(f'\nStage: {name} ({n_channels} canali totali)')
        print(f'{"Frazione canali":>16s} {"# canali attivi":>16s} {"Dice mean":>10s} '
              f'{"RV":>7s} {"MYO":>7s} {"LV":>7s} {"NaN batch":>10s}')

        for frac in sorted(args.fractions, reverse=True):
            k = max(1, round(n_channels * frac))
            mask = torch.zeros(n_channels)
            mask[ranking[:k]] = 1.0
            result = evaluate_with_mask(model, stage, mask, val_loader, device)
            if result is None:
                print(f'{frac:16.3f} {k:16d}  TUTTI I BATCH ESPLOSI IN NaN/Inf')
            else:
                print(f'{frac:16.3f} {k:16d} {result["mean"]:10.3f} {result["rv"]:7.3f} '
                      f'{result["myo"]:7.3f} {result["lv"]:7.3f} {result["n_nan_batches"]:10d}')


if __name__ == '__main__':
    main()