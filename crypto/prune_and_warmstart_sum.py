"""
crypto/prune_and_warmstart_sum.py

Structured channel pruning con warm-start, per il checkpoint a piena
larghezza skip_mode='sum' (Dice ~0.854, results/test_skip_sum_trial50/...,
NESSUNA Weight Standardization -- verificato dal nome della cartella,
nessun suffisso _ws).

DIFFERENZA CHIAVE rispetto a prune_and_warmstart.py (versione per
skip_mode='concat'):

Con concat, encoder e decoder restano indipendenti: l'output di up4 e lo
skip di enc4 vengono AFFIANCATI (concatenati), quindi si puo' scegliere i
migliori canali di ciascuno con un criterio indipendente.

Con sum, invece, l'output di up4 e lo skip di enc4 vengono SOMMATI canale
per canale (il canale i di uno con il canale i dell'altro). Nella rete
originale esiste quindi una corrispondenza appresa specifica tra "canale i
di up4" e "canale i di enc4". Se scegliessimo i migliori canali di up4 e
di enc4 INDIPENDENTEMENTE, rischieremmo di sommare canali che nella rete
allenata non erano mai stati pensati per stare insieme, rompendo
l'informazione che il trapianto dovrebbe preservare.

SOLUZIONE: l'importanza Taylor per la selezione di enc4/up4 viene
calcolata sul tensore GIA' SOMMATO (l'input di dec4.block.0, catturato
con un forward-pre-hook su dec4) -- e lo STESSO identico set di indici
risultante viene usato per selezionare sia i canali di enc4 sia i canali
di up4. Nessun offset "+512" necessario (a differenza della versione
concat): con sum i canali restano nello stesso spazio, non raddoppiano.

CORREZIONE rispetto alla prima versione di questo script: l'OUTPUT di
dec4 (e di conseguenza l'input di up3) e' un tensore completamente
DIVERSO da enc4/up4 -- e' prodotto DOPO le convoluzioni di dec4, senza
alcun vincolo di corrispondenza con i canali in ingresso. Riusare
l'indice condiviso enc4/up4 anche li' era un errore (causava un crollo
di Dice pre-finetuning molto piu' marcato del previsto sulla
configurazione aggressiva: 0.509 invece di un valore vicino a 0.70).
Corretto aggiungendo un terzo hook, questa volta sull'OUTPUT di dec4, e
un indice dedicato (idx_dec4_out) usato SOLO per dec4.block[3],
dec4.block[4] e up3 -- mai per enc4/up4.

Il resto della logica (importanza L1 per i confini interni, trapianto
diretto dei pesi, fine-tuning breve) e' identico alla versione concat.
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


def copy_unchanged_submodule(src, tgt):
    """Copia ricorsivamente i parametri da src a tgt per un sottoalbero
    con la STESSA architettura (nessun pruning). Il checkpoint sum puro
    usa nn.Conv2d normali, non WSConv2d -- copia diretta, nessuna
    conversione di gain necessaria (a differenza della versione concat)."""
    src_modules = dict(src.named_modules())
    tgt_modules = dict(tgt.named_modules())
    assert src_modules.keys() == tgt_modules.keys(), (
        f"Struttura diversa tra sorgente e target: {set(src_modules) ^ set(tgt_modules)}"
    )
    for name, src_mod in src_modules.items():
        tgt_mod = tgt_modules[name]
        if isinstance(src_mod, (nn.Conv2d, nn.ConvTranspose2d)):
            tgt_mod.weight.data.copy_(src_mod.weight.data)
            if src_mod.bias is not None:
                tgt_mod.bias.data.copy_(src_mod.bias.data)
        elif isinstance(src_mod, nn.InstanceNorm2d):
            if src_mod.weight is not None:
                tgt_mod.weight.data.copy_(src_mod.weight.data)
                tgt_mod.bias.data.copy_(src_mod.bias.data)
        elif hasattr(src_mod, 'a') and hasattr(src_mod, 'b') and hasattr(src_mod, 'c'):
            tgt_mod.a.data.copy_(src_mod.a.data)
            tgt_mod.b.data.copy_(src_mod.b.data)
            tgt_mod.c.data.copy_(src_mod.c.data)


def l1_importance_conv_output(conv):
    return conv.weight.detach().abs().sum(dim=[1, 2, 3])


def top_k_sorted(scores, k):
    idx = torch.topk(scores, k).indices
    return torch.sort(idx).values


def compute_taylor_importance_sum(model, val_loader, device, criterion, n_calib_batches=10):
    """
    Importanza Taylor (|attivazione * gradiente|) per TRE confini distinti:

    - 'enc5': hook forward NORMALE (post) su model.enc5 -- confine
      sequenziale standard, enc5 alimenta up4 in cascata, nessuna
      questione di pairing (enc5 non partecipa a nessuna somma).
    - 'dec4_input': hook forward-PRE su model.dec4 -- cattura il tensore
      GIA' SOMMATO (up4_out + enc4_skip) prima che entri in dec4.block.0.
      Questo E' il segnale di importanza sia per i canali di enc4 sia per
      i canali di up4: essendo sommati, condividono lo stesso "significato"
      di canale, quindi condividono anche lo stesso set di indici scelti.
    - 'dec4_output': hook forward NORMALE (post) su model.dec4 -- cattura
      l'output di dec4, DOPO le sue convoluzioni. E' un tensore diverso
      da 'dec4_input', senza alcun vincolo di corrispondenza con
      enc4/up4 -- serve un indice SEPARATO per selezionare i canali di
      uscita di dec4 (e di conseguenza l'ingresso di up3).
    """
    activations = {}

    def make_post_hook(name):
        def hook(module, inputs, output):
            output.retain_grad()
            activations.setdefault(name, []).append(output)
        return hook

    def make_pre_hook(name):
        def hook(module, inputs):
            x = inputs[0]
            x.retain_grad()
            activations.setdefault(name, []).append(x)
        return hook

    handles = [
        model.enc5.register_forward_hook(make_post_hook('enc5')),
        model.dec4.register_forward_pre_hook(make_pre_hook('dec4_input')),
        model.dec4.register_forward_hook(make_post_hook('dec4_output')),
    ]

    model.train()
    importance_sums = {'enc5': None, 'dec4_input': None, 'dec4_output': None}
    n_batches_done = 0

    for i, (imgs, segs) in enumerate(val_loader):
        if i >= n_calib_batches:
            break
        activations.clear()
        model.zero_grad()
        imgs, segs = imgs.to(device), segs.to(device)
        logits = model(imgs)
        if not torch.isfinite(logits).all():
            continue
        loss = criterion(logits, segs)
        loss.backward()

        for name in importance_sums:
            act = activations[name][-1]
            grad = act.grad
            if grad is None:
                continue
            score = (act.detach() * grad).abs().sum(dim=[0, 2, 3])
            if importance_sums[name] is None:
                importance_sums[name] = score
            else:
                importance_sums[name] += score
        n_batches_done += 1

    for h in handles:
        h.remove()
    model.eval()

    return {name: (s / max(1, n_batches_done)).cpu() for name, s in importance_sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint a piena larghezza skip_mode=sum da cui fare pruning '
                             '(es. Dice 0.854, results/test_skip_sum_trial50/.../best_model.pth)')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clamp_values_json', default='crypto/calibrated_clamp_values.json')
    parser.add_argument('--width4', type=int, required=True,
                        help='Larghezza target per enc4/up4 (confine condiviso) E per l\'output '
                             'di dec4 (confine separato, vedi docstring del modulo)')
    parser.add_argument('--width5', type=int, required=True,
                        help='Larghezza target per enc5 (filters[5], indipendente da width4)')
    parser.add_argument('--out', default='crypto/pruned_sum_warmstart.pth')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    # --- Carica il checkpoint SORGENTE (piena larghezza, skip_mode=sum, NO WS) ---
    source = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode='sum',
        weight_standardization=False, filters=[32, 64, 128, 256, 512, 512],
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = source.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'\u26a0\ufe0f  missing={len(missing)} unexpected={len(unexpected)} '
              f'(dovrebbero essere 0 -- verifica che il checkpoint sia quello giusto)')
    source.eval()
    print(f'Checkpoint sorgente caricato: {args.checkpoint}')

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    criterion = DiceCELoss(num_classes=4)

    print('Calcolo importanza Taylor per enc5, per il confine sommato enc4+up4, e per l\'output di dec4...')
    taylor = compute_taylor_importance_sum(source, val_loader, device, criterion)

    K4, K5 = args.width4, args.width5
    # Indice CONDIVISO tra enc4 e up4 -- CRITICO per sum (vedi docstring)
    idx_enc4_up4_shared = top_k_sorted(taylor['dec4_input'], K4).to(device)
    idx_enc5_out = top_k_sorted(taylor['enc5'], K5).to(device)
    # Indice SEPARATO per l'output di dec4 (e quindi l'input di up3) --
    # NON deve mai essere confuso con idx_enc4_up4_shared (vedi docstring).
    idx_dec4_out = top_k_sorted(taylor['dec4_output'], K4).to(device)
    print(f'Canali condivisi enc4/up4 selezionati (Taylor sul tensore sommato): {len(idx_enc4_up4_shared)}')
    print(f'Canali enc5 selezionati (Taylor): {len(idx_enc5_out)}')
    print(f'Canali output dec4 selezionati (Taylor, indipendenti da enc4/up4): {len(idx_dec4_out)}')

    # Importanza L1 per i confini INTERNI (dentro ogni ConvBlock, nessuna
    # questione di pairing -- puramente sequenziali)
    idx_enc4_mid = top_k_sorted(l1_importance_conv_output(source.enc4.block[0]), K4)
    idx_enc5_mid = top_k_sorted(l1_importance_conv_output(source.enc5.block[0]), K5)
    idx_dec4_mid = top_k_sorted(l1_importance_conv_output(source.dec4.block[0]), K4)

    # --- Costruisci il modello TARGET (larghezza ridotta, skip_mode=sum) ---
    target_filters = [32, 64, 128, 256, K4, K5]
    target = hf.HEFriendlyUNet(
        in_channels=1, num_classes=4, act_type='poly', norm_type='instance',
        clamp_values=clamp_values, norm_mode='per_instance', skip_mode='sum',
        weight_standardization=False, filters=target_filters,
    ).to(device)
    print(f'Modello target costruito: filters={target_filters}, '
          f'params={sum(p.numel() for p in target.parameters()):,}')

    # --- Stage NON toccati: copia diretta 1:1 ---
    for stage_name in ['enc0', 'enc1', 'enc2', 'enc3',
                       'dec3', 'dec2', 'dec1', 'dec0',
                       'up0', 'up1', 'up2']:
        copy_unchanged_submodule(getattr(source, stage_name), getattr(target, stage_name))
    target.out_conv.load_state_dict(source.out_conv.state_dict())

    with torch.no_grad():
        # ================= enc4 =================
        # block.0: input pieno (256 da enc3, invariato), output-mid pruning -> K4
        b = source.enc4.block[0]
        target.enc4.block[0].weight.copy_(b.weight[idx_enc4_mid])
        target.enc4.block[0].bias.copy_(b.bias[idx_enc4_mid])
        target.enc4.block[1].weight.copy_(source.enc4.block[1].weight[idx_enc4_mid])
        target.enc4.block[1].bias.copy_(source.enc4.block[1].bias[idx_enc4_mid])
        target.enc4.block[2].a.copy_(source.enc4.block[2].a)
        target.enc4.block[2].b.copy_(source.enc4.block[2].b)
        target.enc4.block[2].c.copy_(source.enc4.block[2].c)
        # block.3: output = idx_enc4_up4_shared (deve corrispondere esattamente
        # a up4 per la somma -- vedi docstring)
        b = source.enc4.block[3]
        target.enc4.block[3].weight.copy_(b.weight[idx_enc4_up4_shared][:, idx_enc4_mid])
        target.enc4.block[3].bias.copy_(b.bias[idx_enc4_up4_shared])
        target.enc4.block[4].weight.copy_(source.enc4.block[4].weight[idx_enc4_up4_shared])
        target.enc4.block[4].bias.copy_(source.enc4.block[4].bias[idx_enc4_up4_shared])
        target.enc4.block[5].a.copy_(source.enc4.block[5].a)
        target.enc4.block[5].b.copy_(source.enc4.block[5].b)
        target.enc4.block[5].c.copy_(source.enc4.block[5].c)

        # ================= enc5 =================
        # block.0: input = enc4 output (idx_enc4_up4_shared), output-mid -> K5
        b = source.enc5.block[0]
        target.enc5.block[0].weight.copy_(b.weight[idx_enc5_mid][:, idx_enc4_up4_shared])
        target.enc5.block[0].bias.copy_(b.bias[idx_enc5_mid])
        target.enc5.block[1].weight.copy_(source.enc5.block[1].weight[idx_enc5_mid])
        target.enc5.block[1].bias.copy_(source.enc5.block[1].bias[idx_enc5_mid])
        target.enc5.block[2].a.copy_(source.enc5.block[2].a)
        target.enc5.block[2].b.copy_(source.enc5.block[2].b)
        target.enc5.block[2].c.copy_(source.enc5.block[2].c)
        b = source.enc5.block[3]
        target.enc5.block[3].weight.copy_(b.weight[idx_enc5_out][:, idx_enc5_mid])
        target.enc5.block[3].bias.copy_(b.bias[idx_enc5_out])
        target.enc5.block[4].weight.copy_(source.enc5.block[4].weight[idx_enc5_out])
        target.enc5.block[4].bias.copy_(source.enc5.block[4].bias[idx_enc5_out])
        target.enc5.block[5].a.copy_(source.enc5.block[5].a)
        target.enc5.block[5].b.copy_(source.enc5.block[5].b)
        target.enc5.block[5].c.copy_(source.enc5.block[5].c)

        # ================= up4 (ConvTranspose2d, no WS) =================
        # input = enc5 output (idx_enc5_out), OUTPUT = idx_enc4_up4_shared
        # (STESSO set di enc4 -- e' la corrispondenza che rende sum corretta)
        w = source.up4.weight[idx_enc5_out]              # seleziona input (dim0)
        w = w[:, idx_enc4_up4_shared]                      # seleziona output (dim1)
        target.up4.weight.copy_(w)
        target.up4.bias.copy_(source.up4.bias[idx_enc4_up4_shared])

        # ================= dec4 =================
        # block.0: input = SOMMA (up4_out + enc4_skip), entrambi gia' ridotti a K4
        # canali NELLO STESSO SPAZIO -- l'input di dec4.block.0 ha gia' K4 canali,
        # nessun offset "+512" necessario (a differenza di concat)
        b = source.dec4.block[0]
        target.dec4.block[0].weight.copy_(b.weight[idx_dec4_mid][:, idx_enc4_up4_shared])
        target.dec4.block[0].bias.copy_(b.bias[idx_dec4_mid])
        target.dec4.block[1].weight.copy_(source.dec4.block[1].weight[idx_dec4_mid])
        target.dec4.block[1].bias.copy_(source.dec4.block[1].bias[idx_dec4_mid])
        target.dec4.block[2].a.copy_(source.dec4.block[2].a)
        target.dec4.block[2].b.copy_(source.dec4.block[2].b)
        target.dec4.block[2].c.copy_(source.dec4.block[2].c)
        # block.3: OUTPUT = idx_dec4_out (indice SEPARATO, mai idx_enc4_up4_shared
        # -- questo e' il bug corretto rispetto alla prima versione)
        b = source.dec4.block[3]
        target.dec4.block[3].weight.copy_(b.weight[idx_dec4_out][:, idx_dec4_mid])
        target.dec4.block[3].bias.copy_(b.bias[idx_dec4_out])
        target.dec4.block[4].weight.copy_(source.dec4.block[4].weight[idx_dec4_out])
        target.dec4.block[4].bias.copy_(source.dec4.block[4].bias[idx_dec4_out])
        target.dec4.block[5].a.copy_(source.dec4.block[5].a)
        target.dec4.block[5].b.copy_(source.dec4.block[5].b)
        target.dec4.block[5].c.copy_(source.dec4.block[5].c)

        # ================= up3 (input = output di dec4, cioe' idx_dec4_out) =================
        target.up3.weight.copy_(source.up3.weight[idx_dec4_out])
        target.up3.bias.copy_(source.up3.bias)  # output invariato (filters[3]=256)

    print('Pruning completato. Verifica rapida su validation set (nessun fine-tuning ancora)...')
    target.eval()
    dice_rv, dice_myo, dice_lv = [], [], []
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            logits = target(imgs)
            if not torch.isfinite(logits).all():
                continue
            preds = logits.argmax(dim=1)
            scores = dice_score(preds, segs)
            dice_rv.append(scores[1]); dice_myo.append(scores[2]); dice_lv.append(scores[3])
    if dice_rv:
        mean_dice = (sum(dice_rv)/len(dice_rv) + sum(dice_myo)/len(dice_myo) + sum(dice_lv)/len(dice_lv)) / 3
        print(f'Dice del modello PRUNATO (prima del fine-tuning): {mean_dice:.3f}')
    else:
        print('\u26a0\ufe0f  Tutti i batch sono esplosi in NaN/Inf -- qualcosa nel pruning non torna.')

    torch.save(target.state_dict(), args.out)
    print(f'\nCheckpoint pruned salvato in: {args.out}')
    print('Pronto per il fine-tuning con: --init_from', args.out)


if __name__ == '__main__':
    main()