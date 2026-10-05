"""
crypto/finetune_noclamp.py

Applica la RICETTA SENZA CLAMP che ha funzionato sul 6 stage sum
(results/sum_noclamp_w1: Dice 0.867 senza clamp, max|logit| 110) a QUALUNQUE
checkpoint della famiglia: 6, 5, 4, 3 stage, skip sum o concat, con o senza
norm bypassate. Si parte dal checkpoint gia' allenato (init_from), come si e'
fatto per il 6 stage.

La ricetta (train.py, run sum_noclamp_w1):
  - clamp interno delle PolyAct SPENTO: soglie 1e5 da crypto/huge_clamp.json
  - penalita' sulle attivazioni (training.train.activation_penalty) con soglia
    GLOBALE 50 e peso 1.0 (NON la soglia per layer: sarebbe incompatibile)
  - norm affini congelate (--freeze_norm), coefficiente 'a' riportato a 0.1
    (--reset_poly_a), lr 3e-5, batch 16, 15 epoche, AdamW weight decay 1e-3

La validazione che si stampa a ogni epoca e' GIA' senza clamp (le soglie sono
1e5): non esiste un Dice "con clamp" da cui dipendere.

Uso, dalla cartella del progetto (PYTHONPATH=. e' necessario):

  # 5 stage, tutte le 18 norm
  PYTHONPATH=. caffeinate -i python3 crypto/finetune_noclamp.py \\
     --init_from results/test_5stage_warmstart_finetuned/best_model.pth \\
     --freeze_norm --reset_poly_a --out_dir results/noclamp_5stage_w1

  # 5 stage con 4 norm bypassate (le stesse dell'ablazione)
  PYTHONPATH=. caffeinate -i python3 crypto/finetune_noclamp.py \\
     --init_from results/test_5stage_no4norm_finetuned/best_model.pth \\
     --bypass dec0.block.1,enc4.block.1,dec2.block.4,enc3.block.1 \\
     --freeze_norm --reset_poly_a --out_dir results/noclamp_5stage_no4norm_w1

Poi si valuta con
  python3 crypto/clamp_sweep.py --clamp_values_json crypto/huge_clamp.json \\
     --ckpt results/<out_dir>/best_model.pth[:layer,bypassati]
Il modello salvato NON contiene il bypass: va passato di nuovo nella valutazione.
"""

import os
import re
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.he_friendly import PolyAct
from training.dataset import ACDCDataset, load_splits
from training.train import DiceCELoss, activation_penalty
from clamp_sweep import infer_config, build_model, evaluate

NORM_AFFINE = re.compile(r"\.block\.[14]\.(weight|bias)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", required=True)
    ap.add_argument("--bypass", default="", help="layer di norm da bypassare, separati da virgola")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clamp_values_json", default="crypto/huge_clamp.json")
    ap.add_argument("--act_penalty_weight", type=float, default=1.0)
    ap.add_argument("--act_penalty_threshold", type=float, default=50.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--freeze_norm", action="store_true")
    ap.add_argument("--reset_poly_a", action="store_true")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=os.path.expanduser("~/Desktop/tesi_acdc/training"))
    ap.add_argument("--splits_path", default=os.path.expanduser("~/Desktop/tesi_acdc/splits_final.json"))
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else \
        (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}   seed {args.seed}")

    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)
    state = torch.load(args.init_from, map_location="cpu", weights_only=False)
    k, filters, skip_mode = infer_config(state)
    model, strict = build_model(k, filters, skip_mode, clamp_values)
    model.load_state_dict(state, strict=strict)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modello: k={k}, filters={filters}, skip={skip_mode}, {n_params:,} parametri, "
          f"clamp da {args.clamp_values_json}")

    mods = dict(model.named_modules())
    bypass = [b for b in args.bypass.split(",") if b]
    for b in bypass:
        assert b in mods, f"layer da bypassare non trovato: {b}"
        mods[b].register_forward_hook(lambda m, i, o: i[0])
    if bypass:
        print(f"Norm bypassate ({len(bypass)}): {','.join(bypass)}")

    polys = [m for m in model.modules() if isinstance(m, PolyAct)]
    thr = [m.clamp_value for m in polys]
    print(f"{len(polys)} PolyAct, soglie di clamp {min(thr):g}..{max(thr):g}")
    if max(thr) < 1e4:
        print("ATTENZIONE: le soglie di clamp NON sono 'enormi': il clamp interno e' attivo. "
              "Questa non e' la ricetta senza clamp (usa --clamp_values_json crypto/huge_clamp.json).")

    if args.reset_poly_a:
        with torch.no_grad():
            for m in polys:
                m.a.fill_(0.1)
        print("Coefficiente 'a' delle PolyAct riportato a 0.1")

    n_frozen = 0
    if args.freeze_norm:
        for name, p in model.named_parameters():
            if NORM_AFFINE.search(name):
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"Norm affini congelate: {n_frozen:,} parametri")
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Parametri allenabili: {sum(p.numel() for p in params):,} / {n_params:,}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = DiceCELoss(num_classes=4)

    train_cases, val_cases = load_splits(args.splits_path, fold=args.fold)
    train_ds = ACDCDataset(args.data_dir, train_cases, patch_size=(256, 224), augment=True)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Fette: train {len(train_ds)}, validazione {len(val_ds)}\n")

    d0, pc0, mx0, ex0 = evaluate(model, val_loader, device, 0)
    print(f"PRIMA del fine-tuning (senza clamp): Dice {d0 if d0 is None else round(d0, 4)}  "
          f"max|logit| {mx0:.1f}  batch non finiti {ex0}\n", flush=True)

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, tot_pen, n_ok, n_skip = 0.0, 0.0, 0, 0
        epoch_max = torch.zeros((), device=device)
        for imgs, segs in train_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            if not torch.isfinite(logits).all():
                n_skip += 1
                continue
            loss = criterion(logits, segs)
            pen = None
            if args.act_penalty_weight > 0:
                pen = activation_penalty(model, penalty_threshold=args.act_penalty_threshold)
                loss = loss + args.act_penalty_weight * pen
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            if pen is not None:
                tot_pen += float(pen)
            n_ok += 1
            with torch.no_grad():
                raws = [m.last_raw.detach().abs().max() for m in polys if getattr(m, "last_raw", None) is not None]
                if raws:
                    epoch_max = torch.maximum(epoch_max, torch.stack(raws).max())
        val, pcls, mxl, exl = evaluate(model, val_loader, device, 0)
        rec = dict(epoch=epoch, loss=tot_loss / max(1, n_ok), penalty=tot_pen / max(1, n_ok),
                   max_raw=float(epoch_max), val_dice=val, val_per_class=pcls, val_max_logit=mxl,
                   val_nonfinite=exl, train_skipped=n_skip, seconds=time.time() - t0)
        history.append(rec)
        mark = ""
        if val is not None and val > best:
            best = val
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
            mark = "  -> saved best"
        v = "n/d" if val is None else f"{val:.3f} (RV/MYO/LV {pcls[0]:.3f}/{pcls[1]:.3f}/{pcls[2]:.3f})"
        print(f"Epoch {epoch:3d} | loss {rec['loss']:.4f} | pen {rec['penalty']:.4f} | max_raw {rec['max_raw']:8.1f} | "
              f"val {v} | max|logit| {mxl:6.1f} | non finiti {exl}/{n_skip} | {rec['seconds']:.0f}s{mark}", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pth"))
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    print(f"\nFatto. Miglior Dice di validazione senza clamp: {best:.4f}   (ultima epoca {history[-1]['val_dice']:.4f})")
    print(f"Salvati in {args.out_dir}: best_model.pth, final_model.pth, history.json")
    if bypass:
        print(f"RICORDA: valutare con :{','.join(bypass)} dopo il percorso del checkpoint.")


if __name__ == "__main__":
    main()