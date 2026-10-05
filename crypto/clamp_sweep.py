"""
crypto/clamp_sweep.py

DA LANCIARE SUL MAC (serve torch, il dataset e i checkpoint).

Il golden ha mostrato che il checkpoint a 5 stage / no4norm perde quasi
tutto il Dice senza clamp (sui primi 6 batch: 0.907 con clamp, 0.351
senza). L'HE non puo' fare il clamp. Questo script misura, su TUTTA la
validazione e per piu' checkpoint, quanto Dice resta senza clamp, per
capire QUALI configurazioni gia' allenate ne sono (poco) dipendenti.

Per ogni checkpoint: deduce k (numero di stage) e filters dallo state_dict,
costruisce il modello, e calcola il Dice con
   ON : clamp calibrato attivo (come misurato finora)
   OFF: clamp disattivato (cio' che l'HE puo' fare, senza approssimazioni)
Stampa anche il valore assoluto massimo dei logit in modalita' OFF: un
indicatore di quanto i valori escono dal regime "ragionevole" per CKKS.

Uso (dalla cartella del progetto):
  python3 crypto/clamp_sweep.py \\
      --ckpt results/test_5stage_no4norm_finetuned/best_model.pth:dec0.block.1,enc4.block.1,dec2.block.4,enc3.block.1 \\
      --ckpt results/test_5stage_warmstart_finetuned/best_model.pth \\
      --ckpt results/test_4stage_warmstart_finetuned/best_model.pth

Formato di --ckpt:  percorso[:layer_da_bypassare,separati,da,virgola]
                    oppure  percorso:@calib.json[+altro.json]  -> i layer bypassati sono dedotti: tutte le norm
                    del modello meno quelle che hanno una voce nei JSON di calibrazione
Il bypass serve solo per le configurazioni "noNnorm" (le norm restano nello
state_dict ma sono spente in valutazione, come negli script di ablazione).
Per i checkpoint di cui non conosci i layer bypassati lascia vuoto: se il
Dice ON risulta diverso da quello che conosci, e' il segnale che serve il bypass.

Opzioni: --max_batches N (default 0 = tutta la validazione), --batch_size 16.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
import models.he_friendly as hf
from models.he_friendly import PolyAct
from training.dataset import ACDCDataset, load_splits
from training.train import dice_score

try:
    from crypto.remove_deep_stages import UNetKStage
except ImportError:
    UNetKStage = None
try:
    from crypto.remove_enc5_stage import UNet5Stage
except ImportError:
    UNet5Stage = None


def infer_config(state):
    k = 0
    while f"enc{k}.block.0.weight" in state:
        k += 1
    if k == 0:
        raise ValueError("nessuna chiave enc0.block.0.weight nello state_dict: non e' un modello di questa famiglia")
    filters = [int(state[f"enc{i}.block.0.weight"].shape[0]) for i in range(k)]
    dec0_in = int(state["dec0.block.0.weight"].shape[1])
    skip_mode = "sum" if dec0_in == filters[0] else "concat"
    return k, filters, skip_mode


def deduce_bypass(state, json_paths):
    """Layer di norm da bypassare = tutte le norm del modello MENO quelle che hanno una voce
    nei JSON di calibrazione (ogni norm attiva ha una voce: Chebyshev o Newton)."""
    import re
    norms = sorted({k[:-len(".weight")] for k in state if re.search(r"\.block\.[14]\.weight$", k)})
    active = set()
    for jp in json_paths:
        with open(jp) as f:
            for key, entry in json.load(f).items():
                if isinstance(entry, dict) and entry.get("schema") == "bypass":
                    continue          # calibrate_kstage_isqrt.py scrive una voce "schema": "bypass" per ogni norm tolta
                active.add(key)
    bypass = [n for n in norms if n not in active]
    unknown = sorted(a for a in active if a not in norms)
    return bypass, norms, active, unknown


def build_model(k, filters, skip_mode, clamp_values):
    if k == 6:
        model = hf.HEFriendlyUNet(in_channels=1, num_classes=4, act_type="poly", norm_type="instance",
                                  clamp_values=clamp_values, norm_mode="per_instance", skip_mode=skip_mode,
                                  weight_standardization=False, filters=filters)
        return model, False          # strict=False, come negli script per il modello a 6 stage
    if UNetKStage is not None and k in (3, 4, 5):
        return UNetKStage(k, filters, clamp_values=clamp_values), True
    if k == 5 and UNet5Stage is not None:
        return UNet5Stage(filters, clamp_values=clamp_values, skip_mode="sum"), True
    raise ValueError(f"nessuna classe disponibile per k={k}")


def evaluate(model, loader, device, max_batches):
    model.eval()
    d = {1: [], 2: [], 3: []}
    max_abs, exploded, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, segs in loader:
            imgs, segs = imgs.to(device), segs.to(device)
            logits = model(imgs)
            n += 1
            if not torch.isfinite(logits).all():
                exploded += 1
            else:
                max_abs = max(max_abs, logits.abs().max().item())
                s = dice_score(logits.argmax(dim=1), segs)
                for c in (1, 2, 3):
                    d[c].append(s[c])
            if max_batches and n >= max_batches:
                break
    if not d[1]:
        return None, None, max_abs, exploded
    per_class = [float(np.mean(d[c])) for c in (1, 2, 3)]
    return float(np.mean(per_class)), per_class, max_abs, exploded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--data_dir", default=os.path.expanduser("~/Desktop/tesi_acdc/training"))
    ap.add_argument("--splits_path", default=os.path.expanduser("~/Desktop/tesi_acdc/splits_final.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--clamp_values_json", default="crypto/calibrated_clamp_values.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=0)
    args = ap.parse_args()

    for path, what in ((args.data_dir, "il dataset"), (args.splits_path, "gli split"),
                       (args.clamp_values_json, "il JSON dei clamp")):
        if not os.path.exists(path):
            raise SystemExit(f"Non trovo {what}: {path}")

    device = torch.device("mps") if torch.backends.mps.is_available() else \
        (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device: {device}")
    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)
    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Validazione: {len(ds)} fette, batch {args.batch_size}"
          f"{'' if not args.max_batches else f', solo i primi {args.max_batches} batch'}\n")

    rows = []
    for spec in args.ckpt:
        path, _, bypass_s = spec.partition(":")
        bypass = None if bypass_s.startswith("@") else [b for b in bypass_s.split(",") if b]
        name = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        print(f"=== {name} ===", flush=True)
        if not os.path.exists(path):
            print(f"  checkpoint non trovato: {path}\n")
            rows.append((name, None))
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            k, filters, skip_mode = infer_config(state)
            if bypass is None:                         # formato  percorso:@calib1.json+calib2.json
                bypass, norms, active, unknown = deduce_bypass(state, bypass_s[1:].split("+"))
                print(f"  bypass dedotto dai JSON: {len(norms)} norm nel modello, {len(active)} attive nei JSON, "
                      f"{len(bypass)} bypassate: {','.join(bypass)}", flush=True)
                if unknown:
                    print(f"  ATTENZIONE: voci dei JSON che non sono norm di questo modello: {unknown}", flush=True)
            model, strict = build_model(k, filters, skip_mode, clamp_values)
            missing, unexpected = model.load_state_dict(state, strict=strict)
            model.to(device)
            mods = dict(model.named_modules())
            for b in bypass:
                assert b in mods, f"layer da bypassare non trovato: {b}"
                mods[b].register_forward_hook(lambda m, i, o: i[0])
            polys = [m for m in model.modules() if isinstance(m, PolyAct)]
            original = [m.clamp_value for m in polys]
            print(f"  k={k}, filters={filters}, skip={skip_mode}, bypass={len(bypass)}, "
                  f"{len(polys)} PolyAct (clamp {min(original):.1f}..{max(original):.1f}), "
                  f"chiavi mancanti/ignorate: {len(missing)}/{len(unexpected)}", flush=True)
            t0 = time.time()
            for m, v in zip(polys, original):
                m.clamp_value = v
            on, on_c, on_max, on_exp = evaluate(model, loader, device, args.max_batches)
            for m in polys:
                m.clamp_value = float("inf")
            off, off_c, off_max, off_exp = evaluate(model, loader, device, args.max_batches)
            print(f"  ON : Dice {on:.3f}  (RV/MYO/LV {on_c[0]:.3f}/{on_c[1]:.3f}/{on_c[2]:.3f})  max|logit| {on_max:.1f}")
            if off is None:
                print(f"  OFF: tutti i batch non finiti ({off_exp})")
            else:
                print(f"  OFF: Dice {off:.3f}  (RV/MYO/LV {off_c[0]:.3f}/{off_c[1]:.3f}/{off_c[2]:.3f})  "
                      f"max|logit| {off_max:.1f}  batch non finiti {off_exp}   [{time.time()-t0:.0f}s]")
            print()
            rows.append((name, (k, len(bypass), on, off, off_max)))
        except Exception as e:
            print(f"  ERRORE: {type(e).__name__}: {e}\n")
            rows.append((name, None))

    print("=== RIEPILOGO ===")
    print(f"  {'checkpoint':<40}{'k':>3}{'bypass':>8}{'ON':>8}{'OFF':>8}{'ON-OFF':>9}{'max|logit| OFF':>16}")
    for name, r in rows:
        if r is None:
            print(f"  {name:<40}  (non valutato)")
            continue
        k, nb, on, off, mx = r
        off_s = f"{off:8.3f}" if off is not None else "     n/d"
        diff = f"{on - off:9.3f}" if off is not None else "      n/d"
        print(f"  {name:<40}{k:3d}{nb:8d}{on:8.3f}{off_s}{diff}{mx:16.1f}")
    print("\nOFF e' il Dice atteso sotto cifratura SENZA le approssimazioni (radice inversa, rumore CKKS).")
    print("Una configurazione con ON-OFF piccolo e max|logit| contenuto e' candidata per l'HE.")


if __name__ == "__main__":
    main()