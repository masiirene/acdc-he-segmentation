"""
crypto/list_checkpoints.py

DA LANCIARE SUL MAC, dalla cartella del progetto:
    python3 crypto/list_checkpoints.py                 # scansiona results/ (ricorsivo)
    python3 crypto/list_checkpoints.py --only_k 6      # solo i modelli a 6 stage

Per ogni file .pth sotto results/ deduce dallo state_dict, SENZA costruire
nessun modello:
  k       numero di stage encoder (enc0..enc{k-1})
  filters canali di ogni stage
  skip    'sum' o 'concat' (dai canali in ingresso di dec0)
  norm    'inst' se non ci sono statistiche di popolazione, 'pop' se ci sono
          running_mean/running_var (modelli con statistiche fisse: NON valutabili
          con lo strumento clamp_sweep, che costruisce norm per-istanza)
  WS      'si' se ci sono chiavi raw_gain (weight standardization attiva)
Serve a ritrovare, ad esempio, il checkpoint a 6 stage "aggressivo+sum".
"""

import os
import re
import sys
import time
import argparse


def inspect(state):
    k = 0
    while f"enc{k}.block.0.weight" in state:
        k += 1
    if k == 0:
        return None
    filters = [int(state[f"enc{i}.block.0.weight"].shape[0]) for i in range(k)]
    dec0 = state.get("dec0.block.0.weight")
    skip = "?"
    if dec0 is not None:
        skip = "sum" if int(dec0.shape[1]) == filters[0] else "concat"
    keys = list(state.keys())
    pop = any("running_mean" in key for key in keys)
    ws = any(key.endswith("raw_gain") for key in keys)
    n_poly = sum(1 for key in keys if re.search(r"\.block\.[25]\.a$", key))
    return dict(k=k, filters=filters, skip=skip, norm="pop" if pop else "inst",
                ws="si" if ws else "no", n_poly=n_poly)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--only_k", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"Cartella non trovata: {args.root} (lancia dalla cartella del progetto, sul Mac)")

    files = []
    for dirpath, dirnames, filenames in os.walk(args.root):          # RICORSIVO: i checkpoint di train.py stanno in sottocartelle
        dirnames.sort()
        for f in sorted(filenames):
            if f.endswith(".pth"):
                files.append((os.path.relpath(dirpath, args.root), f, os.path.join(dirpath, f)))
    print(f"Trovati {len(files)} file .pth sotto {args.root}/ (ricerca ricorsiva) ... lettura in corso", flush=True)
    print("(ogni file e' ~14-80 MB: puo' richiedere un minuto o due)\n", flush=True)

    rows, skipped = [], 0
    for d, f, p in files:
        try:
            state = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state and not any(k.startswith("enc0") for k in state):
                state = state["state_dict"]
            info = inspect(state) if isinstance(state, dict) else None
        except Exception as e:
            info = None
        if info is None:
            skipped += 1
            continue
        rows.append((info["k"], os.path.getmtime(p), d, f, info))

    rows.sort(key=lambda r: (-r[0], r[1]))
    shown = [r for r in rows if not args.only_k or r[0] == args.only_k]
    print(f"{'k':>2} {'data':<11} {'cartella / file':<72} {'skip':<7}{'norm':<5}{'WS':<3} filters")
    print("-" * 130)
    for k, mt, d, f, info in shown:
        # accorcia il nome lungo di train.py, che e' sempre lo stesso suffisso
        d2 = d.replace("/act=poly_norm=instance_mode=per_instance_bs16_lr3e-05_freeze-norm", " /…freeze-norm")
        name = d2 if f == "best_model.pth" else f"{d2}  [{f}]"
        print(f"{k:>2} {time.strftime('%Y-%m-%d', time.localtime(mt)):<11} {name:<72} {info['skip']:<7}{info['norm']:<5}{info['ws']:<3} {info['filters']}")
    print(f"\n{len(shown)} checkpoint mostrati su {len(rows)} riconosciuti ({skipped} file .pth non sono modelli di questa famiglia).")
    by_k = {}
    for r in rows:
        by_k[r[0]] = by_k.get(r[0], 0) + 1
    print("Per numero di stage: " + ", ".join(f"k={k}: {n}" for k, n in sorted(by_k.items(), reverse=True)))


if __name__ == "__main__":
    main()