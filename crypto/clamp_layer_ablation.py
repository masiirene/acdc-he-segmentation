"""
crypto/clamp_layer_ablation.py

DA LANCIARE SUL MAC (serve torch, dataset, checkpoint, e crypto/clamp_sweep.py).

Domanda: la dipendenza dal clamp e' CONCENTRATA in pochi layer o DIFFUSA?
  - concentrata -> si puo' intervenire in pochi punti (attivazione diversa,
    riscalatura) e le soluzioni sono locali;
  - diffusa -> serve cambiare la rete o il training nel suo insieme.

Esperimenti (Dice sulla validazione, tutti con la stessa pipeline di
clamp_sweep.py, che riproduce i Dice gia' noti con clamp attivo):
  ONLY    clamp attivo SOLO nel layer i, spento negli altri
          -> "guadagno": quanto recupera un solo layer da solo
  EXCEPT  clamp attivo ovunque TRANNE nel layer i
          -> "perdita": quanto costa togliere solo quello
  PREFIX  (opzionale) clamp attivo nei primi i layer, spento nei successivi
          -> da che profondita' in poi il clamp non serve piu'
Piu' una DIAGNOSTICA con tutto spento: per ogni PolyAct, il valore massimo
in uscita, la frazione di valori oltre la soglia del clamp e la frazione di
fette (immagini) toccate. Mostra DOVE nascono i valori fuori scala e se
sono pochi outlier o molti valori.

Nota: gli effetti non sono additivi (un layer a valle puo' compensare uno a
monte): ONLY e EXCEPT sono due punti di vista diversi sulla stessa
domanda, non una scomposizione esatta.

Uso (dalla cartella del progetto):
  python3 crypto/clamp_layer_ablation.py \\
     --ckpt results/test_5stage_no4norm_finetuned/best_model.pth:dec0.block.1,enc4.block.1,dec2.block.4,enc3.block.1
Opzioni: --modes only,except,prefix  --max_batches N  --diag_batches 4  --batch_size 16
Durata: ~10 s per valutazione su validazione intera (mps) -> 36 valutazioni = ~6 minuti.
"""

import os
import sys
import json
import time
import argparse


# ============================================================
# Logica pura (nessuna dipendenza da torch): provabile con una funzione finta
# ============================================================

def run_experiments(n_layers, eval_active, modes, log=print):
    """eval_active(set_di_indici_con_clamp_attivo) -> Dice medio."""
    res = {"modes": modes}
    t0 = time.time()
    res["off"] = eval_active(set())
    res["on"] = eval_active(set(range(n_layers)))
    log(f"  baseline: tutto spento {res['off']:.3f}, tutto attivo {res['on']:.3f}  [{time.time()-t0:.0f}s]")
    all_idx = set(range(n_layers))
    if "only" in modes:
        res["only"] = {}
        for i in range(n_layers):
            res["only"][i] = eval_active({i})
        log(f"  ONLY completato  [{time.time()-t0:.0f}s]")
    if "except" in modes:
        res["except"] = {}
        for i in range(n_layers):
            res["except"][i] = eval_active(all_idx - {i})
        log(f"  EXCEPT completato  [{time.time()-t0:.0f}s]")
    if "prefix" in modes:
        res["prefix"] = {}
        for i in range(1, n_layers + 1):
            res["prefix"][i] = eval_active(set(range(i)))
        log(f"  PREFIX completato  [{time.time()-t0:.0f}s]")
    return res


def format_report(names, thr, res, diag=None):
    L = []
    n = len(names)
    off, on = res["off"], res["on"]
    gap = on - off
    if diag is not None:
        L.append("=== DIAGNOSTICA (tutto spento): dove i valori superano la soglia del clamp ===")
        L.append(f"  {'#':>2} {'layer':<18}{'soglia':>7}{'max|uscita|':>13}{'valori oltre soglia':>21}{'fette coinvolte':>17}")
        first = None
        for i in range(n):
            s = diag[i]
            frac = s["exceed"] / max(s["total"], 1)
            sl = s["slices_hit"] / max(s["slices"], 1)
            if first is None and s["exceed"] > 0:
                first = names[i]
            L.append(f"  {i:>2} {names[i]:<18}{thr[i]:7.1f}{s['max']:13.1f}{frac:21.2e}{100 * sl:16.1f}%")
        L.append(f"  -> primo layer con valori oltre la soglia: {first if first else 'nessuno'}")
        L.append("")
    L.append("=== BASELINE ===")
    L.append(f"  tutto spento {off:.3f}   tutto attivo {on:.3f}   divario {gap:+.3f}")
    L.append("")
    if "only" in res:
        L.append("=== ONLY: clamp attivo SOLO nel layer i (gli altri spenti) ===")
        L.append(f"  {'#':>2} {'layer':<18}{'soglia':>7}{'Dice':>8}{'guadagno vs spento':>20}")
        for i in range(n):
            d = res["only"][i]
            L.append(f"  {i:>2} {names[i]:<18}{thr[i]:7.1f}{d:8.3f}{d - off:+20.3f}")
        L.append("")
    if "except" in res:
        L.append("=== EXCEPT: clamp attivo ovunque TRANNE nel layer i ===")
        L.append(f"  {'#':>2} {'layer':<18}{'soglia':>7}{'Dice':>8}{'perdita vs attivo':>20}")
        for i in range(n):
            d = res["except"][i]
            L.append(f"  {i:>2} {names[i]:<18}{thr[i]:7.1f}{d:8.3f}{d - on:+20.3f}")
        L.append("")
    if "prefix" in res:
        L.append("=== PREFIX: clamp attivo nei primi i layer, spento nei successivi ===")
        L.append(f"  {'i':>2} {'ultimo layer attivo':<22}{'Dice':>8}")
        for i in range(1, n + 1):
            L.append(f"  {i:>2} {names[i - 1]:<22}{res['prefix'][i]:8.3f}")
        L.append("")
    L.append("=== LETTURA ===")
    if gap < 0.05:
        L.append(f"  Il divario tra tutto-attivo e tutto-spento e' piccolo ({gap:+.3f}): non c'e' dipendenza forte dal clamp.")
        return "\n".join(L)
    if "only" in res:
        top = sorted(range(n), key=lambda i: -(res["only"][i] - off))[:3]
        L.append("  Piu' utili da soli (ONLY): " + ", ".join(
            f"{names[i]} {res['only'][i] - off:+.3f}" for i in top))
        best = res["only"][top[0]] - off
        if best >= 0.8 * gap:
            L.append(f"  -> UN SOLO layer ({names[top[0]]}) recupera {100 * best / gap:.0f}% del divario: dipendenza CONCENTRATA.")
        elif best >= 0.4 * gap:
            L.append(f"  -> un layer ({names[top[0]]}) recupera {100 * best / gap:.0f}% del divario: dipendenza PARZIALMENTE concentrata.")
        else:
            L.append(f"  -> nessun layer da solo recupera piu' del {100 * best / gap:.0f}% del divario.")
    if "except" in res:
        bot = sorted(range(n), key=lambda i: res["except"][i] - on)[:3]
        L.append("  Piu' costosi da togliere (EXCEPT): " + ", ".join(
            f"{names[i]} {res['except'][i] - on:+.3f}" for i in bot))
        worst = on - res["except"][bot[0]]
        if worst >= 0.5 * gap:
            L.append(f"  -> senza il solo clamp di {names[bot[0]]} il Dice cala di {worst:.3f} ({100 * worst / gap:.0f}% del divario): layer NECESSARIO.")
        elif worst >= max(0.15, 2.0 / n) * gap:     # soglia che scala col numero di layer
            L.append(f"  -> il layer piu' necessario ({names[bot[0]]}) pesa {100 * worst / gap:.0f}% del divario.")
        else:
            L.append(f"  -> togliere un solo clamp costa al massimo {worst:.3f} ({100 * worst / gap:.0f}% del divario): dipendenza DIFFUSA.")
    return "\n".join(L)


# ============================================================
# Parte torch
# ============================================================

def main():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, '.')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.he_friendly import PolyAct
    from training.dataset import ACDCDataset, load_splits
    from clamp_sweep import infer_config, build_model, evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="percorso[:layer,da,bypassare]")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/Desktop/tesi_acdc/training"))
    ap.add_argument("--splits_path", default=os.path.expanduser("~/Desktop/tesi_acdc/splits_final.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--clamp_values_json", default="crypto/calibrated_clamp_values.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--diag_batches", type=int, default=4)
    ap.add_argument("--modes", default="only,except")
    args = ap.parse_args()
    modes = [m for m in args.modes.split(",") if m]

    path, _, bypass_s = args.ckpt.partition(":")
    bypass = [b for b in bypass_s.split(",") if b]
    for p, what in ((path, "il checkpoint"), (args.data_dir, "il dataset"),
                    (args.splits_path, "gli split"), (args.clamp_values_json, "il JSON dei clamp")):
        if not os.path.exists(p):
            raise SystemExit(f"Non trovo {what}: {p}")

    device = torch.device("mps") if torch.backends.mps.is_available() else \
        (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)

    state = torch.load(path, map_location="cpu", weights_only=False)
    k, filters, skip_mode = infer_config(state)
    model, strict = build_model(k, filters, skip_mode, clamp_values)
    model.load_state_dict(state, strict=strict)
    model.to(device).eval()
    mods = dict(model.named_modules())
    for b in bypass:
        assert b in mods, f"layer da bypassare non trovato: {b}"
        mods[b].register_forward_hook(lambda m, i, o: i[0])

    named = [(n, m) for n, m in model.named_modules() if isinstance(m, PolyAct)]
    names = [n for n, _ in named]
    polys = [m for _, m in named]
    thr = [m.clamp_value for m in polys]
    print(f"Device {device}; k={k}, filters={filters}, skip={skip_mode}, bypass={len(bypass)}; "
          f"{len(polys)} PolyAct, soglie {min(thr):.1f}..{max(thr):.1f}")

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Validazione: {len(ds)} fette\n")

    def set_active(active):
        for i, (m, v) in enumerate(zip(polys, thr)):
            m.clamp_value = v if i in active else float("inf")

    def eval_active(active):
        set_active(active)
        d, _, _, exploded = evaluate(model, loader, device, args.max_batches)
        if d is None:
            return float("nan")
        return d

    # ---------- diagnostica: tutto spento, statistiche per layer ----------
    stats = {i: dict(max=0.0, exceed=0, total=0, slices_hit=0, slices=0) for i in range(len(polys))}

    def make_hook(i):
        def hook(mod, inp, out):
            o = out.detach()
            s = stats[i]
            s["max"] = max(s["max"], o.abs().max().item())
            ex = o.abs() > thr[i]
            s["exceed"] += int(ex.sum().item())
            s["total"] += ex.numel()
            s["slices_hit"] += int(ex.flatten(1).any(dim=1).sum().item())
            s["slices"] += o.shape[0]
        return hook

    print("Diagnostica con tutto spento...", flush=True)
    handles = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(polys)]
    set_active(set())
    evaluate(model, loader, device, args.diag_batches)
    for h in handles:
        h.remove()

    print("Esperimenti (puo' richiedere alcuni minuti)...", flush=True)
    res = run_experiments(len(polys), eval_active, modes)
    set_active(set(range(len(polys))))
    print()
    print(format_report(names, thr, res, diag=stats))


if __name__ == "__main__":
    main()