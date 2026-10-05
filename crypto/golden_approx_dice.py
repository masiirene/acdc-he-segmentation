"""
crypto/golden_approx_dice.py

DICE DI UN CHECKPOINT CON LA RADICE INVERSA APPROSSIMATA, senza clamp.

Fino a ora il Dice "senza clamp" (clamp_sweep, 0.867 per sum_noclamp_w1) usa la
radice inversa ESATTA. Sotto cifratura la radice inversa si calcola con
un'approssimazione: polinomio di Chebyshev (monotono: non sovrastima mai) piu'
passi di Newton, e Newton puro per i layer che con Chebyshev non convergono.
Questo script misura il Dice con QUELLA approssimazione, in aritmetica per il
resto esatta: e' il numero che separa "la rete funziona" da "la rete funziona
sotto cifratura". Non include il rumore CKKS (misurato a parte: ~3e-3 per bootstrap).

Legge i pesi direttamente dallo state_dict (nessun .npz da estrarre), forward
float32 su MPS (o float64 su CPU con --dtype float64), skip SUM, nessun clamp.
Per ogni norm usa lo schema dei JSON di calibrazione:
  - voce in --cheb_json:   y0 = serie di Chebyshev (numpy: dominio mappato su [-1,1],
                           primo coefficiente PIENO) su var+eps, poi `post_iter` passi
                           y <- y (1.5 - 0.5 v y^2)
  - voce in --newton_json: y0 scalare, `iterations_needed` passi di Newton
  - nessuna voce:          errore (o layer in --bypass: norm tolta)

Stampa: Dice con radice ESATTA (deve riprodurre clamp_sweep: controllo di coerenza),
Dice con radice APPROSSIMATA, differenza, accordo pixel a pixel tra le due,
e per ogni layer: frazione di varianze FUORI dal dominio di calibrazione ed errore
relativo massimo di 1/std (target della calibrazione: 1%).

Le calibrazioni sono state fatte sul set di VALIDAZIONE (stesso fold 0 usato qui):
il Dice con approssimazione qui e' quindi "in-sample" rispetto alla calibrazione.
Per un numero imparziale va rifatto sul test set a configurazione chiusa.

Uso (dalla cartella del progetto, sul Mac):
  python3 crypto/golden_approx_dice.py \\
     --ckpt results/sum_noclamp_w1/act=poly_norm=instance_mode=per_instance_skip-sum_bs16_lr3e-05_freeze-norm/best_model.pth
"""

import os
import re
import sys
import json
import argparse
import time
import numpy as np

NORM_EPS = 1e-5


# ============================================================
# Radice inversa approssimata. Solo aritmetica elementare: funziona
# identica con array numpy e con tensori torch (cosi' si puo' provare senza torch).
# ============================================================

def chebval(t, c):
    """Serie di Chebyshev sum_k c_k T_k(t), c_0 con peso pieno (ricorrenza di Clenshaw)."""
    b1 = t * 0.0
    b2 = t * 0.0
    for ck in reversed(c[1:]):
        b1, b2 = 2.0 * t * b1 - b2 + ck, b1
    return t * b1 - b2 + c[0]


def newton_steps(y, v, n):
    for _ in range(n):
        y = y * (1.5 - 0.5 * v * y * y)
    return y


def approx_inv_std(v, scheme):
    """v: varianza (+eps). scheme: ('cheb', coef, dominio, n_iter) oppure ('newton', y0, n_iter)."""
    if scheme[0] == "cheb":
        _, coef, dom, n = scheme
        t = (2.0 * v - dom[0] - dom[1]) / (dom[1] - dom[0])
        y = chebval(t, coef)
    else:
        _, y0, n = scheme
        y = v * 0.0 + y0
    return newton_steps(y, v, n)


def load_schemes(cheb_json, newton_json):
    schemes = {}
    with open(cheb_json) as f:
        for name, e in json.load(f).items():
            schemes[name] = ("cheb", [float(c) for c in e["cheb_coeffs"]],
                             [float(e["cheb_domain"][0]), float(e["cheb_domain"][1])], int(e["post_iter"]))
    with open(newton_json) as f:
        for name, e in json.load(f).items():
            schemes[name] = ("newton", float(e["y0"]), int(e["iterations_needed"]))   # il fallback ha la precedenza
    return schemes


def load_schemes_combined(path):
    """Formato unico dei calibration_<k>stage*.json: ogni voce ha "schema": "newton" oppure "cheb"."""
    schemes = {}
    with open(path) as f:
        for name, e in json.load(f).items():
            if e.get("schema") == "bypass":
                continue                                  # norm tolta: nessuno schema
            if e.get("schema") == "newton" or ("y0" in e and "cheb_coeffs" not in e):
                schemes[name] = ("newton", float(e["y0"]), int(e["iterations_needed"]))
            else:
                schemes[name] = ("cheb", [float(c) for c in e["cheb_coeffs"]],
                                 [float(e["cheb_domain"][0]), float(e["cheb_domain"][1])], int(e["post_iter"]))
    return schemes


# ============================================================
# Rete: forward dai pesi dello state_dict
# ============================================================

class Net:
    def __init__(self, state, torch_mod, device, dtype, bypass=()):
        torch = torch_mod
        self.torch, self.F = torch, torch.nn.functional
        self.s = {k: v.to(device=device, dtype=dtype) for k, v in state.items()}
        k = 0
        while f"enc{k}.block.0.weight" in self.s:
            k += 1
        self.k = k
        filters = [self.s[f"enc{i}.block.0.weight"].shape[0] for i in range(k)]
        if self.s["dec0.block.0.weight"].shape[1] != filters[0]:
            raise SystemExit("Questo script supporta solo skip_mode=sum (dec0.block.0 ha il doppio dei canali: concat).")
        self.bypass = set(bypass)
        self.norms = sorted({n[:-len(".weight")] for n in self.s if re.search(r"\.block\.[14]\.weight$", n)})
        self.inv_fn = None            # None = radice esatta; altrimenti funzione (nome, v) -> 1/std approssimato

    def _poly(self, key, C):
        t = self.s[key]
        return t.reshape(1, -1, 1, 1) if t.numel() == C and C > 1 else t.reshape(1, 1, 1, 1)

    def _norm_act(self, x, name, which):
        nb, pb = (1, 2) if which == 1 else (4, 5)
        nname = f"{name}.block.{nb}"
        if nname not in self.bypass:
            mean = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
            v = var + NORM_EPS
            inv = (1.0 / self.torch.sqrt(v)) if self.inv_fn is None else self.inv_fn(nname, v)
            x = (x - mean) * inv * self.s[f"{nname}.weight"].view(1, -1, 1, 1) + self.s[f"{nname}.bias"].view(1, -1, 1, 1)
        C = x.shape[1]
        a, b, c = (self._poly(f"{name}.block.{pb}.{p}", C) for p in "abc")
        return a * x * x + b * x + c                 # PolyAct, nessun clamp

    def _block(self, x, name, stride):
        s, F = self.s, self.F
        x = F.conv2d(x, s[f"{name}.block.0.weight"], s[f"{name}.block.0.bias"], stride=stride, padding=1)
        x = self._norm_act(x, name, 1)
        x = F.conv2d(x, s[f"{name}.block.3.weight"], s[f"{name}.block.3.bias"], stride=1, padding=1)
        return self._norm_act(x, name, 2)

    def __call__(self, x):
        s, F = self.s, self.F
        enc = {}
        h = x
        for i in range(self.k):
            h = self._block(h, f"enc{i}", 1 if i == 0 else 2)
            enc[i] = h
        h = enc[self.k - 1]
        for j in range(self.k - 2, -1, -1):
            h = F.conv_transpose2d(h, s[f"up{j}.weight"], s[f"up{j}.bias"], stride=2)
            h = self._block(h + enc[j], f"dec{j}", 1)
        return F.conv2d(h, s["out_conv.weight"], s["out_conv.bias"])


# ============================================================
# Valutazione
# ============================================================

def main():
    import torch
    sys.path.insert(0, '.')
    from torch.utils.data import DataLoader
    from training.dataset import ACDCDataset, load_splits
    from training.train import dice_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bypass", default="", help="layer di norm bypassati (csv), come in clamp_sweep")
    ap.add_argument("--cheb_json", default="crypto/chebyshev_calibration_monotonic.json")
    ap.add_argument("--newton_json", default="crypto/newton_fallback_full22.json")
    ap.add_argument("--calib_json", default=None,
                    help="formato unico (calibration_5stage_*.json): se dato, sostituisce cheb_json e newton_json")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/Desktop/tesi_acdc/training"))
    ap.add_argument("--splits_path", default=os.path.expanduser("~/Desktop/tesi_acdc/splits_final.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16,
                    help="16 = lo stesso di clamp_sweep (il Dice e' una media per batch: con 8 viene ~0.01 piu' basso)")
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["val", "train", "test"],
                    help="train = fette di training NON usate per calibrare (vedi --skip_first): controllo della coda senza toccare il test set")
    ap.add_argument("--testing_dir", default=os.path.expanduser("~/Desktop/tesi_acdc/testing"))
    ap.add_argument("--test_cases_json", default="crypto/test_cases.json")
    ap.add_argument("--note", default=None, help="OBBLIGATORIA con --split test: perche' si usa il test set ORA")
    ap.add_argument("--log_file", default="crypto/test_set_evaluation_log.json")
    ap.add_argument("--skip_confirmation", action="store_true")
    ap.add_argument("--skip_first", type=int, default=320,
                    help="con --split train: salta le prime N fette (calibrate_cheb_robust.py ne usa 320 per i campioni di varianza)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = ap.parse_args()

    need = [(args.ckpt, "il checkpoint"), (args.data_dir, "il dataset"), (args.splits_path, "gli split")]
    need += [(args.calib_json, "il JSON di calibrazione")] if args.calib_json else \
            [(args.cheb_json, "il JSON Chebyshev"), (args.newton_json, "il JSON Newton")]
    for p, what in need:
        if not os.path.exists(p):
            raise SystemExit(f"Non trovo {what}: {p}")

    if args.dtype == "float64":
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        dtype = torch.float32
    print(f"Device {device}, {args.dtype}")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    bypass = [b for b in args.bypass.split(",") if b]
    net = Net(state, torch, device, dtype, bypass)
    schemes = load_schemes_combined(args.calib_json) if args.calib_json else load_schemes(args.cheb_json, args.newton_json)
    missing = [n for n in net.norms if n not in schemes and n not in net.bypass]
    extra = [n for n in schemes if n not in net.norms]
    print(f"Rete: {net.k} stage, {len(net.norms)} norm ({len(bypass)} bypassate); calibrazioni: "
          f"{sum(1 for s in schemes.values() if s[0] == 'cheb')} Chebyshev + {sum(1 for s in schemes.values() if s[0] == 'newton')} Newton")
    if missing:
        raise SystemExit(f"Norm senza calibrazione nei due JSON: {missing}\n(se sono norm bypassate, passale con --bypass)")
    if extra:
        print(f"ATTENZIONE: voci nei JSON che non sono norm di questo modello: {extra}")

    stats = {n: dict(low=0, high=0, tot=0, err=0.0, worst_v=None, worst_y0=None, nonfinite=0, max_v=0.0)
             for n in net.norms if n not in net.bypass}

    def y0_ratio(vv, sc):
        """y0 / (1/sqrt(vv)) per una varianza scalare: > 1 = sovrastima, > 1.732 = Newton diverge."""
        if sc[0] == "cheb":
            _, coef, dom, _n = sc
            y0 = chebval((2.0 * vv - dom[0] - dom[1]) / (dom[1] - dom[0]), coef)
        else:
            y0 = sc[1]
        return y0 * (vv ** 0.5)

    def inv_approx(name, v):
        sc = schemes[name]
        y = approx_inv_std(v, sc)
        exact = 1.0 / torch.sqrt(v)
        rel = ((y - exact).abs() / exact).flatten()
        st = stats[name]
        st["nonfinite"] += int((~torch.isfinite(y)).sum().item())
        vf = v[torch.isfinite(v)]
        if vf.numel():
            st["max_v"] = max(st["max_v"], float(vf.max()))
        mx, idx = rel.max(0)
        if float(mx) > st["err"] or (st["worst_v"] is None):
            st["err"] = float(mx)
            st["worst_v"] = float(v.flatten()[idx])
            st["worst_y0"] = y0_ratio(st["worst_v"], sc)
        if sc[0] == "cheb":
            st["low"] += (v < sc[2][0]).sum().item()
            st["high"] += (v > sc[2][1]).sum().item()
        st["tot"] += v.numel()
        return y

    train_cases, val_cases = load_splits(args.splits_path, fold=args.fold)
    if args.split == "train":
        from torch.utils.data import Subset
        full = ACDCDataset(args.data_dir, train_cases, patch_size=(256, 224), augment=False)
        ds = Subset(full, range(min(args.skip_first, len(full)), len(full)))
        what = f"fette di TRAINING dalla {args.skip_first}a in poi (non usate per i campioni di calibrazione; il modello pero' le ha viste in training)"
    elif args.split == "test":
        from test_set_guard import confirm_test_set_use
        if not args.note:
            raise SystemExit("--note e' obbligatoria con --split test (regola del progetto: ogni uso del test set va giustificato)")
        if not confirm_test_set_use(args.note, args.skip_confirmation, "Dice con radice inversa esatta e approssimata"):
            return
        with open(args.test_cases_json) as f:
            test_cases = json.load(f)
        ds = ACDCDataset(args.testing_dir, test_cases, patch_size=(256, 224), augment=False)
        what = f"fette del TEST SET ({len(test_cases)} casi)"
    else:
        ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
        what = "fette di validazione (le stesse su cui si e' calibrato)"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"{len(ds)} {what}\n")

    tags = ("esatta", "approssimata")
    dice = {t: {1: [], 2: [], 3: []} for t in tags}
    dice_good = {t: {1: [], 2: [], 3: []} for t in tags}
    n_bad = {t: 0 for t in tags}
    n_slices = 0
    agree, max_dlogit, n_b = [], 0.0, 0
    max_logit = {t: 0.0 for t in tags}
    t0 = time.time()
    with torch.no_grad():
        for imgs, segs in loader:
            imgs, segs = imgs.to(device=device, dtype=dtype), segs.to(device)
            net.inv_fn = None
            lo_e = net(imgs)
            net.inv_fn = inv_approx
            lo_a = net(imgs)
            bad_e = ~torch.isfinite(lo_e).flatten(1).all(dim=1)
            bad_a = ~torch.isfinite(lo_a).flatten(1).all(dim=1)
            n_bad["esatta"] += int(bad_e.sum())
            n_bad["approssimata"] += int(bad_a.sum())
            n_slices += lo_a.shape[0]
            good = ~(bad_a | bad_e)
            for tag, lo in (("esatta", lo_e), ("approssimata", lo_a)):
                fin = lo[torch.isfinite(lo)]
                if fin.numel():
                    max_logit[tag] = max(max_logit[tag], fin.abs().max().item())
                sc = dice_score(lo.argmax(dim=1), segs)
                for c in (1, 2, 3):
                    dice[tag][c].append(sc[c])
                if good.any():
                    sg = dice_score(lo[good].argmax(dim=1), segs[good])
                    for c in (1, 2, 3):
                        dice_good[tag][c].append(sg[c])
            agree.append((lo_e.argmax(1) == lo_a.argmax(1)).float().mean().item())
            d = (lo_e - lo_a).abs()
            d = d[torch.isfinite(d)]
            if d.numel():
                max_dlogit = max(max_dlogit, d.max().item())
            n_b += 1
            if args.max_batches and n_b >= args.max_batches:
                break
    print(f"({n_b} batch in {time.time()-t0:.0f}s)\n")

    def mean3(tag, store=None):
        store = dice if store is None else store
        pc = [float(np.mean(store[tag][c])) for c in (1, 2, 3)]
        return float(np.mean(pc)), pc

    de, pe = mean3("esatta")
    da, pa = mean3("approssimata")
    print("=== DICE (senza clamp, aritmetica esatta tranne la radice inversa) ===")
    print(f"  radice ESATTA        {de:.4f}   (RV/MYO/LV {pe[0]:.3f}/{pe[1]:.3f}/{pe[2]:.3f})   max|logit| {max_logit['esatta']:.1f}")
    print(f"  radice APPROSSIMATA  {da:.4f}   (RV/MYO/LV {pa[0]:.3f}/{pa[1]:.3f}/{pa[2]:.3f})   max|logit| {max_logit['approssimata']:.1f}")
    print(f"  differenza {da - de:+.4f};  pixel con la stessa classe: {100 * np.mean(agree):.2f}%;  max |delta logit| {max_dlogit:.2f}")
    print("  (controllo di coerenza: con --batch_size 16 la riga ESATTA riproduce clamp_sweep)")
    print(f"  slice con logit NON FINITI: esatta {n_bad['esatta']}/{n_slices}, approssimata {n_bad['approssimata']}/{n_slices}")
    if dice_good["esatta"][1]:
        ge, _ = mean3("esatta", dice_good)
        ga, _ = mean3("approssimata", dice_good)
        print(f"  Dice sui soli slice SANI (finiti in entrambe): esatta {ge:.4f}  approssimata {ga:.4f}  differenza {ga - ge:+.4f}")
    print()

    if args.split == "test":
        from test_set_guard import append_log
        append_log(args.log_file, dict(
            kind="golden_approx_dice (test set completo)", checkpoint=args.ckpt, note=args.note, bypass=bypass,
            calibrazione=(args.calib_json or [args.cheb_json, args.newton_json]), n_slices=int(n_slices),
            dice_esatta=de, dice_esatta_per_classe=pe, dice_approssimata=da, dice_approssimata_per_classe=pa,
            slice_non_finiti_esatta=int(n_bad["esatta"]), slice_non_finiti_approssimata=int(n_bad["approssimata"])))

    print("=== PER LAYER: errore della radice inversa e varianze fuori dominio ===")
    print(f"  {'layer':<14}{'schema':<8}{'err.rel.max':>13}{'sotto dom.':>12}{'sopra dom.':>12}"
          f"{'var. caso peggiore':>20}{'y0/vero':>10}{'var.max/lim.alto':>18}")
    rows = sorted(stats.items(), key=lambda kv: -kv[1]["err"])
    for name, st in rows:
        sc = schemes[name]
        tot = max(st["tot"], 1)
        lo = f"{100 * st['low'] / tot:.3f}%" if sc[0] == "cheb" else "-"
        hi = f"{100 * st['high'] / tot:.3f}%" if sc[0] == "cheb" else "-"
        dom = f"[{sc[2][0]:.3f},{sc[2][1]:.3f}]" if sc[0] == "cheb" else ""
        flag = "  <-- >1%" if st["err"] > 0.01 else ""
        nf = f"  NON FINITI: {st['nonfinite']}" if st["nonfinite"] else ""
        rat = f"{st['max_v'] / sc[2][1]:.2f}x" if sc[0] == "cheb" else "-"
        print(f"  {name:<14}{sc[0]:<8}{100 * st['err']:>12.3g}%{lo:>12}{hi:>12}{st['worst_v']:>20.3e}{st['worst_y0']:>10.2f}{rat:>18}  {dom}{flag}{nf}")
    print("\n  LETTURA: 'y0/vero' > 1 = il polinomio SOVRASTIMA 1/std in quel punto; > 1.73 = il passo di Newton "
          "diverge (errore enorme).\n  Se il caso peggiore ha una varianza SOTTO il dominio con y0/vero > 1.73, il limite basso e' troppo "
          "stretto.")
    n_bad = sum(1 for _, st in rows if st["err"] > 0.01)
    print(f"\n  layer con errore massimo sopra l'1%: {n_bad} su {len(rows)}. Un errore grande in pochi punti "
          f"puo' pesare poco sul Dice: la riga del Dice e' la misura che conta.")


if __name__ == "__main__":
    main()