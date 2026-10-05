"""
crypto/calibrate_cheb_robust.py

RICALIBRAZIONE ROBUSTA della radice inversa (Chebyshev monotono + Newton).

PROBLEMA TROVATO (golden_approx_dice.py sul 6 stage sum senza clamp):
la calibrazione originale (simulate_chebyshev_monotonic.py) valuta il polinomio
su varianze TAGLIATE al dominio (np.clip) e poi misura la convergenza di Newton
sulle varianze vere. In HE non si puo' tagliare niente: il polinomio e' valutato
fuori dominio, dove cresce rapidamente, y0 sovrastima 1/sqrt(v) di piu' di 1.73
volte e il passo di Newton diverge. Risultato: 3 fette su 368 con NaN/Inf, errori
fino a 1e17% in dec2.block.1 (variazione 18.7 contro un dominio fino a 2.085).

QUESTO SCRIPT, per ogni layer di norm:
  - dominio [x_min, x_top]: x_min = percentile --x_min_pct (default 0.1, come prima: sotto il
    dominio il polinomio sottostima e Newton converge lentamente ma NON diverge),
    x_top = massimo osservato * --x_max_factor (default 2.0)
  - polinomio di Chebyshev interpolante di 1/sqrt, traslato verso il basso del massimo di
    sovrastima su TUTTA una griglia del dominio (e non solo sui campioni) piu' un margine:
    y0 <= 1/sqrt(v) dentro il dominio, per costruzione
  - valutazione SENZA CLIP sul campione e sulla griglia
  - criteri: 99.9% del campione entro --target (1%) E nessun punto del campione o della griglia
    sopra --tail_max (default 0.5): niente divergenze, niente fallimenti grossolani
  - sceglie il grado e il numero di iterazioni di Newton con profondita' minima (3 n + grado)
  - i layer che non convergono con nessun grado vanno a Newton puro con y0 = 1/sqrt(x_top)
Scrive FILE NUOVI (non sovrascrive le calibrazioni esistenti):
  crypto/chebyshev_calibration_robust.json  (stesso formato di chebyshev_calibration_monotonic.json)
  crypto/newton_fallback_robust.json        (stesso formato di newton_fallback_full22.json)

Uso (dalla cartella del progetto, sul Mac):
  python3 crypto/calibrate_cheb_robust.py --checkpoint <path> [--bypass layer1,layer2] \\
      --clamp_values_json crypto/huge_clamp.json   (k, filtri e skip si deducono dal checkpoint)
Poi si misura con:
  python3 crypto/golden_approx_dice.py --ckpt <path> \\
      --cheb_json crypto/chebyshev_calibration_robust.json --newton_json crypto/newton_fallback_robust.json

LIMITE: le varianze di calibrazione sono quelle della pipeline ESATTA (come nell'originale). In HE le
varianze di un layer dipendono dalle approssimazioni dei layer precedenti: il margine x_max_factor serve a
coprirlo, ma la verifica vera e' golden_approx_dice.py (pipeline approssimata end-to-end). Se dopo questa
ricalibrazione restano fette con NON FINITI, il passo successivo e' una calibrazione layer per layer
sulla pipeline approssimata.
"""

import os
import sys
import json
import argparse
import numpy as np
from numpy.polynomial import chebyshev as C


# ============================================================
# Nucleo numerico (solo numpy: si prova senza torch)
# ============================================================

def rel_err(y, x):
    """Errore relativo di y rispetto a 1/sqrt(x)."""
    return np.abs(y * np.sqrt(x) - 1.0)


def newton(y, x, n):
    for _ in range(n):
        y = y * (1.5 - 0.5 * x * y * y)
    return y


def calibrate_layer(x, degrees, max_iter=15, target=0.01, x_min_pct=0.1, x_max_factor=2.0,
                    tail_max=0.5, crit=0.999, safety=1.4, ext_factor=3.0, min_ratio=0.25, max_ratio=1.6):
    """Ritorna un dict con lo schema scelto ('cheb' o 'newton') o None se non converge nemmeno con Newton."""
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 0]
    x_min = float(np.percentile(x, x_min_pct))
    x_top = float(x.max() * x_max_factor)
    grid = np.geomspace(x_min, x_top, 4000)
    check = np.concatenate([x, grid])

    best = None
    for deg in degrees:
        poly = C.Chebyshev.interpolate(lambda v: 1.0 / np.sqrt(v), deg, domain=[x_min, x_top])
        over = float(np.max(poly(grid) - 1.0 / np.sqrt(grid)))
        shift = max(0.0, over) * safety
        coef = poly.coef.copy()
        coef[0] -= shift
        mono = C.Chebyshev(coef, domain=[x_min, x_top])
        y0_s, y0_c = mono(x), mono(check)               # NESSUN clip: come in HE
        if y0_c.min() <= 1e-6:
            continue                                      # y0 non positivo: Newton non converge
        if ext_factor > 1.0:                              # SICUREZZA ESTESA: stabile anche OLTRE il dominio, fino a ext_factor x x_top
            ext = np.geomspace(x_min, ext_factor * x_top, 6000)
            r_ext = mono(ext) * np.sqrt(ext)              # y0 / (1/sqrt(v)): Newton diverge sopra 1.732
            if r_ext.max() > max_ratio or r_ext.min() < min_ratio:
                continue
        for n in range(max_iter + 1):
            e_s = rel_err(newton(y0_s, x, n), x)
            e_c = rel_err(newton(y0_c, check, n), check)
            if np.mean(e_s < target) >= crit and e_s.max() < tail_max and e_c.max() < tail_max:
                depth = 3 * n + deg
                if best is None or depth < best["total_depth"]:
                    best = dict(schema="cheb", cheb_coeffs=[float(c) for c in coef],
                                cheb_domain=[x_min, x_top], post_iter=n, poly_degree=deg,
                                total_depth=depth, shift_applied=float(shift),
                                err99=float(np.percentile(e_s, 99.9)), err_max=float(e_s.max()))
                break
    if best is not None:
        return best

    # Newton puro con y0 scalare = 1/sqrt(x_top): sottostima per ogni v <= x_top (e converge fino a v < 3 x_top)
    y0 = 1.0 / np.sqrt(x_top)
    for n in range(max_iter + 1):
        e_s = rel_err(newton(np.full_like(x, y0), x, n), x)
        e_c = rel_err(newton(np.full_like(check, y0), check, n), check)
        if np.mean(e_s < target) >= crit and e_s.max() < tail_max and e_c.max() < tail_max:
            return dict(schema="newton", y0=float(y0), iterations_needed=n, total_depth=3 * n,
                        x_top=x_top, err99=float(np.percentile(e_s, 99.9)), err_max=float(e_s.max()))
    return None


def cliff_factor(r, x_max, max_factor=50.0):
    """Primo valore di varianza, in multipli del MASSIMO osservato, per cui lo schema scelto sbaglia di oltre il 50%
    (o diverge). inf = nessun problema fino a max_factor volte il massimo."""
    v = x_max * np.geomspace(1.0, max_factor, 800)
    with np.errstate(all="ignore"):
        if r["schema"] == "cheb":
            y = newton(C.Chebyshev(r["cheb_coeffs"], domain=r["cheb_domain"])(v), v, r["post_iter"])
        else:
            y = newton(np.full_like(v, r["y0"]), v, r["iterations_needed"])
        e = np.abs(y * np.sqrt(v) - 1.0)
    bad = ~np.isfinite(e) | (e > 0.5)
    return float(v[bad.argmax()] / x_max) if bad.any() else float("inf")


def old_style_check(x, degrees, max_iter=15, target=0.01):
    """Riproduce il criterio della calibrazione ORIGINALE (percentile 99.9 * 1.2, valutazione sul clip) e
    poi misura cosa succede valutando il polinomio SENZA clip. Serve solo a mostrare la differenza."""
    x = x[x > 0]
    x_min, x_max = np.percentile(x, 0.1), np.percentile(x, 99.9) * 1.2
    for deg in degrees:
        poly = C.Chebyshev.interpolate(lambda v: 1.0 / np.sqrt(v), deg, domain=[x_min, x_max])
        xc = np.clip(x, x_min, x_max)
        over = np.max(poly(xc) - 1.0 / np.sqrt(xc))
        coef = poly.coef.copy(); coef[0] -= max(0.0, over) * 1.4
        mono = C.Chebyshev(coef, domain=[x_min, x_max])
        y0_clip = np.maximum(mono(xc), 1e-6)
        for n in range(max_iter + 1):
            if np.mean(rel_err(newton(y0_clip, x, n), x) < target) >= 0.999:
                y0_raw = mono(x)                           # come in HE: nessun clip
                e = rel_err(newton(y0_raw, x, n), x)
                with np.errstate(all="ignore"):
                    return dict(deg=deg, n=n, domain=[float(x_min), float(x_max)],
                                err_max_clip=float(rel_err(newton(y0_clip, x, n), x).max()),
                                err_max_unclipped=float(np.nanmax(np.where(np.isfinite(e), e, np.inf))),
                                frac_outside=float(np.mean((x < x_min) | (x > x_max))))
                break
    return None


# ============================================================
# Parte con torch: raccolta delle varianze e scrittura
# ============================================================

def main():
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, '.')
    import torch.nn as nn
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from training.dataset import ACDCDataset, load_splits
    from clamp_sweep import infer_config, build_model

    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--filters', type=int, nargs='*', default=None, help='IGNORATO: k e filtri si deducono dal checkpoint')
    ap.add_argument('--skip_mode', default=None, help='IGNORATO: si deduce dal checkpoint')
    ap.add_argument('--bypass', default='', help='layer di norm bypassati (csv): il bypass viene APPLICATO al forward e i layer non vengono calibrati')
    ap.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    ap.add_argument('--splits_path', default=os.path.expanduser('~/Desktop/tesi_acdc/splits_final.json'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--clamp_values_json', default='crypto/huge_clamp.json')
    ap.add_argument('--cheb_degrees', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument('--max_iter', type=int, default=15)
    ap.add_argument('--target', type=float, default=0.01)
    ap.add_argument('--x_min_pct', type=float, default=0.1)
    ap.add_argument('--x_max_factor', type=float, default=2.0)
    ap.add_argument('--tail_max', type=float, default=0.5)
    ap.add_argument('--include_train_slices', type=int, default=320,
                    help="fette di TRAINING aggiunte ai campioni di varianza (0 = solo validazione): piu' pazienti, coda meglio coperta")
    ap.add_argument('--ext_factor', type=float, default=3.0,
                    help="sicurezza oltre il dominio: il polinomio deve restare tra min_ratio e max_ratio volte il vero fino a "
                         "ext_factor x il limite alto (1 = nessun controllo, come prima di questa opzione)")
    ap.add_argument('--min_ratio', type=float, default=0.25)
    ap.add_argument('--max_ratio', type=float, default=1.6)
    ap.add_argument('--out_cheb', default='crypto/chebyshev_calibration_robust.json')
    ap.add_argument('--out_newton', default='crypto/newton_fallback_robust.json')
    ap.add_argument('--old_cheb', default='crypto/chebyshev_calibration_monotonic.json')
    args = ap.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else \
        (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    with open(args.clamp_values_json) as f:
        clamp_values = json.load(f)
    if max(clamp_values.values()) < 1e4:
        print("ATTENZIONE: i clamp non sono 'enormi': le varianze misurate non sono quelle della rete senza clamp.")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    k, filters, skip_mode = infer_config(state)
    model, strict = build_model(k, filters, skip_mode, clamp_values)
    model.load_state_dict(state, strict=strict)
    model.to(device).eval()
    bypass = [b for b in args.bypass.split(",") if b]
    mods = dict(model.named_modules())
    for b in bypass:
        assert b in mods, f"layer da bypassare non trovato: {b}"
        mods[b].register_forward_hook(lambda m, i, o: i[0])      # il bypass e' parte del forward durante la raccolta
    print(f"Modello dedotto dal checkpoint: k={k}, filters={filters}, skip={skip_mode}; norm bypassate: {len(bypass)}")

    def collect_raw_variance_samples(model, loader, device):
        store, handles = {}, []

        def make(name):
            def hook(module, inputs):
                v = inputs[0].var(dim=[2, 3], unbiased=False)
                store.setdefault(name, []).append(v.detach().flatten().cpu().numpy())
            return hook
        for name, m in model.named_modules():
            if isinstance(m, nn.InstanceNorm2d) and name not in bypass:
                handles.append(m.register_forward_pre_hook(make(name)))
        with torch.no_grad():
            for imgs, _ in loader:
                model(imgs.to(device))
        for h in handles:
            h.remove()
        return {n: np.concatenate(v) for n, v in store.items()}

    train_cases, val_cases = load_splits(args.splits_path, fold=args.fold)
    val_ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print('Raccolgo le varianze per layer (pipeline esatta, come nell\'originale)...\n')
    raw = collect_raw_variance_samples(model, loader, device)
    if args.include_train_slices > 0:
        from torch.utils.data import Subset
        tr_ds = ACDCDataset(args.data_dir, train_cases, patch_size=(256, 224), augment=False)
        n_tr = min(args.include_train_slices, len(tr_ds))
        raw_tr = collect_raw_variance_samples(model, DataLoader(Subset(tr_ds, range(n_tr)), batch_size=8, shuffle=False,
                                                                num_workers=0), device)
        for name in raw:
            if name in raw_tr:
                raw[name] = np.concatenate([np.asarray(raw[name]), np.asarray(raw_tr[name])])
        print(f"  aggiunte {n_tr} fette di training ai campioni di validazione ({len(val_ds)} fette)")

    old = {}
    if os.path.exists(args.old_cheb):
        with open(args.old_cheb) as f:
            old = json.load(f)

    def fmt_cliff(r, x):
        c = cliff_factor(r, float(x.max()))
        return '>50x' if c == float('inf') else f'{c:.1f}x'

    cheb_out, newton_out = {}, {}
    tot_cheb = tot_newton = 0
    print(f"{'layer':<14}{'schema':<8}{'grado':>6}{'iter':>5}{'profondita':>11}{'(prima)':>9}  {'dominio':<22}"
          f"{'max var':>9}{'err99':>8}{'err max':>9}{'precipizio':>12}")
    print('-' * 118)
    for name in sorted(raw.keys()):
        x = np.asarray(raw[name], dtype=np.float64)
        r = calibrate_layer(x, args.cheb_degrees, args.max_iter, args.target, args.x_min_pct,
                            args.x_max_factor, args.tail_max, ext_factor=args.ext_factor,
                            min_ratio=args.min_ratio, max_ratio=args.max_ratio)
        was = old.get(name, {}).get('total_depth', '-')
        if r is None:
            print(f"{name:<14}NESSUNO schema converge (Chebyshev e Newton): da guardare a mano")
            continue
        if r['schema'] == 'cheb':
            cheb_out[name] = {k: r[k] for k in ('cheb_coeffs', 'cheb_domain', 'post_iter', 'poly_degree',
                                                'total_depth', 'shift_applied')}
            tot_cheb += r['total_depth']
            dom = f"[{r['cheb_domain'][0]:.3f},{r['cheb_domain'][1]:.3f}]"
            print(f"{name:<14}{'cheb':<8}{r['poly_degree']:>6}{r['post_iter']:>5}{r['total_depth']:>11}{str(was):>9}  "
                  f"{dom:<22}{x.max():>9.3f}{100 * r['err99']:>7.2f}%{100 * r['err_max']:>8.1f}%{fmt_cliff(r, x):>12}")
        else:
            newton_out[name] = {'y0': r['y0'], 'iterations_needed': r['iterations_needed']}
            tot_newton += r['total_depth']
            print(f"{name:<14}{'newton':<8}{'-':>6}{r['iterations_needed']:>5}{r['total_depth']:>11}{str(was):>9}  "
                  f"{'y0=' + format(r['y0'], '.4f'):<22}{x.max():>9.3f}{100 * r['err99']:>7.2f}%{100 * r['err_max']:>8.1f}%{fmt_cliff(r, x):>12}")
    print('-' * 118)
    print("  'precipizio' = prima varianza (in multipli del massimo osservato) per cui lo schema sbaglia di oltre il 50% o diverge.")
    old_total = sum(e.get('total_depth', 0) for e in old.values())
    print(f"\nProfondita' totale Chebyshev: {tot_cheb} (prima: {old_total} sui layer che convergevano); "
          f"Newton puro: {tot_newton} su {len(newton_out)} layer")
    with open(args.out_cheb, 'w') as f:
        json.dump(cheb_out, f, indent=2)
    with open(args.out_newton, 'w') as f:
        json.dump(newton_out, f, indent=2)
    print(f"Scritti: {args.out_cheb}, {args.out_newton}")


if __name__ == '__main__':
    main()