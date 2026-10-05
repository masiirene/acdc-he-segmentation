"""
crypto/pack_for_zeus.py        (da lanciare sul MAC)

Impacchetta in UN file .npz tutto quello che serve a Zeus per far girare la rete in HE, senza torch:
pesi, struttura (k, filtri, norm bypassate), schemi di radice inversa per layer, un'immagine con la sua etichetta
e i RIFERIMENTI calcolati in chiaro (logit con radice esatta e approssimata, uscite dei primi due canali di ogni stadio).

Due modi:

  --mode real     checkpoint vero + calibrazioni robuste (calibrate_cheb_robust.py) + una fetta del set di validazione
                  (di default quella con piu' pixel di cuore: --slice_index -1)
  --mode narrow   STESSA struttura (stessi nomi di layer, stesso bypass) ma larghezza divisa per --narrow_div
                  (default 8: filtri [4,8,16,32,16]) con pesi casuali; gli schemi si calibrano sulle varianze della
                  rete stessa. Serve a provare il runner in pochi minuti prima di spendere ore sulla rete vera.

Esempi (dalla cartella del progetto):

  # prova a larghezza ridotta
  python3 crypto/pack_for_zeus.py --mode narrow --out crypto/pack_narrow.npz

  # rete vera: 5 stage, 14 norm
  CK5=results/noclamp_5stage_no4norm_w1/best_model.pth
  python3 crypto/pack_for_zeus.py --mode real --ckpt "$CK5" \\
      --bypass dec0.block.1,enc4.block.1,dec2.block.4,enc3.block.1 \\
      --cheb_json crypto/chebyshev_robust_5s14.json --newton_json crypto/newton_robust_5s14.json \\
      --out crypto/pack_5s14.npz

Poi copiare il .npz su Zeus:  scp crypto/pack_5s14.npz masi@zeuslecco:~/acdc-he-segmentation/crypto/

Nel modo real si controlla anche che il riferimento numpy coincida con il golden PyTorch (golden_approx_dice.Net):
se la differenza dei logit e' grande, il pacchetto NON va usato.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import he_network as hn

DEFAULT_BYPASS = "dec0.block.1,enc4.block.1,dec2.block.4,enc3.block.1"
NORM_NAMES = lambda k: [f"enc{i}.block.{b}" for i in range(k) for b in (1, 4)] + \
                       [f"dec{j}.block.{b}" for j in range(k - 2, -1, -1) for b in (1, 4)]


# ============================================================
# Schemi dai JSON delle calibrazioni
# ============================================================

def schemes_from_jsons(cheb_json, newton_json, bypass):
    schemes = {}
    with open(cheb_json) as f:
        for name, e in json.load(f).items():
            if name in bypass or e.get("schema") == "bypass":
                continue
            schemes[name] = {"kind": "cheb", "coef": [float(c) for c in e["cheb_coeffs"]],
                             "dom": [float(e["cheb_domain"][0]), float(e["cheb_domain"][1])], "n": int(e["post_iter"])}
    with open(newton_json) as f:
        for name, e in json.load(f).items():
            if name in bypass:
                continue
            schemes[name] = {"kind": "newton", "y0": float(e["y0"]), "n": int(e["iterations_needed"])}   # il fallback prevale
    return schemes


# ============================================================
# Rete a larghezza ridotta con pesi casuali
# ============================================================

def random_weights(filters, seed=0, num_classes=4):
    rng = np.random.default_rng(seed)
    k = len(filters)
    W = {}

    def conv(name, cin, cout):
        W[name + ".weight"] = rng.normal(size=(cout, cin, 3, 3)) * (0.8 / np.sqrt(cin * 9))
        W[name + ".bias"] = rng.normal(size=cout) * 0.02

    def norm(name, c):
        W[name + ".weight"] = rng.normal(size=c) * 0.1 + 1.0
        W[name + ".bias"] = rng.normal(size=c) * 0.05

    def poly(name):
        W[name + ".a"] = np.array(0.08 + 0.02 * rng.normal())
        W[name + ".b"] = np.array(1.0 + 0.1 * rng.normal())
        W[name + ".c"] = np.array(0.05)

    def block(name, cin, c):
        conv(f"{name}.block.0", cin, c); norm(f"{name}.block.1", c); poly(f"{name}.block.2")
        conv(f"{name}.block.3", c, c); norm(f"{name}.block.4", c); poly(f"{name}.block.5")

    prev = 1
    for i, c in enumerate(filters):
        block(f"enc{i}", prev, c)
        prev = c
    for j in range(k - 2, -1, -1):
        W[f"up{j}.weight"] = rng.normal(size=(filters[j + 1], filters[j], 2, 2)) * (0.8 / np.sqrt(filters[j + 1] * 4))
        W[f"up{j}.bias"] = rng.normal(size=filters[j]) * 0.02
        block(f"dec{j}", filters[j], filters[j])
    W["out_conv.weight"] = rng.normal(size=(num_classes, filters[0], 1, 1)) * (0.8 / np.sqrt(filters[0]))
    W["out_conv.bias"] = rng.normal(size=num_classes) * 0.02
    return W


def calibrate_schemes(W, meta, images, hi_factor=3.0, ext_factor=1.25, log=print):
    """Calibra uno schema per ogni norm con calibrate_cheb_robust.calibrate_layer, sulle varianze della rete esatta."""
    from calibrate_cheb_robust import calibrate_layer
    var_rec = {}
    for im in images:
        hn.reference_forward(W, meta, im, use_schemes=False, var_rec=var_rec)
    schemes = {}
    for name in NORM_NAMES(meta["k"]):
        if name in set(meta["bypass"]):
            continue
        x = np.concatenate(var_rec[name])
        r = calibrate_layer(x, [2, 3, 4, 5, 6], x_max_factor=hi_factor, ext_factor=ext_factor)
        if r is None:
            raise RuntimeError(f"calibrazione fallita per {name}")
        if r["schema"] == "cheb":
            schemes[name] = {"kind": "cheb", "coef": r["cheb_coeffs"], "dom": r["cheb_domain"], "n": r["post_iter"]}
        else:
            schemes[name] = {"kind": "newton", "y0": r["y0"], "n": r["iterations_needed"]}
    n_ch = sum(1 for s in schemes.values() if s["kind"] == "cheb")
    log(f"  schemi calibrati: {n_ch} Chebyshev + {len(schemes) - n_ch} Newton")
    return schemes


def build_narrow(images, label_image, image_target, filters, bypass, seed=0, calib_images=None, hi_factor=3.0,
                 ext_factor=1.25, log=print):
    """images: lista di immagini (1,H,W) per la calibrazione; image_target (1,H,W) + label_image (H,W): la fetta del pacchetto."""
    W = random_weights(filters, seed)
    meta = dict(k=len(filters), filters=list(filters), bypass=list(bypass), schemes={}, mode="narrow", seed=seed)
    meta["schemes"] = calibrate_schemes(W, meta, list(images) + [image_target], hi_factor, ext_factor, log)
    ref = {}
    la = hn.reference_forward(W, meta, image_target, use_schemes=True, record=ref)
    le = hn.reference_forward(W, meta, image_target, use_schemes=False)
    return W, meta, la, le, ref


# ============================================================
# Dati
# ============================================================

def pick_slice(ds, index, seed=0):
    """index >= 0: quella fetta; -1: la fetta con piu' pixel di cuore; -2: una fetta CASUALE con seme fisso (per il test set
    e' la scelta da preferire: non si sceglie la fetta piu' bella)."""
    if index >= 0:
        return index
    if index == -2:
        return int(np.random.default_rng(seed).integers(len(ds)))
    best, best_i = -1, 0
    for i in range(len(ds)):
        seg = np.asarray(ds[i][1])
        fg = int((seg > 0).sum())
        if fg > best:
            best, best_i = fg, i
    return best_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "narrow"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--bypass", default=DEFAULT_BYPASS)
    ap.add_argument("--cheb_json", default="crypto/chebyshev_robust_5s14.json")
    ap.add_argument("--newton_json", default="crypto/newton_robust_5s14.json")
    ap.add_argument("--slice_index", type=int, default=-1, help="indice nel set scelto; -1 = la fetta con piu' cuore; -2 = fetta casuale con --seed (da preferire sul test set)")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--testing_dir", default=os.path.expanduser("~/Desktop/tesi_acdc/testing"))
    ap.add_argument("--test_cases_json", default="crypto/test_cases.json")
    ap.add_argument("--note", default=None, help="OBBLIGATORIA con --split test")
    ap.add_argument("--log_file", default="crypto/test_set_evaluation_log.json")
    ap.add_argument("--skip_confirmation", action="store_true")
    ap.add_argument("--narrow_div", type=int, default=8)
    ap.add_argument("--filters", type=int, nargs="+", default=[32, 64, 128, 256, 128])
    ap.add_argument("--calib_images", type=int, default=12)
    ap.add_argument("--hi_factor", type=float, default=3.0)
    ap.add_argument("--ext_factor", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--data_dir", default=os.path.expanduser("~/Desktop/tesi_acdc/training"))
    ap.add_argument("--splits_path", default=os.path.expanduser("~/Desktop/tesi_acdc/splits_final.json"))
    args = ap.parse_args()

    import torch
    from training.dataset import ACDCDataset, load_splits
    train_cases, val_cases = load_splits(args.splits_path, fold=args.fold)
    if args.split == "test":
        from test_set_guard import confirm_test_set_use
        if not args.note:
            raise SystemExit("--note e' obbligatoria con --split test (regola del progetto)")
        if not confirm_test_set_use(args.note, args.skip_confirmation, "impacchettare una fetta del test set per la prova in HE"):
            return
        with open(args.test_cases_json) as f:
            test_cases = json.load(f)
        ds = ACDCDataset(args.testing_dir, test_cases, patch_size=(256, 224), augment=False)
    else:
        ds = ACDCDataset(args.data_dir, val_cases if args.split == "val" else train_cases, patch_size=(256, 224), augment=False)
    idx = pick_slice(ds, args.slice_index, args.seed)
    img_t, seg_t = ds[idx]
    image = np.asarray(img_t, dtype=np.float64).reshape(1, 256, 224)
    label = np.asarray(seg_t).reshape(256, 224)
    print(f"Fetta scelta: indice {idx} del set {args.split}, pixel di cuore {(label > 0).sum()}")
    bypass = [b for b in args.bypass.split(",") if b]

    if args.mode == "narrow":
        filters = [max(2, f // args.narrow_div) for f in args.filters]
        ds_cal = ds if args.split != "test" else ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
        step = max(1, len(ds_cal) // args.calib_images)
        calib = [np.asarray(ds_cal[i][0], dtype=np.float64).reshape(1, 256, 224) for i in range(0, len(ds_cal), step)][:args.calib_images]
        print(f"Rete a larghezza ridotta: filtri {filters}, bypass {bypass}; calibrazione su {len(calib)} immagini")
        t0 = time.time()
        W, meta, la, le, ref = build_narrow(calib, label, image, filters, bypass, seed=args.seed,
                                            hi_factor=args.hi_factor, ext_factor=args.ext_factor)
        print(f"  fatto in {time.time() - t0:.0f}s")
    else:
        assert args.ckpt, "--ckpt richiesto nel modo real"
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        W = {k: v.double().numpy() for k, v in state.items() if hasattr(v, "numpy")}
        k = 0
        while f"enc{k}.block.0.weight" in W:
            k += 1
        filters = [int(W[f"enc{i}.block.0.weight"].shape[0]) for i in range(k)]
        schemes = schemes_from_jsons(args.cheb_json, args.newton_json, set(bypass))
        meta = dict(k=k, filters=filters, bypass=bypass, schemes=schemes, mode="real", ckpt=args.ckpt)
        missing = [n for n in NORM_NAMES(k) if n not in schemes and n not in set(bypass)]
        if missing:
            raise SystemExit(f"norm senza schema nei JSON: {missing}")
        print(f"Rete vera: k={k}, filtri {filters}, bypass {bypass}, {sum(1 for s in schemes.values() if s['kind'] == 'cheb')} "
              f"Chebyshev + {sum(1 for s in schemes.values() if s['kind'] == 'newton')} Newton")
        t0 = time.time()
        ref = {}
        la = hn.reference_forward(W, meta, image, use_schemes=True, record=ref)
        le = hn.reference_forward(W, meta, image, use_schemes=False)
        print(f"  riferimenti numpy calcolati in {time.time() - t0:.0f}s")
        # controllo incrociato contro il golden PyTorch
        try:
            from golden_approx_dice import Net, approx_inv_std
            net = Net(state, torch, torch.device("cpu"), torch.float64, bypass)
            sch = {n: ("cheb", s["coef"], s["dom"], s["n"]) if s["kind"] == "cheb" else ("newton", s["y0"], s["n"])
                   for n, s in schemes.items()}
            x = torch.from_numpy(image[None]).double()
            with torch.no_grad():
                net.inv_fn = None
                te = net(x)[0].numpy()
                net.inv_fn = lambda name, v: approx_inv_std(v, sch[name])
                ta = net(x)[0].numpy()
            d_e, d_a = np.abs(te - le).max(), np.abs(ta - la).max()
            print(f"  controllo contro PyTorch (golden): differenza massima logit esatti {d_e:.2e}, approssimati {d_a:.2e}")
            if max(d_e, d_a) > 1e-3 * max(1.0, np.abs(te).max()):
                print("  ATTENZIONE: il riferimento numpy NON coincide con PyTorch: non usare questo pacchetto.")
        except Exception as e:
            print(f"  (controllo contro PyTorch non eseguito: {type(e).__name__}: {e})")

    pe, pa_ = le.argmax(0), la.argmax(0)
    d_e, d_a = hn.dice_classes(pe, label), hn.dice_classes(pa_, label)
    print(f"  Dice di questa fetta: esatto {np.mean(d_e):.3f}, approssimato {np.mean(d_a):.3f}; pixel con stessa classe "
          f"esatto/approssimato {100 * np.mean(pe == pa_):.2f}%; scala dei logit {np.abs(la).max():.1f}")
    meta.update(slice_index=int(idx), split=args.split, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    hn.save_pack(args.out, meta, W, image, label, le, la, ref)
    print(f"Scritto {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    if args.split == "test":
        from test_set_guard import append_log
        append_log(args.log_file, dict(kind="pack_for_zeus (una fetta del test set per la prova in HE)", checkpoint=args.ckpt,
                                       note=args.note, slice_index=int(idx), seed=args.seed, out=args.out, mode=args.mode))


if __name__ == "__main__":
    main()