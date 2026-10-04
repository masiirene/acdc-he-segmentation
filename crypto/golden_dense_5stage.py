"""
crypto/golden_dense_5stage.py

DA LANCIARE SUL MAC (serve torch, il repo con crypto/remove_deep_stages.py,
il dataset ACDC e il checkpoint).

Perche': finora i test HE sono stati confrontati con riferimenti numpy NOSTRI.
Mai con il modello vero sui pesi veri. E c'e' una domanda che decide il
risultato della tesi: l'HE NON PUO' FARE IL CLAMP (nessun confronto in CKKS).
Nel log di progetto, per il modello a 6 stage: Dice 0.865 con clamp attivo,
0.706 con clamp disattivato (+0.159 dovuto al clamp). Non so se lo 0.870 del
checkpoint a 5 stage / no4norm sia stato misurato con il clamp attivo. Se si',
il Dice sotto cifratura potrebbe essere molto piu' basso di 0.870.

Questo script costruisce un modello "golden" dai pesi estratti
(real_weights_5stage_no4norm.npz) con la STESSA aritmetica dell'HE
senza approssimazioni: convoluzioni, InstanceNorm esatta (eps 1e-5) sulle
sole 14 norm attive, PolyAct, NESSUN clamp. Poi confronta, sulle stesse
immagini di validazione:

  A  modello PyTorch, clamp calibrato ATTIVO     (come e' stato misurato 0.870?)
  B  modello PyTorch, clamp DISATTIVATO          (cio' che l'HE puo' fare)
  C  golden dai pesi .npz                        (deve coincidere con B)

Se C coincide con B (logit a ~1e-4 o meglio), i pesi estratti sono quelli
effettivi, l'architettura e' capita bene, e il Dice di B/C e' il Dice
atteso sotto cifratura MENO le approssimazioni (radice inversa Chebyshev, rumore).
Se C NON coincide con B, il confronto layer per layer dice dove.

Uso (dalla cartella del progetto, sul Mac):
  python3 crypto/golden_dense_5stage.py                      # veloce: 6 batch
  python3 crypto/golden_dense_5stage.py --max_batches 0      # tutta la validazione
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
# La classe del modello a 5 stage: prima UNetKStage (crypto/remove_deep_stages.py, usata
# negli script di ablazione); se quel file non c'e', UNet5Stage da crypto/remove_enc5_stage.py
# (stessa architettura: ConvBlock di models.he_friendly, skip_mode='sum', nn.Conv2d senza WS).
try:
    from crypto.remove_deep_stages import UNetKStage

    def make_model(clamp_values, filters):
        return UNetKStage(5, filters, clamp_values=clamp_values)
    MODEL_SRC = "crypto.remove_deep_stages.UNetKStage"
except ImportError:
    from crypto.remove_enc5_stage import UNet5Stage

    def make_model(clamp_values, filters):
        return UNet5Stage(filters, clamp_values=clamp_values, skip_mode='sum')
    MODEL_SRC = "crypto.remove_enc5_stage.UNet5Stage (fallback: remove_deep_stages.py non trovato)"
from models.he_friendly import PolyAct
from training.dataset import ACDCDataset, load_splits
from training.train import dice_score

BYPASSED = ["dec0.block.1", "enc4.block.1", "dec2.block.4", "enc3.block.1"]
ENC = ["enc0", "enc1", "enc2", "enc3", "enc4"]
UP = ["up3", "up2", "up1", "up0"]
DEC = ["dec3", "dec2", "dec1", "dec0"]
SKIP = ["enc3", "enc2", "enc1", "enc0"]
FILTERS = [32, 64, 128, 256, 128]
NORM_EPS = 1e-5


# ============================================================
# Golden: forward dai pesi .npz, float64, nessun clamp
# ============================================================

class Golden:
    def __init__(self, npz_path):
        self.z = np.load(npz_path)

    def T(self, key):
        return torch.tensor(self.z[key], dtype=torch.float64)

    def _pv(self, key, C):
        t = self.T(key).reshape(-1)
        return t.view(1, -1, 1, 1) if t.numel() == C else t.view(1, 1, 1, 1)

    def norm_act(self, x, name, idx):
        gkey = f"{name}_norm{idx}_gamma"
        if gkey in self.z.files:                       # norm ATTIVA
            mean = x.mean(dim=(2, 3), keepdim=True)
            var = torch.var(x, dim=(2, 3), keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + NORM_EPS)
            x = x * self.T(gkey).view(1, -1, 1, 1) + self.T(f"{name}_norm{idx}_beta").view(1, -1, 1, 1)
        C = x.shape[1]
        a = self._pv(f"{name}_poly{idx}_a", C)
        b = self._pv(f"{name}_poly{idx}_b", C)
        c = self._pv(f"{name}_poly{idx}_c", C)
        return a * x * x + b * x + c                   # PolyAct, NESSUN clamp

    def conv_block(self, x, name, stride):
        x = F.conv2d(x, self.T(f"{name}_conv1_w"), self.T(f"{name}_conv1_b"), stride=stride, padding=1)
        x = self.norm_act(x, name, 1)
        x = F.conv2d(x, self.T(f"{name}_conv2_w"), self.T(f"{name}_conv2_b"), stride=1, padding=1)
        return self.norm_act(x, name, 2)

    def forward(self, x):
        x = x.to(torch.float64)
        inter, enc = {}, {}
        h = x
        for i, name in enumerate(ENC):
            h = self.conv_block(h, name, 1 if i == 0 else 2)
            enc[name] = inter[name] = h
        h = enc["enc4"]
        for up, dec, sk in zip(UP, DEC, SKIP):
            h = F.conv_transpose2d(h, self.T(f"{up}_w"), self.T(f"{up}_b"), stride=2)
            inter[up] = h
            h = h + enc[sk]
            h = self.conv_block(h, dec, 1)
            inter[dec] = h
        out = F.conv2d(h, self.T("out_conv_w"), self.T("out_conv_b"))
        inter["out"] = out
        return out, inter


# ============================================================
# Modello PyTorch (come negli script di ablazione)
# ============================================================

def build_torch_model(ckpt, clamp_json):
    for path, what in ((ckpt, "il checkpoint"), (clamp_json, "il JSON dei clamp")):
        if not os.path.exists(path):
            raise SystemExit(f"Non trovo {what}: {path}\n"
                             f"Questo script va lanciato dove stanno checkpoint e dataset (il Mac), "
                             f"dalla cartella del progetto.")
    with open(clamp_json) as f:
        clamp_values = json.load(f)
    model = make_model(clamp_values, FILTERS)
    print(f"Classe del modello: {MODEL_SRC}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    mods = dict(model.named_modules())
    for name in BYPASSED:
        assert name in mods, f"layer {name} non trovato nel modello"
        mods[name].register_forward_hook(lambda m, i, o: i[0])     # bypass, come negli script
    polys = [m for m in model.modules() if isinstance(m, PolyAct)]
    original = [m.clamp_value for m in polys]
    return model, polys, original


def set_clamp(polys, original, on):
    for m, v in zip(polys, original):
        m.clamp_value = v if on else float("inf")


def torch_intermediates(model, x):
    store, handles = {}, []
    for name in ENC + UP + DEC:
        mod = dict(model.named_modules())[name]
        handles.append(mod.register_forward_hook(lambda m, i, o, n=name: store.__setitem__(n, o.detach().double())))
    out = model(x)
    store["out"] = out.detach().double()
    for h in handles:
        h.remove()
    return out, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/test_5stage_no4norm_finetuned/best_model.pth")
    ap.add_argument("--npz", default="crypto/real_weights_5stage_no4norm.npz")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/Desktop/tesi_acdc/training"))
    ap.add_argument("--splits_path", default=os.path.expanduser("~/Desktop/tesi_acdc/splits_final.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--clamp_values_json", default="crypto/calibrated_clamp_values.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=6, help="0 = tutta la validazione")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    for path, what in ((args.npz, "i pesi .npz"), (args.data_dir, "il dataset"), (args.splits_path, "gli split")):
        if not os.path.exists(path):
            raise SystemExit(f"Non trovo {what}: {path}\n"
                             f"Questo script va lanciato dove stanno checkpoint e dataset (il Mac).")
    model, polys, original = build_torch_model(args.ckpt, args.clamp_values_json)
    n_default = sum(1 for v in original if not np.isfinite(v) or v == 50.0)
    print(f"Modello PyTorch caricato: {len(polys)} PolyAct, clamp attivi di partenza "
          f"(min/mediana/max): {min(original):.1f} / {np.median(original):.1f} / {max(original):.1f}")
    print(f"  ({n_default} layer con clamp 50.0 = valore di default/tetto)")
    gold = Golden(args.npz)
    n_active = sum(1 for k in gold.z.files if k.endswith("_gamma"))
    print(f"Golden caricato da {args.npz}: {n_active} norm attive (attese 14)\n")

    _, val_cases = load_splits(args.splits_path, fold=args.fold)
    ds = ACDCDataset(args.data_dir, val_cases, patch_size=(256, 224), augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---------- confronto layer per layer sul primo batch ----------
    imgs0, segs0 = next(iter(loader))
    set_clamp(polys, original, on=False)
    out_b, inter_t = torch_intermediates(model, imgs0)
    out_g, inter_g = gold.forward(imgs0)
    print("=== Primo batch: golden contro PyTorch (clamp DISATTIVATO, norm bypassate): errore massimo per stadio ===")
    worst = 0.0
    for name in ENC + UP + DEC + ["out"]:
        d = (inter_t[name] - inter_g[name]).abs().max().item()
        scale = inter_t[name].abs().max().item()
        worst = max(worst, d / max(scale, 1e-12))
        print(f"  {name:<6} errore max {d:10.3e}   (scala dei valori {scale:9.3e}, relativo {d / max(scale, 1e-12):8.1e})")
    verdict = "COINCIDONO: pesi estratti = pesi effettivi, architettura capita." if worst < 1e-4 \
        else "NON COINCIDONO: guarda il primo stadio con errore grande (probabili cause: WSConv2d/pesi non effettivi, eps della norm, ordine skip)."
    print(f"  -> errore relativo peggiore {worst:.1e}: {verdict}\n")

    # ---------- Dice ----------
    res = {"A: PyTorch, clamp ATTIVO": [], "B: PyTorch, clamp DISATTIVATO": [], "C: golden (.npz), nessun clamp": []}
    agree, maxdiff = [], []
    n = 0
    for imgs, segs in loader:
        set_clamp(polys, original, on=True)
        la = model(imgs)
        set_clamp(polys, original, on=False)
        lb = model(imgs)
        lc, _ = gold.forward(imgs)
        for key, lg in zip(res, (la, lb, lc)):
            if not torch.isfinite(lg).all():
                continue
            s = dice_score(lg.argmax(dim=1), segs)
            res[key].append([s[1], s[2], s[3]])
        agree.append((lb.argmax(1) == lc.argmax(1)).double().mean().item())
        maxdiff.append((lb.double() - lc).abs().max().item())
        n += 1
        if args.max_batches and n >= args.max_batches:
            break

    print(f"=== Dice su {n} batch (batch_size {args.batch_size}) ===")
    print(f"  {'configurazione':<36}{'RV':>8}{'MYO':>8}{'LV':>8}{'media':>9}")
    for key, vals in res.items():
        if not vals:
            print(f"  {key:<36}  (tutti i batch non finiti)")
            continue
        m = np.mean(vals, axis=0)
        print(f"  {key:<36}{m[0]:8.3f}{m[1]:8.3f}{m[2]:8.3f}{m.mean():9.3f}")
    print(f"\n  B contro C: pixel con stessa classe {100 * np.mean(agree):.2f}%, differenza massima dei logit {max(maxdiff):.2e}")
    if res["A: PyTorch, clamp ATTIVO"] and res["B: PyTorch, clamp DISATTIVATO"]:
        a = np.mean(res["A: PyTorch, clamp ATTIVO"])
        b = np.mean(res["B: PyTorch, clamp DISATTIVATO"])
        print(f"  Effetto del clamp sul Dice (A - B): {a - b:+.3f}")
        print("  -> B e' il Dice che la rete puo' raggiungere sotto cifratura SENZA le approssimazioni"
              " (radice inversa, rumore CKKS). Se B e' molto sotto A, e' la domanda piu' importante della tesi.")


if __name__ == "__main__":
    main()