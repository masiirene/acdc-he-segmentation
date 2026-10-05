"""
crypto/he_network.py

LA RETE COMPLETA IN HE (schema a un ciphertext per canale con reticolo), per qualunque numero di stage `sum`,
con norm bypassate, radice inversa per layer (Chebyshev+Newton o Newton puro) e bootstrap inseriti in base ai
LIVELLI REALI dei ciphertext.

Pezzi:
  - pacchetto .npz (prodotto sul Mac da pack_for_zeus.py): pesi, schemi di calibrazione, immagine, riferimenti
  - plan(meta): sequenza degli stadi (conv -> [norm] -> PolyAct), upsampling+skip, convoluzione finale 1x1
  - reference_forward(...): lo STESSO calcolo in numpy (radice inversa approssimata come in HE): serve a
    costruire i riferimenti e a provare il runner senza GPU
  - HERunner: esegue il piano in HE. Politica dei bootstrap:
      * prima di ogni stadio, i canali in ingresso si bootstrappano se livello + costo_stadio (+ margine) > 43,
        dove costo_stadio = 11 + profondita' del layer (10 + profondita', misurato, +1 dopo un bootstrap),
        o 6 per uno stadio senza norm; per gli stadi con costo > 22 (la finestra dopo un bootstrap) si
        bootstrappa comunque e si rinfresca a meta' stadio
      * dentro la radice inversa, a ogni iterazione di Newton: se y (o vh = -v/2) non ha abbastanza livelli
        per il passo successivo, si bootstrappa SOLO quello
      * prima di (x - media) * inv_std si bootstrappa inv_std se non c'e' spazio per i 5 livelli rimanenti
    Il bootstrap lavora IN PLACE sulle liste di ciphertext: la skip (stessi oggetti dell'uscita del blocco
    encoder) viene rinfrescata insieme al ramo principale.
  - convenzione dei coefficienti di Chebyshev della funzione isqrt_chebyshev_fhe del progetto: rilevata da una
    mini-prova all'avvio (numpy: c0 pieno; OpenFHE: c0 dimezzato), cosi' non dipende da un'ipotesi.

NON copre: skip concat, PolyAct per canale (qui scalari), norm di popolazione.
"""

import os
import json
import time
import math
import numpy as np
from numpy.polynomial import chebyshev as C

NORM_EPS = 1e-5
DEPTH = 43
WINDOW = 22                 # livelli disponibili dopo un bootstrap (43 - 21)
BOOT_OUT = 21
S_NONORM = 6                # conv + PolyAct + maschera senza norm (5 misurati, +1 dopo un bootstrap)


# ============================================================
# Pacchetto
# ============================================================

def save_pack(path, meta, W, image, label, logits_exact, logits_approx, ref):
    arrs = {"meta": np.array(json.dumps(meta)), "image": np.asarray(image, dtype=np.float64),
            "label": np.asarray(label, dtype=np.int16), "logits_exact": np.asarray(logits_exact, dtype=np.float64),
            "logits_approx": np.asarray(logits_approx, dtype=np.float64)}
    for k, v in W.items():
        arrs["W__" + k] = np.asarray(v, dtype=np.float64)
    for k, v in ref.items():
        arrs["ref__" + k] = np.asarray(v, dtype=np.float64)
    np.savez_compressed(path, **arrs)


def load_pack(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(d["meta"].item())
    return dict(meta=meta, W={k[3:]: d[k] for k in d.files if k.startswith("W__")},
                ref={k[5:]: d[k] for k in d.files if k.startswith("ref__")},
                image=d["image"], label=d["label"], logits_exact=d["logits_exact"], logits_approx=d["logits_approx"])


def get_scheme(meta, name):
    e = meta["schemes"].get(name)
    if e is None:
        return None
    if e["kind"] == "cheb":
        return ("cheb", [float(c) for c in e["coef"]], [float(e["dom"][0]), float(e["dom"][1])], int(e["n"]))
    return ("newton", float(e["y0"]), int(e["n"]))


def stage_cost(scheme):
    if scheme is None:
        return S_NONORM
    if scheme[0] == "cheb":
        return 11 + 3 * scheme[3] + (len(scheme[1]) - 1)
    return 11 + 3 * scheme[2]


# ============================================================
# Piano della rete
# ============================================================

def plan(meta):
    k, f, byp = meta["k"], meta["filters"], set(meta["bypass"])
    steps = []
    prev = 1
    for i in range(k):
        c = f[i]
        for which, cin in ((1, prev), (2, c)):
            nm = f"enc{i}.block.{1 if which == 1 else 4}"
            first_stride = (i > 0 and which == 1)
            steps.append(dict(type="stage", block=f"enc{i}", which=which, cin=cin, cout=c, stride2=first_stride,
                              level_in=(i - 1) if first_stride else i,
                              conv=f"enc{i}.block.{0 if which == 1 else 3}", norm=None if nm in byp else nm,
                              poly=f"enc{i}.block.{2 if which == 1 else 5}"))
        prev = c
    for j in range(k - 2, -1, -1):
        steps.append(dict(type="up", j=j, level_in=j + 1, cin=f[j + 1], cout=f[j]))
        c = f[j]
        for which in (1, 2):
            nm = f"dec{j}.block.{1 if which == 1 else 4}"
            steps.append(dict(type="stage", block=f"dec{j}", which=which, cin=c, cout=c, stride2=False, level_in=j,
                              conv=f"dec{j}.block.{0 if which == 1 else 3}", norm=None if nm in byp else nm,
                              poly=f"dec{j}.block.{2 if which == 1 else 5}"))
    steps.append(dict(type="out"))
    return steps


def scalar(W, key):
    a = np.asarray(W[key]).reshape(-1)
    if a.size != 1:
        raise ValueError(f"{key}: PolyAct per canale non supportata qui (forma {np.asarray(W[key]).shape})")
    return float(a[0])


# ============================================================
# Riferimento numpy (stessa radice inversa approssimata dell'HE)
# ============================================================

def chebval(t, c):
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
    if scheme[0] == "cheb":
        _, coef, dom, n = scheme
        t = (2.0 * v - dom[0] - dom[1]) / (dom[1] - dom[0])
        y = chebval(t, coef)
    else:
        _, y0, n = scheme
        y = v * 0.0 + y0
    return newton_steps(y, v, n)


def reference_forward(W, meta, image, use_schemes=True, record=None, var_rec=None):
    """image (1,H,W). Ritorna i logit (4,H,W). record: dict che raccoglie i primi 2 canali di ogni uscita di stadio;
    var_rec: dict che raccoglie le varianze (+eps) viste da ogni norm."""
    from prototype_lattice_resample import conv_ref, upconv_ref
    h = np.asarray(image, dtype=np.float64)
    if h.ndim == 2:
        h = h[None]
    skips = {}
    for st in plan(meta):
        if st["type"] == "stage":
            y = conv_ref(h, W[st["conv"] + ".weight"], W[st["conv"] + ".bias"], stride=2 if st["stride2"] else 1)
            if st["norm"] is not None:
                flat = y.reshape(y.shape[0], -1)
                mean, var = flat.mean(axis=1), flat.var(axis=1)
                v = var + NORM_EPS
                if var_rec is not None:
                    var_rec.setdefault(st["norm"], []).append(v.copy())
                inv = approx_inv_std(v, get_scheme(meta, st["norm"])) if use_schemes else 1.0 / np.sqrt(v)
                y = ((y - mean[:, None, None]) * inv[:, None, None] * W[st["norm"] + ".weight"][:, None, None]
                     + W[st["norm"] + ".bias"][:, None, None])
            a, b, c = (scalar(W, st["poly"] + s) for s in (".a", ".b", ".c"))
            h = a * y * y + b * y + c
            if record is not None:
                record[st["poly"]] = h[:2].copy()
            if st["block"].startswith("enc") and st["which"] == 2:
                skips[int(st["block"][3:])] = h
        elif st["type"] == "up":
            h = upconv_ref(h, W[f"up{st['j']}.weight"], W[f"up{st['j']}.bias"]) + skips[st["j"]]
        else:
            w = W["out_conv.weight"][:, :, 0, 0]
            h = np.einsum("oc,chw->ohw", w, h) + W["out_conv.bias"][:, None, None]
    return h


def dice_classes(pred, label):
    out = []
    for c in (1, 2, 3):
        a, b = pred == c, label == c
        s = a.sum() + b.sum()
        out.append(1.0 if s == 0 else 2.0 * (a & b).sum() / s)
    return out


# ============================================================
# Operazioni HE aggiuntive (le stesse della mini U-Net, con il lavaggio opzionale)
# ============================================================

def he_upconv(he, chans, w, b, level_in):
    """ConvTranspose2d(k=2, s=2) sul reticolo: ingresso a level_in, uscita a level_in - 1; 3 rotazioni per canale."""
    cc, g = he.cc, he.g
    lo = level_in - 1
    s = 1 << lo
    shifts = [0, -s, -s * g.Wg, -s * (g.Wg + 1)]
    mask = he.mask_pt(lo)
    cin, cout = len(chans), w.shape[1]
    acc = [None] * cout
    for ci in range(cin):
        x = chans[ci]
        rots = [x if sh == 0 else cc.EvalRotate(x, sh) for sh in shifts]
        for co in range(cout):
            a_ = acc[co]
            for ph in range(4):
                t = cc.EvalMult(rots[ph], float(w[ci, co, ph // 2, ph % 2]))
                a_ = t if a_ is None else cc.EvalAdd(a_, t)
            acc[co] = a_
        x.Offload()
        del rots
        cc.TrimGPUMemoryPool()
    outs = []
    for co in range(cout):
        o = he.mask_mult(cc.EvalAdd(acc[co], float(b[co])), mask)
        o.Offload()
        outs.append(o)
    cc.TrimGPUMemoryPool()
    return outs


def he_add(he, A, B):
    out = []
    for x, y in zip(A, B):
        s = he.cc.EvalAdd(x, y)
        s.Offload()
        out.append(s)
    he.cc.TrimGPUMemoryPool()
    return out


def he_conv1x1(he, chans, w, b):
    cc = he.cc
    outs = []
    for co in range(w.shape[0]):
        acc = None
        for ci, x in enumerate(chans):
            t = cc.EvalMult(x, float(w[co, ci]))
            acc = t if acc is None else cc.EvalAdd(acc, t)
        o = cc.EvalAdd(acc, float(b[co]))
        o.Offload()
        outs.append(o)
    cc.TrimGPUMemoryPool()
    return outs


# ============================================================
# Runner
# ============================================================

class HERunner:
    def __init__(self, he, pack, log=print, mem=None, margin=1, conv_b=8, check=False, stop_after=None):
        self.he, self.cc, self.g = he, he.cc, he.g
        self.pack, self.W, self.meta = pack, pack["W"], pack["meta"]
        self.log = log
        self.mem = mem or (lambda: -1)
        self.margin, self.conv_b, self.check, self.stop_after = margin, conv_b, check, stop_after
        self.nb = dict(entrata=0, newton=0, inversa=0, altro=0)
        self.max_level = 0
        self.mem_peak = 0
        self.c0_factor = 1.0
        self.cheb_post0 = True
        self.skips = {}
        self.rows = []
        self.first_stage = True
        self.t_conv_tot = 0.0
        self.t_norm_tot = 0.0

    # ---------- livelli e bootstrap ----------
    def _note(self, ct):
        self.max_level = max(self.max_level, ct.GetLevel())
        return ct

    def boot(self, ct):
        nb = self.cc.EvalBootstrap(ct)
        ct.Offload()
        nb.Offload()
        return nb

    def ensure1(self, ct, need, tag):
        if ct.GetLevel() + need + self.margin > DEPTH:
            self.nb[tag] += 1
            return self.boot(ct)
        return ct

    def ensure_list(self, cts, need, tag):
        n = 0
        for i, ct in enumerate(cts):
            if ct.GetLevel() + need + self.margin > DEPTH:
                cts[i] = self.boot(ct)
                n += 1
                if n % 4 == 0:
                    self.cc.TrimGPUMemoryPool()
        self.nb[tag] += n
        if n:
            self.cc.TrimGPUMemoryPool()
        return n

    # ---------- convenzione di Chebyshev ----------
    def detect_cheb_convention(self):
        """isqrt_chebyshev_fhe del progetto: i coefficienti dei JSON sono in convenzione numpy (c0 pieno). Si prova con una
        mini-valutazione se la funzione li vuole cosi' o con c0 raddoppiato (convenzione OpenFHE)."""
        from crypto.fhe_ops_single_ciphertext import isqrt_chebyshev_fhe
        scheme = next((get_scheme(self.meta, n) for n in self.meta["schemes"]
                       if self.meta["schemes"][n]["kind"] == "cheb"), None)
        if scheme is None:
            self.log("  nessun layer Chebyshev: convenzione non necessaria")
            return
        _, coef, dom, _n = scheme
        v = np.linspace(dom[0], dom[1], 2048)
        vec = np.zeros(self.g.N)
        vec[:v.size] = v
        ct = self.he.enc(vec)
        exp0 = C.Chebyshev(coef, domain=dom)(v)
        best = None
        for factor in (1.0, 2.0):
            coef_he = [coef[0] * factor] + list(coef[1:])
            post0 = True
            try:
                y = isqrt_chebyshev_fhe(self.cc, ct, coef_he, list(dom), 0)
            except Exception:
                post0 = False
                y = isqrt_chebyshev_fhe(self.cc, ct, coef_he, list(dom), 1)
            got = self.he.dec(y)[:v.size]
            exp = exp0 if post0 else newton_steps(exp0, v, 1)
            err = float(np.max(np.abs(got - exp)))
            self.log(f"  prova convenzione: c0 x {factor:g} (post_iter {'0' if post0 else '1'}): errore {err:.2e}")
            if best is None or err < best[0]:
                best = (err, factor, post0)
        if best[0] > 1e-2:
            raise RuntimeError(f"nessuna convenzione dei coefficienti riproduce il polinomio (errore {best[0]:.2e}): "
                               f"guardare isqrt_chebyshev_fhe")
        self.c0_factor, self.cheb_post0 = best[1], best[2]
        self.log(f"  convenzione scelta: c0 x {best[1]:g}, post_iter=0 {'supportato' if best[2] else 'NON supportato (si usa 1)'}")

    # ---------- radice inversa ----------
    def isqrt(self, var, scheme):
        from crypto.fhe_ops_single_ciphertext import isqrt_chebyshev_fhe
        cc = self.cc
        vh = cc.EvalMult(var, -0.5)
        if scheme[0] == "cheb":
            _, coef, dom, n = scheme
            coef_he = [coef[0] * self.c0_factor] + list(coef[1:])
            if self.cheb_post0:
                y, done = isqrt_chebyshev_fhe(cc, var, coef_he, list(dom), 0), 0
            else:
                y, done = isqrt_chebyshev_fhe(cc, var, coef_he, list(dom), 1), 1
        else:
            y0, n = scheme[1], scheme[2]
            if n == 0:
                y, done = cc.EvalAdd(cc.EvalMult(var, 0.0), y0), 0
            else:
                y, done = cc.EvalAdd(cc.EvalMult(var, -0.5 * y0 ** 3), 1.5 * y0), 1    # prima iterazione in forma chiusa
        for _ in range(done, n):
            y = self.ensure1(y, 3, "newton")
            vh = self.ensure1(vh, 2, "newton")
            y2 = cc.EvalMult(y, y)
            t = cc.EvalMult(vh, y2)
            u = cc.EvalAdd(t, 1.5)
            y = cc.EvalMult(y, u)
        return y

    # ---------- un canale: norm + PolyAct + maschera ----------
    def channel(self, x, scheme, gamma_c, beta_c, pa, mask, n_valid):
        from crypto.fhe_ops_single_ciphertext import poly_act_fhe
        from lattice_conv import sum_all_slots
        cc, N = self.cc, self.g.N
        if scheme is None:
            x = self.ensure1(x, 3, "altro")
            return self.he.mask_mult(poly_act_fhe(cc, x, *pa), mask)
        x = self.ensure1(x, 3, "altro")
        mean = cc.EvalMult(sum_all_slots(cc, x, N), 1.0 / n_valid)
        mean_sq = cc.EvalMult(sum_all_slots(cc, cc.EvalMult(x, x), N), 1.0 / n_valid)
        var = cc.EvalAdd(cc.EvalSub(mean_sq, cc.EvalMult(mean, mean)), NORM_EPS)
        inv = self.isqrt(var, scheme)
        xc = cc.EvalSub(x, mean)
        xc = self.ensure1(xc, 5, "inversa")
        inv = self.ensure1(inv, 5, "inversa")
        scaled = cc.EvalAdd(cc.EvalMult(cc.EvalMult(xc, inv), gamma_c), beta_c)
        return self.he.mask_mult(poly_act_fhe(cc, scaled, *pa), mask)

    # ---------- passi ----------
    def _sample_mem(self):
        self.mem_peak = max(self.mem_peak, self.mem())

    def run_stage(self, chans, st):
        he, cc, g, W = self.he, self.cc, self.g, self.W
        scheme = get_scheme(self.meta, st["norm"]) if st["norm"] else None
        cost = stage_cost(scheme) - (1 if self.first_stage else 0)
        nb0 = dict(self.nb)
        lvl_in = max(c.GetLevel() for c in chans)
        self.ensure_list(chans, min(cost, WINDOW), "entrata")
        out_level = st["level_in"] + (1 if st["stride2"] else 0)
        t0 = time.time()
        conv = he.conv3x3_blocked(chans, W[st["conv"] + ".weight"], W[st["conv"] + ".bias"], st["level_in"],
                                  st["stride2"], B=self.conv_b)
        he.dec(conv[-1])
        t_conv = time.time() - t0
        mask = he.mask_pt(out_level)
        Hh, Ww = g.size(out_level)
        pa = tuple(scalar(W, st["poly"] + s) for s in (".a", ".b", ".c"))
        gamma = W[st["norm"] + ".weight"] if scheme else None
        beta = W[st["norm"] + ".bias"] if scheme else None
        t0 = time.time()
        outs = []
        for c, x in enumerate(conv):
            out = self.channel(x, scheme, None if gamma is None else float(gamma[c]),
                               None if beta is None else float(beta[c]), pa, mask, Hh * Ww)
            out.Offload()
            x.Offload()
            outs.append(self._note(out))
            if c % 4 == 3:
                cc.TrimGPUMemoryPool()
        cc.TrimGPUMemoryPool()
        vec0 = he.dec(outs[0])
        t_norm = time.time() - t0
        self.t_conv_tot += t_conv
        self.t_norm_tot += t_norm
        self._sample_mem()
        name = f"{st['block']}.{st['which']}"
        row = dict(name=name, kind=("senza norm" if scheme is None else scheme[0]), cin=st["cin"], cout=st["cout"],
                   lvl_in=lvl_in, lvl_out=max(c.GetLevel() for c in outs),
                   boots={k: self.nb[k] - nb0[k] for k in self.nb}, t_conv=t_conv, t_norm=t_norm, mem=self.mem(), err=None)
        if self.check:
            ref = self.pack["ref"].get(st["poly"])
            if ref is not None:
                row["err"] = float(np.abs(g.unpack(vec0, out_level) - ref[0]).max())
        self.rows.append(row)
        self._log_row(row)
        return outs

    def run_up(self, chans, st):
        he, g, W = self.he, self.g, self.W
        nb0 = dict(self.nb)
        lvl_in = max(c.GetLevel() for c in chans)
        self.ensure_list(chans, 2, "altro")
        t0 = time.time()
        up = he_upconv(he, chans, W[f"up{st['j']}.weight"], W[f"up{st['j']}.bias"], st["level_in"])
        out = he_add(he, up, self.skips[st["j"]])
        he.dec(out[0])
        t = time.time() - t0
        self._sample_mem()
        row = dict(name=f"up{st['j']}+skip", kind="upsampling", cin=st["cin"], cout=st["cout"], lvl_in=lvl_in,
                   lvl_out=max(c.GetLevel() for c in out), boots={k: self.nb[k] - nb0[k] for k in self.nb},
                   t_conv=t, t_norm=0.0, mem=self.mem(), err=None)
        self.t_conv_tot += t
        self.rows.append(row)
        self._log_row(row)
        return out

    def run_out(self, chans):
        he, g, W = self.he, self.g, self.W
        self.ensure_list(chans, 2, "altro")
        cts = he_conv1x1(he, chans, W["out_conv.weight"][:, :, 0, 0], W["out_conv.bias"])
        self.final_level = max(c.GetLevel() for c in cts)
        return np.stack([g.unpack(he.dec(c), 0) for c in cts])

    def _log_row(self, r):
        b = r["boots"]
        err = "" if r["err"] is None else f" | errore canale 0 {r['err']:.2e}"
        self.log(f"  {r['name']:<12}{r['kind']:<11}{r['cin']:>4}->{r['cout']:<4} livelli {r['lvl_in']:>2}->{r['lvl_out']:<2} | "
                 f"bootstrap ingresso {b['entrata']:>4} newton {b['newton']:>4} inversa {b['inversa']:>4} altro {b['altro']:>3} | "
                 f"conv {r['t_conv']:7.1f}s norm {r['t_norm']:7.1f}s | GPU {r['mem']} MiB{err}")

    # ---------- esecuzione ----------
    def run(self):
        g, he, pack = self.g, self.he, self.pack
        t_start = time.time()
        self.log("\nConvenzione dei coefficienti di Chebyshev ...")
        self.detect_cheb_convention()
        image = pack["image"]
        chans = he.enc_offloaded([g.pack(image[0] if image.ndim == 3 else image, 0)])
        self.log(f"\n  {'stadio':<12}{'schema':<11}{'canali':>9} {'livelli':<15}{'bootstrap':<62}{'tempi':<27}")
        logits = None
        for st in plan(self.meta):
            if st["type"] == "stage":
                chans = self.run_stage(chans, st)
                self.first_stage = False
                if st["block"].startswith("enc") and st["which"] == 2:
                    self.skips[int(st["block"][3:])] = chans
                if self.stop_after and st["block"] == self.stop_after and st["which"] == 2:
                    self.log(f"\n  STOP_AFTER={self.stop_after}: esecuzione parziale")
                    break
            elif st["type"] == "up":
                chans = self.run_up(chans, st)
            else:
                logits = self.run_out(chans)
        self.t_total = time.time() - t_start
        return self.report(logits)

    def report(self, logits):
        n_boot = sum(self.nb.values())
        L = self.log
        L("\n=== RIEPILOGO ===")
        L(f"  tempo totale: {self.t_total / 60:.1f} min (conv {self.t_conv_tot / 60:.1f} min, norm+attivazione "
          f"{self.t_norm_tot / 60:.1f} min)")
        L(f"  bootstrap totali: {n_boot}  (ingresso agli stadi {self.nb['entrata']}, dentro la radice inversa "
          f"{self.nb['newton']}, su inv_std {self.nb['inversa']}, altro {self.nb['altro']})")
        L(f"  livello massimo raggiunto: {self.max_level} (limite {DEPTH}); picco memoria GPU: {self.mem_peak} MiB")
        res = dict(n_boot=n_boot, max_level=self.max_level, mem_peak=self.mem_peak, t_total=self.t_total)
        if logits is None:
            return res
        la, le, lab = self.pack["logits_approx"], self.pack["logits_exact"], self.pack["label"]
        err = np.abs(logits - la)
        pred = logits.argmax(0)
        L(f"  livello dei logit: {self.final_level}")
        L(f"  logit HE vs riferimento APPROSSIMATO: errore max {err.max():.3e}, medio {err.mean():.3e} "
          f"(scala dei logit {np.abs(la).max():.1f})")
        L(f"  classe per pixel: HE = approssimato {100 * np.mean(pred == la.argmax(0)):.2f}% | HE = esatto "
          f"{100 * np.mean(pred == le.argmax(0)):.2f}% | HE = etichetta {100 * np.mean(pred == lab):.2f}%")
        d_he, d_a, d_e = dice_classes(pred, lab), dice_classes(la.argmax(0), lab), dice_classes(le.argmax(0), lab)
        L(f"  Dice di questa fetta (RV/MYO/LV): HE {np.mean(d_he):.3f} ({d_he[0]:.3f}/{d_he[1]:.3f}/{d_he[2]:.3f})  "
          f"approssimato {np.mean(d_a):.3f}  esatto {np.mean(d_e):.3f}")
        res.update(err_max=float(err.max()), err_mean=float(err.mean()), agree_approx=float(np.mean(pred == la.argmax(0))),
                   dice_he=float(np.mean(d_he)), dice_approx=float(np.mean(d_a)), dice_exact=float(np.mean(d_e)))
        return res