"""
crypto/test_stage_levels_lattice_zeus.py

DOMANDA: nello schema a UN CIPHERTEXT PER CANALE con reticolo, quanti LIVELLI
costa uno stadio completo  conv 3x3 -> InstanceNorm -> PolyAct -> maschera?
(Dopo un bootstrap restano ~22 livelli: uno stadio ci sta solo se costa meno.
Il dato che abbiamo e' del formato a righe: 20 livelli. Qui si misura quello
vero dello schema che stiamo usando.)

Quattro stadi in HE vero, caso piccolo (immagine 16x12, N=256 slot):
  1) conv 3->4, livello 0          2) conv 4->4, livello 0
  3) conv 4->6 STRIDE 2 (0 -> 1)   4) conv 6->6, livello 1
Gli stadi 3 e 4 verificano in CKKS vero le statistiche della norm a
risoluzione RIDOTTA (n_valid = 8x6 pixel sul reticolo di passo 2).

Statistiche: somma su TUTTI gli slot con log2(N) passi rotazione+somma
(N = batch CKKS esatto, quindi il giro si chiude su N: ogni slot riceve la
somma totale, senza broadcast ne' AccumulateSum). Gli slot fuori dal
reticolo valido sono zero (mascherati), quindi non contribuiscono.
Radice inversa: Chebyshev grado 3 + 1 Newton (isqrt_chebyshev_fhe del progetto),
dominio per stadio dalle varianze (in un test; nella rete vera dalla calibrazione).

Per ogni stadio si stampa la TRACCIA DEI LIVELLI del canale 0 passo per passo,
il tempo, e l'errore contro DUE riferimenti numpy: "approssimato" (stessa
radice inversa dell'HE: misura la pipeline) ed "esatto" (include il costo
dell'approssimazione).

REFRESH SIMULATO tra gli stadi (decifra+ricifra con la chiave segreta):
NON e' un bootstrap. Serve a misurare ogni stadio PARTENDO DA LIVELLO 0 e a non
uscire dai 43 livelli. Il costo misurato e' quello che va confrontato con i
~22 livelli disponibili dopo un bootstrap vero.

Uso (in tmux): python3 crypto/test_stage_levels_lattice_zeus.py
Richiede nella stessa cartella: prototype_lattice_resample.py e test_lattice_resample_zeus.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import poly_act_fhe, isqrt_chebyshev_fhe, fit_monotonic_isqrt_coeffs
from prototype_lattice_resample import Grid, conv_ref
from test_lattice_resample_zeus import LatticeHE, level_offsets

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
H0, W0, L = 16, 12, 2
A_, B_, C_ = 0.1, 1.0, 0.5
BOOTSTRAP_BUDGET = 22          # livelli disponibili dopo un bootstrap (43 - 21, misurato)


# ============================================================
# Riferimenti numpy
# ============================================================

def norm_act_ref(y, gamma, beta, inv_fn):
    c = y.shape[0]
    flat = y.reshape(c, -1)
    mean, var = flat.mean(axis=1), flat.var(axis=1)
    inv = inv_fn(var)
    s = (y - mean[:, None, None]) * inv[:, None, None] * gamma[:, None, None] + beta[:, None, None]
    return A_ * s ** 2 + B_ * s + C_


def make_emulator(coeffs, domain):
    x_min, x_max = domain
    scale = 2.0 / (x_max - x_min)
    offset = -(x_min + x_max) / (x_max - x_min)

    def emu(var_vals):
        out = []
        for v in var_vals:
            t = scale * v + offset
            b1 = b2 = 0.0
            for c in reversed(coeffs[1:]):
                b1, b2 = 2.0 * t * b1 - b2 + c, b1
            y = t * b1 - b2 + coeffs[0] / 2.0
            out.append(y * (1.5 - 0.5 * v * y * y))       # un passo di Newton
        return np.array(out)
    return emu


def domain_from_var(var):
    return (max(0.01, 0.5 * var.min()), 2.0 * var.max())


# ============================================================
# Un intero stadio in HE
# ============================================================

def sum_all_slots(cc, ct, N):
    """Somma di tutti gli N slot in OGNI slot: log2(N) passi rotazione+somma.
    Corretta perche' il batch CKKS e' esattamente N (il giro si chiude su N)."""
    s = 1
    while s < N:
        ct = cc.EvalAdd(ct, cc.EvalRotate(ct, s))
        s *= 2
    return ct


def he_stage(he, chans, w, b, gamma, beta, level_in, stride2, coeffs, domain):
    cc, g = he.cc, he.g
    N = g.N
    out_level = level_in + (1 if stride2 else 0)
    Hh, Ww = g.size(out_level)
    n_valid = Hh * Ww
    mask_pt = he.mask_pt(out_level)

    t0 = time.time()
    conv = he.conv3x3(chans, w, b, level_in, stride2)      # prodotti per scalare + bias + maschera
    t_conv = time.time() - t0

    trace = {}
    outs = []
    t0 = time.time()
    for c, x in enumerate(conv):
        rec = trace if c == 0 else None
        if rec is not None:
            rec["conv + bias + maschera"] = x.GetLevel()
        total = sum_all_slots(cc, x, N)
        mean = cc.EvalMult(total, 1.0 / n_valid)
        total2 = sum_all_slots(cc, cc.EvalMult(x, x), N)
        mean_sq = cc.EvalMult(total2, 1.0 / n_valid)
        var = cc.EvalSub(mean_sq, cc.EvalMult(mean, mean))
        if rec is not None:
            rec["media"] = mean.GetLevel()
            rec["varianza"] = var.GetLevel()
        inv = isqrt_chebyshev_fhe(cc, var, coeffs, list(domain), post_iter=1)
        if rec is not None:
            rec["radice inversa (Chebyshev+Newton)"] = inv.GetLevel()
        normalized = cc.EvalMult(cc.EvalSub(x, mean), inv)
        if rec is not None:
            rec["(x - media) * inv_std"] = normalized.GetLevel()
        scaled = cc.EvalAdd(cc.EvalMult(normalized, float(gamma[c])), float(beta[c]))
        if rec is not None:
            rec["* gamma + beta"] = scaled.GetLevel()
        act = poly_act_fhe(cc, scaled, A_, B_, C_)
        if rec is not None:
            rec["PolyAct"] = act.GetLevel()
        out = cc.EvalMult(act, mask_pt)
        if rec is not None:
            rec["maschera finale"] = out.GetLevel()
        outs.append(out)
        cc.TrimGPUMemoryPool()
    t_norm = time.time() - t0
    return outs, trace, t_conv, t_norm


def main():
    g = Grid(H0, W0, L)
    N = g.N
    print(f"=== Livelli di uno stadio completo, schema per canale + reticolo, HE vero ===")
    print(f"Immagine {H0}x{W0}, N={N} slot, {L} livelli (passi 1 e 2)\n")
    assert N & (N - 1) == 0

    rng = np.random.default_rng(41)
    x_img = rng.normal(size=(3, H0, W0))
    spec = [  # (Cin, Cout, stride2, livello di ingresso)
        (3, 4, False, 0), (4, 4, False, 0), (4, 6, True, 0), (6, 6, False, 1),
    ]
    W_, B_, G_, BE_ = [], [], [], []
    for cin, cout, _, _ in spec:
        W_.append(rng.normal(size=(cout, cin, 3, 3)) * (0.5 / np.sqrt(cin * 9)))
        B_.append(rng.normal(size=cout) * 0.05)
        G_.append(rng.normal(size=cout) * 0.2 + 1.0)
        BE_.append(rng.normal(size=cout) * 0.1)

    # ---------- catena numpy: approssimata (come l'HE) ed esatta ----------
    ref_a, ref_e, domains, coeffs_l = [], [], [], []
    xa = x_img.copy()
    xe = x_img.copy()
    for s, (cin, cout, st2, lvl) in enumerate(spec):
        ya = conv_ref(xa, W_[s], B_[s], stride=2 if st2 else 1)
        var = ya.reshape(cout, -1).var(axis=1)
        dom = domain_from_var(var)
        co, _ = fit_monotonic_isqrt_coeffs(fhe, dom[0], dom[1], degree=3, extra_safety=1.2)
        domains.append(dom)
        coeffs_l.append(co)
        xa = norm_act_ref(ya, G_[s], BE_[s], make_emulator(co, dom))
        ye = conv_ref(xe, W_[s], B_[s], stride=2 if st2 else 1)
        xe = norm_act_ref(ye, G_[s], BE_[s], lambda v: 1.0 / np.sqrt(v))
        ref_a.append(xa)
        ref_e.append(xe)
    print("Domini Chebyshev per stadio: " + "; ".join(f"[{d[0]:.2f},{d[1]:.2f}]" for d in domains))
    print(f"Differenza numpy esatto vs approssimato all'uscita dello stadio 4: {np.abs(ref_a[-1] - ref_e[-1]).max():.2e}\n")

    # ---------- contesto HE ----------
    rots = set()
    for lvl in range(L):
        rots |= set(level_offsets(g, lvl))
    s = 1
    while s < N:
        rots.add(s)
        s *= 2
    rots = sorted(r for r in rots if r != 0)
    from test_lattice_resample_zeus import build_context
    print(f"Costruzione contesto ({len(rots)} chiavi di rotazione)...", flush=True)
    t0 = time.time()
    cc, keys = build_context(g, rots)
    he = LatticeHE(cc, keys, g)
    print(f"Contesto pronto in {time.time()-t0:.1f}s.\n", flush=True)

    chans = [he.enc(g.pack(x_img[c], 0)) for c in range(3)]
    costs = []
    for s_i, (cin, cout, st2, lvl) in enumerate(spec):
        print(f"=== STADIO {s_i + 1}: conv {cin}->{cout}{' STRIDE 2' if st2 else ''}, livello {lvl}"
              f"{' -> ' + str(lvl + 1) if st2 else ''} ===", flush=True)
        in_level = max(c.GetLevel() for c in chans)
        outs, trace, t_conv, t_norm = he_stage(he, chans, W_[s_i], B_[s_i], G_[s_i], BE_[s_i],
                                               lvl, st2, coeffs_l[s_i], domains[s_i])
        out_level = max(c.GetLevel() for c in outs)
        out_lvl_grid = lvl + (1 if st2 else 0)
        res = he.unpack_all(outs, out_lvl_grid)
        e_a = np.abs(res - ref_a[s_i]).max()
        e_e = np.abs(res - ref_e[s_i]).max()
        print(f"  {'passo':<38}{'livello (canale 0)':>20}")
        prev = in_level
        for k, v in trace.items():
            print(f"  {k:<38}{v:>14d}   (+{v - prev})")
            prev = v
        cost = out_level - in_level
        costs.append(cost)
        print(f"  --> COSTO DELLO STADIO: {cost} livelli (da {in_level} a {out_level})   "
              f"[tempo: conv {t_conv:.1f}s, norm+act {t_norm:.1f}s]")
        print(f"  --> errore vs numpy APPROSSIMATO {e_a:.2e}   vs ESATTO {e_e:.2e}\n", flush=True)
        # refresh simulato: riparte da livello 0 (NON e' un bootstrap)
        chans = [he.enc(he.dec(c)) for c in outs]

    print("=== RIEPILOGO ===")
    print(f"  costo per stadio (livelli): {costs}")
    worst = max(costs)
    print(f"  stadio piu' caro: {worst} livelli; disponibili dopo un bootstrap: {BOOTSTRAP_BUDGET} "
          f"-> {'ENTRA' if worst <= BOOTSTRAP_BUDGET else 'NON ENTRA'} (margine {BOOTSTRAP_BUDGET - worst:+d})")
    print(f"  (riferimento: formato a righe, misurato prima: 20 livelli per stadio)")
    if worst <= BOOTSTRAP_BUDGET:
        print("  Conseguenza: un bootstrap per stadio basta. Con costo <= 11, due stadi per bootstrap.")
    else:
        print("  Conseguenza: uno stadio non entra nei livelli dopo un bootstrap: serve un bootstrap a meta' stadio "
              "o ridurre il costo.")


if __name__ == '__main__':
    main()