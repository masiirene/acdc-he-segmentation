"""
crypto/test_mini_unet_fullres_zeus.py     (PROVA 3)

UNA PICCOLA U-NET COMPLETA IN HE VERO, a risoluzione reale (256x224, 65.536
slot), schema a UN CIPHERTEXT PER CANALE con reticolo. Mette insieme per la
prima volta, nello stesso contesto con bootstrap vero:
  - stadi conv 3x3 -> InstanceNorm (Chebyshev+Newton) -> PolyAct (lattice_conv.LHE.stage)
  - stride 2 nella prima conv di due blocchi (livello 0 -> 1 -> 2 del reticolo)
  - upsampling ConvTranspose2d(k=2, s=2): 3 rotazioni per canale in ingresso, 4 fasi su slot disgiunti
  - skip connection a SOMMA tra encoder e decoder (livelli diversi dei due rami)
  - convoluzione finale 1x1 (4 classi)
  - BOOTSTRAP INSERITI AUTOMATICAMENTE: prima di ogni stadio, ogni canale il cui livello
    + 17 supererebbe 43 viene bootstrappato (17 = costo misurato di uno stadio dopo bootstrap)

    S1 1->4   S2 4->4   [skip0]          livello 0   256x224
    S3 4->8 (stride 2)  S4 8->8 [skip1]  livello 1   128x112
    S5 8->16 (stride 2) S6 16->16        livello 2    64x56
    U1 16->8 + skip1    S7 8->8  S8 8->8 livello 1
    U0 8->4  + skip0    S9 4->4  S10 4->4 livello 0
    OUT 1x1 4->4 (logit)

Pesi casuali, stessa architettura in numpy (riferimento APPROSSIMATO: stessa radice
inversa dell'HE, ed ESATTO). Si stampa per ogni passo: livelli in ingresso/uscita,
bootstrap eseguiti, tempo, errore del canale 0 contro numpy; alla fine errore sui
logit e percentuale di pixel con la stessa classe.

Cosa NON e' in questa prova: canali reali (qui 4-16), norm con pesi della rete,
calibrazione dei domini Chebyshev (qui dai dati), concat (qui sum).
Ipotesi che la prova puo' smentire: (a) EvalAdd tra ciphertext a livelli diversi (skip) e'
supportato; (b) i 3 offset dell'upsampling coincidono con chiavi gia' generate per la conv;
(c) i bootstrap totali sono meno di quelli stimati con "uno per stadio + uno per skip", perche'
il ramo di skip viene bootstrappato insieme al ramo principale.

Uso (in tmux, GPU libera):  python3 crypto/test_mini_unet_fullres_zeus.py
Durata attesa: ~6-8 minuti (contesto con bootstrap ~1.5 min, ~60 bootstrap ~2 min).
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import fit_monotonic_isqrt_coeffs
from prototype_lattice_resample import Grid, conv_ref
from lattice_conv import LHE, level_offsets
from test_stage_bootstrap_fullres_zeus import norm_act_ref, make_emulator, domain_from_var, gpu_mem

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
H0, W0 = 256, 224
N_LEVELS = 3
WID = [4, 8, 16]
STAGE_NEED = 17                      # livelli di uno stadio dopo bootstrap (misurato: 17; da livello 0: 16)
LEVEL_BUDGET = [4, 4]
BSGS_DIM = [8, 8]

SEQ = [  # nome, cin, cout, livello del reticolo in ingresso, stride2 (None per U e OUT)
    ("S1", 1, WID[0], 0, False), ("S2", WID[0], WID[0], 0, False),
    ("S3", WID[0], WID[1], 0, True), ("S4", WID[1], WID[1], 1, False),
    ("S5", WID[1], WID[2], 1, True), ("S6", WID[2], WID[2], 2, False),
    ("U1", WID[2], WID[1], 2, None),
    ("S7", WID[1], WID[1], 1, False), ("S8", WID[1], WID[1], 1, False),
    ("U0", WID[1], WID[0], 1, None),
    ("S9", WID[0], WID[0], 0, False), ("S10", WID[0], WID[0], 0, False),
    ("OUT", WID[0], 4, 0, None),
]
SKIP_FROM = {"U1": "S4", "U0": "S2"}


def needed_rotations(g):
    rots = set()
    for l in range(g.n_levels):
        rots |= set(level_offsets(g, l))          # conv 3x3 e upsampling (offset negativi gia' inclusi)
    s = 1
    while s < g.N:
        rots.add(s)                               # somme su tutti gli slot per le statistiche della norm
        s *= 2
    rots.discard(0)
    return sorted(rots)


# ============================================================
# Riferimenti numpy
# ============================================================

def convT_ref(x, w, b):
    """ConvTranspose2d(k=2, s=2): x (Cin,H,W), w (Cin,Cout,2,2) -> (Cout,2H,2W)."""
    cin, H, W = x.shape
    cout = w.shape[1]
    out = np.zeros((cout, 2 * H, 2 * W))
    for co in range(cout):
        for ci in range(cin):
            for a in range(2):
                for bb in range(2):
                    out[co, a::2, bb::2] += w[ci, co, a, bb] * x[ci]
        out[co] += b[co]
    return out


def make_params(rng):
    P = {}
    for name, cin, cout, lvl, st in SEQ:
        if name.startswith("S"):
            P[name] = dict(w=rng.normal(size=(cout, cin, 3, 3)) * (0.5 / np.sqrt(cin * 9)),
                           b=rng.normal(size=cout) * 0.05,
                           g=rng.normal(size=cout) * 0.2 + 1.0, be=rng.normal(size=cout) * 0.1)
        elif name.startswith("U"):
            P[name] = dict(w=rng.normal(size=(cin, cout, 2, 2)) * (0.5 / np.sqrt(cin * 4)),
                           b=rng.normal(size=cout) * 0.05)
        else:
            P[name] = dict(w=rng.normal(size=(cout, cin)) * (0.5 / np.sqrt(cin)),
                           b=rng.normal(size=cout) * 0.05)
    return P


def numpy_chain(x0, P):
    outs_a, outs_e, doms, coefs = {}, {}, {}, {}
    a = e = x0
    for name, cin, cout, lvl, st in SEQ:
        p = P[name]
        if name.startswith("S"):
            stride = 2 if st else 1
            ya = conv_ref(a, p["w"], p["b"], stride=stride)
            dom = domain_from_var(ya.reshape(cout, -1).var(axis=1))
            co, _ = fit_monotonic_isqrt_coeffs(fhe, dom[0], dom[1], degree=3, extra_safety=1.2)
            a = norm_act_ref(ya, p["g"], p["be"], make_emulator(co, dom))
            ye = conv_ref(e, p["w"], p["b"], stride=stride)
            e = norm_act_ref(ye, p["g"], p["be"], lambda v: 1.0 / np.sqrt(v))
            doms[name], coefs[name] = dom, co
        elif name.startswith("U"):
            a = convT_ref(a, p["w"], p["b"]) + outs_a[SKIP_FROM[name]]
            e = convT_ref(e, p["w"], p["b"]) + outs_e[SKIP_FROM[name]]
        else:
            a = np.einsum('oc,chw->ohw', p["w"], a) + p["b"][:, None, None]
            e = np.einsum('oc,chw->ohw', p["w"], e) + p["b"][:, None, None]
        outs_a[name], outs_e[name] = a, e
    return outs_a, outs_e, doms, coefs


# ============================================================
# Operazioni HE aggiuntive
# ============================================================

def he_upconv(he, chans, w, b, level_in):
    """ConvTranspose2d(k=2, s=2) sul reticolo: l'ingresso sta al livello level_in, l'uscita al livello
    level_in-1. Le 4 fasi (a,b) sono l'ingresso spostato di s*(a*Wg+b) (s = passo del reticolo di
    uscita): 3 rotazioni per canale in ingresso, nessuna maschera tra le fasi (slot disgiunti).
    Maschera finale solo per azzerare il bias fuori dal reticolo (+1 livello)."""
    cc, g = he.cc, he.g
    lo = level_in - 1
    s = 1 << lo
    shifts = [0, -s, -s * g.Wg, -s * (g.Wg + 1)]          # fase (a,b) = (0,0),(0,1),(1,0),(1,1)
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
        o = cc.EvalMult(cc.EvalAdd(acc[co], float(b[co])), mask)
        o.Offload()
        outs.append(o)
    cc.TrimGPUMemoryPool()
    return outs


def he_add(he, A, B):
    cc = he.cc
    out = []
    for i, (x, y) in enumerate(zip(A, B)):
        try:
            s = cc.EvalAdd(x, y)
        except Exception as ex:
            raise RuntimeError(f"EvalAdd tra ciphertext a livelli {x.GetLevel()} e {y.GetLevel()} fallita "
                               f"({ex}): la somma delle skip a livelli diversi non e' supportata cosi'; "
                               f"serve allineare i livelli prima.")
        s.Offload()
        out.append(s)
    cc.TrimGPUMemoryPool()
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


def ensure_budget(he, chans, need):
    """Bootstrap, IN PLACE, dei canali che non hanno 'need' livelli disponibili. Torna quanti."""
    cc = he.cc
    n = 0
    for i, ct in enumerate(chans):
        if ct.GetLevel() + need > DEPTH:
            nb = cc.EvalBootstrap(ct)
            ct.Offload()
            nb.Offload()
            chans[i] = nb
            n += 1
            if n % 4 == 0:
                cc.TrimGPUMemoryPool()
    cc.TrimGPUMemoryPool()
    return n


def levels(chans):
    lv = [c.GetLevel() for c in chans]
    return min(lv), max(lv)


# ============================================================
# Contesto e prova
# ============================================================

def build_context(g):
    N = g.N
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(N)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetDevices([0])
    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    t0 = time.time()
    cc.EvalBootstrapSetup(LEVEL_BUDGET, BSGS_DIM, N)
    cc.EvalBootstrapKeyGen(keys.secretKey, N)
    print(f"  Bootstrap setup+keygen (bsgsDim={BSGS_DIM}): {time.time()-t0:.1f}s", flush=True)
    rots = needed_rotations(g)
    t0 = time.time()
    cc.EvalRotateKeyGen(keys.secretKey, rots)
    print(f"  chiavi di rotazione extra: {len(rots)} ({time.time()-t0:.1f}s)", flush=True)
    cc.SetRotationKeyCache(int(float(os.environ.get("ROT_CACHE_GIB", "6")) * GiB))
    cc.SetBootstrapCache(1 * GiB)
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s", flush=True)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(int(float(os.environ.get("CT_CACHE_GIB", "6")) * GiB))
    return cc, keys


def run(cc, keys, g, log=print):
    he = LHE(cc, keys, g)
    rng = np.random.default_rng(23)
    P = make_params(rng)
    x0 = rng.normal(size=(1, H0, W0)) * 0.5

    log("\nCatena numpy (approssimata ed esatta) ...")
    t0 = time.time()
    ref_a, ref_e, doms, coefs = numpy_chain(x0, P)
    log(f"  fatto in {time.time()-t0:.0f}s")

    chans = he.enc_offloaded([g.pack(x0[0], 0)])
    outs_he = {}
    rows = []
    total_boot = 0
    t_start = time.time()
    for name, cin, cout, lvl, st in SEQ:
        p = P[name]
        t0 = time.time()
        n_boot = 0
        if name.startswith("S"):
            n_boot = ensure_budget(he, chans, STAGE_NEED)
            lin = levels(chans)
            chans, info = he.stage(chans, p["w"], p["b"], p["g"], p["be"], lvl, st, coefs[name], doms[name])
            out_lvl_grid = lvl + (1 if st else 0)
        elif name.startswith("U"):
            n_boot = ensure_budget(he, chans, 2)
            lin = levels(chans)
            up = he_upconv(he, chans, p["w"], p["b"], lvl)
            skip = outs_he[SKIP_FROM[name]]
            lskip = levels(skip)
            chans = he_add(he, up, skip)
            out_lvl_grid = lvl - 1
            log(f"  [{name}] up-conv {lin} ; skip da {SKIP_FROM[name]} ai livelli {lskip}")
        else:
            lin = levels(chans)
            chans = he_conv1x1(he, chans, p["w"], p["b"])
            out_lvl_grid = 0
        lout = levels(chans)
        outs_he[name] = chans                    # stesso oggetto: un bootstrap successivo aggiorna anche la skip
        total_boot += n_boot
        if name != "OUT":
            err = np.abs(g.unpack(he.dec(chans[0]), out_lvl_grid) - ref_a[name][0]).max()
        else:
            err = float("nan")
        dt = time.time() - t0
        rows.append((name, lin, n_boot, lout, dt, err))
        log(f"  {name:<4} livelli in {lin[0]:>2}-{lin[1]:<2} | bootstrap {n_boot:>2} | uscita {lout[0]:>2}-{lout[1]:<2} | "
            f"{dt:6.1f}s | errore canale 0 vs numpy approssimato {err:.2e}")

    # ---------------- logit ----------------
    he_logits = np.stack([g.unpack(he.dec(c), 0) for c in chans])
    la, le = ref_a["OUT"], ref_e["OUT"]
    err_a = np.abs(he_logits - la)
    agree_a = float(np.mean(he_logits.argmax(0) == la.argmax(0)))
    agree_e = float(np.mean(he_logits.argmax(0) == le.argmax(0)))
    agree_ae = float(np.mean(la.argmax(0) == le.argmax(0)))

    log("\n=== RIEPILOGO ===")
    log(f"  tempo totale delle operazioni HE: {time.time()-t_start:.0f}s ; bootstrap eseguiti: {total_boot} canali "
        f"(~{total_boot * 2:.0f}s)")
    log(f"  livello finale dei logit: {rows[-1][3][1]} (limite {DEPTH})")
    log(f"  logit: errore max {err_a.max():.2e}, medio {err_a.mean():.2e} contro numpy APPROSSIMATO "
        f"(scala dei logit {np.abs(la).max():.2f})")
    log(f"  classe per pixel: HE = approssimato {100 * agree_a:.2f}%  |  HE = esatto {100 * agree_e:.2f}%  "
        f"|  (approssimato = esatto {100 * agree_ae:.2f}%)")
    chain_names = [f"S{i}" for i in range(2, 10)]                      # uscite degli stadi 2..9
    chain_est = sum(c for n, _, c, _, _ in SEQ if n in chain_names)
    skip_est = WID[0] + WID[1]
    log(f"  stima precedente 'bootstrap a ogni uscita di stadio (2..9) + uno per ogni canale di skip': "
        f"{chain_est} + {skip_est} = {chain_est + skip_est}; effettivi: {total_boot}")
    ok = err_a.max() < 5e-2 and agree_a > 0.999
    log("  -> " + ("la mini U-Net funziona in HE vero, a risoluzione piena, con bootstrap automatici."
                    if ok else "DIFFERENZE sopra le attese: guardare l'errore per passo qui sopra per localizzare dove nasce."))
    return rows, total_boot


def main():
    g = Grid(H0, W0, N_LEVELS)
    print(f"=== PROVA 3: mini U-Net {WID} a risoluzione piena {H0}x{W0}, N={g.N}, {N_LEVELS} livelli del reticolo ===")
    assert g.N == 65536
    gpu_mem("inizio")
    cc, keys = build_context(g)
    gpu_mem("dopo LoadContext")
    run(cc, keys, g)
    gpu_mem("fine")


if __name__ == '__main__':
    main()