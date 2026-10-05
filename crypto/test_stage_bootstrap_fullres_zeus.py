"""
crypto/test_stage_bootstrap_fullres_zeus.py     (PROVA 2)

Lo scenario che la rete vera ripete decine di volte, per la prima volta
tutto insieme in HE vero, a risoluzione reale e 65.536 slot:

    ingresso fresco (livello 0)
      -> STADIO 1  (conv 3x3 -> InstanceNorm -> PolyAct -> maschera)   ~16 livelli
      -> BOOTSTRAP VERO di ogni canale (bsgsDim=[8,8], level_budget [4,4]) -> livello ~21
      -> STADIO 2  dopo il bootstrap                                    ~+16 -> ~37 (<= 43)

Si verifica: (1) i livelli (16 / 21 / 37); (2) che lo stadio 2 funzioni DOPO
un bootstrap vero, con il rumore del bootstrap (~3e-3) in ingresso: la norm
divide per la deviazione standard, quindi puo' amplificarlo; (3) tempi e
memoria per canale; (4) correttezza contro numpy (riferimento approssimato,
stessa radice inversa dell'HE, e riferimento esatto).

Argomenti:  python3 crypto/test_stage_bootstrap_fullres_zeus.py [C [LIVELLO]]
  C        canali (default 32)
  LIVELLO  livello del reticolo (default 0 = 256x224; 2 = 64x56, come enc2/dec2)
Tempi attesi: C=32 livello 0: ~8-10 minuti; C=128 livello 2: ~35-45 minuti
(la parte lunga e' la convoluzione: 147k termini a ~5-10 ms per stadio).

Nota per la lettura: la differenza "esatto vs approssimato" in un test come
questo include la scelta GROSSOLANA del dominio Chebyshev (0.5*min .. 2*max
delle varianze, con pochi canali): non e' rappresentativa della rete vera,
dove i domini vengono dalla calibrazione. La pipeline si giudica contro
l'APPROSSIMATO.
"""

import os
import sys
import time
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import fit_monotonic_isqrt_coeffs
from prototype_lattice_resample import Grid, conv_ref
from lattice_conv import LHE, level_offsets

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
H0, W0 = 256, 224
LEVEL_BUDGET = [4, 4]
BSGS_DIM = [8, 8]
A_, B_, C_ = 0.1, 1.0, 0.5
C_DEFAULT, LEVEL_DEFAULT = 32, 0
# Cache GPU configurabili: la prima versione usava 2 GiB di cache ciphertext e lo stadio 1 (ciphertext a
# tutti i limbi) e' andato a 54 ms/termine invece di ~3 (thrash, stesso fenomeno del benchmark a 1 GiB).
CT_CACHE_GIB = float(os.environ.get("CT_CACHE_GIB", "6"))
ROT_CACHE_GIB = float(os.environ.get("ROT_CACHE_GIB", "6"))
START_BOOT = os.environ.get("START_BOOT", "0") == "1"      # 1 = bootstrap iniziale: lo stadio 1 parte da livello 21 come gli stadi profondi
CONV_B = int(os.environ.get("CONV_B", "8"))      # blocco di canali d'uscita della conv: meno = meno memoria


def gpu_mem(label, log=print):
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                       '--format=csv,noheader,nounits'], text=True).strip()
        used, total = out.split(',')
        log(f"    [GPU MEM] {label}: {used.strip()} / {total.strip()} MiB")
    except Exception as e:
        log(f"    [GPU MEM] {label}: impossibile leggere ({e})")


# ---------- riferimenti numpy ----------

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
            out.append(y * (1.5 - 0.5 * v * y * y))
        return np.array(out)
    return emu


def domain_from_var(var):
    return (max(0.01, 0.5 * var.min()), 2.0 * var.max())


def build_context(g, level):
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
    rots = set(r for r in level_offsets(g, level) if r != 0)
    s = 1
    while s < N:
        rots.add(s)                         # somme su tutti gli slot: 16 chiavi
        s *= 2
    cc.EvalRotateKeyGen(keys.secretKey, sorted(rots))
    print(f"  chiavi di rotazione extra: {len(rots)}", flush=True)
    cc.SetRotationKeyCache(int(ROT_CACHE_GIB * GiB))   # le 16 potenze di 2 + 8 offset: devono stare in cache
    cc.SetBootstrapCache(1 * GiB)
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s", flush=True)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(int(CT_CACHE_GIB * GiB))
    print(f"  cache: ciphertext {CT_CACHE_GIB} GiB, rotazioni {ROT_CACHE_GIB} GiB, blocco conv B={CONV_B}", flush=True)
    return cc, keys


def run(cc, keys, g, C, level, log=print):
    he = LHE(cc, keys, g)
    Hl, Wl = g.size(level)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(C, Hl, Wl)) * 0.5
    W1 = rng.normal(size=(C, C, 3, 3)) * (0.5 / np.sqrt(C * 9))
    W2 = rng.normal(size=(C, C, 3, 3)) * (0.5 / np.sqrt(C * 9))
    b1, b2 = rng.normal(size=C) * 0.05, rng.normal(size=C) * 0.05
    g1, g2 = rng.normal(size=C) * 0.2 + 1.0, rng.normal(size=C) * 0.2 + 1.0
    be1, be2 = rng.normal(size=C) * 0.1, rng.normal(size=C) * 0.1

    log(f"\nRiferimenti numpy (conv densa {C}x{C} su {Hl}x{Wl}, due stadi) ...")
    t0 = time.time()
    y1 = conv_ref(x, W1, b1)
    d1 = domain_from_var(y1.reshape(C, -1).var(axis=1))
    c1, _ = fit_monotonic_isqrt_coeffs(fhe, d1[0], d1[1], degree=3, extra_safety=1.2)
    x1a = norm_act_ref(y1, g1, be1, make_emulator(c1, d1))
    y2 = conv_ref(x1a, W2, b2)
    d2 = domain_from_var(y2.reshape(C, -1).var(axis=1))
    c2, _ = fit_monotonic_isqrt_coeffs(fhe, d2[0], d2[1], degree=3, extra_safety=1.2)
    x2a = norm_act_ref(y2, g2, be2, make_emulator(c2, d2))
    x1e = norm_act_ref(y1, g1, be1, lambda v: 1.0 / np.sqrt(v))
    x2e = norm_act_ref(conv_ref(x1e, W2, b2), g2, be2, lambda v: 1.0 / np.sqrt(v))
    log(f"  fatto in {time.time()-t0:.0f}s; domini [{d1[0]:.3f},{d1[1]:.3f}] e [{d2[0]:.3f},{d2[1]:.3f}]")

    log(f"\nCifratura di {C} canali d'ingresso ...")
    t0 = time.time()
    chans = he.enc_offloaded([g.pack(x[c], level) for c in range(C)])
    log(f"  fatto in {time.time()-t0:.0f}s")
    gpu_mem("dopo la cifratura", log)
    if START_BOOT:
        log("\nBootstrap iniziale dei canali d'ingresso (START_BOOT=1): lo stadio 1 parte da livello 21, "
            "come ogni stadio profondo della rete (errore atteso sullo stadio 1: ~3e-3, non 1e-9)")
        t0 = time.time()
        for i, ct in enumerate(chans):
            nb = cc.EvalBootstrap(ct)
            ct.Offload()
            nb.Offload()
            chans[i] = nb
            if i % 4 == 3:
                cc.TrimGPUMemoryPool()
        cc.TrimGPUMemoryPool()
        log(f"  {len(chans)} bootstrap in {time.time()-t0:.0f}s; livello {chans[0].GetLevel()}")
        gpu_mem("dopo il bootstrap iniziale", log)

    def check(outs, ref, label, n_check=None):
        idx = range(len(outs)) if n_check is None else range(min(n_check, len(outs)))
        err = max(np.abs(g.unpack(he.dec(outs[c]), level) - ref[c]).max() for c in idx)
        mean = np.mean([np.abs(g.unpack(he.dec(outs[c]), level) - ref[c]).mean() for c in idx])
        return err, mean

    res = {}
    # ---------------- stadio 1 ----------------
    log("\n=== STADIO 1 (da ingresso fresco) ===")
    t0 = time.time()
    outs1, info1 = he.stage(chans, W1, b1, g1, be1, level, False, c1, d1, B=CONV_B)
    log(f"  tempo: conv {info1['t_conv']:.0f}s + norm/act {info1['t_norm']:.0f}s = {time.time()-t0:.0f}s ; "
        f"livello in uscita: {info1['level']}")
    log(f"  conv: {info1['t_conv'] / (C * C * 9) * 1000:.1f} ms/termine ({C * C * 9} termini); "
        f"norm+act: {info1['t_norm'] / C:.2f} s/canale")
    gpu_mem("dopo lo stadio 1", log)
    e_max, e_mean = check(outs1, x1a, "stadio 1", n_check=4)
    log(f"  errore vs numpy APPROSSIMATO (4 canali): max {e_max:.2e}  medio {e_mean:.2e}")
    res["lvl1"] = info1["level"]

    # ---------------- bootstrap vero ----------------
    log("\n=== BOOTSTRAP VERO di ogni canale ===")
    t0 = time.time()
    boot = []
    lvl_after = None
    for i, ct in enumerate(outs1):
        nb = cc.EvalBootstrap(ct)
        if lvl_after is None:
            lvl_after = nb.GetLevel()
        ct.Offload()
        nb.Offload()
        boot.append(nb)
        if i % 4 == 3:
            cc.TrimGPUMemoryPool()
    cc.TrimGPUMemoryPool()
    t_boot = time.time() - t0
    log(f"  {len(boot)} bootstrap in {t_boot:.0f}s ({t_boot / len(boot):.2f} s/canale); livello dopo: {lvl_after}")
    gpu_mem("dopo i bootstrap", log)
    e_max, e_mean = check(boot, x1a, "dopo bootstrap", n_check=4)
    log(f"  errore dopo il bootstrap vs numpy (4 canali): max {e_max:.2e}  medio {e_mean:.2e}")
    res["lvl_boot"] = lvl_after

    # ---------------- stadio 2 ----------------
    log("\n=== STADIO 2 (dopo il bootstrap) ===")
    t0 = time.time()
    outs2, info2 = he.stage(boot, W2, b2, g2, be2, level, False, c2, d2, B=CONV_B)
    log(f"  tempo: conv {info2['t_conv']:.0f}s + norm/act {info2['t_norm']:.0f}s = {time.time()-t0:.0f}s ; "
        f"livello in uscita: {info2['level']} (limite {DEPTH})")
    log(f"  conv: {info2['t_conv'] / (C * C * 9) * 1000:.1f} ms/termine ({C * C * 9} termini); "
        f"norm+act: {info2['t_norm'] / C:.2f} s/canale")
    gpu_mem("dopo lo stadio 2", log)
    e_a, m_a = check(outs2, x2a, "stadio 2 vs approssimato")
    e_e, m_e = check(outs2, x2e, "stadio 2 vs esatto")
    log(f"  errore vs numpy APPROSSIMATO ({C} canali): max {e_a:.2e}  medio {m_a:.2e}")
    log(f"  errore vs numpy ESATTO       ({C} canali): max {e_e:.2e}  medio {m_e:.2e}")
    res.update(lvl2=info2["level"], err_a=e_a, mean_a=m_a, err_e=e_e)

    log("\n=== RIEPILOGO ===")
    log(f"  livelli: stadio 1 -> {res['lvl1']}; dopo bootstrap -> {res['lvl_boot']}; stadio 2 -> {res['lvl2']} (limite {DEPTH})")
    cost2 = res["lvl2"] - res["lvl_boot"]
    log(f"  costo dello stadio dopo il bootstrap: {cost2} livelli; fuori dai {DEPTH}? "
        f"{'NO, ci sta' if res['lvl2'] <= DEPTH else 'SI: NON CI STA'}")
    log(f"  precisione dopo bootstrap + stadio 2 contro il riferimento approssimato: "
        f"max {res['err_a']:.1e}, medio {res['mean_a']:.1e}")
    if res["err_a"] < 5e-2:
        log("  -> la catena stadio / bootstrap / stadio funziona a risoluzione reale.")
    else:
        log("  -> errore alto dopo il bootstrap: la norm amplifica il rumore. Da indagare "
            "(dominio Chebyshev, varianza piccola, ampiezza).")
    return res


def main():
    C = int(sys.argv[1]) if len(sys.argv) > 1 else C_DEFAULT
    level = int(sys.argv[2]) if len(sys.argv) > 2 else LEVEL_DEFAULT
    g = Grid(H0, W0, level + 1)
    Hl, Wl = g.size(level)
    print(f"=== PROVA 2: stadio -> bootstrap vero -> stadio, C={C}, livello {level} ({Hl}x{Wl}), N={g.N} ===")
    assert g.N == 65536
    gpu_mem("inizio")
    cc, keys = build_context(g, level)
    gpu_mem("dopo LoadContext")
    run(cc, keys, g, C, level)
    gpu_mem("fine")


if __name__ == '__main__':
    main()