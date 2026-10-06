"""
crypto/smoke_test_runner.py        (SOLO CPU: nessuna GPU, nessuna libreria HE, funziona anche sul Mac)

Fa girare il RUNNER (he_network.py + lattice_conv.py) con una libreria CKKS FINTA che simula le operazioni slot per slot
(rotazioni cicliche, prodotti, somme, livelli, bootstrap) e con stub delle funzioni HE del progetto. Serve a verificare, prima
di toccare la GPU, che:
  - i file copiati siano coerenti tra loro (versioni, importazioni, nomi);
  - il pacchetto .npz si carichi e il piano della rete si esegua fino in fondo (stride, upsampling, skip, bypass, Newton);
  - tutte le rotazioni richieste abbiano una chiave tra quelle che il contesto vero genererebbe;
  - i logit coincidano con il riferimento (con bootstrap senza rumore: errore ~1e-12).
NON verifica: i costi in livelli della libreria vera, la memoria, i tempi, ne' la funzione isqrt_chebyshev_fhe vera
(qui c'e' uno stub): questo lo fa la prova su Zeus.

Uso:
  python3 crypto/smoke_test_runner.py crypto/pack_narrow.npz                    # ~10 secondi
  STOP_AFTER=enc1 python3 crypto/smoke_test_runner.py crypto/pack_5s14.npz       # rete vera, primi due blocchi
  python3 crypto/smoke_test_runner.py crypto/pack_5s14.npz                       # rete vera intera (CPU: alcuni minuti)
Variabili: FAKE_NOISE (default 0: rumore per bootstrap), STOP_AFTER, MASK_LAUNDER.
"""

import os
import sys
import time
import types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, ".")

# ---- stub delle librerie HE: devono stare in sys.modules PRIMA di importare il resto ----
sys.modules["fideslib_py"] = types.ModuleType("fideslib_py")
_ops = types.ModuleType("crypto.fhe_ops_single_ciphertext")


def poly_act_fhe(cc, x, a, b, c):
    x2 = cc.EvalMult(x, x)
    return cc.EvalAdd(cc.EvalAdd(cc.EvalMult(x2, a), cc.EvalMult(x, b)), c)


def isqrt_chebyshev_fhe(cc, ct_var, coeffs, dom, post_iter):
    x_min, x_max = dom
    sc = 2.0 / (x_max - x_min)
    off = -(x_min + x_max) / (x_max - x_min)
    y = cc.EvalChebyshevSeries(cc.EvalAdd(cc.EvalMult(ct_var, sc), off), coeffs, -1.0, 1.0)
    for _ in range(post_iter):
        y2 = cc.EvalMult(y, y)
        xy2 = cc.EvalMult(ct_var, y2)
        y = cc.EvalMult(y, cc.EvalAdd(cc.EvalMult(xy2, -0.5), 1.5))
    return y


_ops.poly_act_fhe, _ops.isqrt_chebyshev_fhe = poly_act_fhe, isqrt_chebyshev_fhe
sys.modules["crypto"] = types.ModuleType("crypto")
sys.modules["crypto.fhe_ops_single_ciphertext"] = _ops

from prototype_lattice_resample import Grid
from lattice_conv import LHE, level_offsets
import he_network as hn


# ---- libreria CKKS finta ----
class PT:
    def __init__(self, v):
        self.v = v


class CT:
    def __init__(self, v, l=0):
        self.v, self.l = v, l

    def GetLevel(self):
        return self.l

    def Offload(self):
        pass


class Out:
    def __init__(self, v):
        self.v, self.n = v, len(v)

    def SetLength(self, n):
        self.n = n

    def GetRealPackedValue(self):
        return list(self.v[:self.n])


class Keys:
    publicKey = "pk"
    secretKey = "sk"


class FakeCC:
    def __init__(self, N, allowed, noise=0.0, seed=1):
        self.N, self.allowed, self.used, self.nboot = N, set(allowed), set(), 0
        self.noise, self.rng, self.max_level = noise, np.random.default_rng(seed), 0

    def _mk(self, v, l):
        self.max_level = max(self.max_level, l)
        return CT(v, l)

    def MakeCKKSPackedPlaintext(self, vals):
        v = np.zeros(self.N)
        v[:len(vals)] = vals
        return PT(v)

    def Encrypt(self, pk, p):
        return CT(p.v.copy(), 0)

    def Decrypt(self, sk, c):
        return Out(c.v.copy())

    def EvalRotate(self, c, k):
        assert k in self.allowed, f"ROTAZIONE {k} SENZA CHIAVE nel contesto vero"
        self.used.add(k)
        return self._mk(np.roll(c.v, -k), c.l)

    def _v(self, b):
        return b.v if isinstance(b, (PT, CT)) else float(b)

    def _l(self, a, b):
        return max(a.l, b.l if isinstance(b, CT) else 0)

    def EvalMult(self, a, b):
        return self._mk(a.v * self._v(b), self._l(a, b) + 1)

    def EvalAdd(self, a, b):
        return self._mk(a.v + self._v(b), self._l(a, b))

    def EvalSub(self, a, b):
        return self._mk(a.v - self._v(b), self._l(a, b))

    def EvalChebyshevSeries(self, c, co, lo, hi):                      # convenzione OpenFHE: c0 dimezzato
        co = list(co)
        if os.environ.get("FAKE_CHEB_QUIRK", "1") == "1":
            # Comportamento OSSERVATO su Zeus (batteria del 5 ott): gli zeri finali dei coefficienti vengono scartati prima
            # di scegliere il percorso (stessi errori con e senza riempimento a zeri); per i gradi 1 e 2 il risultato e'
            # sbagliato; un polinomio costante lancia un'eccezione.
            while len(co) > 1 and co[-1] == 0:
                co.pop()
            if len(co) == 1:
                raise RuntimeError("polinomio costante")
            if len(co) <= 3:
                return self._mk(3.0 * c.v + 2.0, c.l + 8)                  # valore sbagliato, come sul grado 1 reale
        t = c.v
        b1 = np.zeros_like(t)
        b2 = np.zeros_like(t)
        for k in reversed(co[1:]):
            b1, b2 = 2.0 * t * b1 - b2 + k, b1
        return self._mk(t * b1 - b2 + co[0] / 2.0, c.l + 8)

    def EvalBootstrap(self, c):
        self.nboot += 1
        return CT(c.v + self.rng.normal(size=self.N) * self.noise if self.noise else c.v.copy(), 21)

    def GetCiphertextCacheResidentBytes(self):
        return self.ct_resident * 1048576

    def OffloadCiphertexts(self):
        self.ct_resident = 0

    def GetDeviceObjectCounts(self):
        return {"ciphertexts": 7, "plaintexts": 3}

    def GetAuxiliaryPolyPoolSize(self):
        return self.aux_calls

    def ClearAuxiliaryPolyPool(self):
        self.aux_calls += 1

    def Synchronize(self):
        pass

    def TrimGPUMemoryPool(self):
        pass


def needed_rotations(g):
    rots = set()
    for l in range(g.n_levels):
        rots |= set(level_offsets(g, l))
    s = 1
    while s < g.N:
        rots.add(s)
        s *= 2
    rots.discard(0)
    return sorted(rots)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("uso: python3 crypto/smoke_test_runner.py <pacchetto.npz>")
    pack = hn.load_pack(sys.argv[1])
    meta = pack["meta"]
    H0, W0 = pack["image"].shape[-2:]
    g = Grid(H0, W0, meta["k"])
    noise = float(os.environ.get("FAKE_NOISE", "0"))
    stop = os.environ.get("STOP_AFTER") or None
    print(f"=== TEST DI FUMO (solo CPU, libreria CKKS finta): {sys.argv[1]} ===")
    print(f"  k={meta['k']}, filtri {meta['filters']}, bypass {len(meta['bypass'])}, schemi "
          f"{sum(1 for s in meta['schemes'].values() if s['kind'] == 'cheb')} Chebyshev + "
          f"{sum(1 for s in meta['schemes'].values() if s['kind'] == 'newton')} Newton; immagine {H0}x{W0}; rumore bootstrap finto {noise}")
    cc = FakeCC(g.N, needed_rotations(g), noise)
    cc.aux_calls = 0
    cc.ct_resident = 640
    cc = hn.PoolCC(cc, os.environ.get('AUX_CLEAR', '0') == '1')
    he = LHE(cc, Keys(), g)
    guard = int(os.environ.get('SMOKE_GUARD', '0'))                # prova della protezione di memoria: limite finto
    runner = hn.HERunner(he, pack, log=lambda s: print(s, flush=True), mem=lambda: 100 if guard else 0, check=True,
                         stop_after=stop, mem_guard=guard // 2,
                         evict_ct=os.environ.get('EVICT_CT', '0') == '1')
    t0 = time.time()
    res = runner.run()
    print(f"\n  [fumo] tempo CPU {time.time() - t0:.0f}s; chiavi di rotazione usate {len(cc.used)} su {len(needed_rotations(g))} "
          f"che il contesto vero genera; bootstrap simulati {cc.nboot}")
    if stop:
        errs = [r["err"] for r in runner.rows if r["err"] is not None]
        ok = bool(errs) and max(errs) < (1e-6 if noise == 0 else 5e-2)
        print(f"  [fumo] esecuzione parziale (STOP_AFTER={stop}): errore massimo per stadio {max(errs):.2e} -> {'PASS' if ok else 'FAIL'}")
    else:
        scale = float(np.abs(pack['logits_approx']).max())
        tol = 1e-6 * max(1.0, scale) if noise == 0 else 5e-2 * max(1.0, scale)
        ok = res.get("err_max", 1e9) < tol and res.get("agree_approx", 0) > (0.999 if noise == 0 else 0.98)
        print(f"  [fumo] logit: errore {res.get('err_max', float('nan')):.2e} (soglia {tol:.1e}), classi uguali "
              f"{100 * res.get('agree_approx', 0):.2f}% -> {'PASS' if ok else 'FAIL'}")
    print("  [fumo] ricorda: questo NON prova costi in livelli, memoria e tempi della libreria vera.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()