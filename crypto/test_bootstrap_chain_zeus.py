"""
crypto/test_bootstrap_chain_zeus.py

DOMANDA: ogni bootstrap a 65.536 slot (bsgsDim=[8,8]) aggiunge ~3e-3 di
errore. Nella rete vera un ciphertext attraversera' circa 15-20 bootstrap
di fila, con operazioni non lineari in mezzo. L'errore si accumula in modo
tollerabile (come sqrt(k)) o esplode (come k, o peggio)?

Due catene, K bootstrap ciascuna:

  A  "solo bootstrap":  x <- bootstrap(0.999 * x)
     (la moltiplicazione per 0.999 e' solo per non bootstrappare un
      ciphertext appena cifrato; e' applicata anche al riferimento)

  C  "con non linearita' neutra":  x <- bootstrap( x - 0.005*x^2 + 0.00045 )
     La derivata 1-0.01x sta entro 1 +- 1% sul range dei dati e la deriva
     su 30 iterazioni e' trascurabile: la funzione NON amplifica ne'
     smorza il rumore, cosi' si misura l'accumulo del bootstrap e non un
     artefatto della funzione (una prima versione con coefficiente 0.05
     scappava verso l'infinito per x < -0.3: scartata). Include un
     prodotto ct x ct (con relinearizzazione) e consuma livelli come una
     attivazione.

LIMITE: misura l'accumulo di rumore DEL BOOTSTRAP. L'amplificazione dovuta
ai layer veri (normalizzazione, attivazioni con derivata > 1) non e'
catturata: si misura nel test end-to-end.

Riferimento: stessa iterazione in numpy (float64). Errore = |HE - numpy|
sulla regione dei dati (primi 58.308 slot) e, a parte, sul padding.

Interpretazione: stampa err_k / err_1 e lo confronta con sqrt(k) e con k.
  - vicino a sqrt(k): rumore che si somma in modo indipendente (benigno)
  - vicino a k: errore sistematico che si accumula (preoccupante)
  - ben oltre k: la catena amplifica (problema serio)

Valori dei dati: N(0, 0.3), come ordine di grandezza di un'uscita normalizzata
(la normalizzazione nella rete porta le attivazioni a varianza ~1; qui
usiamo 0.3 per restare dentro il dominio del bootstrap con margine).

Uso (in tmux, GPU libera):
  python3 crypto/test_bootstrap_chain_zeus.py [K [BSGS1 BSGS2]]
  default: K=20, bsgsDim=[8,8]
Durata: ~2 bootstrap/5s -> ~3-4 minuti in tutto, piu' ~1 minuto di setup.
"""

import sys
import time
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
BATCH = 65536
LEVEL_BUDGET = [4, 4]
REAL_LEN = 258 * 226          # immagine con alone, come nel resto del progetto
K_ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BSGS_DIM = [int(sys.argv[2]), int(sys.argv[3])] if len(sys.argv) > 3 else [8, 8]
CHECKPOINTS = [k for k in (1, 2, 3, 5, 8, 10, 15, 20, 30) if k <= K_ITERS]
if K_ITERS not in CHECKPOINTS:
    CHECKPOINTS.append(K_ITERS)


def gpu_mem(label):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'], text=True).strip()
        used, total = out.split(',')
        print(f"    [GPU MEM] {label}: {used.strip()} / {total.strip()} MiB", flush=True)
    except Exception as e:
        print(f"    [GPU MEM] {label}: impossibile leggere ({e})", flush=True)


def make_context():
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(BATCH)
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
    cc.EvalBootstrapSetup(LEVEL_BUDGET, BSGS_DIM, BATCH)
    cc.EvalBootstrapKeyGen(keys.secretKey, BATCH)
    print(f"  Bootstrap setup+keygen (bsgsDim={BSGS_DIM}): {time.time()-t0:.1f}s", flush=True)
    cc.SetRotationKeyCache(1 * GiB)
    cc.SetBootstrapCache(1 * GiB)
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s", flush=True)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(1 * GiB)
    return cc, keys


def run(cc, keys, label_extra=""):
    rng = np.random.default_rng(7)
    data = np.zeros(BATCH)
    data[:REAL_LEN] = np.clip(rng.normal(size=REAL_LEN) * 0.3, -0.9, 0.9)

    def pt(vec):
        return cc.MakeCKKSPackedPlaintext([float(v) for v in vec])

    def dec_full(ct):
        p = cc.Decrypt(keys.secretKey, ct)
        p.SetLength(BATCH)
        return np.array(p.GetRealPackedValue())

    def step_A_he(ct):
        return cc.EvalMult(ct, 0.999)

    def step_A_np(x):
        return x * 0.999

    def step_C_he(ct):
        sq = cc.EvalMult(ct, ct)
        t = cc.EvalMult(sq, -0.005)
        return cc.EvalAdd(cc.EvalAdd(ct, t), 0.00045)

    def step_C_np(x):
        return x - 0.005 * x * x + 0.00045

    tables = {}
    for name, step_he, step_np in (("A: solo bootstrap", step_A_he, step_A_np),
                                    ("C: con non linearita' neutra", step_C_he, step_C_np)):
        print(f"\n=== Catena {name}: {K_ITERS} bootstrap ===", flush=True)
        ct = cc.Encrypt(keys.publicKey, pt(data))
        x_ref = data.copy()
        rows = []
        t_chain = time.time()
        for k in range(1, K_ITERS + 1):
            pre = step_he(ct)
            new = cc.EvalBootstrap(pre)
            if k == 1:
                print(f"  livello dopo il 1o bootstrap: {new.GetLevel()}", flush=True)
            ct.Offload()
            pre.Offload()
            cc.TrimGPUMemoryPool()
            ct = new
            x_ref = step_np(x_ref)
            if k in CHECKPOINTS:
                he = dec_full(ct)
                err = np.abs(he - x_ref)
                e_max = err[:REAL_LEN].max()
                e_mean = err[:REAL_LEN].mean()
                e_pad = err[REAL_LEN:].max()
                rows.append((k, e_max, e_mean, e_pad))
                print(f"  k={k:2d}  errore max {e_max:.3e}  medio {e_mean:.3e}  padding max {e_pad:.3e}"
                      f"   (valore tipico |x| ~ {np.abs(x_ref[:REAL_LEN]).mean():.3f})", flush=True)
        print(f"  catena completata in {time.time()-t_chain:.0f}s", flush=True)
        tables[name] = rows
        del ct
        cc.TrimGPUMemoryPool()
        gpu_mem(f"dopo la catena {name[0]}")

    print("\n=== RIEPILOGO: come cresce l'errore massimo? ===")
    for name, rows in tables.items():
        print(f"\n  Catena {name}")
        print(f"  {'k':>3} {'err max':>11} {'err medio':>11} {'max/max(1)':>11} {'sqrt(k)':>8} {'k':>4}")
        e1 = rows[0][1]
        for k, e_max, e_mean, e_pad in rows:
            print(f"  {k:3d} {e_max:11.3e} {e_mean:11.3e} {e_max/e1:11.2f} {k**0.5:8.2f} {k:4d}")
        ks = np.array([r[0] for r in rows], dtype=float)
        em = np.array([r[2] for r in rows], dtype=float)
        if len(ks) >= 3:
            p = float(np.polyfit(np.log(ks), np.log(em), 1)[0])
            if p < 0.65:
                verdict = "compatibile con sqrt(k): rumore che si somma in modo indipendente (benigno)"
            elif p < 0.85:
                verdict = "tra sqrt(k) e lineare: parte sistematica che si accumula"
            elif p < 1.15:
                verdict = "LINEARE in k: errore sistematico che si accumula (preoccupante)"
            else:
                verdict = "PIU' CHE LINEARE: la catena amplifica l'errore (problema serio)"
            print(f"  -> esponente di crescita dell'errore MEDIO (fit log-log su k): {p:.2f}  =>  {verdict}")
        print(f"  -> a k={rows[-1][0]}: errore max {rows[-1][1]:.2e}, medio {rows[-1][2]:.2e} "
              f"(primo bootstrap: max {rows[0][1]:.2e}, medio {rows[0][2]:.2e})")


def main():
    print(f"=== Catena di bootstrap, 65.536 slot, bsgsDim={BSGS_DIM}, K={K_ITERS} ===\n")
    gpu_mem("inizio")
    cc, keys = make_context()
    gpu_mem("dopo LoadContext")
    run(cc, keys)
    gpu_mem("fine")


if __name__ == '__main__':
    main()