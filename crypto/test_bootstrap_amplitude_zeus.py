"""
crypto/test_bootstrap_amplitude_zeus.py

DOMANDA: il bootstrap CKKS a 65.536 slot (bsgsDim=[8,8]) e' preciso solo
per valori piccoli, o regge anche valori di ordine 10-1000?

Perche': tutti i test di bootstrap fatti finora usano dati di ampiezza
~0.1-3. Ma nelle reti vere, anche in quella allenata SENZA clamp con la
penalita' sulle attivazioni (log di phase3_penalty_noclamp_trial15_v2),
i valori grezzi arrivano a: p99 ~ 18, massimo assoluto ~ 100-300 per epoca;
nelle reti dipendenti dal clamp, spente, a migliaia. Il bootstrap CKKS
approssima una riduzione modulare con un polinomio valido su un intervallo
limitato: fuori da quello, l'errore puo' esplodere (non e' graduale).

Per ogni ampiezza A: dati ~ N(0, A/3) troncati a +-A, K bootstrap di fila
(con una moltiplicazione per 0.999 prima di ognuno, applicata anche al
riferimento). Si stampa errore massimo, medio, e relativo ad A, dopo il
1o bootstrap e dopo il K-esimo.

LETTURA:
  - errore assoluto ~ costante (~3e-3) al crescere di A: il bootstrap regge
    quella ampiezza, l'errore relativo diventa ancora piu' piccolo;
  - errore che cresce con A (relativo ~ costante): precisione relativa fissa;
  - errore dell'ordine di A: il bootstrap FALLISCE per quella ampiezza
    (valori fuori dall'intervallo dell'approssimazione).
Il test dice dove stanno i confini: per sapere se serve riscalare le
attivazioni prima del bootstrap.

Uso (in tmux, GPU libera):
  python3 crypto/test_bootstrap_amplitude_zeus.py [K [A1 A2 ...]]
  default: K=3, A in 0.3 3 10 30 100 300 1000   (~2-3 minuti piu' ~1 di setup)
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
BSGS_DIM = [8, 8]
REAL_LEN = 258 * 226
K_ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
AMPLITUDES = [float(a) for a in sys.argv[2:]] if len(sys.argv) > 2 else [0.3, 3, 10, 30, 100, 300, 1000]


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


def run(cc, keys, amplitudes=AMPLITUDES, k_iters=K_ITERS, log=print):
    rng = np.random.default_rng(11)

    def pt(vec):
        return cc.MakeCKKSPackedPlaintext([float(v) for v in vec])

    def dec_full(ct):
        p = cc.Decrypt(keys.secretKey, ct)
        p.SetLength(BATCH)
        return np.array(p.GetRealPackedValue())

    rows = []
    for A in amplitudes:
        data = np.zeros(BATCH)
        data[:REAL_LEN] = np.clip(rng.normal(size=REAL_LEN) * (A / 3.0), -A, A)
        ct = cc.Encrypt(keys.publicKey, pt(data))
        ref = data.copy()
        log(f"\n  Ampiezza A = {A:g}  (|x| medio {np.abs(data[:REAL_LEN]).mean():.3g}, max {np.abs(data).max():.3g})")
        entry = {"A": A}
        t0 = time.time()
        for k in range(1, k_iters + 1):
            pre = cc.EvalMult(ct, 0.999)
            new = cc.EvalBootstrap(pre)
            ct.Offload()
            pre.Offload()
            cc.TrimGPUMemoryPool()
            ct = new
            ref = ref * 0.999
            if k == 1 or k == k_iters:
                he = dec_full(ct)
                err = np.abs(he - ref)
                e_max, e_mean, e_pad = err[:REAL_LEN].max(), err[:REAL_LEN].mean(), err[REAL_LEN:].max()
                entry[k] = (e_max, e_mean, e_pad)
                log(f"    dopo {k} bootstrap: errore max {e_max:.3e}  medio {e_mean:.3e}  "
                    f"max/A {e_max / A:.2e}  padding max {e_pad:.3e}")
        entry["time"] = time.time() - t0
        rows.append(entry)
        del ct
        cc.TrimGPUMemoryPool()
    return rows


def summarize(rows, k_iters=K_ITERS, log=print):
    log("\n=== RIEPILOGO ===")
    log(f"  {'A':>7} {'err max (1 boot)':>17} {'err medio':>11} {'max/A':>10}   {f'err max ({k_iters} boot)':>17}   esito")
    base_abs = rows[0][1][0]            # errore assoluto alla piu' piccola ampiezza
    for r in rows:
        A = r["A"]
        e1 = r[1]
        ek = r[k_iters] if k_iters in r else r[1]
        if e1[0] > 0.2 * A:
            verdict = "FALLITO: errore dell'ordine di A"
        elif e1[0] > 10 * base_abs:
            verdict = "errore assoluto cresce con A (precisione ~relativa)"
        else:
            verdict = "ok (errore assoluto ~ costante)"
        log(f"  {A:7g} {e1[0]:17.3e} {e1[1]:11.3e} {e1[0] / A:10.2e}   {ek[0]:17.3e}   {verdict}")
    ok = [r["A"] for r in rows if r[1][0] <= 0.2 * r["A"]]
    if ok:
        log(f"\n  Ampiezza massima per cui il bootstrap regge (errore < 20% di A): {max(ok):g}")
    log("  Se la rete reale ha valori (anche in pochi punti) sopra quell'ampiezza, serve riscalare "
        "prima del bootstrap o contenere le attivazioni.")


def main():
    print(f"=== Bootstrap vs ampiezza dei dati: 65.536 slot, bsgsDim={BSGS_DIM}, K={K_ITERS} ===", flush=True)
    gpu_mem("inizio")
    cc, keys = make_context()
    gpu_mem("dopo LoadContext")
    rows = run(cc, keys)
    summarize(rows)
    gpu_mem("fine")


if __name__ == '__main__':
    main()