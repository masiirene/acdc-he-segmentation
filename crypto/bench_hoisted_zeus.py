"""
crypto/bench_hoisted_zeus.py

SECONDO benchmark. Il primo (bench_ops_zeus.py) ha mostrato che il formato
a righe costa 130-210 ms per termine contro ~30-39 ms dello schema
originale, quindi non porta un vantaggio di velocita'. Qui si misura se lo
schema ORIGINALE (un ciphertext per canale, immagine intera dentro) puo'
andare piu' veloce con HOISTING: ruotare ogni canale d'ingresso UNA volta
per posizione di kernel e riusare la rotazione per molti canali d'uscita,
invece di rifarla per ogni canale d'uscita.

Il problema di stamattina con questa idea ("fast", 64 canali): teneva vivi
Cin*9 ciphertext ruotati insieme -> troppa memoria, e piu' lenta. Qui il
working set e' tenuto piccolo in tre modi, tutti misurati sullo stesso
contesto:

  REF       ciclo originale (rotazione dentro il ciclo interno), su Cout ridotto
  BLOCCO-B  accumulatori per B canali d'uscita alla volta (B = 8, 32); le
            rotazioni si rifanno una volta per blocco (Cout/B volte)
  OFFLOAD   TUTTI gli accumulatori su RAM host, uno solo alla volta in GPU
            (Offload() dopo ogni gruppo di 9 termini); le rotazioni si
            fanno UNA volta sola

Per ogni variante: tempo per termine (prodotto per scalare + addizione),
verifica di correttezza di 2 canali d'uscita contro numpy, e PROIEZIONE
sulla rete intera (3.18 M termini naive nei 9 stage con risoluzioni reali).
La proiezione ASSUME che il costo per termine misurato qui valga ovunque
(contesto a 65.536 slot): e' un'ipotesi, non un dato.

Uso (in tmux, UN LANCIO ALLA VOLTA):
  python3 crypto/bench_hoisted_zeus.py 1     # cache ciphertext 1 GiB
  python3 crypto/bench_hoisted_zeus.py 6     # cache ciphertext 6 GiB

Ordine: REF, BLOCCO-8, BLOCCO-32, OFFLOAD (l'ultima e' quella con l'API
Offload/Reload, la piu' rischiosa: se la libreria crasha, le altre sono
gia' stampate).
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
WP = 226
K = 3
OFFSETS = [ky * WP + kx for ky in range(K) for kx in range(K)]   # 9 posizioni
ROT_LIST = sorted(set(OFFSETS) - {0})                            # 8 chiavi

CIN = 8
COUT = 64
COUT_REF = 16
BLOCK_SIZES = [8, 32]
NAIVE_TERMS_NETWORK = 3_179_808      # somma 9*Cin*Cout sui 9 stage (calcolata prima)

CT_CACHE_GIB = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
ROT_CACHE_GIB = 4.0                  # 8 chiavi * ~354 MB = 2.8 GB: entrano, non e' la variabile


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
    cc.EvalRotateKeyGen(keys.secretKey, ROT_LIST)
    cc.SetRotationKeyCache(int(ROT_CACHE_GIB * GiB))
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s", flush=True)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(int(CT_CACHE_GIB * GiB))
    return cc, keys


def run(cc, keys):
    rng = np.random.default_rng(5)
    x_vals = [rng.normal(size=BATCH) * 0.5 for _ in range(CIN)]
    w = rng.normal(size=(COUT, CIN, K * K)) * 0.1

    def pt(vec):
        return cc.MakeCKKSPackedPlaintext([float(v) for v in vec])

    x = [cc.Encrypt(keys.publicKey, pt(v)) for v in x_vals]

    def decrypt_full(ct):
        p = cc.Decrypt(keys.secretKey, ct)
        p.SetLength(BATCH)
        return np.array(p.GetRealPackedValue())

    def sync(ct):
        p = cc.Decrypt(keys.secretKey, ct)
        p.SetLength(4)
        p.GetRealPackedValue()

    sync(x[0])
    t0 = time.time()
    for _ in range(3):
        sync(x[0])
    sync_cost = (time.time() - t0) / 3
    print(f"  (costo sincronizzazione: {sync_cost:.2f}s, sottratto)\n", flush=True)

    def reference_channel(co):
        ref = np.zeros(BATCH)
        for ci in range(CIN):
            for k, off in enumerate(OFFSETS):
                ref += w[co, ci, k] * np.roll(x_vals[ci], -off)
        return ref

    def check(outs, label):
        errs = []
        for co in (0, len(outs) - 1):
            errs.append(np.max(np.abs(decrypt_full(outs[co]) - reference_channel(co))))
        print(f"    correttezza ({label}, canali 0 e {len(outs)-1}): errore max {max(errs):.2e}", flush=True)
        return max(errs)

    results = {}

    def report(label, n_terms, n_rot, elapsed):
        per_term = elapsed / n_terms * 1000
        proj_h = per_term / 1000 * NAIVE_TERMS_NETWORK / 3600
        results[label] = (per_term, proj_h)
        print(f"  {label:<34} {per_term:7.1f} ms/termine   ({n_terms} termini, {n_rot} rotazioni, {elapsed:.0f}s)"
              f"   -> rete intera ~{proj_h:.1f} h", flush=True)

    def rotated_inputs(ci):
        return [x[ci] if off == 0 else cc.EvalRotate(x[ci], off) for off in OFFSETS]

    # ---------------- REF: ciclo originale ----------------
    print("--- REF: ciclo originale (rotazione dentro il ciclo interno) ---", flush=True)
    t0 = time.time()
    outs_ref = []
    for co in range(COUT_REF):
        acc = None
        for ci in range(CIN):
            for k, off in enumerate(OFFSETS):
                sh = x[ci] if off == 0 else cc.EvalRotate(x[ci], off)
                t = cc.EvalMult(sh, float(w[co, ci, k]))
                acc = t if acc is None else cc.EvalAdd(acc, t)
        acc.Offload()
        outs_ref.append(acc)
        cc.TrimGPUMemoryPool()
    sync(outs_ref[-1])
    el = time.time() - t0 - sync_cost
    report(f"REF (Cout={COUT_REF})", COUT_REF * CIN * 9, COUT_REF * CIN * 8, el)
    check(outs_ref, "REF")
    del outs_ref
    cc.TrimGPUMemoryPool()
    gpu_mem("dopo REF")

    # ---------------- BLOCCO-B ----------------
    for B in BLOCK_SIZES:
        print(f"\n--- BLOCCO-{B}: {B} accumulatori vivi, rotazioni rifatte {COUT // B} volte ---", flush=True)
        t0 = time.time()
        outs = [None] * COUT
        n_rot = 0
        for b0 in range(0, COUT, B):
            cos = list(range(b0, min(b0 + B, COUT)))
            acc = {co: None for co in cos}
            for ci in range(CIN):
                rot = rotated_inputs(ci)
                n_rot += 8
                for co in cos:
                    a = acc[co]
                    for k in range(9):
                        t = cc.EvalMult(rot[k], float(w[co, ci, k]))
                        a = t if a is None else cc.EvalAdd(a, t)
                    acc[co] = a
                del rot
                cc.TrimGPUMemoryPool()
            for co in cos:
                acc[co].Offload()
                outs[co] = acc[co]
            del acc
            cc.TrimGPUMemoryPool()
        sync(outs[-1])
        el = time.time() - t0 - sync_cost
        report(f"BLOCCO-{B}", COUT * CIN * 9, n_rot, el)
        check(outs, f"BLOCCO-{B}")
        del outs
        cc.TrimGPUMemoryPool()
        gpu_mem(f"dopo BLOCCO-{B}")

    # ---------------- OFFLOAD ----------------
    print("\n--- OFFLOAD: tutti gli accumulatori su RAM, uno alla volta in GPU, rotazioni UNA volta ---", flush=True)
    t0 = time.time()
    acc = [None] * COUT
    n_rot = 0
    for ci in range(CIN):
        rot = rotated_inputs(ci)
        n_rot += 8
        for co in range(COUT):
            a = acc[co]          # se e' su host, si ricarica da solo al primo uso
            for k in range(9):
                t = cc.EvalMult(rot[k], float(w[co, ci, k]))
                a = t if a is None else cc.EvalAdd(a, t)
            a.Offload()
            acc[co] = a
            if co % 8 == 7:
                cc.TrimGPUMemoryPool()
        del rot
        cc.TrimGPUMemoryPool()
    sync(acc[-1])
    el = time.time() - t0 - sync_cost
    report("OFFLOAD", COUT * CIN * 9, n_rot, el)
    check(acc, "OFFLOAD")
    del acc
    cc.TrimGPUMemoryPool()
    gpu_mem("dopo OFFLOAD")

    print("\n=== RIEPILOGO ===")
    print(f"Cache ciphertext {CT_CACHE_GIB} GiB, Cin={CIN}, Cout={COUT}, batch {BATCH}\n")
    print(f"  {'variante':<26}{'ms/termine':>12}{'rete intera (h)':>18}")
    for k, (pt_ms, proj) in results.items():
        print(f"  {k:<26}{pt_ms:12.1f}{proj:18.1f}")
    print("\n  Riferimenti: naive misurato ieri 29-39 ms/termine (26-34 h);")
    print("               formato a righe misurato oggi 130-210 ms/termine (44-73 h).")
    print("  La colonna 'rete intera' assume lo stesso costo per termine in tutti gli stage.")


def main():
    print(f"=== Benchmark hoisting (ring 2^{RING_POW}, batch {BATCH}, depth {DEPTH}) ===")
    print(f"Cache ciphertext {CT_CACHE_GIB} GiB, cache rotazioni {ROT_CACHE_GIB} GiB, {len(ROT_LIST)} chiavi\n")
    gpu_mem("inizio")
    cc, keys = make_context()
    gpu_mem("dopo LoadContext")
    print()
    run(cc, keys)
    gpu_mem("fine")


if __name__ == '__main__':
    main()