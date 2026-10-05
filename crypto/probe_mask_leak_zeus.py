"""
crypto/probe_mask_leak_zeus.py

COSA SAPPIAMO (sonde precedenti, tutte su Zeus):
  - probe_offload_levels (un processo per misura): ciphertext prodotti da EvalAdd(ct, scalare) e parcheggiati
    con Offload() NON occupano memoria GPU, a nessun livello (1, 16, 34). L'ipotesi "Offload non libera i
    ciphertext profondi" e' SMENTITA.
  - nelle stesse sonde la modalita' 'keep' cresce sempre di 6144 MiB = la dimensione della cache dei ciphertext:
    quei valori per ciphertext (128/83/30) sono solo 6144/n, non una misura.
  - ma tutte le uscite che invece LASCIANO memoria dopo Offload (circa 64 MiB ciascuna, a livello 2 come a 17:
    conv, norm+attivazione) hanno in comune l'ultima operazione: EvalMult(ciphertext, plaintext) con la MASCHERA.
  - il backtrace dell'out of memory passa da  Ciphertext::multPt -> Plaintext::adjustPlaintextToCiphertext ->
    Plaintext::copy -> GPUmalloc: ogni moltiplicazione per plaintext fa una COPIA del plaintext adattata al livello.
  64 MiB = 44 limbi x 1 MiB x 1.5 (fattore del pool, visto nelle sonde): la dimensione di un polinomio a lunghezza piena.

IPOTESI H2 (da verificare): il risultato di EvalMult(ct, plaintext) si porta dietro la copia del plaintext, che
Offload() non libera; viene liberata solo quando il ciphertext e' distrutto.

QUESTA SONDA (un solo processo, ~4 minuti; le liste restano VIVE fino alla fine, cosi' la crescita di ogni variante
si somma e il riuso del pool non la nasconde; tutte in modalita' Offload):
  A  EvalMult(x, scalare)                          atteso ~0 sia sotto H2 sia no
  B  EvalMult(x, maschera)                         H2: ~64 MiB per ciphertext
  C  EvalMult(x, maschera creata AL LIVELLO di x)  se l'API lo permette: evita la copia?
  D  EvalMult(x, maschera) poi EvalAdd(., 0.0), si tiene solo il risultato dell'add  ("lavaggio"): sparisce?
  F  uscite di bootstrap (dopo un bootstrap di riscaldamento che carica i dati una tantum)
LETTURA: B >> A conferma H2; se D ~ 0 il lavaggio e' la correzione (una riga, nessun livello);
         se C ~ 0 si puo' evitare la copia a monte.

Uso (in tmux, GPU libera):   python3 crypto/probe_mask_leak_zeus.py
"""

import os
import sys
import gc
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fideslib_py as fhe
from prototype_lattice_resample import Grid
from lattice_conv import LHE
from test_stage_bootstrap_fullres_zeus import build_context

H0, W0 = 256, 224
LEVEL = 2
N_CT = 64
N_BOOT = 24


def used_mib():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                      text=True).strip()
        return int(out.split()[0])
    except Exception:
        return -1


def run(cc, keys, g, log=print, mem=used_mib, n_ct=N_CT, n_boot=N_BOOT):
    he = LHE(cc, keys, g)
    rng = np.random.default_rng(21)
    x0 = he.enc_offloaded([g.pack(rng.normal(size=g.size(LEVEL)) * 0.5, LEVEL)])[0]
    x = cc.EvalMult(cc.EvalMult(x0, 1.0001), 1.0001)                  # livello 2, come le uscite di conv
    he.dec(x)
    cc.TrimGPUMemoryPool()
    mask_vec = g.valid_mask(LEVEL)
    mask_pt = he.pt(mask_vec)
    lvl = x.GetLevel()
    keep = {}
    results = []

    def measure(label, make, n, note=""):
        cc.TrimGPUMemoryPool()
        m0 = mem()
        lst = []
        for i in range(n):
            c = make(i)
            c.Offload()
            lst.append(c)
            if i % 8 == 7:
                cc.TrimGPUMemoryPool()
        cc.TrimGPUMemoryPool()
        m1 = mem()
        keep[label] = lst                                              # tenute vive
        per = (m1 - m0) / n
        results.append((label, per))
        log(f"  {label:<44} {n:>3} ciphertext: {m0} -> {m1} MiB   {per:7.1f} MiB per ciphertext {note}")
        return per

    log(f"\n  ciphertext a livello {lvl}; tutte le varianti in modalita' Offload, liste vive\n")
    pa = measure("A  EvalMult(x, scalare)", lambda i: cc.EvalMult(x, 1.0 + 1e-6 * (i + 1)), n_ct)
    pb = measure("B  EvalMult(x, maschera)", lambda i: cc.EvalMult(x, mask_pt), n_ct)
    # C: plaintext creato direttamente al livello del ciphertext (se l'API del wrapper lo consente)
    try:
        pt_l = cc.MakeCKKSPackedPlaintext([float(v) for v in mask_vec], 1, lvl)
        pc = measure("C  EvalMult(x, maschera creata al livello)", lambda i: cc.EvalMult(x, pt_l), n_ct)
    except Exception as e:
        pc = None
        log(f"  C  EvalMult(x, maschera creata al livello)   NON ESEGUIBILE: {type(e).__name__}: {str(e)[:90]}")

    def launder(i):
        y = cc.EvalMult(x, mask_pt)
        z = cc.EvalAdd(y, 0.0)
        del y
        return z
    pd = measure("D  EvalMult(x, maschera) + EvalAdd(., 0.0)", launder, n_ct, "(si tiene solo il risultato dell'add)")

    log("\n  riscaldamento del bootstrap (carica i dati una tantum, non va contato) ...")
    w = cc.EvalBootstrap(x)
    w.Offload()
    keep["warm"] = [w]
    cc.TrimGPUMemoryPool()
    pf = measure("F  uscite di EvalBootstrap", lambda i: cc.EvalBootstrap(x), n_boot)

    log("\n=== RIEPILOGO (MiB trattenuti per ciphertext dopo Offload) ===")
    for label, per in results:
        log(f"  {label:<46}{per:8.1f}")
    log("\n  LETTURA:")
    if pb > 30 and pa < 0.25 * pb:
        log(f"   - CONFERMATO H2: la moltiplicazione per plaintext lascia ~{pb:.0f} MiB per ciphertext che Offload non libera "
            f"(la moltiplicazione per scalare no: {pa:.0f}).")
        if pd is not None and pd < 0.25 * pb:
            log(f"   - il 'lavaggio' con EvalAdd(., 0.0) elimina quasi tutto ({pd:.0f} MiB): e' la correzione, senza costo di livelli. "
                f"In lattice_conv si attiva con MASK_LAUNDER=1.")
        else:
            log(f"   - il lavaggio NON basta ({pd:.0f} MiB): serve un'altra via (maschera al livello, o evitare la copia).")
        if pc is not None and pc < 0.25 * pb:
            log(f"   - creare la maschera gia' al livello del ciphertext evita la copia ({pc:.0f} MiB).")
    else:
        log(f"   - H2 NON confermata: B ({pb:.0f}) non si distingue da A ({pa:.0f}). La memoria trattenuta ha un'altra origine.")
    if pf is not None and pf > 30:
        log(f"   - anche le uscite del bootstrap trattengono ~{pf:.0f} MiB ciascuna.")
    return results


def main():
    g = Grid(H0, W0, LEVEL + 1)
    print("=== SONDA MASCHERA: la moltiplicazione per plaintext lascia memoria dopo Offload? ===")
    cc, keys = build_context(g, LEVEL)
    run(cc, keys, g)


if __name__ == '__main__':
    main()