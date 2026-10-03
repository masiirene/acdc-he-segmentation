"""
crypto/test_conv_plus_bootstrap_rows_zeus.py

Prima integrazione vera: convoluzione nel formato a righe (metodo
delle diagonali, verificato oggi) SEGUITA da bootstrap su ogni riga
di output -- unendo i due pezzi testati finora separatamente.

Larghezza REALE ma ancora moderata: Cin=Cout=32 (enc0/dec0), non
ancora 256 -- un passo alla volta, come sempre in questo progetto.

Numero di righe LIMITATO (Hp=10, non le 258 reali) per restare
leggeri mentre si verifica che l'integrazione funzioni -- niente
ancora sulla scala vera, solo sulla CORRETTEZZA dell'unione.

Pattern di offload/trim per riga, stessa disciplina di ieri sera
(Offload() + TrimGPUMemoryPool() prima/dopo ogni bootstrap).

Uso: python3 crypto/test_conv_plus_bootstrap_rows_zeus.py
"""

import sys
import math
import time
import subprocess
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe

GiB = 1 << 30
DEPTH = 43
RING_POW = 17
LEVEL_BUDGET = [4, 4]
BSGS_DIM = [4, 4]
CACHE_GIB = 1

CHANNELS = 64  # prossimo stage vero (enc1/dec1) -- stesso schema appena
               # confermato a 32, verifichiamo che regga anche qui
IMG_W = 224
HALO = 1
WP = IMG_W + 2 * HALO  # 226
H_SMALL = 4  # ridotta ulteriormente (da 8 a 4) per prudenza, dopo il
             # crash SSH -- rialziamo una volta confermato che la
             # correzione della cache dei plaintext risolve il problema
K = 3


def print_gpu_mem(label):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'], text=True
        ).strip()
        used, total = out.split(',')
        print(f"    [GPU MEM] {label}: {used.strip()} / {total.strip()} MiB")
    except Exception as e:
        print(f"    [GPU MEM] {label}: impossibile leggere ({e})")


def build_context(batch_size):
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    params.SetRingDim(1 << RING_POW)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(batch_size)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetDevices([0])

    cc = fhe.GenCryptoContext(params)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    print("  Bootstrap setup...")
    t0 = time.time()
    cc.EvalBootstrapSetup(LEVEL_BUDGET, BSGS_DIM, batch_size)
    cc.EvalBootstrapKeyGen(keys.secretKey, batch_size)
    print(f"    fatto in {time.time()-t0:.1f}s")

    # Chiavi di rotazione per lo schema a righe: shift orizzontali
    # (kx*n, con n=CHANNELS) e le d/(d-n) per la correzione a blocchi.
    n = CHANNELS
    rot_offsets = set()
    for kx in range(K):
        rot_offsets.add(kx * n)
    for d in range(1, n):
        rot_offsets.add(d)
        rot_offsets.add(d - n)
    rot_list = sorted(rot_offsets)
    print(f"  Chiavi di rotazione extra (non-bootstrap): {len(rot_list)}")
    cc.EvalRotateKeyGen(keys.secretKey, rot_list)

    cc.SetRotationKeyCache(CACHE_GIB * GiB)
    cc.SetBootstrapCache(CACHE_GIB * GiB)
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    print(f"  LoadContext: {time.time()-t0:.1f}s")
    cc.SetPlaintextCache(CACHE_GIB * GiB)
    cc.SetCiphertextCache(CACHE_GIB * GiB)

    return cc, keys


def make_mask(total_len, block_size, d, want_main):
    rel_pos = np.arange(total_len) % block_size
    if want_main:
        return (rel_pos < (block_size - d)).astype(float)
    else:
        return (rel_pos >= (block_size - d)).astype(float)


def diagonal_mix_channels_fhe(cc, ct_row, W_k, n, wp, total_len, mask_cache, diag_cache, kernel_pos_key):
    """
    diag_cache: memorizza i plaintext delle diagonali per (posizione
    kernel, d) -- NON dipendono dalla riga, quindi vanno calcolati UNA
    SOLA VOLTA e riusati per tutte le Hp righe, non ricreati ogni volta
    (bug trovato dopo un crash SSH: 2.880 plaintext creati invece di
    288 -- probabilmente la causa del sovraccarico di RAM di sistema).

    SECONDO fix, dopo un secondo crash SSH: anche i ciphertext
    TEMPORANEI creati qui dentro (rotated_main, rotated_wrap, term_*)
    restano nel pool di FIDESlib finche' non si chiama
    TrimGPUMemoryPool() esplicitamente -- con ~12.000 operazioni totali
    nell'intera convoluzione e NESSUN trim nel mezzo, l'accumulo cresce
    senza controllo. Puliamo ogni poche diagonali, non solo a fine riga.
    """
    W_padded = W_k  # gia' quadrata (Cin==Cout==n in questo test)
    acc = None
    for d in range(n):
        # Trim interno RIMOSSO per questo esperimento di velocita' --
        # proviamo ad affidarci SOLO al trim dopo ogni posizione di
        # kernel (9 volte per riga, molto meno frequente) per vedere
        # se basta a tenere la memoria piatta, recuperando velocita'.
        cache_key = (kernel_pos_key, d)
        if cache_key not in diag_cache:
            diag = np.array([W_padded[i, (i + d) % n] for i in range(n)])
            diag_full = np.zeros(total_len)
            diag_full[:wp * n] = np.tile(diag, wp)
            diag_cache[cache_key] = cc.MakeCKKSPackedPlaintext(diag_full.tolist())
        diag_pt = diag_cache[cache_key]

        if d == 0:
            row_rot = ct_row
        else:
            if d not in mask_cache:
                m_main = make_mask(total_len, n, d, True)
                m_wrap = make_mask(total_len, n, d, False)
                mask_cache[d] = (cc.MakeCKKSPackedPlaintext(m_main.tolist()),
                                  cc.MakeCKKSPackedPlaintext(m_wrap.tolist()))
            mask_main_pt, mask_wrap_pt = mask_cache[d]
            rotated_main = cc.EvalRotate(ct_row, d)
            rotated_wrap = cc.EvalRotate(ct_row, d - n)
            term_main = cc.EvalMult(rotated_main, mask_main_pt)
            term_wrap = cc.EvalMult(rotated_wrap, mask_wrap_pt)
            row_rot = cc.EvalAdd(term_main, term_wrap)

        term = cc.EvalMult(row_rot, diag_pt)
        acc = term if acc is None else cc.EvalAdd(acc, term)
    return acc


def main():
    n = CHANNELS
    real_len = WP * n
    batch_size = 1 << math.ceil(math.log2(real_len))
    Hp = H_SMALL + 2 * HALO

    print(f"=== Conv + Bootstrap uniti, formato a righe, {CHANNELS} canali, {Hp} righe ===")
    print(f"Wp={WP}, n={n}, real_len={real_len}, batch={batch_size}\n")

    print_gpu_mem("prima del contesto")
    t0 = time.time()
    cc, keys = build_context(batch_size)
    print(f"Contesto pronto in {time.time()-t0:.1f}s.")
    print_gpu_mem("dopo LoadContext")

    rng = np.random.default_rng(5)
    x = rng.normal(size=(n, H_SMALL, IMG_W))
    x_padded = np.pad(x, ((0, 0), (HALO, HALO), (HALO, HALO)))
    weight = rng.normal(size=(n, n, K, K)) * (0.1 / math.sqrt(n * K * K))
    bias = rng.normal(size=(n,)) * 0.01

    def pack_row(r):
        row = x_padded[:, r, :].T  # (Wp, n)
        flat = row.flatten()
        full = np.zeros(batch_size)
        full[:len(flat)] = flat
        return full

    print("\nCifratura righe di input...")
    rows_in = []
    for r in range(Hp):
        pt = cc.MakeCKKSPackedPlaintext(pack_row(r).tolist())
        rows_in.append(cc.Encrypt(keys.publicKey, pt))
    print_gpu_mem("dopo cifratura input")

    print("\nConvoluzione (una sola, senza norm/att -- solo per verificare conv+bootstrap uniti)...")
    mask_cache = {}
    diag_cache = {}  # (ky,kx,d) -> plaintext, calcolato UNA VOLTA, riusato per tutte le righe
    t0 = time.time()
    rows_out = [None] * Hp
    for r_out in range(Hp):
        for ky in range(K):
            r_in = r_out + ky
            if not (0 <= r_in < Hp):
                continue
            for kx in range(K):
                shift = kx * n
                row_shifted = rows_in[r_in] if shift == 0 else cc.EvalRotate(rows_in[r_in], shift)
                W_k = weight[:, :, ky, kx]
                contrib = diagonal_mix_channels_fhe(cc, row_shifted, W_k, n, WP, batch_size,
                                                     mask_cache, diag_cache, kernel_pos_key=(ky, kx))
                rows_out[r_out] = contrib if rows_out[r_out] is None else cc.EvalAdd(rows_out[r_out], contrib)
                cc.TrimGPUMemoryPool()  # pulizia dopo OGNI posizione di kernel
        print_gpu_mem(f"dopo riga di output {r_out}")
    print(f"Convoluzione completata in {time.time()-t0:.2f}s.")
    print_gpu_mem("dopo la convoluzione")
    if rows_out[0] is not None:
        print(f"Livello riga di output 0: {rows_out[0].GetLevel()}")

    print("\nBootstrap su ogni riga di output valida...")
    t0 = time.time()
    rows_boot = []
    for r in range(H_SMALL):
        if rows_out[r] is None:
            continue
        ct_b = cc.EvalBootstrap(rows_out[r])
        rows_boot.append(ct_b)
        rows_out[r].Offload()
        cc.TrimGPUMemoryPool()
    print(f"Bootstrap di {len(rows_boot)} righe completato in {time.time()-t0:.2f}s.")
    print_gpu_mem("dopo tutti i bootstrap")
    print(f"Livello dopo bootstrap: {rows_boot[0].GetLevel() if rows_boot else 'N/A'}")

    print("\nVerifica correttezza contro riferimento numpy...")

    def conv2d_direct(x, W, b, K=3):
        Cin_, Hp_, Wp_ = x.shape
        Cout_ = W.shape[0]
        out = np.zeros((Cout_, Hp_, Wp_))
        for co in range(Cout_):
            for ci in range(Cin_):
                for ky in range(K):
                    for kx in range(K):
                        shifted = np.roll(np.roll(x[ci], -ky, axis=0), -kx, axis=1)
                        out[co] += W[co, ci, ky, kx] * shifted
        for co in range(Cout_):
            out[co] += b[co]
        return out

    ref = conv2d_direct(x_padded, weight, bias, K=K)
    # NOTA: il bias non e' ancora stato aggiunto nella pipeline HE in
    # questo test (solo conv+bootstrap, per isolare l'integrazione) --
    # lo sottraiamo dal riferimento per un confronto equo.
    ref_no_bias = ref - bias[:, None, None]

    max_err = 0.0
    for r in range(H_SMALL):
        pt = cc.Decrypt(keys.secretKey, rows_boot[r])
        pt.SetLength(batch_size)
        he_row = np.array(pt.GetRealPackedValue())[:WP * n].reshape(WP, n)[:IMG_W, :n]
        ref_row = ref_no_bias[:, r, :IMG_W].T
        err = np.max(np.abs(he_row - ref_row))
        max_err = max(max_err, err)
        print(f"  Riga {r}: errore max = {err:.6e}")

    print(f"\nErrore massimo: {max_err:.6e}")
    if max_err < 1e-2:
        print("\n=== CONVOLUZIONE + BOOTSTRAP UNITI FUNZIONANO, formato a righe, larghezza reale. ===")
    else:
        print("\n=== ATTENZIONE: errore grande -- da investigare prima di scalare oltre. ===")


if __name__ == '__main__':
    main()