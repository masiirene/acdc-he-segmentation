"""
crypto/test_row_packed_convblock_small_zeus.py

ConvBlock COMPLETO a due stadi (Conv->Norm->Att->Conv->Norm->Att) nel
formato a righe, in HE VERO, su un caso piccolo. E' la traduzione del
prototipo numpy prototype_row_packed_convblock.py (che coincide con il
riferimento a 3.5e-15), con le scelte che in HE non sono banali:

- ALONE VERTICALE: non si azzerano ciphertext-riga. Nel ciclo della
  convoluzione si SALTANO i termini con r_in >= H (equivale a zero).
- BORDO ORIZZONTALE: moltiplicazione per una maschera plaintext dopo
  ogni stadio (costa un livello).
- STATISTICHE di norm: solo sulle H righe valide.
- RADICE INVERSA: Chebyshev grado 3 + 1 Newton, dominio per stadio.
  Il riferimento numpy e' DOPPIO: "approssimato" (stessa approssimazione
  dell'HE: misura la pipeline) ed "esatto" (misura il costo
  dell'approssimazione).

PROFONDITA': senza bootstrap, due stadi potrebbero non entrare in
DEPTH=43. Il test misura il costo del primo stadio e, se due stadi non
entrano, fa un "REFRESH SIMULATO" tra i due (decifra e ricifra con la
chiave segreta: SOLO per verificare la logica dello stadio 2, NON e' un
bootstrap e la sua precisione non dice nulla sul bootstrap vero). Lo
stampa chiaramente.

Checkpoint intermedi stampati: conv1, statistiche, uscita stadio 1,
uscita stadio 2 -- se qualcosa non torna sappiamo dove.

Uso: python3 crypto/test_row_packed_convblock_small_zeus.py
"""

import sys
import math
import time
import numpy as np

sys.path.insert(0, '/home/masi/PyFIDESlib')
sys.path.insert(0, '/home/masi/acdc-he-segmentation')

import fideslib_py as fhe
from crypto.fhe_ops_single_ciphertext import (
    poly_act_fhe, isqrt_chebyshev_fhe, fit_monotonic_isqrt_coeffs,
)

GiB = 1 << 30
DEPTH = 43
RING_POW = 17

# Caso piccolo, come il prototipo numpy
CIN, COUT = 4, 6
N = max(CIN, COUT)        # larghezza del blocco-pixel (qui Cin<Cout=N)
H, W = 5, 5
HALO = 1
K = 3
HP, WP = H + 2 * HALO, W + 2 * HALO
REAL_LEN = WP * N
BATCH = 1 << math.ceil(math.log2(REAL_LEN))
A_, B_, C_ = 0.1, 1.0, 0.5


# ============================================================
# Riferimenti numpy (convenzione del progetto: valida in alto a sx,
# alone a ZERO a destra/sotto)
# ============================================================

def conv_ref(x, w, b):
    cin, h, wd = x.shape
    cout = w.shape[0]
    xp = np.pad(x, ((0, 0), (0, K - 1), (0, K - 1)))
    out = np.zeros((cout, h, wd))
    for co in range(cout):
        for ci in range(cin):
            for ky in range(K):
                for kx in range(K):
                    out[co] += w[co, ci, ky, kx] * xp[ci, ky:ky + h, kx:kx + wd]
        out[co] += b[co]
    return out


def norm_act_ref(y, gamma, beta, inv_fn):
    c = y.shape[0]
    flat = y.reshape(c, -1)
    mean = flat.mean(axis=1)
    var = flat.var(axis=1)
    inv = inv_fn(var)
    s = (y - mean[:, None, None]) * inv[:, None, None] * gamma[:, None, None] + beta[:, None, None]
    return A_ * s ** 2 + B_ * s + C_


def make_emulator(coeffs, domain):
    """Stessa approssimazione di isqrt_chebyshev_fhe (convenzione di
    clenshaw_unit in fit_monotonic_isqrt_coeffs + 1 passo di Newton)."""
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
            y = y * (1.5 - 0.5 * v * y * y)
            out.append(y)
        return np.array(out)
    return emu


def domain_from_var(var):
    return (max(0.01, 0.5 * var.min()), 2.0 * var.max())


def build_numpy_reference(x, p):
    """Calcola i riferimenti esatto e approssimato, e i domini/coefficienti
    Chebyshev per stadio (derivati dalle varianze, come in un test: nella
    rete vera vengono dalla calibrazione)."""
    y1 = conv_ref(x, p['w1'], p['b1'])
    var1 = y1.reshape(COUT, -1).var(axis=1)
    dom1 = domain_from_var(var1)
    coeffs1, _ = fit_monotonic_isqrt_coeffs(fhe, dom1[0], dom1[1], degree=3, extra_safety=1.2)
    emu1 = make_emulator(coeffs1, dom1)
    y1n_a = norm_act_ref(y1, p['g1'], p['be1'], emu1)

    y2_a = conv_ref(y1n_a, p['w2'], p['b2'])
    var2 = y2_a.reshape(COUT, -1).var(axis=1)
    dom2 = domain_from_var(var2)
    coeffs2, _ = fit_monotonic_isqrt_coeffs(fhe, dom2[0], dom2[1], degree=3, extra_safety=1.2)
    emu2 = make_emulator(coeffs2, dom2)
    out_a = norm_act_ref(y2_a, p['g2'], p['be2'], emu2)

    exact = lambda v: 1.0 / np.sqrt(v)
    y1n_e = norm_act_ref(y1, p['g1'], p['be1'], exact)
    y2_e = conv_ref(y1n_e, p['w2'], p['b2'])
    out_e = norm_act_ref(y2_e, p['g2'], p['be2'], exact)

    return dict(y1=y1, var1=var1, dom1=dom1, coeffs1=coeffs1,
                y1n_a=y1n_a, var2=var2, dom2=dom2, coeffs2=coeffs2,
                out_a=out_a, out_e=out_e)


# ============================================================
# Layout a righe e helper HE
# ============================================================

def tile_full(vec_n):
    """Vettore per-canale (lunghezza N) ripetuto per i WP blocchi, zero
    oltre REAL_LEN, lungo BATCH."""
    v = np.zeros(N)
    v[:len(vec_n)] = vec_n
    full = np.zeros(BATCH)
    full[:REAL_LEN] = np.tile(v, WP)
    return full


def valid_cols_mask():
    m = np.zeros((WP, N))
    m[:W, :] = 1.0
    full = np.zeros(BATCH)
    full[:REAL_LEN] = m.flatten()
    return full


def pack_row(x_c_w):
    """(C, W) -> vettore lunghezza BATCH, blocchi di N canali, colonne >= W a zero."""
    c = x_c_w.shape[0]
    row = np.zeros((WP, N))
    row[:W, :c] = x_c_w.T
    full = np.zeros(BATCH)
    full[:REAL_LEN] = row.flatten()
    return full


def build_context(rot_list):
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
    cc.EvalRotateKeyGen(keys.secretKey, rot_list)
    cc.SetRotationKeyCache(1 * GiB)
    cc.LoadContext(keys.publicKey)
    cc.SetPlaintextCache(1 * GiB)
    cc.SetCiphertextCache(1 * GiB)
    return cc, keys


def required_rotations():
    s = set()
    for kx in range(1, K):
        s.add(kx * N)                       # shift orizzontale conv
    for d in range(1, N):
        s.add(d)
        s.add(d - N)                        # correzione a blocchi (diagonali)
    for k in range(1, WP):
        if k * N < REAL_LEN:
            s.add(k * N)                    # somma a passi (statistiche)
            s.add(k * N - REAL_LEN)         # correzione del wraparound
    return sorted(r for r in s if r != 0)


class HE:
    def __init__(self, cc, keys):
        self.cc = cc
        self.keys = keys
        self.diag_cache = {}
        self.bmask_cache = {}
        self.rc_cache = {}

    def pt(self, vec):
        return self.cc.MakeCKKSPackedPlaintext([float(v) for v in vec])

    def dec_rows(self, rows, cout):
        """Lista di H ciphertext-riga -> array (cout, H, W) (regione valida)."""
        out = np.zeros((cout, len(rows), W))
        for r, ct in enumerate(rows):
            p = self.cc.Decrypt(self.keys.secretKey, ct)
            p.SetLength(BATCH)
            vals = np.array(p.GetRealPackedValue())[:REAL_LEN].reshape(WP, N)
            out[:, r, :] = vals[:W, :cout].T
        return out

    def refresh_simulated(self, rows):
        """SOLO TEST: decifra e ricifra. NON e' un bootstrap."""
        new_rows = []
        for ct in rows:
            p = self.cc.Decrypt(self.keys.secretKey, ct)
            p.SetLength(BATCH)
            vals = np.array(p.GetRealPackedValue())
            new_rows.append(self.cc.Encrypt(self.keys.publicKey, self.pt(vals)))
        return new_rows

    # ---------------- convoluzione ----------------
    def diag_mix(self, ct_row, W_k, cin, cout, stage, pos):
        cc = self.cc
        W_pad = np.zeros((N, N))
        W_pad[:cout, :cin] = W_k
        acc = None
        for d in range(N):
            key = (stage, pos, d)
            if key not in self.diag_cache:
                diag = np.array([W_pad[i, (i + d) % N] for i in range(N)])
                full = np.zeros(BATCH)
                full[:REAL_LEN] = np.tile(diag, WP)
                self.diag_cache[key] = self.pt(full)
            dpt = self.diag_cache[key]
            if d == 0:
                row_rot = ct_row
            else:
                if d not in self.bmask_cache:
                    rel = np.arange(BATCH) % N
                    mm = (rel < (N - d)).astype(float)
                    self.bmask_cache[d] = (self.pt(mm), self.pt(1.0 - mm))
                m_main, m_wrap = self.bmask_cache[d]
                t_main = cc.EvalMult(cc.EvalRotate(ct_row, d), m_main)
                t_wrap = cc.EvalMult(cc.EvalRotate(ct_row, d - N), m_wrap)
                row_rot = cc.EvalAdd(t_main, t_wrap)
            term = cc.EvalMult(row_rot, dpt)
            acc = term if acc is None else cc.EvalAdd(acc, term)
        return acc

    def conv(self, rows_in, weight, bias, cin, cout, stage):
        cc = self.cc
        bias_pt = self.pt(tile_full(bias))
        rows_out = []
        for r_out in range(H):
            acc = None
            for ky in range(K):
                r_in = r_out + ky
                if r_in >= H:
                    continue          # alone verticale = zero: termine assente
                for kx in range(K):
                    shift = kx * N
                    rs = rows_in[r_in] if shift == 0 else cc.EvalRotate(rows_in[r_in], shift)
                    contrib = self.diag_mix(rs, weight[:, :, ky, kx], cin, cout, stage, (ky, kx))
                    acc = contrib if acc is None else cc.EvalAdd(acc, contrib)
            rows_out.append(cc.EvalAdd(acc, bias_pt))
        return rows_out

    # ---------------- statistiche di norm ----------------
    def _rotate_correct(self, ct, shift):
        cc = self.cc
        if shift == 0:
            return ct
        if shift not in self.rc_cache:
            idx = np.arange(BATCH)
            m_main = ((idx + shift < REAL_LEN) & (idx < REAL_LEN)).astype(float)
            m_wrap = ((idx + shift >= REAL_LEN) & (idx < REAL_LEN)).astype(float)
            self.rc_cache[shift] = (self.pt(m_main), self.pt(m_wrap))
        m_main, m_wrap = self.rc_cache[shift]
        t_main = cc.EvalMult(cc.EvalRotate(ct, shift), m_main)
        t_wrap = cc.EvalMult(cc.EvalRotate(ct, shift - REAL_LEN), m_wrap)
        return cc.EvalAdd(t_main, t_wrap)

    def sum_row_strided(self, ct_row):
        cc = self.cc
        result = None
        partial = ct_row
        remaining = WP
        shift_base = 0
        power = 1
        while remaining > 0:
            if remaining & 1:
                shifted = self._rotate_correct(partial, shift_base * N)
                result = shifted if result is None else cc.EvalAdd(result, shifted)
                shift_base += power
            remaining >>= 1
            if remaining > 0:
                partial = cc.EvalAdd(partial, self._rotate_correct(partial, power * N))
                power *= 2
        return result

    def norm_stats(self, rows_m):
        cc = self.cc
        nvalid = H * W
        sums = [self.sum_row_strided(r) for r in rows_m]
        total = sums[0]
        for s in sums[1:]:
            total = cc.EvalAdd(total, s)
        mean = cc.EvalMult(total, 1.0 / nvalid)
        sq = [cc.EvalMult(r, r) for r in rows_m]
        sums2 = [self.sum_row_strided(r) for r in sq]
        total2 = sums2[0]
        for s in sums2[1:]:
            total2 = cc.EvalAdd(total2, s)
        mean_sq = cc.EvalMult(total2, 1.0 / nvalid)
        var = cc.EvalSub(mean_sq, cc.EvalMult(mean, mean))
        return mean, var

    def norm_act(self, rows_m, mean, inv_std, gamma, beta, mask_after):
        cc = self.cc
        gamma_pt = self.pt(tile_full(gamma))
        beta_pt = self.pt(tile_full(beta))
        mask_pt = self.pt(valid_cols_mask()) if mask_after else None
        out = []
        for r in rows_m:
            centered = cc.EvalSub(r, mean)
            normalized = cc.EvalMult(centered, inv_std)
            scaled = cc.EvalAdd(cc.EvalMult(normalized, gamma_pt), beta_pt)
            act = poly_act_fhe(cc, scaled, A_, B_, C_)
            out.append(cc.EvalMult(act, mask_pt) if mask_after else act)
        return out

    def stage(self, rows_in, weight, bias, gamma, beta, cin, stage_id, coeffs, domain,
              ref_y=None, ref_var=None, mask_after=True):
        """Un intero stadio Conv->Norm->Att. Se ref_y/ref_var sono dati, stampa
        i checkpoint conv e statistiche contro numpy."""
        cc = self.cc
        t0 = time.time()
        rows_c = self.conv(rows_in, weight, bias, cin, COUT, stage_id)
        print(f"  [stadio {stage_id}] conv: {time.time()-t0:.1f}s, livello {rows_c[0].GetLevel()}", flush=True)
        if ref_y is not None:
            err = np.max(np.abs(self.dec_rows(rows_c, COUT) - ref_y))
            print(f"  [stadio {stage_id}] checkpoint conv (regione valida): errore {err:.3e}", flush=True)

        mask_pt = self.pt(valid_cols_mask())
        rows_m = [cc.EvalMult(r, mask_pt) for r in rows_c]
        mean, var = self.norm_stats(rows_m)
        print(f"  [stadio {stage_id}] statistiche: livello var = {var.GetLevel()}", flush=True)
        if ref_var is not None:
            p = cc.Decrypt(self.keys.secretKey, var)
            p.SetLength(N)
            err = np.max(np.abs(np.array(p.GetRealPackedValue())[:COUT] - ref_var))
            print(f"  [stadio {stage_id}] checkpoint varianza: errore {err:.3e}", flush=True)

        inv_std = isqrt_chebyshev_fhe(cc, var, coeffs, list(domain), post_iter=1)
        print(f"  [stadio {stage_id}] radice inversa: livello {inv_std.GetLevel()}", flush=True)
        out = self.norm_act(rows_m, mean, inv_std, gamma, beta, mask_after)
        print(f"  [stadio {stage_id}] uscita: livello {out[0].GetLevel()}", flush=True)
        return out


def main():
    print("=== ConvBlock COMPLETO a 2 stadi, HE vero, formato a righe (caso piccolo) ===\n")
    print(f"Cin={CIN}, Cout={COUT}, N={N}, immagine {H}x{W} (+alone) -> Hp={HP}, Wp={WP}")
    print(f"lunghezza reale riga = {REAL_LEN}, batch CKKS = {BATCH}, DEPTH = {DEPTH}\n")

    rng = np.random.default_rng(13)
    x = rng.normal(size=(CIN, H, W))
    p = dict(
        w1=rng.normal(size=(COUT, CIN, K, K)) * 0.15, b1=rng.normal(size=COUT) * 0.05,
        w2=rng.normal(size=(COUT, COUT, K, K)) * 0.15, b2=rng.normal(size=COUT) * 0.05,
        g1=rng.normal(size=COUT) * 0.2 + 1.0, be1=rng.normal(size=COUT) * 0.1,
        g2=rng.normal(size=COUT) * 0.2 + 1.0, be2=rng.normal(size=COUT) * 0.1,
    )

    print("Riferimenti numpy (esatto + approssimato) e domini Chebyshev per stadio...")
    ref = build_numpy_reference(x, p)
    print(f"  dominio stadio 1: [{ref['dom1'][0]:.3f}, {ref['dom1'][1]:.3f}]   "
          f"dominio stadio 2: [{ref['dom2'][0]:.3f}, {ref['dom2'][1]:.3f}]")
    print(f"  differenza numpy esatto vs approssimato in uscita: "
          f"{np.max(np.abs(ref['out_a'] - ref['out_e'])):.3e}  (costo atteso dell'approssimazione)\n")

    rots = required_rotations()
    print(f"Costruzione contesto ({len(rots)} chiavi di rotazione)...")
    t0 = time.time()
    cc, keys = build_context(rots)
    he = HE(cc, keys)
    print(f"Contesto pronto in {time.time()-t0:.1f}s.\n")

    rows_x = [cc.Encrypt(keys.publicKey, he.pt(pack_row(x[:, r, :]))) for r in range(H)]

    print("=== STADIO 1 ===")
    t0 = time.time()
    rows_1 = he.stage(rows_x, p['w1'], p['b1'], p['g1'], p['be1'], CIN, 1,
                      ref['coeffs1'], ref['dom1'], ref_y=ref['y1'], ref_var=ref['var1'])
    L1 = rows_1[0].GetLevel()
    err1 = np.max(np.abs(he.dec_rows(rows_1, COUT) - ref['y1n_a']))
    print(f"  [stadio 1] USCITA vs numpy APPROSSIMATO: errore {err1:.3e}   (tempo {time.time()-t0:.1f}s)")
    print(f"  Costo di un stadio: {L1} livelli su {DEPTH}.\n")

    need = 2 * L1 + 1
    if need <= DEPTH:
        print(f"Due stadi entrano senza bootstrap (stima {need} <= {DEPTH}). Procedo direttamente.\n")
        rows_in2 = rows_1
    else:
        print(f"ATTENZIONE: due stadi NON entrano senza bootstrap (stima {need} > {DEPTH}).")
        print("  -> serve un bootstrap tra i due stadi nella pipeline vera.")
        print("  -> qui uso un REFRESH SIMULATO (decifra+ricifra): NON e' un bootstrap,")
        print("     serve solo a verificare la logica dello stadio 2.\n")
        rows_in2 = he.refresh_simulated(rows_1)

    print("=== STADIO 2 ===")
    t0 = time.time()
    rows_2 = he.stage(rows_in2, p['w2'], p['b2'], p['g2'], p['be2'], COUT, 2,
                      ref['coeffs2'], ref['dom2'], mask_after=False)
    out_he = he.dec_rows(rows_2, COUT)
    print(f"  (tempo stadio 2: {time.time()-t0:.1f}s)\n")

    e_a = np.max(np.abs(out_he - ref['out_a']))
    e_e = np.max(np.abs(out_he - ref['out_e']))
    print("=== RISULTATO FINALE ===")
    print(f"Errore max vs riferimento APPROSSIMATO (misura la pipeline HE): {e_a:.3e}")
    print(f"Errore max vs riferimento ESATTO (include l'approssimazione):   {e_e:.3e}")
    if e_a < 1e-4:
        print("\n=== PIPELINE HE DEL CONVBLOCK CORRETTA (a meno del refresh simulato, se usato). ===")
    else:
        print("\n=== ATTENZIONE: errore grande anche vs l'approssimato: guarda i checkpoint sopra "
              "per capire in quale passaggio nasce. ===")


if __name__ == '__main__':
    main()