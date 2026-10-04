"""
crypto/prototype_lattice_resample.py

PROTOTIPO IN CHIARO (numpy) -- stride 2, upsampling (ConvTranspose2d k=2
s=2), skip connection e conv 1x1 nello schema a UN CIPHERTEXT PER CANALE,
SENZA spostare pixel: ogni canale resta in un vettore piatto di N slot
(come un ciphertext), con la griglia a risoluzione piena (righe di Wg slot).

IDEA ("reticolo"). Al livello l (passo s = 2^l) i pixel validi stanno alle
posizioni (s*r, s*c) della griglia piena:  slot = (s*r)*Wg + s*c.
  - convoluzione 3x3 al livello l: stessi offset di sempre ma moltiplicati
    per s:  d = ky*s*Wg + kx*s,  ky,kx in {-1,0,+1}  (rotazioni = np.roll).
  - stride 2: si fa la conv normale e poi si tiene solo il reticolo del
    livello l+1 (maschera). NESSUN dato si muove.
  - upsampling k=2 s=2: 4 pesi (a,b) -> il valore del pixel (r',c') va negli
    slot idx + s*a*Wg + s*b (3 rotazioni, la fase (0,0) non si sposta). Le
    4 fasi cadono su slot DISGIUNTI, quindi NON serve nessuna maschera.
  - skip connection: somma di due vettori allo stesso livello.
  - conv 1x1: prodotti per scalare e somme, nessuna rotazione.

CONVENZIONE: offset CENTRATI (-1,0,+1), cioe' quella di PyTorch con
padding=1. (I test HE di oggi usavano offset 0..2, "valida in alto a
sinistra": con pesi veri e' uno spostamento di un pixel per convoluzione
rispetto a PyTorch, da non lasciar accumulare su 18 convoluzioni. Qui la
questione non si pone.) Gli offset negativi richiedono chiavi di rotazione
negative, gia' usate nei test di oggi.

RIFERIMENTI: implementazioni numpy DENSE della semantica di PyTorch
(Conv2d k=3 pad=1 con stride 1 o 2; ConvTranspose2d k=2 s=2). Torch non e'
disponibile in questo ambiente: i riferimenti codificano la semantica che
conosco, vanno rivisti contro torch su Zeus/Mac con un confronto diretto.

Uso: python3 crypto/prototype_lattice_resample.py
"""

import numpy as np


# ============================================================
# Griglia, reticolo, rotazione
# ============================================================

class Grid:
    def __init__(self, H0, W0, n_levels, N=None):
        assert H0 % (1 << (n_levels - 1)) == 0 and W0 % (1 << (n_levels - 1)) == 0
        self.H0, self.W0, self.n_levels = H0, W0, n_levels
        self.Hg, self.Wg = H0 + 2, W0 + 2
        s_max = 1 << (n_levels - 1)
        # Requisito su N: la zona dati Hg*Wg deve starci. Il vicino "sopra" della
        # riga 0 (slot (i - s*Wg) mod N) cade nelle ultime righe, che sono SEMPRE
        # alone o non-reticolo: l'ultima riga valida al livello l e' H0 - s, e la
        # lettura atterra alla riga N/Wg - s >= Hg - s = H0 - s + 2. Quindi basta
        # N >= Hg*Wg: lo stesso requisito dei test HE di oggi (58.308 <= 65.536).
        self.min_N = self.Hg * self.Wg
        self.N = N if N is not None else 1 << int(np.ceil(np.log2(self.min_N)))

    def size(self, level):
        return self.H0 >> level, self.W0 >> level

    def idx(self, level, r, c):
        s = 1 << level
        return (s * r) * self.Wg + s * c

    def _rc(self, level):
        H, W = self.size(level)
        return np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    def valid_mask(self, level):
        rr, cc = self._rc(level)
        m = np.zeros(self.N)
        m[self.idx(level, rr, cc)] = 1.0
        return m

    def pack(self, img, level):
        rr, cc = self._rc(level)
        v = np.zeros(self.N)
        v[self.idx(level, rr, cc)] = img
        return v

    def unpack(self, vec, level):
        rr, cc = self._rc(level)
        return vec[self.idx(level, rr, cc)]


class Counter:
    def __init__(self):
        self.rot = 0
        self.offsets = set()
        self.terms = 0

    def reset(self):
        self.rot = 0; self.terms = 0


CNT = Counter()


def rot(v, d):
    """Equivalente a EvalRotate(ct, d): new[i] = old[(i+d) mod N]."""
    if d == 0:
        return v
    CNT.rot += 1
    CNT.offsets.add(d)
    return np.roll(v, -d)


# ============================================================
# Operazioni sul reticolo
# ============================================================

def conv3x3(g, chans, w, b, level, stride2):
    """chans: Cin vettori piatti al livello `level` (zero fuori dal reticolo
    valido). w: (Cout,Cin,3,3). Restituisce Cout vettori al livello
    level+1 se stride2 altrimenti `level`. Rotazioni 'hoisted': Cin*8."""
    s = 1 << level
    offs = [(ky - 1) * s * g.Wg + (kx - 1) * s for ky in range(3) for kx in range(3)]
    rotated = [[rot(x, d) for d in offs] for x in chans]
    out_level = level + (1 if stride2 else 0)
    mask = g.valid_mask(out_level)
    outs = []
    for co in range(w.shape[0]):
        acc = np.zeros(g.N)
        for ci in range(len(chans)):
            for k in range(9):
                acc = acc + w[co, ci, k // 3, k % 3] * rotated[ci][k]
                CNT.terms += 1
        outs.append((acc + b[co]) * mask)       # il bias solo sui pixel validi, poi maschera
    return outs


def upconv2x2(g, chans, w, b, level_in, check_disjoint=False):
    """ConvTranspose2d(k=2, s=2). w: (Cin,Cout,2,2) come PyTorch. Da level_in
    a level_in-1. Rotazioni 'hoisted': 3 per canale d'ingresso. Nessuna
    maschera: le 4 fasi cadono su slot disgiunti dentro il reticolo valido."""
    lo = level_in - 1
    s = 1 << lo
    shifts = {(a, c): -(s * a * g.Wg + s * c) for a in (0, 1) for c in (0, 1)}
    rotated = [{ab: rot(x, d) for ab, d in shifts.items()} for x in chans]
    maskout = g.valid_mask(lo)
    outs = []
    for co in range(w.shape[1]):
        acc = np.zeros(g.N)
        for ci in range(len(chans)):
            for (a, c) in shifts:
                acc = acc + w[ci, co, a, c] * rotated[ci][(a, c)]
                CNT.terms += 1
        if check_disjoint:
            off_lattice = np.abs(acc * (1.0 - maskout)).max()
            assert off_lattice == 0.0, f"upconv: valori fuori dal reticolo ({off_lattice})"
        outs.append(acc + b[co] * maskout)       # bias come plaintext sui soli pixel validi
    return outs


def conv1x1(g, chans, w, b, level):
    mask = g.valid_mask(level)
    outs = []
    for k in range(w.shape[0]):
        acc = np.zeros(g.N)
        for ci in range(len(chans)):
            acc = acc + w[k, ci] * chans[ci]
            CNT.terms += 1
        outs.append((acc + b[k]) * mask)
    return outs


def polyact_masked(g, chans, level, a=0.1, b=1.0, c=0.5, apply_mask=True):
    """PolyAct + maschera: act(0)=c != 0, quindi la maschera e' NECESSARIA per
    riportare a zero tutto cio' che non e' pixel valido (e' la stessa maschera
    che si applica dopo ogni stadio nella pipeline). apply_mask=False serve
    solo al controllo negativo."""
    mask = g.valid_mask(level)
    return [(a * x * x + b * x + c) * (mask if apply_mask else 1.0) for x in chans]


def skip_add(chans_a, chans_b):
    return [x + y for x, y in zip(chans_a, chans_b)]


def lattice_norm_stats(g, chans, level):
    """Media/varianza per canale: somma sul vettore mascherato / n_valid."""
    H, W = g.size(level)
    n = H * W
    mask = g.valid_mask(level)
    means, vars_ = [], []
    for x in chans:
        xm = x * mask
        mu = xm.sum() / n
        means.append(mu)
        vars_.append((xm * xm).sum() / n - mu * mu)
    return np.array(means), np.array(vars_)


# ============================================================
# Riferimenti DENSI (semantica PyTorch)
# ============================================================

def conv_ref(x, w, b, stride=1):
    """x (Cin,H,W); Conv2d k=3, padding=1, stride 1 o 2."""
    cin, H, W = x.shape
    cout = w.shape[0]
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    Ho, Wo = H // stride, W // stride
    out = np.zeros((cout, Ho, Wo))
    for co in range(cout):
        for ci in range(cin):
            for ky in range(3):
                for kx in range(3):
                    out[co] += w[co, ci, ky, kx] * xp[ci, ky:ky + stride * Ho:stride, kx:kx + stride * Wo:stride]
        out[co] += b[co]
    return out


def upconv_ref(x, w, b):
    """x (Cin,H,W); w (Cin,Cout,2,2); ConvTranspose2d k=2, s=2."""
    cin, H, W = x.shape
    cout = w.shape[1]
    out = np.zeros((cout, 2 * H, 2 * W))
    for co in range(cout):
        for a in range(2):
            for c in range(2):
                out[co, a::2, c::2] = sum(w[ci, co, a, c] * x[ci] for ci in range(cin))
        out[co] += b[co]
    return out


def act_ref(x, a=0.1, b=1.0, c=0.5):
    return a * x * x + b * x + c


# ============================================================
# Test
# ============================================================

def rand_chans(rng, g, C, level):
    H, W = g.size(level)
    imgs = rng.normal(size=(C, H, W))
    return imgs, [g.pack(imgs[c], level) for c in range(C)]


def unpack_all(g, chans, level):
    return np.stack([g.unpack(v, level) for v in chans])


def main():
    rng = np.random.default_rng(21)
    H0, W0, L = 32, 24, 3
    g = Grid(H0, W0, L)
    print(f"Griglia: immagine {H0}x{W0}, righe di Wg={g.Wg} slot, {L} livelli (passi 1,2,4), "
          f"N={g.N} slot (minimo richiesto {g.min_N} = Hg*Wg)\n")
    all_ok = True

    def report(name, err, extra=""):
        nonlocal all_ok
        ok = err < 1e-10
        all_ok &= ok
        print(f"  {name:<58} errore {err:9.2e}  {'OK' if ok else 'ERRORE'}  {extra}")

    print("=== Test per singola operazione ===")
    for level in range(L):
        Cin, Cout = 3, 5
        imgs, chans = rand_chans(rng, g, Cin, level)
        w = rng.normal(size=(Cout, Cin, 3, 3)) * 0.2
        b = rng.normal(size=Cout) * 0.1
        CNT.reset()
        out = conv3x3(g, chans, w, b, level, stride2=False)
        err = np.abs(unpack_all(g, out, level) - conv_ref(imgs, w, b, 1)).max()
        report(f"conv 3x3 stride 1, livello {level} (passo {1 << level})", err, f"[{CNT.rot} rot, {CNT.terms} termini]")

    for level in range(L - 1):
        Cin, Cout = 3, 5
        imgs, chans = rand_chans(rng, g, Cin, level)
        w = rng.normal(size=(Cout, Cin, 3, 3)) * 0.2
        b = rng.normal(size=Cout) * 0.1
        CNT.reset()
        out = conv3x3(g, chans, w, b, level, stride2=True)
        err = np.abs(unpack_all(g, out, level + 1) - conv_ref(imgs, w, b, 2)).max()
        report(f"conv 3x3 STRIDE 2, livello {level} -> {level + 1}", err, f"[{CNT.rot} rot, {CNT.terms} termini]")

    for level_in in range(1, L):
        Cin, Cout = 4, 3
        imgs, chans = rand_chans(rng, g, Cin, level_in)
        w = rng.normal(size=(Cin, Cout, 2, 2)) * 0.3
        b = rng.normal(size=Cout) * 0.1
        CNT.reset()
        out = upconv2x2(g, chans, w, b, level_in, check_disjoint=True)
        err = np.abs(unpack_all(g, out, level_in - 1) - upconv_ref(imgs, w, b)).max()
        report(f"ConvTranspose k2 s2, livello {level_in} -> {level_in - 1} (senza maschere)", err,
               f"[{CNT.rot} rot, {CNT.terms} termini]")

    imgs, chans = rand_chans(rng, g, 6, 0)
    w1 = rng.normal(size=(2, 6)) * 0.3
    b1 = rng.normal(size=2) * 0.1
    out = conv1x1(g, chans, w1, b1, 0)
    ref = np.einsum('kc,chw->khw', w1, imgs) + b1[:, None, None]
    report("conv 1x1 (6 -> 2 canali), 0 rotazioni", np.abs(unpack_all(g, out, 0) - ref).max())

    imgs, chans = rand_chans(rng, g, 4, 2)
    mu, var = lattice_norm_stats(g, chans, 2)
    report("statistiche di norm al livello 2 (media)", np.abs(mu - imgs.reshape(4, -1).mean(1)).max())
    report("statistiche di norm al livello 2 (varianza)", np.abs(var - imgs.reshape(4, -1).var(1)).max())

    print("\n=== Mini U-Net a 3 livelli, tutto sul reticolo (conv-act ... stride ... up + skip ... 1x1) ===")
    C0, C1, C2, NCLS = 3, 4, 5, 2
    x_img = rng.normal(size=(1, H0, W0))
    Wt = {
        'e0a': rng.normal(size=(C0, 1, 3, 3)) * 0.3, 'e0b': rng.normal(size=(C0, C0, 3, 3)) * 0.2,
        'e1a': rng.normal(size=(C1, C0, 3, 3)) * 0.2, 'e1b': rng.normal(size=(C1, C1, 3, 3)) * 0.2,
        'e2a': rng.normal(size=(C2, C1, 3, 3)) * 0.2, 'e2b': rng.normal(size=(C2, C2, 3, 3)) * 0.2,
        'u1': rng.normal(size=(C2, C1, 2, 2)) * 0.3, 'd1a': rng.normal(size=(C1, C1, 3, 3)) * 0.2,
        'u0': rng.normal(size=(C1, C0, 2, 2)) * 0.3, 'd0a': rng.normal(size=(C0, C0, 3, 3)) * 0.2,
        'out': rng.normal(size=(NCLS, C0)) * 0.3,
    }
    Bt = {k: rng.normal(size=v.shape[0] if k != 'u1' and k != 'u0' else v.shape[1]) * 0.05 for k, v in Wt.items()}

    # --- riferimento denso ---
    r = act_ref(conv_ref(x_img, Wt['e0a'], Bt['e0a']))
    r = act_ref(conv_ref(r, Wt['e0b'], Bt['e0b'])); skip0_r = r
    r = act_ref(conv_ref(r, Wt['e1a'], Bt['e1a'], stride=2))
    r = act_ref(conv_ref(r, Wt['e1b'], Bt['e1b'])); skip1_r = r
    r = act_ref(conv_ref(r, Wt['e2a'], Bt['e2a'], stride=2))
    r = act_ref(conv_ref(r, Wt['e2b'], Bt['e2b']))
    r = upconv_ref(r, Wt['u1'], Bt['u1']) + skip1_r
    r = act_ref(conv_ref(r, Wt['d1a'], Bt['d1a']))
    r = upconv_ref(r, Wt['u0'], Bt['u0']) + skip0_r
    r = act_ref(conv_ref(r, Wt['d0a'], Bt['d0a']))
    ref_logits = np.einsum('kc,chw->khw', Wt['out'], r) + Bt['out'][:, None, None]

    # --- reticolo ---
    def lattice_unet(apply_mask):
        P = lambda c, lvl: polyact_masked(g, c, lvl, apply_mask=apply_mask)
        x = [g.pack(x_img[0], 0)]
        x = P(conv3x3(g, x, Wt['e0a'], Bt['e0a'], 0, False), 0)
        x = P(conv3x3(g, x, Wt['e0b'], Bt['e0b'], 0, False), 0); skip0 = x
        x = P(conv3x3(g, x, Wt['e1a'], Bt['e1a'], 0, True), 1)
        x = P(conv3x3(g, x, Wt['e1b'], Bt['e1b'], 1, False), 1); skip1 = x
        x = P(conv3x3(g, x, Wt['e2a'], Bt['e2a'], 1, True), 2)
        x = P(conv3x3(g, x, Wt['e2b'], Bt['e2b'], 2, False), 2)
        x = skip_add(upconv2x2(g, x, Wt['u1'], Bt['u1'], 2, check_disjoint=apply_mask), skip1)
        x = P(conv3x3(g, x, Wt['d1a'], Bt['d1a'], 1, False), 1)
        x = skip_add(upconv2x2(g, x, Wt['u0'], Bt['u0'], 1, check_disjoint=apply_mask), skip0)
        x = P(conv3x3(g, x, Wt['d0a'], Bt['d0a'], 0, False), 0)
        return unpack_all(g, conv1x1(g, x, Wt['out'], Bt['out'], 0), 0)

    CNT.reset(); CNT.offsets.clear()
    logits = lattice_unet(apply_mask=True)
    err = np.abs(logits - ref_logits).max()
    report("mini U-Net completa (logits finali vs riferimento denso)", err,
           f"[{CNT.rot} rot, {CNT.terms} termini, {len(CNT.offsets)} offset di rotazione distinti]")
    used_offsets = set(CNT.offsets)

    # chiavi: 8 offset per livello (conv); gli offset dell'upsampling sono un sottoinsieme
    expected = set()
    for lvl in range(L):
        s = 1 << lvl
        expected |= {(ky - 1) * s * g.Wg + (kx - 1) * s for ky in range(3) for kx in range(3)} - {0}
    print(f"\n  Chiavi di rotazione: usate {len(used_offsets)}, attese 8 per livello x {L} livelli = {len(expected)}; "
          f"usate contenute nelle attese: {used_offsets <= expected}")
    all_ok &= used_offsets <= expected

    print("\n=== Controllo negativo: stessa rete SENZA la maschera dopo la PolyAct ===")
    err_bad = np.abs(lattice_unet(apply_mask=False) - ref_logits).max()
    print(f"  errore {err_bad:.2e}  -> "
          f"{'il confronto SA accorgersi di un errore; la maschera dopo ogni stadio e\' NECESSARIA' if err_bad > 1e-3 else 'INCONCLUDENTE: il confronto non rileva la differenza (da indagare)'}")
    all_ok &= err_bad > 1e-3

    print("\n" + ("=== TUTTO COINCIDE con la semantica di PyTorch (riferimenti numpy). ===" if all_ok
                  else "=== ATTENZIONE: qualche operazione NON coincide. ==="))

    print(f"\nRequisito su N alla scala vera: Hg*Wg = 258*226 = {258 * 226} <= 65536 -> "
          f"{'OK' if 258 * 226 <= 65536 else 'NON ENTRA'} (identico ai test di oggi, per ogni numero di livelli)")


if __name__ == '__main__':
    main()