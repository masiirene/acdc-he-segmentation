"""
crypto/dimensioning.py

Calcolo dei requisiti di ciphertext per la HEFriendlyUNet con lo schema di
packing a griglia fissa (crypto/packing.py). Parametri CKKS: N=2^16 di
default (32.768 slot/ciphertext) -- valore di partenza ragionevole, da
confermare con Aurora quando si sceglieranno i parametri HE definitivi
(punto D del suo schema: profondita' moltiplicativa, bootstrap placement).

--------------------------------------------------------------------------
Due correzioni rispetto alla prima versione:
--------------------------------------------------------------------------
1. GRIGLIA OTTIMALE, non fissata a mano. La griglia deve dividere
   ESATTAMENTE la risoluzione ad ogni stage della rete, inclusa quella
   piu' piccola (dopo 5 stride-2: H/32 x W/32). Con H=256, W=224, la
   risoluzione minima e' 8x7 -- e 7 e' primo, quindi sulla larghezza la
   griglia puo' essere solo 1 o 7 tile. Cerchiamo automaticamente la
   griglia con MENO tile possibile che (a) divide esattamente H_min e
   W_min, (b) fa entrare tile+halo nel budget di slot a risoluzione
   PIENA (il caso peggiore, dove i tile sono piu' grandi).

2. DUE METRICHE SEPARATE, non una sola "totale":
   - `total_tile_ops`: somma di (canali x tile) su tutti i layer.
     Proxy per il NUMERO DI OPERAZIONI (quindi la LATENZA): ogni tile di
     ogni canale richiede una convoluzione indipendente.
   - `peak_concurrent_ciphertexts`: quanti ciphertext devono essere vivi
     CONTEMPORANEAMENTE nel punto piu' critico dell'esecuzione. Questo,
     non il totale, e' il proxy corretto per la RAM: gran parte dei
     layer intermedi puo' essere liberata non appena non serve piu',
     TRANNE le feature map delle skip connection, che devono restare in
     memoria dall'encoder fino alla concatenazione nel decoder.

Nota sulla stima in GB: NON la includiamo. La dimensione reale di un
ciphertext CKKS dipende dalla profondita' moltiplicativa scelta (numero
di "limb" RNS) -- che dipende a sua volta da quanti layer si susseguono
tra un bootstrap e l'altro. Con tanti PolyAct/Conv2d in cascata, un
ciphertext potrebbe pesare molto piu' di una stima "a spanne". Aspettiamo
i parametri reali da Aurora (punto D del suo schema) prima di quotare
numeri di RAM in GB.
"""

import math


# ---------------------------------------------------------------------------
# Ricerca della griglia ottimale
# ---------------------------------------------------------------------------

def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def find_optimal_grid(H, W, n_stride2_stages, slots_budget, halo=1):
    """
    Cerca la griglia (n_tiles_h, n_tiles_w) con MENO tile totali che:
      (a) divide esattamente sia (H, W) sia la risoluzione minima della
          rete (H / 2^n_stride2_stages, W / 2^n_stride2_stages) -- per
          avere tile di dimensione intera ad ogni stage;
      (b) fa entrare tile+halo nel budget di slot A RISOLUZIONE PIENA
          (il caso peggiore: i tile sono piu' grandi li').

    Ritorna (n_tiles_h, n_tiles_w) oppure solleva ValueError se nessuna
    griglia valida esiste con questo budget.
    """
    H_min = H // (2 ** n_stride2_stages)
    W_min = W // (2 ** n_stride2_stages)

    candidates = []
    for nh in divisors(H_min):
        for nw in divisors(W_min):
            tile_h, tile_w = H // nh, W // nw
            slots = (tile_h + 2 * halo) * (tile_w + 2 * halo)
            if slots <= slots_budget:
                candidates.append((nh * nw, nh, nw, slots))

    if not candidates:
        raise ValueError(
            f"Nessuna griglia valida trova posto nel budget ({slots_budget} slot) "
            f"rispettando la divisibilita' fino a {H_min}x{W_min}. "
            f"Serve un N piu' grande, o un halo/kernel piu' piccolo."
        )

    candidates.sort(key=lambda c: c[0])  # meno tile possibile
    n_tiles, nh, nw, slots = candidates[0]
    utilization = 100 * slots / slots_budget
    print(f"Griglia ottimale trovata: {nh}x{nw} ({n_tiles} tile), "
          f"{slots:,} slot/tile a ris. piena ({utilization:.1f}% del budget)")
    return nh, nw


# ---------------------------------------------------------------------------
# Dimensionamento per layer
# ---------------------------------------------------------------------------

class HEDimensioner:
    def __init__(self, N=2 ** 16, n_tiles_h=1, n_tiles_w=1):
        self.slots_per_ct = N // 2
        self.n_tiles_h = n_tiles_h
        self.n_tiles_w = n_tiles_w
        self.total_tiles = n_tiles_h * n_tiles_w
        self.total_tile_ops = 0
        self._layer_ct = {}  # nome layer -> numero di ciphertext (per il calcolo del picco)

    def check_tile_capacity(self, h, w, halo=1, name=""):
        tile_h = h // self.n_tiles_h
        tile_w = w // self.n_tiles_w
        slots_used = (tile_h + 2 * halo) * (tile_w + 2 * halo)
        if slots_used > self.slots_per_ct:
            raise ValueError(f"[{name}] Overflow! Tile richiede {slots_used} slot, max {self.slots_per_ct}")
        utilization = (slots_used / self.slots_per_ct) * 100
        return slots_used, utilization

    def dimension_layer(self, name, channels, h, w, halo=1, verbose=True):
        slots_used, util = self.check_tile_capacity(h, w, halo, name)
        ct_layer = self.total_tiles * channels
        self.total_tile_ops += ct_layer
        self._layer_ct[name] = ct_layer
        if verbose:
            print(f"{name:10s} | {channels:3d} canali | Ris: {h:3d}x{w:3d} | "
                  f"tile-conv: {ct_layer:4d} | slot/tile: {slots_used:5d} ({util:4.1f}%)")
        return ct_layer

    def ct(self, name):
        return self._layer_ct[name]


# ---------------------------------------------------------------------------
# Dimensionamento della rete reale (filters = [32,64,128,256,512,512])
# ---------------------------------------------------------------------------

def run_acdc_dimensioning():
    H, W = 256, 224
    n_stride2 = 5  # enc1..enc5
    slots_budget = 2 ** 16 // 2

    nh, nw = find_optimal_grid(H, W, n_stride2, slots_budget, halo=1)
    print()

    dim = HEDimensioner(N=2 ** 16, n_tiles_h=nh, n_tiles_w=nw)
    filters = [32, 64, 128, 256, 512, 512]  # come in models/he_friendly.py

    print("=== ENCODER ===")
    res = [(H, W)]
    for i in range(1, 6):
        prev_h, prev_w = res[-1]
        res.append((prev_h // 2, prev_w // 2))

    for i, (c, (h, w)) in enumerate(zip(filters, res)):
        dim.dimension_layer(f"enc{i}", c, h, w)

    print("\n=== DECODER (canali post-concat coerenti col forward() del modello) ===")
    # dec4: concat(up(enc5), enc4) -> Conv -> filters[4]=512 canali in output
    # dec3: concat(up(dec4), enc3) -> filters[3]=256
    # dec2: concat(up(dec3), enc2) -> filters[2]=128
    # dec1: concat(up(dec2), enc1) -> filters[1]=64
    # dec0: concat(up(dec1), enc0) -> filters[0]=32
    dec_out_channels = [512, 256, 128, 64, 32]
    dec_res = res[4::-1]  # risoluzioni enc4..enc0, nell'ordine giusto per il decoder
    for j, (c, (h, w)) in enumerate(zip(dec_out_channels, dec_res)):
        dim.dimension_layer(f"dec{4-j}", c, h, w)

    dim.dimension_layer("out_1x1", 4, H, W, halo=0)  # conv 1x1, nessun halo necessario

    print("\n" + "=" * 78)
    print(f"OPERAZIONI TOTALI (proxy latenza -- somma di canali x tile su tutti i layer): "
          f"{dim.total_tile_ops:,}")

    # -----------------------------------------------------------------
    # Picco di memoria concorrente: simuliamo la timeline di esecuzione,
    # tenendo vive le skip connection (enc0..enc4) finche' non vengono
    # consumate dal decoder corrispondente.
    # -----------------------------------------------------------------
    print("\n=== PICCO DI MEMORIA CONCORRENTE (proxy RAM, con skip connection) ===")
    timeline = [
        ("dopo enc0", {"enc0"}),
        ("dopo enc1", {"enc0", "enc1"}),
        ("dopo enc2", {"enc0", "enc1", "enc2"}),
        ("dopo enc3", {"enc0", "enc1", "enc2", "enc3"}),
        ("dopo enc4", {"enc0", "enc1", "enc2", "enc3", "enc4"}),
        ("dopo enc5 (bottleneck)", {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5"}),
        ("dopo dec4 (enc4,enc5 liberati)", {"enc0", "enc1", "enc2", "enc3", "dec4"}),
        ("dopo dec3 (enc3 liberato)", {"enc0", "enc1", "enc2", "dec3"}),
        ("dopo dec2 (enc2 liberato)", {"enc0", "enc1", "dec2"}),
        ("dopo dec1 (enc1 liberato)", {"enc0", "dec1"}),
        ("dopo dec0 (enc0 liberato)", {"dec0"}),
        ("dopo out_1x1", {"out_1x1"}),
    ]

    peak = 0
    peak_label = ""
    for label, alive_set in timeline:
        total = sum(dim.ct(name) for name in alive_set)
        marker = ""
        if total > peak:
            peak = total
            peak_label = label
            marker = "  <-- picco finora"
        print(f"  {label:35s}: {total:5,} ciphertext vivi{marker}")

    print(f"\nPICCO DI MEMORIA: {peak:,} ciphertext ({peak_label})")
    print(f"Rapporto picco/totale: {100*peak/dim.total_tile_ops:.1f}% "
          f"(la RAM reale e' molto minore del conteggio 'totale')")
    print("\nNOTA: nessuna stima in GB -- serve prima la dimensione reale di un")
    print("ciphertext CKKS con i parametri di profondita' scelti (da discutere con Aurora).")


if __name__ == "__main__":
    print("=== Dimensionamento Ciphertext HEFriendlyUNet ===\n")
    run_acdc_dimensioning()