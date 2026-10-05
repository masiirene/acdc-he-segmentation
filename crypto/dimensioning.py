"""
crypto/dimensioning.py

[... docstring originale invariata fino a qui ...]

AGGIORNAMENTO: supporto per skip_mode ('concat' vs 'sum').

La metrica principale (total_tile_ops, peak_ciphertexts) conta le
operazioni SOLO in base ai canali di USCITA di ogni stage (Cout), non a
quelli di INGRESSO (Cin) -- una semplificazione gia' presente nella
versione precedente di questo script (quella usata per i numeri gia'
riportati ad Aurora: 5.000/3.464/2.568). Con skip_mode diverso ma stessi
filters, questa metrica principale NON CAMBIA: sia concat sia sum
producono lo stesso numero di canali di uscita per ogni stage decoder.

Questo pero' nasconde una differenza reale: con concat, il primo strato
convoluzionale di ogni blocco decoder riceve in INGRESSO il doppio dei
canali (skip + upsampling AFFIANCATI) rispetto a sum (skip + upsampling
gia' SOMMATI, stesso numero di canali). Per rendere conto di questo
vantaggio reale di sum, senza alterare la metrica principale gia'
riportata, aggiungiamo una riga SEPARATA: 'sovraccarico da concatenazione'
-- il costo extra, in termini della stessa unita' di misura (tile-conv),
dovuto ai canali raddoppiati in ingresso al primo conv di ogni stage
decoder, presente SOLO con skip_mode='concat', assente con 'sum'.
"""

import math


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def find_optimal_grid(H, W, n_stride2_stages, slots_budget, halo=1, verbose=True):
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
            f"rispettando la divisibilita' fino a {H_min}x{W_min}."
        )

    candidates.sort(key=lambda c: c[0])
    n_tiles, nh, nw, slots = candidates[0]
    utilization = 100 * slots / slots_budget
    if verbose:
        print(f"Griglia ottimale trovata: {nh}x{nw} ({n_tiles} tile), "
              f"{slots:,} slot/tile a ris. piena ({utilization:.1f}% del budget)")
    return nh, nw


class HEDimensioner:
    def __init__(self, N=2 ** 16, n_tiles_h=1, n_tiles_w=1):
        self.slots_per_ct = N // 2
        self.n_tiles_h = n_tiles_h
        self.n_tiles_w = n_tiles_w
        self.total_tiles = n_tiles_h * n_tiles_w
        self.total_tile_ops = 0
        self.concat_overhead_ops = 0  # NUOVO: accumulatore separato
        self._layer_ct = {}

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

    def add_concat_overhead(self, name, extra_channels, verbose=True):
        """
        Sovraccarico da concatenazione: canali EXTRA in ingresso al primo
        conv di uno stage decoder, presenti SOLO con skip_mode='concat'.
        extra_channels = filters[i] (la meta' aggiuntiva che sum non ha,
        perche' sum somma invece di affiancare -- lo skip e l'upsampling
        hanno GIA' lo stesso numero di canali dell'output desiderato).
        """
        overhead = self.total_tiles * extra_channels
        self.concat_overhead_ops += overhead
        if verbose:
            print(f"{name:10s} | +{extra_channels:3d} canali extra (da concat) | "
                  f"sovraccarico: {overhead:4d} tile-conv")
        return overhead

    def ct(self, name):
        return self._layer_ct[name]


def run_dimensioning(filters, nh, nw, skip_mode='concat', label="", N=2 ** 16, H=256, W=224, verbose=True):
    """
    Dimensiona la rete per una data lista 'filters' [enc0..enc5] e uno
    skip_mode ('concat' o 'sum'). Vedi nota nel docstring del modulo sul
    perche' total_tile_ops NON distingue concat da sum (dipende solo da
    Cout), mentre concat_overhead_ops si', ed e' riportato separatamente.
    """
    if verbose:
        print(f"\n{'='*78}")
        print(f"=== Configurazione: {label} -- filters={filters}, skip_mode={skip_mode} ===")
        print(f"{'='*78}")

    n_stride2 = 5
    dim = HEDimensioner(N=N, n_tiles_h=nh, n_tiles_w=nw)

    if verbose:
        print("\n--- ENCODER ---")
    res = [(H, W)]
    for i in range(1, 6):
        prev_h, prev_w = res[-1]
        res.append((prev_h // 2, prev_w // 2))

    for i, (c, (h, w)) in enumerate(zip(filters, res)):
        dim.dimension_layer(f"enc{i}", c, h, w, verbose=verbose)

    if verbose:
        print("\n--- DECODER ---")
    dec_out_channels = list(reversed(filters[:5]))
    dec_res = res[4::-1]
    for j, (c, (h, w)) in enumerate(zip(dec_out_channels, dec_res)):
        dim.dimension_layer(f"dec{4-j}", c, h, w, verbose=verbose)
        if skip_mode == 'concat':
            # Sovraccarico: con concat, il primo conv di questo stage
            # riceve filters[i] canali extra (lo skip, affiancato invece
            # che sommato all'upsampling) -- assente con sum.
            dim.add_concat_overhead(f"dec{4-j}_concat_overhead", c, verbose=verbose)

    dim.dimension_layer("out_1x1", 4, H, W, halo=0, verbose=verbose)

    if verbose:
        print(f"\nOPERAZIONI TOTALI (metrica principale, Cout-only, comparabile ai numeri "
              f"gia' riportati): {dim.total_tile_ops:,}")
        if skip_mode == 'concat':
            print(f"SOVRACCARICO DA CONCATENAZIONE (canali extra in ingresso, riga separata): "
                  f"{dim.concat_overhead_ops:,}")
            print(f"OPERAZIONI TOTALI + sovraccarico concat: "
                  f"{dim.total_tile_ops + dim.concat_overhead_ops:,}")
        print("\n--- PICCO DI MEMORIA CONCORRENTE ---")

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
    for line_label, alive_set in timeline:
        total = sum(dim.ct(name) for name in alive_set)
        marker = ""
        if total > peak:
            peak = total
            peak_label = line_label
            marker = "  <-- picco finora"
        if verbose:
            print(f"  {line_label:35s}: {total:5,} ciphertext vivi{marker}")

    if verbose:
        print(f"\nPICCO DI MEMORIA: {peak:,} ciphertext ({peak_label})")

    return {
        "label": label,
        "skip_mode": skip_mode,
        "filters": filters,
        "total_tile_ops": dim.total_tile_ops,
        "concat_overhead_ops": dim.concat_overhead_ops,
        "total_with_overhead": dim.total_tile_ops + dim.concat_overhead_ops,
        "peak_ciphertexts": peak,
        "peak_label": peak_label,
    }


def run_comparison():
    H, W = 256, 224
    n_stride2 = 5
    slots_budget = 2 ** 16 // 2

    print("=== Dimensionamento Ciphertext HEFriendlyUNet -- Confronto concat vs sum ===\n")
    nh, nw = find_optimal_grid(H, W, n_stride2, slots_budget, halo=1)

    configs = [
        ([32, 64, 128, 256, 512, 512], 'concat', "Piena larghezza, concat (WS, Dice 0.851)"),
        ([32, 64, 128, 256, 512, 512], 'sum',    "Piena larghezza, sum (Dice 0.854)"),
        ([32, 64, 128, 256, 256, 256], 'concat', "Moderato, concat (WS, Dice 0.846)"),
        ([32, 64, 128, 256, 256, 256], 'sum',    "Moderato, sum (Dice 0.852)"),
        ([32, 64, 128, 256, 128, 64],  'concat', "Aggressivo, concat (WS, Dice 0.855)"),
        ([32, 64, 128, 256, 128, 64],  'sum',    "Aggressivo, sum (Dice 0.863)"),
    ]

    results = []
    for filters, skip_mode, label in configs:
        r = run_dimensioning(filters, nh, nw, skip_mode=skip_mode, label=label, H=H, W=W, verbose=True)
        results.append(r)

    baseline = results[0]  # piena larghezza, concat -- stesso riferimento gia' usato con Aurora

    print(f"\n{'='*115}")
    print("=== TABELLA RIASSUNTIVA ===")
    print(f"{'='*115}")
    header = (f"{'Configurazione':42s} {'Op. (Cout-only)':>16s} {'+ overhead concat':>18s} "
              f"{'Riduzione tot.':>15s} {'Picco ct':>10s} {'Riduzione':>10s}")
    print(header)
    print("-" * len(header))
    for r in results:
        total_reduction = 100 * (1 - r["total_with_overhead"] / baseline["total_with_overhead"])
        peak_reduction = 100 * (1 - r["peak_ciphertexts"] / baseline["peak_ciphertexts"])
        print(f"{r['label']:42s} {r['total_tile_ops']:16,} {r['total_with_overhead']:18,} "
              f"{total_reduction:14.1f}% {r['peak_ciphertexts']:10,} {peak_reduction:9.1f}%")

    print("\nNOTA: 'Op. (Cout-only)' e' la metrica gia' riportata (identica tra concat/sum a")
    print("parita' di filters, per costruzione). '+ overhead concat' aggiunge il costo reale")
    print("dei canali raddoppiati in ingresso al primo conv decoder -- zero per sum, quindi")
    print("la colonna 'Riduzione tot.' e' il confronto onesto che mostra il vantaggio di sum.")

    return results


if __name__ == "__main__":
    run_comparison()