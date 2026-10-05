"""
crypto/bootstrap_placement.py (v2 -- corretta)

Fix rispetto alla v1:
1. Norm depths per la config a 16 IN (6 rimosse) HARDCODED qui sotto,
   presi dall'ultima esecuzione CORRETTA e verificata di
   estimate_total_depth_no6norm.py (quella che dava Norm=221, TOTALE=271)
   -- non piu' letti da crypto/chebyshev_calibration_no6norm.json, che e'
   stato sovrascritto nel frattempo dall'esperimento sui 7 layer rimossi.
2. Il gap di allineamento delle skip connection e' ora calcolato
   correttamente come "quanto il ramo principale ha consumato dall'ULTIMO
   bootstrap", non dall'inizio assoluto della rete -- dato che il ramo
   principale viene comunque bootstrappato periodicamente, il gap e'
   per costruzione limitato da K, non dalla profondita' totale.
"""

import argparse

# Valori CONFERMATI (somma = 221, verificata) per la config a 16 IN, 6
# stage, 6 layer rimossi -- gia' include il +2 (varsq+apply) per ognuno.
NORM_DEPTH_16IN = {
    'enc0.block.1': 29, 'enc0.block.4': 12,   # enc0.block.1: newton fallback
    'enc1.block.1': 13, 'enc1.block.4': 11,
    'enc2.block.1': 12, 'enc2.block.4': 11,
    'enc3.block.1': 0,  'enc3.block.4': 10,   # enc3.block.1: bypass
    'enc4.block.1': 0,  'enc4.block.4': 0,    # entrambi bypass
    'enc5.block.1': 0,  'enc5.block.4': 0,    # entrambi bypass
    'dec4.block.1': 12, 'dec4.block.4': 11,
    'dec3.block.1': 10, 'dec3.block.4': 11,
    'dec2.block.1': 10, 'dec2.block.4': 12,
    'dec1.block.1': 11, 'dec1.block.4': 14,
    'dec0.block.1': 0,  'dec0.block.4': 32,   # dec0.block.1: bypass; dec0.block.4: newton fallback
}
# +2 gia' incluso per i valori Chebyshev; per enc0.block.1 e dec0.block.4
# (newton) il +2 e' gia' incluso nei numeri 29 e 32 sopra.

BLOCK_ORDER = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
               'dec4', 'dec3', 'dec2', 'dec1', 'dec0']
SKIP_PAIRS = [('enc4', 'dec4'), ('enc3', 'dec3'), ('enc2', 'dec2'),
              ('enc1', 'dec1'), ('enc0', 'dec0')]


def build_step_sequence():
    steps = []
    enc_output_marker = {}
    for name in BLOCK_ORDER:
        if name.startswith('dec'):
            steps.append((f'up_{name}', 1))
            steps.append((f'{name}_skip_add', 0))
        n1 = NORM_DEPTH_16IN[f'{name}.block.1']
        n2 = NORM_DEPTH_16IN[f'{name}.block.4']
        steps.append((f'{name}_conv1', 1))
        if n1 > 0:
            steps.append((f'{name}_norm1', n1))
        steps.append((f'{name}_act1', 1))
        steps.append((f'{name}_conv2', 1))
        if n2 > 0:
            steps.append((f'{name}_norm2', n2))
        steps.append((f'{name}_act2', 1))
        if name.startswith('enc'):
            enc_output_marker[name] = f'{name}_act2'
    steps.append(('out_conv', 1))
    return steps, enc_output_marker


def place_bootstraps_and_track_skip_levels(steps, enc_output_marker, K):
    """Un unico cammino: piazza bootstrap sul percorso principale E
    registra, per ogni encoder, quanto vale 'since_last_bootstrap' nel
    momento in cui quell'encoder produce il suo output (cio' che lo
    skip 'porta con se' fino alla fusione)."""
    bootstrap_points = []
    since_last_bootstrap = 0
    skip_level_at_origin = {}

    marker_to_enc = {v: k for k, v in enc_output_marker.items()}

    for name, cost in steps:
        if since_last_bootstrap + cost > K:
            bootstrap_points.append(name)
            since_last_bootstrap = 0
        since_last_bootstrap += cost
        if name in marker_to_enc:
            skip_level_at_origin[marker_to_enc[name]] = since_last_bootstrap

    return bootstrap_points, skip_level_at_origin, since_last_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, required=True)
    args = parser.parse_args()
    K = args.K

    steps, enc_output_marker = build_step_sequence()
    total_depth = sum(cost for _, cost in steps)
    print(f"Profondita' totale del percorso principale: {total_depth} livelli (atteso: 271)\n")

    bootstrap_points, skip_level_at_origin, final_level = place_bootstraps_and_track_skip_levels(
        steps, enc_output_marker, K)

    print(f"--- Piazzamento bootstrap sul percorso principale (K={K}) ---")
    for bp in bootstrap_points:
        print(f"  bootstrap PRIMA di: {bp}")
    print(f"\nBootstrap sul percorso principale: {len(bootstrap_points)}")

    # Per calcolare il gap alla fusione serve sapere il livello del ramo
    # principale ANCHE al momento della fusione stessa (non solo
    # all'origine dello skip) -- rifacciamo un secondo passaggio leggero
    # che registra il livello 'since_last_bootstrap' anche ai marker
    # 'up_decX' (subito prima della somma).
    since_last_bootstrap = 0
    level_at_fusion = {}
    for name, cost in steps:
        if since_last_bootstrap + cost > K:
            since_last_bootstrap = 0
        since_last_bootstrap += cost
        if name.startswith('up_dec'):
            dec_name = name.replace('up_', '')
            level_at_fusion[dec_name] = since_last_bootstrap

    print(f"\n--- Allineamento delle skip connection (K={K}) ---")
    extra_skip_bootstraps = 0
    for origin, fusion in SKIP_PAIRS:
        origin_level = skip_level_at_origin[origin]
        fusion_level = level_at_fusion[fusion]
        gap = abs(fusion_level - origin_level)
        needs_extra = gap > K  # non dovrebbe mai succedere per costruzione, ma verifichiamo
        print(f"  {origin} -> {fusion}: livello origine={origin_level}, livello fusione={fusion_level}, "
              f"gap={gap} {'-> serve bootstrap extra!' if needs_extra else '-> allineamento economico, nessun bootstrap extra'}")
        if needs_extra:
            extra_skip_bootstraps += 1

    print(f"\nBootstrap extra necessari per le skip: {extra_skip_bootstraps}")
    print(f"\n=== TOTALE BOOTSTRAP PER L'INTERA RETE: {len(bootstrap_points) + extra_skip_bootstraps} ===")


if __name__ == '__main__':
    main()