"""
crypto/bootstrap_placement_kstage.py

Versione generica del piazzamento bootstrap, per qualunque k (6,5,4),
che legge i dati REALI dal file di calibrazione (niente numeri scritti
a mano -- evita l'errore di trascrizione/file corrotto gia' incontrato
con la versione a 6 stage).

Stessa logica corretta di bootstrap_placement.py:
- Bootstrap sul percorso principale ogni volta che si supererebbe K
  dall'ultimo bootstrap.
- Il gap di allineamento delle skip e' calcolato rispetto all'ULTIMO
  bootstrap sul ramo principale, non dall'inizio della rete -- per
  costruzione resta piccolo (<= K), quasi mai serve un bootstrap dedicato.

I file di calibrazione (crypto/calibrate_kstage_isqrt.py) salvano
'total_depth' SENZA il +2 di varsq/apply -- questo script lo aggiunge
qui, in un unico posto, per evitare ambiguita'.
"""

import argparse
import json


def build_step_sequence(k, norm_depths):
    enc_names = [f'enc{i}' for i in range(k)]
    dec_names = [f'dec{i}' for i in range(k - 2, -1, -1)]  # es. k=5 -> dec3,dec2,dec1,dec0

    steps = []
    enc_output_marker = {}

    for name in enc_names:
        n1 = norm_depths.get(f'{name}.block.1', 0)
        n2 = norm_depths.get(f'{name}.block.4', 0)
        steps.append((f'{name}_conv1', 1))
        if n1 > 0:
            steps.append((f'{name}_norm1', n1))
        steps.append((f'{name}_act1', 1))
        steps.append((f'{name}_conv2', 1))
        if n2 > 0:
            steps.append((f'{name}_norm2', n2))
        steps.append((f'{name}_act2', 1))
        enc_output_marker[name] = f'{name}_act2'

    for name in dec_names:
        steps.append((f'up_{name}', 1))
        steps.append((f'{name}_skip_add', 0))
        n1 = norm_depths.get(f'{name}.block.1', 0)
        n2 = norm_depths.get(f'{name}.block.4', 0)
        steps.append((f'{name}_conv1', 1))
        if n1 > 0:
            steps.append((f'{name}_norm1', n1))
        steps.append((f'{name}_act1', 1))
        steps.append((f'{name}_conv2', 1))
        if n2 > 0:
            steps.append((f'{name}_norm2', n2))
        steps.append((f'{name}_act2', 1))

    steps.append(('out_conv', 1))

    # Coppie skip: enc(k-2)->dec(k-2), ..., enc0->dec0
    skip_pairs = [(f'enc{i}', f'dec{i}') for i in range(k - 2, -1, -1)]

    return steps, enc_output_marker, skip_pairs


def place_and_track(steps, enc_output_marker, K):
    bootstrap_points = []
    since_last = 0
    skip_level_at_origin = {}
    level_at_fusion = {}
    marker_to_enc = {v: k for k, v in enc_output_marker.items()}

    for name, cost in steps:
        if since_last + cost > K:
            bootstrap_points.append(name)
            since_last = 0
        since_last += cost
        if name in marker_to_enc:
            skip_level_at_origin[marker_to_enc[name]] = since_last
        if name.startswith('up_dec'):
            level_at_fusion[name.replace('up_', '')] = since_last

    return bootstrap_points, skip_level_at_origin, level_at_fusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, choices=[4, 5, 6])
    parser.add_argument('--calibration_json', required=True,
                        help='Output di crypto/calibrate_kstage_isqrt.py per questo k')
    parser.add_argument('--K', type=int, required=True,
                        help='Budget di livelli tra un bootstrap e il successivo')
    args = parser.parse_args()

    with open(args.calibration_json) as f:
        cal = json.load(f)

    # total_depth nel file NON include il +2 di varsq/apply -- lo aggiungiamo qui.
    norm_depths = {name: entry['total_depth'] + 2 for name, entry in cal.items()}

    steps, enc_output_marker, skip_pairs = build_step_sequence(args.k, norm_depths)
    total_depth = sum(c for _, c in steps)
    print(f"k={args.k}: profondita' totale del percorso principale = {total_depth} livelli\n")

    bootstrap_points, skip_level_at_origin, level_at_fusion = place_and_track(
        steps, enc_output_marker, args.K)

    print(f"--- Bootstrap sul percorso principale (K={args.K}) ---")
    for bp in bootstrap_points:
        print(f"  bootstrap PRIMA di: {bp}")
    print(f"\nBootstrap sul percorso principale: {len(bootstrap_points)}")

    print(f"\n--- Allineamento skip connection ---")
    extra = 0
    for origin, fusion in skip_pairs:
        o = skip_level_at_origin.get(origin, 0)
        fu = level_at_fusion.get(fusion, 0)
        gap = abs(fu - o)
        need_extra = gap > args.K
        print(f"  {origin} -> {fusion}: origine={o}, fusione={fu}, gap={gap}"
              f"{' -> serve bootstrap extra!' if need_extra else ' -> economico, nessun extra'}")
        if need_extra:
            extra += 1

    total_bootstraps = len(bootstrap_points) + extra
    print(f"\n=== TOTALE BOOTSTRAP (k={args.k}, K={args.K}): {total_bootstraps} ===")


if __name__ == '__main__':
    main()