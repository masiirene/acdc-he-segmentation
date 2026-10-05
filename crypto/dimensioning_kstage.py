"""
crypto/dimensioning_kstage.py

Numero TOTALE DI MOLTIPLICAZIONI (non la profondita') per un UNetKStage
con k arbitrario -- stesso schema di conteggio gia' usato ieri per la
rete a 6 stage (crypto/dimensioning_full_comparison.py), generalizzato:

- Convoluzioni: canali di uscita x numero di tile (stesso proxy di
  crypto/dimensioning.py).
- Normalizzazione: canali x profondita' per-layer (dalla calibrazione
  gia' salvata da crypto/calibrate_kstage_isqrt.py) -- ogni canale
  richiede il proprio calcolo scalare di 1/sqrt(varianza).
- Riduzione (media/varianza, AccumulateSum): stima log2(pixel-per-tile)
  x canali x tile x2 -- zero per i layer bypassati (nessuna statistica
  calcolata).
"""

import argparse
import json
import math


def stage_channels(k, filters):
    enc = {f'enc{i}': filters[i] for i in range(k)}
    dec = {f'dec{i}': filters[i] for i in range(k - 1)}
    return {**enc, **dec}


def stage_resolution(k, base_h=256, base_w=224):
    res = {}
    for i in range(k):
        h = base_h // (2 ** i)
        w = base_w // (2 ** i)
        res[f'enc{i}'] = (h, w)
        if i < k - 1:
            res[f'dec{i}'] = (h, w)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, choices=[3, 4, 5, 6])
    parser.add_argument('--filters', type=int, nargs='+', required=True)
    parser.add_argument('--calibration_json', required=True,
                        help='Output di calibrate_kstage_isqrt.py')
    parser.add_argument('--n_tiles', type=int, default=2)
    args = parser.parse_args()

    assert len(args.filters) == args.k

    ch = stage_channels(args.k, args.filters)
    res = stage_resolution(args.k)

    with open(args.calibration_json) as f:
        cal = json.load(f)
    cal = {k_: v for k_, v in cal.items() if not k_.startswith('_')}

    conv_ops = 0
    for prefix, c in ch.items():
        conv_ops += args.n_tiles * c
        conv_ops += args.n_tiles * c
    conv_ops += args.n_tiles * 4
    up_channels = [ch[f'enc{i}'] for i in range(args.k - 2, -1, -1)]
    conv_ops += sum(args.n_tiles * c for c in up_channels)

    print(f"{'Layer':16s} {'canali':>7s} {'schema':>8s} {'depth/canale':>13s} {'norm ops':>10s} {'riduz. ops':>11s}")
    print("-" * 72)

    norm_ops = 0
    reduction_ops = 0
    for name, entry in sorted(cal.items()):
        prefix = name.split('.')[0]
        if prefix not in ch:
            print(f"\u26a0\ufe0f  Stage '{prefix}' non presente in k={args.k} -- ignorato")
            continue
        c = ch[prefix]
        depth = entry.get('total_depth', 0)
        schema = entry.get('schema', '?')

        layer_norm_ops = c * depth
        norm_ops += layer_norm_ops

        if depth > 0:
            h, w = res[prefix]
            pixels_per_tile = (h * w) // args.n_tiles
            log2_px = max(1, math.ceil(math.log2(max(pixels_per_tile, 2))))
            layer_reduction_ops = c * args.n_tiles * log2_px * 2
        else:
            layer_reduction_ops = 0
        reduction_ops += layer_reduction_ops

        print(f"{name:16s} {c:7d} {schema:>8s} {depth:13d} {layer_norm_ops:10d} {layer_reduction_ops:11d}")

    print("-" * 72)
    total = conv_ops + norm_ops + reduction_ops
    print(f"\nConvoluzioni:      {conv_ops:>10,}  ({100*conv_ops/total:4.1f}%)")
    print(f"Normalizzazione:   {norm_ops:>10,}  ({100*norm_ops/total:4.1f}%)")
    print(f"Riduzione:         {reduction_ops:>10,}  ({100*reduction_ops/total:4.1f}%)")
    print(f"TOTALE (numero di moltiplicazioni): {total:,}")


if __name__ == '__main__':
    main()