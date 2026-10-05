"""
crypto/dimensioning_full_comparison.py

Confronto omogeneo (convoluzioni + normalizzazione + riduzione) tra le
tre configurazioni gia' verificate end-to-end:

1. 22 normalizzazioni, Newton-Raphson puro
   -- crypto/newton_raphson_simulation_v2.json
2. 22 normalizzazioni, Chebyshev monotono (iter>=2)
   -- crypto/chebyshev_calibration_full_dataset.json
3. 16 normalizzazioni (6 rimosse), schema misto Chebyshev/Newton/bypass
   -- crypto/chebyshev_calibration_no6norm.json +
      crypto/newton_fallback_no6norm.json

Stesso identico schema di conteggio per tutte e tre, letto direttamente
dai file di calibrazione reali (nessun numero ricopiato a mano) --
cosi' il confronto e' onesto e verificabile.

Per le config 1 e 2, TUTTI i 22 layer sono attivi (nessun bypass). Per
la config 3, 6 layer sono bypassati (costo zero, esclusi anche dal
conteggio di riduzione, dato che non si calcola alcuna statistica per
loro).
"""

import json
import math
import os

STAGE_CHANNELS = {
    'enc0': 32, 'enc1': 64, 'enc2': 128, 'enc3': 256, 'enc4': 128, 'enc5': 64,
    'dec4': 128, 'dec3': 256, 'dec2': 128, 'dec1': 64, 'dec0': 32,
}
BLOCK_PREFIXES = list(STAGE_CHANNELS.keys())
ALL_NORM_LAYERS = [f'{p}.block.1' for p in BLOCK_PREFIXES] + \
                  [f'{p}.block.4' for p in BLOCK_PREFIXES]
STAGE_RES = {
    'enc0': (256, 224), 'enc1': (128, 112), 'enc2': (64, 56),
    'enc3': (32, 28), 'enc4': (16, 14), 'enc5': (8, 7),
    'dec4': (16, 14), 'dec3': (32, 28), 'dec2': (64, 56),
    'dec1': (128, 112), 'dec0': (256, 224),
}
N_TILES = 2


def compute_conv_ops():
    """Identico per tutte e tre le configurazioni -- stessi filters,
    stesso skip_mode, nessun canale cambia tra le tre config."""
    conv_ops = 0
    for prefix, ch in STAGE_CHANNELS.items():
        conv_ops += N_TILES * ch  # block.0
        conv_ops += N_TILES * ch  # block.3
    conv_ops += N_TILES * 4  # out_conv, 4 classi
    up_channels = [STAGE_CHANNELS['enc4'], STAGE_CHANNELS['enc3'],
                   STAGE_CHANNELS['enc2'], STAGE_CHANNELS['enc1'], STAGE_CHANNELS['enc0']]
    conv_ops += sum(N_TILES * ch for ch in up_channels)
    return conv_ops


def compute_norm_and_reduction(depth_lookup, bypass_layers):
    """
    depth_lookup: dict {layer_name: depth_per_channel}, gia' calcolato
    (2 + 3*iter per Newton, 2 + total_depth per Chebyshev).
    bypass_layers: set di layer con costo zero (nessuna statistica calcolata).
    """
    norm_ops = 0
    reduction_ops = 0
    for name in ALL_NORM_LAYERS:
        prefix = name.split('.')[0]
        ch = STAGE_CHANNELS[prefix]

        if name in bypass_layers:
            continue  # costo zero, nessuna statistica calcolata

        depth_per_channel = depth_lookup[name]
        norm_ops += ch * depth_per_channel

        h, w = STAGE_RES[prefix]
        pixels_per_tile = (h * w) // N_TILES
        log2_px = math.ceil(math.log2(pixels_per_tile))
        reduction_ops += ch * N_TILES * log2_px * 2  # x2: media e varianza

    return norm_ops, reduction_ops


def load_newton_depths(path):
    with open(path) as f:
        data = json.load(f)
    return {name: 2 + 3 * info['iterations_needed']
            for name, info in data.items() if not name.startswith('_')}


def load_chebyshev_depths(path):
    with open(path) as f:
        data = json.load(f)
    return {name: 2 + info['total_depth'] for name, info in data.items()}


def report(label, norm_ops, reduction_ops, conv_ops, baseline_total=None):
    total = conv_ops + norm_ops + reduction_ops
    print(f"\n=== {label} ===")
    print(f"  Convoluzioni:  {conv_ops:>8,}  ({100*conv_ops/total:4.1f}%)")
    print(f"  Normalizzazione: {norm_ops:>8,}  ({100*norm_ops/total:4.1f}%)")
    print(f"  Riduzione:     {reduction_ops:>8,}  ({100*reduction_ops/total:4.1f}%)")
    print(f"  TOTALE:        {total:>8,}")
    if baseline_total:
        print(f"  Riduzione vs riferimento: {100*(1 - total/baseline_total):.1f}%")
    return total


def main():
    conv_ops = compute_conv_ops()

    results = {}

    # --- Config 1: 22 norm, Newton puro ---
    path1 = 'crypto/newton_raphson_simulation_v2.json'
    if os.path.exists(path1):
        depths1 = load_newton_depths(path1)
        norm1, red1 = compute_norm_and_reduction(depths1, bypass_layers=set())
        results['newton22'] = report("22 normalizzazioni, Newton puro (riferimento)",
                                      norm1, red1, conv_ops)
    else:
        print(f"\u26a0\ufe0f  {path1} non trovato -- salto config 1")

    # --- Config 2: 22 norm, Chebyshev monotono + fallback Newton per i
    # 2 layer (enc0.block.1, dec0.block.4) che non convergono con Chebyshev
    # (stesso problema strutturale gia' scoperto nella config a 16 norm) ---
    path2 = 'crypto/chebyshev_calibration_full_dataset.json'
    path2_fallback = 'crypto/newton_fallback_full22.json'
    if os.path.exists(path2):
        depths2 = load_chebyshev_depths(path2)
        if os.path.exists(path2_fallback):
            with open(path2_fallback) as f:
                fallback2 = json.load(f)
            for name, info in fallback2.items():
                depths2[name] = 2 + 3 * info['iterations_needed']
        else:
            print(f"\u26a0\ufe0f  {path2_fallback} non trovato -- i layer mancanti da "
                  f"{path2} verranno esclusi (numero SOTTOSTIMATO)")
        missing = set(ALL_NORM_LAYERS) - set(depths2.keys())
        if missing:
            print(f"\u26a0\ufe0f  Layer ancora mancanti in config 2: {missing} -- esclusi dal conteggio")
        norm2, red2 = compute_norm_and_reduction(depths2, bypass_layers=set())
        results['cheb22'] = report("22 normalizzazioni, Chebyshev monotono + fallback Newton",
                                    norm2, red2, conv_ops,
                                    baseline_total=results.get('newton22'))
    else:
        print(f"\u26a0\ufe0f  {path2} non trovato -- salto config 2")

    # --- Config 3: 16 norm, schema misto ---
    path3a = 'crypto/chebyshev_calibration_no6norm.json'
    path3b = 'crypto/newton_fallback_no6norm.json'
    bypass_layers = {'enc4.block.1', 'enc5.block.1', 'dec0.block.1',
                      'enc5.block.4', 'enc3.block.1', 'enc4.block.4'}
    if os.path.exists(path3a) and os.path.exists(path3b):
        with open(path3a) as f:
            cheb3 = json.load(f)
        with open(path3b) as f:
            newton3 = json.load(f)
        depths3 = {}
        for name in ALL_NORM_LAYERS:
            if name in bypass_layers:
                continue
            if name in newton3:
                depths3[name] = 2 + 3 * newton3[name]['iterations_needed']
            elif name in cheb3:
                depths3[name] = 2 + cheb3[name]['total_depth']
        norm3, red3 = compute_norm_and_reduction(depths3, bypass_layers)
        results['mixed16'] = report("16 normalizzazioni (6 rimosse), schema misto",
                                     norm3, red3, conv_ops,
                                     baseline_total=results.get('newton22'))
    else:
        print(f"\u26a0\ufe0f  {path3a} o {path3b} non trovato -- salto config 3")

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    for label, total in results.items():
        print(f"  {label:12s}: {total:>10,} operazioni totali")


if __name__ == '__main__':
    main()