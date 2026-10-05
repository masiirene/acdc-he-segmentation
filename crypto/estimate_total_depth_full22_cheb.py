"""
crypto/estimate_total_depth_full22_cheb.py

Profondita' moltiplicativa TOTALE (non operazioni pesate per canale) per
la configurazione 22 normalizzazioni, Chebyshev monotono + fallback
Newton (enc0.block.1, dec0.block.4). Legge i file di calibrazione REALI
gia' su disco, per avere il numero esatto da riportare ad Aurora.
"""

import json

BLOCK_PREFIXES = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
                   'dec4', 'dec3', 'dec2', 'dec1', 'dec0']
ALL_NORM_LAYERS = [f'{p}.block.1' for p in BLOCK_PREFIXES] + \
                  [f'{p}.block.4' for p in BLOCK_PREFIXES]

with open('crypto/chebyshev_calibration_full_dataset.json') as f:
    cheb = json.load(f)
with open('crypto/newton_fallback_full22.json') as f:
    newton_fallback = json.load(f)

total_conv = total_act = total_norm = 0
print(f"{'Layer':16s} {'schema':>8s} {'norm_depth':>11s}")
print("-" * 40)
for name in ALL_NORM_LAYERS:
    total_conv += 1
    total_act += 1
    if name in newton_fallback:
        depth = 2 + 3 * newton_fallback[name]['iterations_needed']
        schema = 'newton'
    elif name in cheb:
        depth = 2 + cheb[name]['total_depth']
        schema = 'cheb'
    else:
        raise KeyError(f"Layer '{name}' non trovato in nessun file di calibrazione")
    total_norm += depth
    print(f"{name:16s} {schema:>8s} {depth:11d}")
print("-" * 40)

total_upsample = 5
total_out_conv = 1
grand_total = total_conv + total_act + total_norm + total_upsample + total_out_conv

print(f"\nConv: {total_conv}  Act: {total_act}  Norm: {total_norm}  "
      f"Upsample: {total_upsample}  out_conv: {total_out_conv}")
print(f"\nPROFONDITA' TOTALE (22 norm, Chebyshev+fallback): {grand_total} livelli")
print(f"Confronto: 22 norm Newton puro = 439 | 16 norm schema misto = 271")