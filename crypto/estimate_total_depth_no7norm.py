"""
crypto/estimate_total_depth_no7norm.py

Profondita' totale per la configurazione a 15 normalizzazioni attive
(7 rimosse: i 6 originali + enc0.block.1), schema misto Chebyshev (14
layer) + Newton fallback (dec0.block.4).
"""

import json

BLOCK_PREFIXES = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
                   'dec4', 'dec3', 'dec2', 'dec1', 'dec0']
ALL_NORM_LAYERS = [f'{p}.block.1' for p in BLOCK_PREFIXES] + \
                  [f'{p}.block.4' for p in BLOCK_PREFIXES]
BYPASS_LAYERS = {'enc4.block.1', 'enc5.block.1', 'dec0.block.1',
                  'enc5.block.4', 'enc3.block.1', 'enc4.block.4',
                  'enc0.block.1'}

with open('crypto/chebyshev_calibration_no6norm.json') as f:
    cheb = json.load(f)
with open('crypto/newton_fallback_no7norm.json') as f:
    newton_fallback = json.load(f)

total_conv = total_act = total_norm = 0
print(f"{'Layer':16s} {'schema':>8s} {'norm_depth':>11s}")
print("-" * 40)
for name in ALL_NORM_LAYERS:
    total_conv += 1
    total_act += 1
    if name in BYPASS_LAYERS:
        depth, schema = 0, 'bypass'
    elif name in newton_fallback:
        depth = 2 + 3 * newton_fallback[name]['iterations_needed']
        schema = 'newton'
    elif name in cheb:
        depth = 2 + cheb[name]['total_depth']
        schema = 'cheb'
    else:
        raise KeyError(f"Layer '{name}' non trovato in nessun file")
    total_norm += depth
    print(f"{name:16s} {schema:>8s} {depth:11d}")
print("-" * 40)

total_upsample = 5
total_out_conv = 1
grand_total = total_conv + total_act + total_norm + total_upsample + total_out_conv

print(f"\nConv: {total_conv}  Act: {total_act}  Norm: {total_norm}  "
      f"Upsample: {total_upsample}  out_conv: {total_out_conv}")
print(f"\nPROFONDITA' TOTALE (15 norm attive, 7 bypassate): {grand_total} livelli")
print(f"\nConfronto:")
print(f"  22 norm, Newton puro:              439")
print(f"  22 norm, Chebyshev+fallback:        332")
print(f"  16 norm (6 rimosse), schema misto:  271  (Dice 0.8744)")
print(f"  15 norm (7 rimosse), schema misto:  {grand_total}  (Dice 0.8688)")