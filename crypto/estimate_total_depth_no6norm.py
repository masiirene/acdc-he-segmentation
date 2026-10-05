"""
crypto/estimate_total_depth_no6norm.py

Profondita' moltiplicativa totale per il checkpoint con 6 InstanceNorm
rimosse (crypto/finetune_without_6_instancenorm.py, Dice 0.8744).

Schema MISTO per i 16 layer di normalizzazione rimanenti:
- 14 layer: Chebyshev monotono (crypto/chebyshev_calibration_no6norm.json)
- 2 layer (enc0.block.1, dec0.block.4): Newton-Raphson puro, y0 scalare
  (crypto/newton_fallback_no6norm.json) -- range di varianza troppo ampio
  (174x e 582x) per un singolo polinomio di Chebyshev entro l'1% di errore.
- 6 layer: bypassati del tutto, costo zero.

Somma i contributi di ogni componente lungo il percorso principale della
rete, stessa metodologia di crypto/estimate_total_depth.py.
"""

import json

BLOCK_PREFIXES = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
                   'dec4', 'dec3', 'dec2', 'dec1', 'dec0']
ALL_NORM_LAYERS = [f'{p}.block.1' for p in BLOCK_PREFIXES] + \
                  [f'{p}.block.4' for p in BLOCK_PREFIXES]
BYPASS_LAYERS = {'enc4.block.1', 'enc5.block.1', 'dec0.block.1',
                  'enc5.block.4', 'enc3.block.1', 'enc4.block.4'}

with open('crypto/chebyshev_calibration_no6norm.json') as f:
    cheb = json.load(f)
with open('crypto/newton_fallback_no6norm.json') as f:
    newton_fallback = json.load(f)

total_conv = total_act = total_norm = 0
print(f"{'Layer':16s} {'schema':>10s} {'norm_depth':>11s}")
print("-" * 42)
for name in ALL_NORM_LAYERS:
    total_conv += 1  # Conv2d di quel "mezzo blocco", sempre presente
    total_act += 1   # PolyAct di quel "mezzo blocco", sempre presente

    if name in BYPASS_LAYERS:
        norm_depth = 0
        schema = 'bypass'
    elif name in newton_fallback:
        # 1 (x^2 per varianza) + 3*iter (Newton puro) + 1 (applicazione finale)
        norm_depth = 1 + 3 * newton_fallback[name]['iterations_needed'] + 1
        schema = 'newton'
    elif name in cheb:
        # 1 (x^2 per varianza) + [Chebyshev+Newton] + 1 (applicazione finale)
        norm_depth = 1 + cheb[name]['total_depth'] + 1
        schema = 'cheb'
    else:
        raise KeyError(f"Layer '{name}' non trovato in nessuna calibrazione -- "
                        f"controlla che tutti e 3 gli schemi (bypass/cheb/newton) "
                        f"coprano insieme tutti i 22 layer, senza buchi.")

    total_norm += norm_depth
    print(f"{name:16s} {schema:>10s} {norm_depth:11d}")
print("-" * 42)

total_upsample = 5   # 5 ConvTranspose2d (up0..up4)
total_out_conv = 1   # conv 1x1 finale

grand_total = total_conv + total_act + total_norm + total_upsample + total_out_conv

print(f"\nConv (22 operazioni):                  {total_conv:5d} livelli "
      f"({100*total_conv/grand_total:4.1f}%)")
print(f"PolyAct (22 attivazioni):               {total_act:5d} livelli "
      f"({100*total_act/grand_total:4.1f}%)")
print(f"Normalizzazione (14 cheb + 2 newton + 6 bypass): {total_norm:5d} livelli "
      f"({100*total_norm/grand_total:4.1f}%)")
print(f"ConvTranspose2d (upsampling):            {total_upsample:5d} livelli "
      f"({100*total_upsample/grand_total:4.1f}%)")
print(f"out_conv (1x1 finale):                   {total_out_conv:5d} livelli "
      f"({100*total_out_conv/grand_total:4.1f}%)")
print(f"\nTOTALE: {grand_total} livelli")

print(f"\nConfronto con la configurazione originale (22 normalizzazioni, Newton puro): 439 livelli")
print(f"Riduzione: {100*(1 - grand_total/439):.1f}%")