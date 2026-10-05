"""
crypto/dimensioning_no6norm.py

Dimensionamento HE completo (convoluzioni + normalizzazione) per la
configurazione con 6 InstanceNorm rimosse e schema misto Chebyshev/
Newton/bypass -- estende crypto/dimensioning.py (che contava SOLO le
operazioni di convoluzione, mai il costo della normalizzazione) con un
conteggio esplicito delle operazioni di isqrt per canale.

CONCETTO CHIAVE: ogni InstanceNorm calcola 1/sqrt(varianza) UNA VOLTA PER
CANALE (non per pixel) -- la varianza e' uno scalare per canale/istanza.
Quindi il costo totale di normalizzazione per uno stage e' (numero di
canali in quello stage) x (profondita'/operazioni sequenziali per quel
layer specifico, gia' calcolata in crypto/estimate_total_depth_no6norm.py
-- essendo una catena sequenziale di moltiplicazioni scalari, il numero
di "livelli" e il numero di "operazioni" coincidono qui: non c'e'
parallelismo da sfruttare in una catena, quindi depth == ops).

NON include il costo di AccumulateSum (la riduzione per calcolare media/
varianza a partire dai pixel) nel totale principale -- quel costo dipende
dalla risoluzione spaziale di ogni stage (numero di pixel per tile), non
dallo schema di approssimazione della radice, quindi e' IDENTICO in tutte
e tre le configurazioni gia' confrontate (Newton puro su 22, Chebyshev
puro su 22, schema misto su 16) e non cambia la classifica relativa tra
loro -- riportato in fondo come riferimento separato.
"""

import json
import math

# Canali per stage (aggressivo+sum, [32,64,128,256,128,64]) -- entrambi i
# norm slot di un blocco (block.1 e block.4) lavorano sullo stesso numero
# di canali (l'output di quello stage), per costruzione di ConvBlock.
STAGE_CHANNELS = {
    'enc0': 32, 'enc1': 64, 'enc2': 128, 'enc3': 256, 'enc4': 128, 'enc5': 64,
    'dec4': 128, 'dec3': 256, 'dec2': 128, 'dec1': 64, 'dec0': 32,
}
BLOCK_PREFIXES = list(STAGE_CHANNELS.keys())
ALL_NORM_LAYERS = [f'{p}.block.1' for p in BLOCK_PREFIXES] + \
                  [f'{p}.block.4' for p in BLOCK_PREFIXES]
BYPASS_LAYERS = {'enc4.block.1', 'enc5.block.1', 'dec0.block.1',
                  'enc5.block.4', 'enc3.block.1', 'enc4.block.4'}

# Risoluzione per stage (H, W), per il conteggio delle pixel-per-canale
# usato nel proxy di AccumulateSum (riferimento separato, vedi sopra)
STAGE_RES = {
    'enc0': (256, 224), 'enc1': (128, 112), 'enc2': (64, 56),
    'enc3': (32, 28), 'enc4': (16, 14), 'enc5': (8, 7),
    'dec4': (16, 14), 'dec3': (32, 28), 'dec2': (64, 56),
    'dec1': (128, 112), 'dec0': (256, 224),
}


def main():
    with open('crypto/chebyshev_calibration_no6norm.json') as f:
        cheb = json.load(f)
    with open('crypto/newton_fallback_no6norm.json') as f:
        newton_fallback = json.load(f)

    # --- Operazioni di CONVOLUZIONE: stesso conteggio "Cout-only" gia'
    # usato in crypto/dimensioning.py -- IDENTICO per costruzione a
    # prima (filters invariati), riportato qui solo per completezza. ---
    n_tiles = 2  # griglia 2x1
    conv_ops = 0
    for prefix, ch in STAGE_CHANNELS.items():
        conv_ops += n_tiles * ch  # block.0
        conv_ops += n_tiles * ch  # block.3 (stesso numero di canali in uscita)
    conv_ops += n_tiles * 4  # out_conv, 4 classi
    # upsampling: canali di ARRIVO di ogni up (= canali dello stage encoder corrispondente)
    up_channels = [STAGE_CHANNELS['enc4'], STAGE_CHANNELS['enc3'],
                   STAGE_CHANNELS['enc2'], STAGE_CHANNELS['enc1'], STAGE_CHANNELS['enc0']]
    conv_ops += sum(n_tiles * ch for ch in up_channels)

    # --- Operazioni di NORMALIZZAZIONE (isqrt), pesate per canale ---
    print(f"{'Layer':16s} {'canali':>7s} {'schema':>8s} {'depth/canale':>13s} {'ops totali':>12s}")
    print("-" * 62)

    norm_ops = 0
    for name in ALL_NORM_LAYERS:
        prefix = name.split('.')[0]
        ch = STAGE_CHANNELS[prefix]

        if name in BYPASS_LAYERS:
            depth_per_channel = 0
            schema = 'bypass'
        elif name in newton_fallback:
            depth_per_channel = 1 + 3 * newton_fallback[name]['iterations_needed'] + 1
            schema = 'newton'
        elif name in cheb:
            depth_per_channel = 1 + cheb[name]['total_depth'] + 1
            schema = 'cheb'
        else:
            raise KeyError(f"Layer '{name}' non coperto da nessuno schema")

        layer_ops = ch * depth_per_channel
        norm_ops += layer_ops
        print(f"{name:16s} {ch:7d} {schema:>8s} {depth_per_channel:13d} {layer_ops:12d}")

    print("-" * 62)
    print(f"\nOperazioni di convoluzione (proxy Cout-only, invariato rispetto a prima): {conv_ops:,}")
    print(f"Operazioni di normalizzazione (isqrt, pesate per canale):                  {norm_ops:,}")
    print(f"TOTALE (conv + norm):                                                      {conv_ops + norm_ops:,}")

    # --- Riferimento separato: costo AccumulateSum (riduzione media/varianza) ---
    print("\n--- Riferimento separato: costo di riduzione (AccumulateSum) ---")
    print("Indipendente dallo schema isqrt -- stesso in tutte le configurazioni gia'")
    print("confrontate, non cambia la classifica relativa tra loro.")
    reduction_ops = 0
    for name in ALL_NORM_LAYERS:
        if name in BYPASS_LAYERS:
            continue  # nessuna riduzione calcolata se il layer e' bypassato
        prefix = name.split('.')[0]
        ch = STAGE_CHANNELS[prefix]
        h, w = STAGE_RES[prefix]
        pixels_per_tile = (h * w) // n_tiles
        # AccumulateSum: circa log2(pixels_per_tile) rotazioni+somme, x2
        # (una per la media, una per la varianza) x canali x tile
        log2_px = math.ceil(math.log2(pixels_per_tile))
        reduction_ops += ch * n_tiles * log2_px * 2
    print(f"Operazioni di riduzione stimate: {reduction_ops:,}")

    print(f"\nNOTA: il vecchio dimensionamento (crypto/dimensioning.py) NON includeva")
    print("il costo di normalizzazione -- il confronto diretto con la configurazione")
    print("originale (22 norm) va fatto applicando questo stesso schema di conteggio")
    print("anche a quella, non con il numero '2.568' gia' riportato in precedenza")
    print("(quello copriva solo la parte di convoluzione).")


if __name__ == '__main__':
    main()