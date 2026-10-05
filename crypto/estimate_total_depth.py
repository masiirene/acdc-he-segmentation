"""
crypto/estimate_total_depth.py

Stima la profondita' moltiplicativa TOTALE lungo il percorso principale
della rete (encoder -> bottleneck -> decoder -> out_conv), sommando i
contributi di Conv2d, PolyAct e InstanceNorm (Newton-Raphson calibrato
per layer), usando i dati reali gia' misurati in
crypto/newton_raphson_simulation_v2.json.

ASSUNZIONI ESPLICITE (da confermare con Aurora prima di usare questo
numero per il dimensionamento reale dei parametri CKKS):

1. Conv2d/ConvTranspose2d: 1 livello per operazione (moltiplicazione
   ciphertext per scalare in chiaro -- i pesi sono noti, non cifrati).
2. PolyAct: 1 livello (il quadrato x*x, ciphertext-ciphertext). Le
   moltiplicazioni per le costanti a, b sono trattate come SENZA costo
   di livello aggiuntivo (assunzione ottimistica basata su "lazy
   rescaling" -- se la libreria HE scelta non lo supporta, aggiungere
   +1 livello per PolyAct, cioe' CONST_ACT_DEPTH=2 invece di 1).
3. InstanceNorm (per-istanza, Newton-Raphson): 1 livello per calcolare
   x^2 (necessario per la varianza) + 3 livelli per iterazione di
   Newton-Raphson (vedi crypto/packing.py:newton_raphson_isqrt) + 1
   livello per l'applicazione finale (x-mean)*inv_std -- QUESTO livello
   e' spesso trascurato: inv_std NON e' una costante in chiaro (e'
   calcolato su dato cifrato, specifico per istanza), quindi la
   moltiplicazione finale e' ciphertext-ciphertext, non uno scalare.

Il conteggio NON include il costo di livello per riportare allo stesso
livello i due rami di una skip connection prima della concatenazione
(l'encoder produce quel tensore con MOLTA meno profondita' accumulata
rispetto al decoder, quando arriva il momento di concatenarli molti
layer piu' avanti) -- e' un problema separato, aperto, che riguarda il
piazzamento del bootstrap piu' che la profondita' "pura" del calcolo,
segnalato esplicitamente in fondo all'output di questo script.
"""

import json
import argparse


CONST_CONV_DEPTH = 1     # Conv2d/ConvTranspose2d: mult per scalare in chiaro
CONST_ACT_DEPTH = 1       # PolyAct: quadrato (assunzione ottimistica, vedi sopra)
CONST_NORM_VARSQ_DEPTH = 1   # InstanceNorm: x^2 per la varianza
CONST_NORM_APPLY_DEPTH = 1   # InstanceNorm: (x-mean)*inv_std, ciphertext-ciphertext
CONST_NEWTON_MULTS_PER_ITER = 3  # newton_raphson_isqrt: y*y, x*(y*y), y*(correzione)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isqrt_json', default='crypto/newton_raphson_simulation_v2.json')
    args = parser.parse_args()

    with open(args.isqrt_json) as f:
        isqrt_data = json.load(f)

    layers = {k: v for k, v in isqrt_data.items() if not k.startswith('_')}

    print(f"{'Layer':16s} {'n_iter':>8s} {'depth Norm':>12s} {'depth Conv':>12s} "
          f"{'depth Act':>11s} {'depth totale':>14s}")
    print("-" * 78)

    total_conv = total_act = total_norm = 0
    for name in sorted(layers.keys()):
        n_iter = layers[name]['iterations_needed']
        if n_iter is None:
            print(f"\u26a0\ufe0f  {name}: iterations_needed=None (layer non convergente nella "
                  f"calibrazione -- controlla crypto/simulate_newton_raphson_isqrt_v2.py)")
            continue

        norm_depth = CONST_NORM_VARSQ_DEPTH + CONST_NEWTON_MULTS_PER_ITER * n_iter + CONST_NORM_APPLY_DEPTH
        conv_depth = CONST_CONV_DEPTH
        act_depth = CONST_ACT_DEPTH
        block_total = conv_depth + norm_depth + act_depth

        print(f"{name:16s} {n_iter:8d} {norm_depth:12d} {conv_depth:12d} "
              f"{act_depth:11d} {block_total:14d}")

        total_conv += conv_depth
        total_act += act_depth
        total_norm += norm_depth

    # 5 ConvTranspose2d (up0..up4), non contati sopra (non hanno una
    # InstanceNorm/PolyAct associata direttamente, sono un layer a parte)
    n_upsample = 5
    total_upsample = n_upsample * CONST_CONV_DEPTH

    # out_conv: 1x1 conv finale, nessuna norm/act dopo
    total_out_conv = CONST_CONV_DEPTH

    grand_total = total_conv + total_act + total_norm + total_upsample + total_out_conv

    print("-" * 78)
    print(f"\nProfondita' totale lungo il percorso principale (encoder -> bottleneck -> "
          f"decoder -> out_conv):\n")
    print(f"  Conv2d (nei ConvBlock, {len(layers)} operazioni):        {total_conv:5d} livelli "
          f"({100*total_conv/grand_total:4.1f}%)")
    print(f"  PolyAct ({len(layers)} attivazioni):                    {total_act:5d} livelli "
          f"({100*total_act/grand_total:4.1f}%)")
    print(f"  InstanceNorm/Newton-Raphson ({len(layers)} normaliz.):  {total_norm:5d} livelli "
          f"({100*total_norm/grand_total:4.1f}%)")
    print(f"  ConvTranspose2d (upsampling, {n_upsample} operazioni):    {total_upsample:5d} livelli "
          f"({100*total_upsample/grand_total:4.1f}%)")
    print(f"  out_conv (1x1 finale):                        {total_out_conv:5d} livelli "
          f"({100*total_out_conv/grand_total:4.1f}%)")
    print(f"  {'-'*60}")
    print(f"  TOTALE:                                        {grand_total:5d} livelli\n")

    print("\u26a0\ufe0f  NON incluso in questo totale: il costo di allineamento di livello per le")
    print("   skip connection. L'encoder produce ogni tensore di skip con MOLTA meno")
    print("   profondita' accumulata rispetto a quando il decoder lo concatena (molti")
    print("   layer piu' avanti nel grafo) -- prima della concatenazione, i due rami")
    print("   devono essere allo stesso livello, il che richiede o mantenere il ramo")
    print("   skip 'in attesa' con moltiplicazioni fittizie per farlo scendere di livello,")
    print("   o un bootstrap dedicato su quel ramo. Questo e' un problema di PIAZZAMENTO")
    print("   (dove/quando bootstrap-are), non di profondita' 'pura' del calcolo -- da")
    print("   discutere esplicitamente con Aurora nel contesto del punto D.")


if __name__ == '__main__':
    main()