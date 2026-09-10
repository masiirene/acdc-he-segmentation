"""
Analizza results/.../polyact_drift.json (prodotto da training.train con
--diagnostics_out) per capire QUALE layer PolyAct, tra tutti i 22, mostra
un vero cambiamento di comportamento a cavallo del collasso -- non solo il
"vincitore" globale ad ogni epoca (che puo' cambiare identita' e mascherare
il trend di un singolo layer, come successo nel run diag2: dec1.block.5
era il worst-case fino a epoch 6, poi il ranking e' passato a dec0.block.5,
nascondendo l'andamento di dec1.block.5 stesso nei print a runtime).

Uso:
    python3 analyze_polyact_drift.py path/to/polyact_drift.json

Nessuna dipendenza esterna (solo libreria standard) -- gira anche senza
torch/numpy installati, utile per un'analisi rapida separata dal training.
"""
import json
import sys


def main(path):
    with open(path) as f:
        history = json.load(f)

    if not history:
        print('File vuoto.')
        return

    epochs = [h['epoch'] for h in history]
    layer_names = sorted(history[0]['polyact'].keys())

    print(f'Epoche disponibili: {epochs[0]}..{epochs[-1]} ({len(epochs)} totali)')
    print(f'Layer PolyAct tracciati: {len(layer_names)}')
    print()

    # Per ogni layer, costruisci la serie temporale di raw_p99_abs e raw_max_abs
    series_p99 = {name: [] for name in layer_names}
    series_max = {name: [] for name in layer_names}
    series_a   = {name: [] for name in layer_names}

    for h in history:
        for name in layer_names:
            entry = h['polyact'][name]
            series_p99[name].append(entry.get('raw_p99_abs') or 0.0)
            series_max[name].append(entry.get('raw_max_abs') or 0.0)
            series_a[name].append(entry.get('a', 0.0))

    # Trova l'epoca (indice) col salto piu' grande di clamp/collasso -- qui la
    # identifichiamo semplicemente come il punto in cui p99 GLOBALE MAX cambia
    # di piu' rispetto all'epoca precedente in valore assoluto tra layer.
    # Piu' robusto: chiediamo all'utente quale epoca guardare, di default
    # mostriamo l'intera tabella e lasciamo scegliere a vista.

    print('=== Variazione relativa di raw_p99_abs (valore grezzo pre-clamp, percentile 99) ===')
    print('Confronto tra la PRIMA e l\'ULTIMA epoca disponibile, ordinato per crescita relativa:')
    print()
    growth = []
    for name in layer_names:
        first = series_p99[name][0]
        last = series_p99[name][-1]
        rel = (last - first) / (abs(first) + 1e-8)
        growth.append((name, first, last, rel))
    growth.sort(key=lambda t: t[3], reverse=True)

    print(f'{"layer":30s} {"epoch1":>10s} {"epochN":>10s} {"var.relativa":>14s}')
    for name, first, last, rel in growth:
        print(f'{name:30s} {first:10.3f} {last:10.3f} {rel:+13.1%}')

    print()
    print('=== Serie temporale completa di raw_p99_abs per i 5 layer con maggiore variazione ===')
    for name, *_ in growth[:5]:
        vals = ' '.join(f'{v:7.2f}' for v in series_p99[name])
        print(f'{name:20s}: {vals}')

    print()
    print('=== Serie temporale completa di raw_max_abs per gli stessi 5 layer ===')
    for name, *_ in growth[:5]:
        vals = ' '.join(f'{v:9.1f}' for v in series_max[name])
        print(f'{name:20s}: {vals}')

    print()
    print('=== Coefficiente "a" (termine quadratico) per gli stessi 5 layer ===')
    for name, *_ in growth[:5]:
        vals = ' '.join(f'{v:8.5f}' for v in series_a[name])
        print(f'{name:20s}: {vals}')

    print()
    print('=== Varianza minima InstanceNorm per epoca (se presente nel file) ===')
    for h in history:
        min_var_dict = h.get('instance_norm_min_var', {})
        if min_var_dict:
            worst_layer = min(min_var_dict.items(), key=lambda kv: kv[1])
            print(f"  epoch {h['epoch']:3d}: min_var={worst_layer[1]:.4e} (layer {worst_layer[0]})")

    # --- Coefficiente di variazione della varianza tra istanze (test ipotesi
    # "risoluzione spaziale bassa -> stima di varianza rumorosa") ---
    spatial = history[-1].get('instance_norm_spatial_hw', {})
    if spatial and any(h.get('instance_norm_max_cv') for h in history):
        print()
        print('=== Coefficiente di variazione (CV) della varianza tra istanze, per layer ===')
        print('(CV alto = la varianza per-istanza "salta" molto da un\'immagine all\'altra nello')
        print(' stesso batch -- atteso piu\' alto nei layer a bassa risoluzione spaziale)')
        print()
        cv_names = sorted(spatial.keys(), key=lambda n: spatial[n])
        print(f'{"layer":20s} {"pixel/istanza":>14s}  ' +
              '  '.join(f'ep{h["epoch"]}' for h in history))
        for name in cv_names:
            hw = spatial.get(name, '?')
            vals = []
            for h in history:
                cv_dict = h.get('instance_norm_max_cv', {})
                v = cv_dict.get(name)
                vals.append(f'{v:5.2f}' if v is not None else '   - ')
            print(f'{name:20s} {hw:>14} ' + ' '.join(vals))



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Uso: python3 analyze_polyact_drift.py path/to/polyact_drift.json')
        sys.exit(1)
    main(sys.argv[1])