"""
crypto/test_block_rotation_issue.py

Il prototipo precedente (prototype_row_packed_conv.py) simulava la
rotazione "lungo l'asse canale" rimodellando la riga in una matrice
2D (Wp x Cin) e ruotando solo quell'asse -- comodo in numpy, ma NON
corrisponde a nessuna vera primitiva CKKS. EvalRotate ruota l'INTERO
ciphertext come un vettore piatto di slot, senza sapere che esiste
una struttura a blocchi (un pixel ogni Cin slot) al suo interno.

Questo script:
1. Dimostra il problema vero: una rotazione GLOBALE (np.roll su un
   array 1D, equivalente a EvalRotate) di un vettore impacchettato a
   blocchi NON produce una rotazione indipendente dentro ogni blocco
   -- contamina i bordi tra un pixel e il successivo.
2. Implementa e verifica la correzione standard (usata in letteratura
   per questo esatto problema, es. GAZELLE/HElib): combinare DUE
   rotazioni globali con maschere complementari, per ricostruire la
   vera rotazione "a blocchi indipendenti" a partire da sole
   operazioni che EvalRotate puo' davvero fare.

Uso: python3 crypto/test_block_rotation_issue.py
"""

import numpy as np


def global_rotate(flat_vec, d):
    """Equivalente a EvalRotate(ct, d): ruota l'INTERO vettore piatto,
    nessuna nozione di blocchi."""
    return np.roll(flat_vec, -d)


def naive_block_rotate_via_global(flat_vec, d, block_size):
    """Quello che il prototipo precedente faceva IMPLICITAMENTE
    (sbagliato): tratta la rotazione come se fosse locale a ogni
    blocco, usando una sola rotazione globale -- corretto SOLO se
    d==0 o se non attraversa mai un confine di blocco."""
    return global_rotate(flat_vec, d)


def correct_block_rotate(flat_vec, d, block_size, n_blocks):
    """
    La correzione standard: combina DUE rotazioni globali con
    maschere complementari.

    Dentro ogni blocco di 'block_size' slot, vogliamo che la nuova
    posizione i (relativa al blocco) contenga il valore che era alla
    posizione (i+d) mod block_size DELLO STESSO BLOCCO -- una
    rotazione "ciclica locale", non globale.

    - rotated_main = ruota l'intero vettore di d: per le posizioni
      relative [0, block_size-d) dentro ogni blocco, il valore e'
      GIA' quello giusto (viene dalla stessa blocco, non attraversa
      il confine).
    - rotated_wrap = ruota l'intero vettore di (d - block_size): per
      le posizioni relative [block_size-d, block_size) dentro ogni
      blocco, QUESTA rotazione porta il valore giusto (il "giro"
      ciclico dentro lo stesso blocco), perche' lo spostamento
      negativo extra di un blocco intero compensa esattamente
      l'attraversamento di confine.
    - Si combinano con due maschere complementari (quali posizioni,
      relative al blocco, usare dall'una o dall'altra).
    """
    total_len = len(flat_vec)
    rotated_main = global_rotate(flat_vec, d)
    rotated_wrap = global_rotate(flat_vec, d - block_size)

    # Maschera: posizione i (assoluta) usa "main" se la sua posizione
    # RELATIVA al blocco e' < block_size - d, altrimenti "wrap".
    rel_pos = np.arange(total_len) % block_size
    mask_main = (rel_pos < (block_size - d)).astype(float)
    mask_wrap = 1.0 - mask_main

    return mask_main * rotated_main + mask_wrap * rotated_wrap


def main():
    print("=== Dimostrazione del problema e verifica della correzione ===\n")

    rng = np.random.default_rng(3)
    block_size = 6   # Cin, piccolo per leggere i numeri a mano
    n_blocks = 4      # 4 "pixel"
    total_len = block_size * n_blocks

    # Costruiamo un vettore dove ogni blocco ha valori facilmente
    # riconoscibili: blocco p, canale c -> valore p*10 + c
    flat = np.array([p * 10 + c for p in range(n_blocks) for c in range(block_size)], dtype=float)
    print(f"Vettore originale ({n_blocks} blocchi da {block_size}):")
    print(flat.reshape(n_blocks, block_size))
    print()

    d = 2  # vogliamo ruotare ogni blocco di 2 posizioni (ciclicamente, DENTRO il blocco)

    expected = np.array([
        [(p * 10 + ((c + d) % block_size)) for c in range(block_size)]
        for p in range(n_blocks)
    ], dtype=float).flatten()
    print(f"Risultato ATTESO (rotazione di {d} DENTRO ogni blocco, indipendente):")
    print(expected.reshape(n_blocks, block_size))
    print()

    naive = naive_block_rotate_via_global(flat, d, block_size)
    print(f"Risultato con UNA SOLA rotazione globale (quello che faceva il prototipo precedente):")
    print(naive.reshape(n_blocks, block_size))
    err_naive = np.max(np.abs(naive - expected))
    print(f"Errore rispetto all'atteso: {err_naive:.2f}  <-- SBAGLIATO se non e' zero\n")

    corrected = correct_block_rotate(flat, d, block_size, n_blocks)
    print(f"Risultato con la CORREZIONE (due rotazioni globali + maschere):")
    print(corrected.reshape(n_blocks, block_size))
    err_corrected = np.max(np.abs(corrected - expected))
    print(f"Errore rispetto all'atteso: {err_corrected:.2e}\n")

    if err_corrected < 1e-8:
        print("=== La correzione FUNZIONA: 2 rotazioni globali + 2 maschere == rotazione a blocchi indipendenti. ===")
    else:
        print("=== ATTENZIONE: anche la correzione e' sbagliata, da rivedere. ===")

    print(f"\n=== Costo della correzione, rispetto al prototipo precedente (che era sbagliato) ===")
    print(f"Prima (sbagliato): 1 rotazione per ogni diagonale d")
    print(f"Ora (corretto):    2 rotazioni + 2 moltiplicazioni per maschera, per ogni diagonale d")
    print(f"Il fattore di riduzione rispetto all'originale (Cin*Cout) resta comunque enorme:")
    print(f"raddoppiare un numero gia' ridotto di {134:.0f}x lo porta a circa {134/2:.0f}x -- ancora una vittoria netta.")


if __name__ == '__main__':
    main()