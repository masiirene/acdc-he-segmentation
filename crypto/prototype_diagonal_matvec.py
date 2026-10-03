"""
crypto/prototype_diagonal_matvec.py

PROTOTIPO IN CHIARO (numpy puro, nessuna crittografia) -- primo
mattone verso una convoluzione multi-canale piu' veloce in HE.

Problema attuale: per calcolare y[co] = sum_ci( W[co,ci] * x[ci] ) per
ogni co, la pipeline HE fa Cout*Cin moltiplicazioni separate (una per
ogni coppia co,ci). Con 256 canali, 65.536 moltiplicazioni -- il
collo di bottiglia misurato ieri sera.

Il metodo delle diagonali (Halevi-Shoup): se i Cin valori di x sono
TUTTI impacchettati nei slot di UN SOLO ciphertext (invece di un
ciphertext per canale, come oggi), si puo' calcolare l'intero
prodotto matrice-vettore y=Wx con sole O(Cin) rotazioni+moltiplicazioni
PLAINTEXT (economiche), invece di O(Cin*Cout) moltiplicazioni
CIPHERTEXT-SCALARE.

Idea: la diagonale d di W e' il vettore [W[0,d], W[1,d+1], ...]
(con wraparound). Allora:
    y = sum_{d=0}^{Cin-1} diag_d * rotate(x, d)
dove diag_d e' un vettore PLAINTEXT di lunghezza Cout (richiede
Cout=Cin per la forma piu' semplice; per Cout!=Cin serve il padding
o la variante rettangolare, vedi sotto).

Questo script:
1. Implementa la versione "quadrata" (Cin==Cout) e la verifica contro
   il prodotto matriciale diretto (numpy), su dimensioni piccole.
2. Implementa la variante rettangolare (Cin!=Cout, il caso vero nella
   rete: es. enc1 ha Cin=32,Cout=64) via padding alla dimensione
   massima, e la verifica allo stesso modo.
3. Conta le operazioni di entrambi gli approcci (diretto vs diagonale)
   per farsi un'idea della riduzione attesa PRIMA di portarlo in HE.

Uso: python3 crypto/prototype_diagonal_matvec.py
"""

import numpy as np


def direct_matvec(W, x):
    """Il modo 'diretto' -- quello che fa oggi la pipeline HE:
    Cout*Cin moltiplicazioni scalari + altrettante somme."""
    Cout, Cin = W.shape
    y = np.zeros(Cout)
    n_mults = 0
    for co in range(Cout):
        for ci in range(Cin):
            y[co] += W[co, ci] * x[ci]
            n_mults += 1
    return y, n_mults


def diagonal_matvec_square(W, x):
    """Metodo delle diagonali, caso quadrato (Cin == Cout).
    Simula esattamente cosa farebbero rotate+mult+add in HE:
    ogni 'rotazione' qui e' np.roll (equivalente in chiaro a
    EvalRotate), ogni moltiplicazione e' elemento-per-elemento con
    un vettore PLAINTEXT (la diagonale), economica in HE."""
    n = W.shape[0]
    assert W.shape[1] == n, "versione quadrata: serve Cin==Cout"
    y = np.zeros(n)
    n_rotations = 0
    n_plaintext_mults = 0
    for d in range(n):
        diag = np.array([W[i, (i + d) % n] for i in range(n)])
        x_rotated = np.roll(x, -d)  # equivalente a EvalRotate(x, d)
        y += diag * x_rotated
        n_rotations += 1 if d != 0 else 0
        n_plaintext_mults += 1
    return y, n_rotations, n_plaintext_mults


def diagonal_matvec_rectangular(W, x):
    """Variante per Cin != Cout (il caso vero nella rete, es.
    enc1: Cin=32 -> Cout=64). Si fa padding alla dimensione massima
    n = max(Cin, Cout), si applica il metodo quadrato, poi si
    ritaglia/estende il risultato alla vera lunghezza Cout."""
    Cout, Cin = W.shape
    n = max(Cin, Cout)

    W_padded = np.zeros((n, n))
    W_padded[:Cout, :Cin] = W

    x_padded = np.zeros(n)
    x_padded[:Cin] = x

    y_padded, n_rot, n_mult = diagonal_matvec_square(W_padded, x_padded)
    return y_padded[:Cout], n_rot, n_mult


def main():
    print("=== Test 1: caso quadrato, Cin=Cout=8 ===")
    rng = np.random.default_rng(42)
    n = 8
    W = rng.normal(size=(n, n))
    x = rng.normal(size=n)

    y_direct, n_mults_direct = direct_matvec(W, x)
    y_diag, n_rot, n_pmult = diagonal_matvec_square(W, x)

    err = np.max(np.abs(y_direct - y_diag))
    print(f"Errore max tra diretto e diagonale: {err:.2e}")
    print(f"Diretto: {n_mults_direct} moltiplicazioni ciphertext-scalare")
    print(f"Diagonale: {n_rot} rotazioni + {n_pmult} moltiplicazioni PLAINTEXT (economiche)")
    assert err < 1e-10, "ERRORE: le due versioni non coincidono!"
    print("=> IDENTICHE.\n")

    print("=== Test 2: caso rettangolare, Cin=32 -> Cout=64 (come enc1 vero) ===")
    Cin, Cout = 32, 64
    W = rng.normal(size=(Cout, Cin))
    x = rng.normal(size=Cin)

    y_direct, n_mults_direct = direct_matvec(W, x)
    y_diag, n_rot, n_pmult = diagonal_matvec_rectangular(W, x)

    err = np.max(np.abs(y_direct - y_diag))
    print(f"Errore max tra diretto e diagonale: {err:.2e}")
    print(f"Diretto: {n_mults_direct} moltiplicazioni ciphertext-scalare")
    print(f"Diagonale: {n_rot} rotazioni + {n_pmult} moltiplicazioni PLAINTEXT")
    assert err < 1e-10, "ERRORE: le due versioni non coincidono!"
    print("=> IDENTICHE.\n")

    print("=== Proiezione sui canali reali della rete (per UNA posizione di kernel) ===")
    print(f"{'Stage':<8} {'Cin':>5} {'Cout':>5} {'Diretto (ciphertext-mult)':>26} {'Diagonale (rotazioni)':>23}")
    real_stages = [
        ("enc0", 1, 32), ("enc1", 32, 64), ("enc2", 64, 128),
        ("enc3", 128, 256), ("enc4", 256, 128),
        ("dec3", 256, 256), ("dec2", 128, 128), ("dec1", 64, 64), ("dec0", 32, 32),
    ]
    tot_direct, tot_diag = 0, 0
    for name, cin, cout in real_stages:
        n = max(cin, cout)
        direct_cost = cin * cout
        diag_cost = n  # rotazioni, per UNA posizione di kernel (x9 per il kernel 3x3 completo)
        tot_direct += direct_cost
        tot_diag += diag_cost
        print(f"{name:<8} {cin:>5} {cout:>5} {direct_cost:>26} {diag_cost:>23}")
    print(f"\nTotale (1 convoluzione per stage, 1 posizione kernel su 9):")
    print(f"  Diretto:   {tot_direct}")
    print(f"  Diagonale: {tot_diag}")
    print(f"  Riduzione: {tot_direct/tot_diag:.1f}x")
    print(f"\n(Il fattore 9 del kernel si applica a entrambi allo stesso modo -- non cambia il rapporto)")


if __name__ == '__main__':
    main()