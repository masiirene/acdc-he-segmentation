"""
crypto/prototype_diagonal_conv_spatial.py

PROTOTIPO IN CHIARO (numpy puro) -- estende prototype_diagonal_matvec.py
aggiungendo la dimensione SPAZIALE (il kernel 3x3, un'immagine vera),
non piu' solo un singolo valore per canale.

Schema di impacchettamento proposto: "pixel-major" -- invece di un
ciphertext per canale con l'intera immagine dentro (schema attuale),
un ciphertext per OGNI PIXEL, con tutti i Cin valori di canale di
quel pixel impacchettati nei suoi slot. La convoluzione spaziale
(spostare la "finestra" del kernel) diventa allora un'operazione sui
singoli PIXEL (quali ciphertext-pixel combinare), mentre la mescolanza
dei canali per ciascuna posizione di kernel usa il metodo delle
diagonali (gia' verificato in prototype_diagonal_matvec.py).

Questo script:
1. Implementa conv2d con lo schema "pixel-major" in numpy puro
   (ogni pixel e' un vettore di lunghezza Cin, la convoluzione scorre
   sui pixel vicini usando gli stessi indici che userebbe
   conv2d_multichannel_fhe, applicando pero' il metodo delle diagonali
   per la combinazione dei canali invece del ciclo diretto).
2. Verifica che dia lo STESSO risultato della convoluzione diretta
   (quella che gia' avete, in forma numpy equivalente), su un'immagine
   piccola.
3. Conta le operazioni di entrambi gli approcci.

Uso: python3 crypto/prototype_diagonal_conv_spatial.py
"""

import numpy as np


def conv2d_direct(x, W, b, K=3):
    """
    Convoluzione diretta, equivalente in chiaro a
    conv2d_multichannel_fhe: x ha forma (Cin, Hp, Wp) -- GIA' con
    l'alone (padding) -- W ha forma (Cout, Cin, K, K). Restituisce
    (Cout, Hp, Wp) (stessa dimensione, bordo "sporco" dal wraparound
    circolare -- esattamente come nella versione HE vera).
    """
    Cin, Hp, Wp = x.shape
    Cout = W.shape[0]
    out = np.zeros((Cout, Hp, Wp))
    n_mults = 0
    for co in range(Cout):
        for ci in range(Cin):
            for ky in range(K):
                for kx in range(K):
                    shifted = np.roll(np.roll(x[ci], -ky, axis=0), -kx, axis=1)
                    out[co] += W[co, ci, ky, kx] * shifted
                    n_mults += 1
    for co in range(Cout):
        out[co] += b[co]
    return out, n_mults


def conv2d_pixel_major_diagonal(x, W, b, K=3):
    """
    Stesso calcolo, ma riorganizzato: per OGNI posizione di kernel
    (ky,kx), la mescolanza Cin->Cout e' fatta con il metodo delle
    diagonali SU TUTTI I PIXEL SIMULTANEAMENTE (sfruttando il fatto
    che numpy, come CKKS in SIMD, applica un'operazione vettoriale a
    tutti gli elementi insieme) -- la parte "costosa" (le Cin*Cout
    combinazioni) diventa O(n) diagonali invece di O(Cin*Cout) termini,
    MOLTIPLICATA per il numero di pixel (che pero' viaggiano insieme
    in un colpo solo via SIMD, non serve un ciclo pixel per pixel).
    """
    Cin, Hp, Wp = x.shape
    Cout = W.shape[0]
    n = max(Cin, Cout)

    # Padding della matrice dei pesi, come nel prototipo precedente,
    # per ogni posizione di kernel separatamente.
    out = np.zeros((Cout, Hp, Wp))
    n_rotations = 0
    n_plaintext_mults = 0

    for ky in range(K):
        for kx in range(K):
            # 1. Sposta spazialmente TUTTI i canali per questa posizione
            #    di kernel (equivalente a ruotare ogni ciphertext-canale
            #    di offset=ky*Wp+kx, come gia' fa la versione originale)
            x_shifted = np.stack([
                np.roll(np.roll(x[ci], -ky, axis=0), -kx, axis=1)
                for ci in range(Cin)
            ])  # (Cin, Hp, Wp)

            # 2. Mescola i canali con il metodo delle diagonali PER
            #    QUESTA posizione di kernel -- applicato a TUTTI i
            #    pixel insieme (broadcasting numpy = SIMD in CKKS)
            W_k = W[:, :, ky, kx]  # (Cout, Cin) -- la matrice per
                                    # questa sola posizione di kernel
            W_padded = np.zeros((n, n))
            W_padded[:Cout, :Cin] = W_k

            x_padded = np.zeros((n, Hp, Wp))
            x_padded[:Cin] = x_shifted

            y_padded = np.zeros((n, Hp, Wp))
            for d in range(n):
                diag = np.array([W_padded[i, (i + d) % n] for i in range(n)])
                x_rot = np.roll(x_padded, -d, axis=0)  # rotazione lungo l'asse CANALE
                y_padded += diag[:, None, None] * x_rot
                n_rotations += 1 if d != 0 else 0
                n_plaintext_mults += 1

            out += y_padded[:Cout]

    for co in range(Cout):
        out[co] += b[co]
    return out, n_rotations, n_plaintext_mults


def main():
    print("=== Test: convoluzione 2D con canali, schema diretto vs pixel-major+diagonale ===\n")

    rng = np.random.default_rng(7)
    Cin, Cout = 6, 10
    H, W_img = 12, 10  # immagine piccola apposta, veloce da testare
    halo = 1
    K = 3
    Hp, Wp = H + 2*halo, W_img + 2*halo

    x = rng.normal(size=(Cin, H, W_img))
    x_padded = np.pad(x, ((0, 0), (0, 2*halo), (0, 2*halo)))

    weight = rng.normal(size=(Cout, Cin, K, K)) * 0.1
    bias = rng.normal(size=(Cout,)) * 0.05

    print(f"Immagine: Cin={Cin}, Cout={Cout}, {H}x{W_img} (+halo={halo})\n")

    out_direct, n_mults_direct = conv2d_direct(x_padded, weight, bias, K=K)
    out_diag, n_rot, n_pmult = conv2d_pixel_major_diagonal(x_padded, weight, bias, K=K)

    err = np.max(np.abs(out_direct - out_diag))
    print(f"Errore massimo tra le due versioni: {err:.2e}")
    print(f"Diretto: {n_mults_direct} moltiplicazioni ciphertext-scalare (per TUTTI i pixel insieme via SIMD)")
    print(f"Pixel-major+diagonale: {n_rot} rotazioni + {n_pmult} moltiplicazioni PLAINTEXT")

    if err < 1e-10:
        print("\n=== IDENTICHE: la logica regge anche con la dimensione spaziale. ===")
    else:
        print("\n=== ATTENZIONE: differenza significativa -- la logica ha un bug, da correggere prima di procedere. ===")

    print(f"\n=== Proiezione sui canali reali (1 convoluzione completa, 9 posizioni di kernel) ===")
    real_stages = [
        ("enc0", 1, 32), ("enc1", 32, 64), ("enc2", 64, 128),
        ("enc3", 128, 256), ("enc4", 256, 128),
        ("dec3", 256, 256), ("dec2", 128, 128), ("dec1", 64, 64), ("dec0", 32, 32),
    ]
    tot_direct, tot_diag = 0, 0
    for name, cin, cout in real_stages:
        n = max(cin, cout)
        direct_cost = cin * cout * 9
        diag_cost = n * 9
        tot_direct += direct_cost
        tot_diag += diag_cost
    print(f"Diretto (come oggi):      {tot_direct} operazioni costose (rotazioni)")
    print(f"Pixel-major + diagonale:  {tot_diag} operazioni costose (rotazioni)")
    print(f"Riduzione: {tot_direct/tot_diag:.1f}x")
    print(f"\nNOTA IMPORTANTE: questo conta le rotazioni, non il tempo reale --")
    print(f"il test di stamattina su Zeus ha mostrato che MENO rotazioni non")
    print(f"garantisce automaticamente PIU' veloce, se lo schema di dati")
    print(f"richiede di tenere troppi ciphertext vivi insieme in memoria.")
    print(f"Questo schema pixel-major e' DIVERSO da quello di stamattina:")
    print(f"qui si ruota lungo l'asse CANALE (dentro un ciphertext che")
    print(f"impacchetta i canali), non si tengono N ciphertext separati")
    print(f"vivi insieme -- ma va comunque verificato con attenzione quanti")
    print(f"ciphertext distinti servono per pixel/gruppi di pixel prima di")
    print(f"fidarsi che risolva anche il problema di memoria.")


if __name__ == '__main__':
    main()