"""
crypto/bootstrap_utils.py

Gestione del bootstrap DURANTE l'esecuzione reale della pipeline, non
piu' uno schema statico precalcolato (bootstrap_placement_kstage.py
resta utile per STIMARE quanti bootstrap serviranno in totale, ma per
il codice vero e' piu' robusto controllare il livello REALE del
ciphertext -- si auto-corregge anche se le stime di profondita' fossero
leggermente sbagliate, come e' gia' successo con un file di calibrazione
corrotto).

API confermate su Zeus (vedi tests/diag_bootstrap.py):
- ct.GetLevel() -> livello attuale del ciphertext
- cc.EvalBootstrap(ct) -> nuovo ciphertext, livello ripristinato
"""


def ensure_level(cc, ct, min_remaining_levels, verbose=False):
    """
    Se il ciphertext ha meno di min_remaining_levels livelli residui
    prima di esaurire la profondita' del contesto, lo bootstrappa.
    Altrimenti lo ritorna invariato.

    min_remaining_levels: quanti livelli vuoi ANCORA disponibili prima
    della prossima operazione pesante (es. una convoluzione + norm
    completa) -- deve essere >= al costo della prossima sezione di
    pipeline che seguira', altrimenti rischi comunque di esaurire la
    profondita' PRIMA del prossimo controllo.

    NOTA: qui non conosciamo la profondita' massima del contesto
    direttamente da ct -- va passata separatamente se serve calcolarla
    come 'livelli residui = depth_totale - ct.GetLevel()'. Se invece
    GetLevel() restituisce gia' 'quanti livelli sono stati CONSUMATI'
    (coerente con l'uso visto in diag_bootstrap.py: 'Encrypt ok (level
    21 of 22)' poi dopo EvalBootstrap 'level 19'), il confronto va
    adattato di conseguenza -- VERIFICARE con un test rapido quale
    convenzione usa la vostra versione prima di fidarsi ciecamente
    della soglia sotto.
    """
    current_level = ct.GetLevel()
    if current_level >= min_remaining_levels:
        # NOTA: qui assumo GetLevel() crescente = piu' vicino
        # all'esaurimento (coerente con 'level 21 of 22' -> quasi
        # esaurito). Se la convenzione fosse invertita, il confronto
        # va capovolto -- vedi test diagnostico sotto.
        if verbose:
            print(f"  [bootstrap] livello {current_level} troppo alto (vicino al limite), bootstrap...")
        ct = cc.EvalBootstrap(ct)
        if verbose:
            print(f"  [bootstrap] fatto, nuovo livello: {ct.GetLevel()}")
    return ct


def ensure_level_list(cc, ct_list, min_remaining_levels, verbose=False):
    """Come ensure_level, ma su una lista di ciphertext (es. tutti i
    canali di uno stage) -- ciascuno controllato indipendentemente,
    dato che canali diversi potrebbero (in teoria) trovarsi a livelli
    leggermente diversi se le operazioni non sono state perfettamente
    identiche."""
    return [ensure_level(cc, ct, min_remaining_levels, verbose) for ct in ct_list]


def align_for_add(cc, ct_a, ct_b, verbose=False):
    """
    Prepara due ciphertext per una somma (es. skip connection): CKKS
    richiede che i due operandi siano allo STESSO livello. Se non lo
    sono, ABBASSA quello piu' 'fresco' (meno livelli consumati) fino al
    livello dell'altro, con moltiplicazioni per 1.0 (ciascuna consuma
    esattamente un livello via rescale, senza cambiare il valore).

    Se il divario fosse cosi' grande da non poter essere colmato senza
    esaurire la profondita' residua di quello piu' 'fresco', bootstrappa
    prima quello piu' 'consumato' (riportandolo in alto), poi riallinea.

    *** DA VERIFICARE: quale dei due (EvalAdd tra livelli diversi lancia
    errore, o la libreria lo gestisce da sola con un adjust automatico)
    -- prova prima SENZA questa funzione, un semplice cc.EvalAdd(ct_a,
    ct_b) con livelli diversi, e guarda se lancia un'eccezione chiara o
    un segfault silenzioso. Se lancia un'eccezione chiara, sappiamo che
    serve davvero allineare a mano; se funziona da sola, questa funzione
    non serve. ***
    """
    level_a, level_b = ct_a.GetLevel(), ct_b.GetLevel()
    if level_a == level_b:
        return ct_a, ct_b

    if verbose:
        print(f"  [align] livelli diversi: a={level_a}, b={level_b} -- allineamento...")

    # Abbassa quello con livello piu' basso (piu' 'fresco', meno consumato)
    # fino al livello di quello piu' alto (piu' 'vecchio', piu' consumato).
    if level_a < level_b:
        gap = level_b - level_a
        for _ in range(gap):
            ct_a = cc.EvalMult(ct_a, 1.0)  # consuma 1 livello via rescale, valore invariato
    else:
        gap = level_a - level_b
        for _ in range(gap):
            ct_b = cc.EvalMult(ct_b, 1.0)

    if verbose:
        print(f"  [align] fatto, nuovi livelli: a={ct_a.GetLevel()}, b={ct_b.GetLevel()}")
    return ct_a, ct_b