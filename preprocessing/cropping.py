import numpy as np
from scipy.ndimage import binary_fill_holes


def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    """
    Crea una maschera booleana dove i dati sono non-zero.

    L'input deve avere shape (C, X, Y, Z) — con la dimensione
    canale davanti, come vuole nnU-Net. Su ACDC abbiamo un solo
    canale MRI, quindi shape (1, X, Y, Z).

    binary_fill_holes riempie i buchi interni alla maschera.
    Questo è importante su ACDC: il sangue dentro il cuore
    può avere intensità bassa e sembrare background — fill_holes
    lo include correttamente nella maschera.
    """
    assert data.ndim in (3, 4), "data deve avere shape (C, X, Y, Z) o (C, X, Y)"

    # unione dei canali: un voxel è foreground se è non-zero
    # in almeno uno dei canali
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0

    return binary_fill_holes(nonzero_mask)


def get_bbox_from_mask(mask: np.ndarray):
    """
    Calcola la bounding box minima che contiene tutti i True nella maschera.

    Restituisce una lista di slice, una per ogni asse.
    Es. per una maschera 3D: [slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)]
    """
    bbox = []
    for ax in range(mask.ndim):
        # proietta la maschera su questo asse
        # np.any lungo tutti gli altri assi
        axes = tuple(i for i in range(mask.ndim) if i != ax)
        projection = np.any(mask, axis=axes)
        nonzero = np.where(projection)[0]
        bbox.append(slice(int(nonzero[0]), int(nonzero[-1]) + 1))
    return bbox


def crop_to_nonzero(data: np.ndarray, seg: np.ndarray = None, nonzero_label: int = -1):
    """
    Ritaglia data (e seg) alla bounding box dei voxel non-zero.

    Parametri
    ---------
    data          : array (C, X, Y, Z) — immagine
    seg           : array (1, X, Y, Z) — segmentazione (opzionale)
    nonzero_label : valore scritto nella seg per i voxel fuori
                    dalla nonzero mask. Default -1.
                    Questo -1 viene poi usato in ZScoreNormalization:
                    mask = seg >= 0  →  esclude i voxel fuori dal cuore

    Restituisce
    -----------
    data croppata, seg croppata (o generata), bbox usata
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)

    # applica la bbox alla maschera stessa
    slicer = tuple(bbox)
    nonzero_mask = nonzero_mask[slicer][None]  # (1, X', Y', Z')

    # applica la bbox a data e seg
    # (slice(None),) seleziona tutti i canali
    slicer_with_channel = (slice(None),) + slicer
    data = data[slicer_with_channel]

    if seg is not None:
        seg = seg[slicer_with_channel]
        # i voxel che erano background (0) e stanno fuori
        # dalla nonzero mask diventano -1
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        # se non abbiamo la seg (es. test set), la generiamo:
        # 0 dentro la maschera, -1 fuori
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))

    return data, seg, bbox
