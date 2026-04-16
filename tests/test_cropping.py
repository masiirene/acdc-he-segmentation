import numpy as np
from preprocessing.cropping import create_nonzero_mask, get_bbox_from_mask, crop_to_nonzero


def test_create_nonzero_mask_base():
    """
    Caso semplice: un cubo 3D con un blocco non-zero al centro.
    La maschera deve essere True solo dove i dati sono non-zero.
    """
    data = np.zeros((1, 10, 10, 10), dtype=np.float32)
    data[0, 3:7, 3:7, 3:7] = 1.0

    mask = create_nonzero_mask(data)

    assert mask.shape == (10, 10, 10)
    assert mask[5, 5, 5] == True    # dentro il blocco
    assert mask[0, 0, 0] == False   # fuori dal blocco


def test_create_nonzero_mask_fill_holes():
    """
    Verifica che binary_fill_holes funzioni.
    Creiamo un cubo cavo (solo le pareti sono non-zero)
    e verifichiamo che l'interno venga riempito.
    """
    data = np.zeros((1, 10, 10, 10), dtype=np.float32)
    # pareti del cubo
    data[0, 3:7, 3:7, 3] = 1.0   # faccia anteriore
    data[0, 3:7, 3:7, 6] = 1.0   # faccia posteriore
    data[0, 3:7, 3,  3:7] = 1.0  # faccia sinistra
    data[0, 3:7, 6,  3:7] = 1.0  # faccia destra
    data[0, 3,   3:7, 3:7] = 1.0 # faccia top
    data[0, 6,   3:7, 3:7] = 1.0 # faccia bottom

    mask = create_nonzero_mask(data)

    # l'interno del cubo deve essere True grazie a fill_holes
    assert mask[4, 4, 4] == True
    # fuori dal cubo deve essere False
    assert mask[0, 0, 0] == False


def test_get_bbox_from_mask():
    """
    La bbox deve contenere esattamente il blocco non-zero.
    """
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:6, 3:7, 1:5] = True

    bbox = get_bbox_from_mask(mask)

    assert bbox[0] == slice(2, 6)
    assert bbox[1] == slice(3, 7)
    assert bbox[2] == slice(1, 5)


def test_crop_to_nonzero_con_seg():
    """
    Con segmentazione presente:
    - data viene croppato alla bbox
    - i voxel fuori dalla nonzero mask nella seg diventano -1
    """
    data = np.zeros((1, 10, 10, 10), dtype=np.float32)
    data[0, 3:7, 3:7, 3:7] = 1.0

    seg = np.zeros((1, 10, 10, 10), dtype=np.int8)
    seg[0, 4:6, 4:6, 4:6] = 1  # LV per esempio

    data_crop, seg_crop, bbox = crop_to_nonzero(data, seg)

    # la shape deve essere quella del blocco non-zero
    assert data_crop.shape == (1, 4, 4, 4)
    assert seg_crop.shape  == (1, 4, 4, 4)

    # dentro al blocco, dove la seg era 1, deve rimanere 1
    assert seg_crop[0, 1, 1, 1] == 1

    # dentro al blocco ma fuori dalla seg (era 0) → rimane 0
    # perché è dentro la nonzero mask
    assert seg_crop[0, 0, 0, 0] == 0


def test_crop_to_nonzero_senza_seg():
    """
    Senza segmentazione (test set):
    - deve generare una seg automatica
    - 0 dentro la nonzero mask, -1 fuori
    """
    data = np.zeros((1, 10, 10, 10), dtype=np.float32)
    data[0, 3:7, 3:7, 3:7] = 1.0

    data_crop, seg_crop, bbox = crop_to_nonzero(data, seg=None)

    assert data_crop.shape == (1, 4, 4, 4)
    # dentro la mask → 0
    assert seg_crop[0, 0, 0, 0] == 0
    # non ci sono -1 perché dopo il crop tutto è dentro la mask
    assert np.all(seg_crop >= 0)


def test_crop_to_nonzero_bbox_corretta():
    """
    Verifica che la bbox restituita sia corretta
    e possa essere usata per ricostruire la posizione originale.
    """
    data = np.zeros((1, 20, 20, 20), dtype=np.float32)
    data[0, 5:15, 5:15, 5:15] = 1.0

    _, _, bbox = crop_to_nonzero(data)

    assert bbox[0] == slice(5, 15)
    assert bbox[1] == slice(5, 15)
    assert bbox[2] == slice(5, 15)


if __name__ == "__main__":
    test_create_nonzero_mask_base()
    test_create_nonzero_mask_fill_holes()
    test_get_bbox_from_mask()
    test_crop_to_nonzero_con_seg()
    test_crop_to_nonzero_senza_seg()
    test_crop_to_nonzero_bbox_corretta()
    print("Tutti i test passati.")