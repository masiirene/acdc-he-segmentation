import os
import json
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


def zscore_normalize(img: np.ndarray) -> np.ndarray:
    """Z-score normalization on nonzero voxels (same as nnU-Net)."""
    mask = img != 0
    if mask.sum() == 0:
        return img
    mean = img[mask].mean()
    std  = img[mask].std()
    return (img - mean) / (std + 1e-8)


def pad_or_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center pad or crop a 2D slice to target size."""
    h, w = img.shape
    # Pad if smaller
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2),
                           (pad_w//2, pad_w - pad_w//2)), mode='constant')
    # Crop if larger
    h, w = img.shape
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return img[start_h:start_h+target_h, start_w:start_w+target_w]


class ACDCDataset(Dataset):
    """
    ACDC 2D slice dataset.

    Loads ED and ES frames for each patient, extracts 2D slices,
    applies Z-score normalization and pad/crop to patch_size.

    Args:
        data_dir:   path to ACDC training/ folder
        case_list:  list of case names to include (e.g. ['patient001_frame01'])
        patch_size: (H, W) tuple, default (256, 224)
        augment:    if True, applies random horizontal flip
    """

    def __init__(self, data_dir: str, case_list: list,
                 patch_size=(256, 224), augment=False):
        self.data_dir   = data_dir
        self.patch_size = patch_size
        self.augment    = augment
        self.slices     = []  # list of (img_path, seg_path, slice_idx)

        for case in case_list:
            # case = 'patient001_frame01'
            patient_id = case[:10]       # 'patient001'
            frame_part = case[11:]       # 'frame01'
            frame_num  = frame_part.replace('frame', '')  # '01'

            patient_dir = os.path.join(data_dir, patient_id)
            img_path = os.path.join(patient_dir,
                                    f'{patient_id}_frame{frame_num}.nii.gz')
            seg_path = os.path.join(patient_dir,
                                    f'{patient_id}_frame{frame_num}_gt.nii.gz')

            if not os.path.exists(img_path) or not os.path.exists(seg_path):
                continue

            img = nib.load(img_path).get_fdata()
            n_slices = img.shape[2]
            for s in range(n_slices):
                self.slices.append((img_path, seg_path, s))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, seg_path, s = self.slices[idx]

        img = nib.load(img_path).get_fdata()[:, :, s].astype(np.float32)
        seg = nib.load(seg_path).get_fdata()[:, :, s].astype(np.int64)

        # Normalize
        img = zscore_normalize(img)

        # Pad/crop to patch size
        img = pad_or_crop(img, *self.patch_size)
        seg = pad_or_crop(seg, *self.patch_size)

        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            seg = np.fliplr(seg).copy()

        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        seg = torch.from_numpy(seg)               # (H, W)

        return img, seg


def load_splits(splits_path: str, fold: int = 0):
    """Load train/val case lists from splits_final.json."""
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    return splits[fold]['train'], splits[fold]['val']