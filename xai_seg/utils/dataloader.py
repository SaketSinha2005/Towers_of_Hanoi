"""
utils/dataloader.py — BraTS2020 Kaggle H5 Dataset Loader
========================================================
Handles the awsaf49/brats2020-training-data Kaggle dataset format:

  data/raw/BraTS2020_training_data/data/data/
    volume_1_slice_0.h5
    volume_1_slice_1.h5
    ...
    volume_369_slice_154.h5

Each .h5 file contains:
  "image" : float32 [4, 240, 240]  — 4 MRI modalities (FLAIR, T1, T1ce, T2)
  "mask"  : float32 [240, 240]     — labels: 0=bg, 1=necrotic, 2=edema, 4=enhancing

Key advantages over NIfTI pipeline:
  - No preprocessing step needed (already sliced + normalized by Kaggle)
  - Much faster loading (small h5 files vs full 3D volumes)
  - Split is done at VOLUME level to prevent data leakage
"""

import os
import re
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# --- Label remapping ---------------------------------------------------------
# BraTS labels: {0, 1, 2, 4} -> {0, 1, 2, 3}  (close the gap at index 4)
def remap_labels(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.int64)
    out[mask == 0] = 0
    out[mask == 1] = 1
    out[mask == 2] = 2
    out[mask == 4] = 3
    return out


# --- Per-slice normalization -------------------------------------------------
def normalize_slice(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each modality channel independently.
    Ignores zero-background pixels (skull-stripped brain mask).
    image: [4, H, W]  ->  [4, H, W] float32
    """
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        ch   = image[c]
        mask = ch > 0
        if mask.sum() == 0:
            continue
        mu    = ch[mask].mean()
        sigma = ch[mask].std()
        if sigma < 1e-8:
            sigma = 1.0
        out[c] = np.where(mask, (ch - mu) / sigma, 0.0)
    return out


# --- Robust mask flattening --------------------------------------------------
def flatten_mask(mask: np.ndarray) -> np.ndarray:
    """
    Robustly convert any mask shape to 2D [H, W].

    Handles all possible H5 mask layouts:
      [H, W]        -> return as-is
      [1, H, W]     -> squeeze dim 0
      [H, W, 1]     -> squeeze dim -1
      [H, W, C]     -> take argmax over channels (one-hot encoded mask)
      [C, H, W]     -> take argmax over channels (one-hot encoded mask)

    The BraTS Kaggle dataset (awsaf49) stores masks as [H, W, 4] one-hot
    encoded tensors for some versions, which is why the raw mask shape
    arrives as [240, 240, 3] or [240, 240, 4] at the DataLoader.
    Taking argmax over the channel axis recovers the class-index mask.
    """
    mask = np.array(mask, dtype=np.float32)

    if mask.ndim == 2:
        # Already [H, W] — perfect
        return mask

    if mask.ndim == 3:
        # Case 1: [1, H, W]
        if mask.shape[0] == 1:
            return mask.squeeze(0)

        # Case 2: [H, W, 1]
        if mask.shape[-1] == 1:
            return mask.squeeze(-1)

        # Case 3: [H, W, C] — channels last, C > 1
        #   e.g. [240, 240, 3] or [240, 240, 4] — one-hot or multi-channel
        #   argmax over last axis -> [H, W] class indices
        if mask.shape[-1] < mask.shape[0]:
            return np.argmax(mask, axis=-1).astype(np.float32)

        # Case 4: [C, H, W] — channels first, C > 1
        #   e.g. [3, 240, 240] or [4, 240, 240]
        #   argmax over first axis -> [H, W] class indices
        if mask.shape[0] < mask.shape[1]:
            return np.argmax(mask, axis=0).astype(np.float32)

    # Fallback: squeeze everything to 2D by taking first channel
    # This should never be reached for valid BraTS data
    while mask.ndim > 2:
        mask = mask[0]
    return mask


# --- Dataset -----------------------------------------------------------------
class BraTS2020H5Dataset(Dataset):
    """
    Loads 2D slices directly from pre-sliced H5 files.

    Each __getitem__ returns:
        image  : Tensor [4, H, W]   float32  (4 MRI modalities)
        mask   : Tensor [H, W]      int64    (class indices 0-3)
        volume : int                (volume/patient index)
        slice  : int                (slice index within that volume)
    """

    def __init__(self, file_paths: list, augment: bool = False):
        self.files   = sorted(file_paths)
        self.augment = augment
        print(f"  [Dataset] {len(self.files)} slices | augment={augment}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]

        with h5py.File(fpath, "r") as f:
            image = f["image"][:]   # may be [H, W, 4] or [4, H, W]
            mask  = f["mask"][:]    # may be [H, W], [H, W, C], [C, H, W], etc.

        image = image.astype(np.float32)

        # ── Fix image layout ────────────────────────────────────────────────
        # H5 files store channels LAST: [H, W, 4] — convert to [4, H, W]
        if image.ndim == 3 and image.shape[-1] == 4:
            image = np.transpose(image, (2, 0, 1))   # [H, W, 4] -> [4, H, W]
        elif image.ndim == 3 and image.shape[0] == 4:
            pass   # already [4, H, W]
        else:
            raise ValueError(
                f"Unexpected image shape {image.shape} in {fpath}. "
                f"Expected [H, W, 4] or [4, H, W]."
            )

        # Normalize image channels
        image = normalize_slice(image)

        # ── Fix mask layout — robustly flatten to [H, W] ────────────────────
        # This handles ALL known BraTS Kaggle H5 mask formats:
        #   [H, W], [1, H, W], [H, W, 1], [H, W, C], [C, H, W]
        mask = flatten_mask(mask)   # -> [H, W] float32

        # ── Resize if needed ─────────────────────────────────────────────────
        if image.shape[1:] != tuple(config.SLICE_SIZE):
            image = self._resize_image(image, config.SLICE_SIZE)
            mask  = self._resize_mask(mask,   config.SLICE_SIZE)

        # ── Remap BraTS labels 0,1,2,4 -> 0,1,2,3 ───────────────────────────
        mask = remap_labels(mask.astype(np.int64))   # [H, W] int64

        # ── Sanity check ─────────────────────────────────────────────────────
        assert mask.ndim == 2, (
            f"Mask must be 2D after processing, got shape {mask.shape} "
            f"from file {fpath}"
        )

        img_t  = torch.from_numpy(image)           # [4, H, W]  float32
        mask_t = torch.from_numpy(mask).long()     # [H, W]     int64

        if self.augment:
            img_t, mask_t = self._augment(img_t, mask_t)

        # ── Parse volume and slice index from filename ───────────────────────
        fname = os.path.basename(fpath)
        nums  = re.findall(r"\d+", fname)
        vol   = int(nums[0]) if len(nums) > 0 else 0
        slc   = int(nums[1]) if len(nums) > 1 else 0

        return {
            "image":   img_t,
            "mask":    mask_t,
            "volume":  vol,
            "slice":   slc,
            "patient": f"volume_{vol}",   # compatibility key for XAI code
        }

    @staticmethod
    def _resize_image(arr: np.ndarray, size: tuple) -> np.ndarray:
        from skimage.transform import resize
        return np.stack([
            resize(arr[c], size, order=1, preserve_range=True,
                   anti_aliasing=True).astype(np.float32)
            for c in range(arr.shape[0])
        ], axis=0)

    @staticmethod
    def _resize_mask(arr: np.ndarray, size: tuple) -> np.ndarray:
        from skimage.transform import resize
        return resize(arr, size, order=0, preserve_range=True,
                      anti_aliasing=False).astype(np.float32)

    def _augment(self, img: torch.Tensor, mask: torch.Tensor):
        # Random horizontal flip
        if random.random() < config.FLIP_PROB:
            img  = TF.hflip(img)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        # Random vertical flip
        if random.random() < 0.3:
            img  = TF.vflip(img)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        # Random rotation
        if random.random() < config.ROTATE_PROB:
            angle = random.uniform(-config.ROTATE_DEGREES, config.ROTATE_DEGREES)
            img   = TF.rotate(img, angle,
                              interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.rotate(mask.unsqueeze(0), angle,
                              interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        # Brightness / contrast jitter (image only)
        if random.random() < config.BRIGHTNESS_PROB:
            img = img * random.uniform(0.8, 1.2)

        # Gaussian noise
        if random.random() < config.GAUSSIAN_PROB:
            img = img + torch.randn_like(img) * 0.05

        return img, mask


# --- Factory -----------------------------------------------------------------
def get_dataloaders(data_root: str = None):
    """
    1. Scans data_root for all volume_*_slice_*.h5 files
    2. Groups by volume index
    3. Splits volumes (not slices) into train/val/test  <- prevents leakage
    4. Returns DataLoaders

    The split is volume-level so no slices from the same patient
    appear in both train and val/test sets.
    """
    if data_root is None:
        data_root = config.DATA_RAW

    # Find all h5 files
    pattern = os.path.join(data_root, "volume_*_slice_*.h5")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        raise FileNotFoundError(
            f"\n[DataLoader] No h5 files found at:\n  {pattern}\n\n"
            f"Expected files like: volume_1_slice_0.h5\n"
            f"Current DATA_RAW in config.py: {config.DATA_RAW}\n\n"
            f"Fix: update DATA_RAW in config.py to point to the folder "
            f"containing volume_*_slice_*.h5 files."
        )

    print(f"[DataLoader] Found {len(all_files):,} h5 slice files")

    # Group files by volume index
    volume_map: dict = {}
    for fpath in all_files:
        fname = os.path.basename(fpath)
        nums  = re.findall(r"\d+", fname)
        vol   = int(nums[0]) if nums else 0
        volume_map.setdefault(vol, []).append(fpath)

    all_volumes = sorted(volume_map.keys())
    print(f"[DataLoader] Unique volumes (patients): {len(all_volumes)}")

    # Volume-level split (prevents data leakage)
    train_vols, test_vols = train_test_split(
        all_volumes,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED
    )
    train_vols, val_vols = train_test_split(
        train_vols,
        test_size=config.VAL_SPLIT / (1 - config.TEST_SPLIT),
        random_state=config.RANDOM_SEED
    )

    # Expand volumes -> file lists
    def vol_to_files(vols):
        files = []
        for v in vols:
            files.extend(volume_map[v])
        return files

    train_files = vol_to_files(train_vols)
    val_files   = vol_to_files(val_vols)
    test_files  = vol_to_files(test_vols)

    print(f"  Train: {len(train_vols)} volumes | {len(train_files):,} slices")
    print(f"  Val  : {len(val_vols)}  volumes | {len(val_files):,} slices")
    print(f"  Test : {len(test_vols)}  volumes | {len(test_files):,} slices")

    train_ds = BraTS2020H5Dataset(train_files, augment=config.AUGMENT_TRAIN)
    val_ds   = BraTS2020H5Dataset(val_files,   augment=False)
    test_ds  = BraTS2020H5Dataset(test_files,  augment=False)

    mp_context = "spawn" if config.NUM_WORKERS > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=False,
        multiprocessing_context=mp_context,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=False,
        multiprocessing_context=mp_context,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, test_vols