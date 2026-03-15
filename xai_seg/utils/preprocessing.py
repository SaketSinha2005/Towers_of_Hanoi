"""
utils/preprocessing.py — NOT NEEDED for Kaggle H5 dataset
==========================================================
The awsaf49/brats2020-training-data Kaggle dataset is already:
  - Skull-stripped
  - Sliced into 2D axial slices
  - Saved as H5 files with "image" [4,240,240] and "mask" [240,240]

NO PREPROCESSING REQUIRED. This file is kept as a stub so imports
don't break, but the actual pipeline reads H5 files directly.

If you somehow have the raw NIfTI version, see the original preprocessing.py.
"""


def preprocess_all(*args, **kwargs):
    print(
        "\n[Preprocessing] SKIPPED.\n"
        "Your dataset (awsaf49/brats2020-training-data) is pre-processed H5 format.\n"
        "No preprocessing needed — the dataloader reads .h5 files directly.\n"
        "Jump straight to: python main.py --mode train\n"
    )