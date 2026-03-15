import numpy as np
import cv2
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate

# ── CONFIG — edit these ──────────────────────────────────────────────────────
FLAIR_PATH   = r"D:/Hackdata/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_006/BraTS20_Validation_006_flair.nii"
T1CE_PATH    = r"D:/Hackdata/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_006/BraTS20_Validation_006_t1ce.nii"
WEIGHTS_PATH = r"D:/Hackdata/model/best_model.weights.h5"
OUTPUT_PATH  = "result.png"

IMG_SIZE        = 128
VOLUME_START_AT = 22
VOLUME_SLICES   = 100
DISPLAY_SLICE   = 60   # which slice to visualize (0–99)
# ─────────────────────────────────────────────────────────────────────────────


def build_unet(input_shape=(128, 128, 2)):
    from tensorflow.keras import backend as K
    inputs = Input(input_shape)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)
    c4 = Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(256, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)
    c5 = Conv2D(512, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(512, 3, activation='relu', padding='same')(c5)
    c5 = Dropout(0.2)(c5)
    u6 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D()(c5))
    u6 = concatenate([c4, u6])
    c6 = Conv2D(256, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(256, 3, activation='relu', padding='same')(c6)
    u7 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D()(c6))
    u7 = concatenate([c3, u7])
    c7 = Conv2D(128, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(128, 3, activation='relu', padding='same')(c7)
    u8 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D()(c7))
    u8 = concatenate([c2, u8])
    c8 = Conv2D(64, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(64, 3, activation='relu', padding='same')(c8)
    u9 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D()(c8))
    u9 = concatenate([c1, u9])
    c9 = Conv2D(32, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(32, 3, activation='relu', padding='same')(c9)
    out = Conv2D(4, (1, 1), activation='softmax')(c9)
    return Model(inputs, out)


def load_and_preprocess(flair_path, t1ce_path):
    scaler = MinMaxScaler()

    print("Loading FLAIR...")
    flair = nib.load(flair_path).get_fdata()
    flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)

    print("Loading T1CE...")
    t1ce = nib.load(t1ce_path).get_fdata()
    t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

    available = min(VOLUME_SLICES, flair.shape[2] - VOLUME_START_AT)
    print(f"Using {available} slices starting from slice {VOLUME_START_AT}")

    X = np.zeros((available, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)
    for j in range(available):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT].astype(np.float32), (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(t1ce[:, :, j + VOLUME_START_AT].astype(np.float32),  (IMG_SIZE, IMG_SIZE))

    return X, flair, available


def generate_image(predictions, flair, num_slices, output_path):
    safe_idx = min(DISPLAY_SLICE, num_slices - 1)
    orig_z   = min(safe_idx + VOLUME_START_AT, flair.shape[2] - 1)

    # Background MRI slice
    bg = cv2.resize(flair[:, :, orig_z].astype(np.float32), (IMG_SIZE, IMG_SIZE))
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-6)

    pred      = predictions[safe_idx]     # (128, 128, 4)
    core      = pred[:, :, 1]
    edema     = pred[:, :, 2]
    enhancing = pred[:, :, 3]
    all_tumor = pred[:, :, 1:4]

    fig = plt.figure(figsize=(24, 10))
    fig.patch.set_facecolor("#0f1117")

    # ── TOP ROW: 6 context slices ────────────────────────────────────────────
    for col, ci in enumerate(np.linspace(0, num_slices - 1, 6, dtype=int)):
        ax  = fig.add_subplot(2, 6, col + 1)
        z   = min(ci + VOLUME_START_AT, flair.shape[2] - 1)
        t   = cv2.resize(flair[:, :, z].astype(np.float32), (IMG_SIZE, IMG_SIZE))
        t   = (t - t.min()) / (t.max() - t.min() + 1e-6)
        ax.imshow(t, cmap="gray")
        ax.set_title(f"z={z}" + (" ◀" if ci == safe_idx else ""), color="white", fontsize=9)
        ax.axis("off")
        if ci == safe_idx:
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor("#3b82f6"); s.set_linewidth(3)

    # ── BOTTOM ROW: segmentation overlays ───────────────────────────────────
    panels = [
        ("Original FLAIR",  None,      "gray"),
        ("All Classes",      all_tumor, None),
        ("Not Tumor",        pred[:,:,0], "Blues"),
        ("Necrotic/Core",   core,      "OrRd"),
        ("Edema",           edema,     "OrRd"),
        ("Enhancing",       enhancing, "OrRd"),
    ]
    for col, (title, overlay, cmap) in enumerate(panels):
        ax = fig.add_subplot(2, 6, col + 7)
        ax.imshow(bg, cmap="gray", interpolation="none")
        if overlay is not None:
            ax.imshow(overlay, cmap=cmap, interpolation="none", alpha=0.35)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"NeuroVision AI — Brain Tumour Segmentation  (axial z={orig_z})",
        color="white", fontsize=13, fontweight="bold", y=0.50
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nImage saved → {output_path}")


if __name__ == "__main__":
    # Optionally pass paths as command-line args:
    # python generate_image.py flair.nii t1ce.nii weights.h5 output.png
    flair_path   = sys.argv[1] if len(sys.argv) > 1 else FLAIR_PATH
    t1ce_path    = sys.argv[2] if len(sys.argv) > 2 else T1CE_PATH
    weights_path = sys.argv[3] if len(sys.argv) > 3 else WEIGHTS_PATH
    output_path  = sys.argv[4] if len(sys.argv) > 4 else OUTPUT_PATH

    print("Building model...")
    model = build_unet()
    model.load_weights(weights_path)
    print("Weights loaded.")

    X, flair, num_slices = load_and_preprocess(flair_path, t1ce_path)

    print("Running prediction...")
    predictions = model.predict(X, verbose=1)

    print("Generating image...")
    generate_image(predictions, flair, num_slices, output_path)