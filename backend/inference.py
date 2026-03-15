import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt

from backend.unet import build_unet

# ===============================
# CONFIG
# ===============================

WEIGHTS_PATH = "D:/Hackdata/model/best_model.weights.h5"

IMG_SIZE = 128
VOLUME_START_AT = 22
SLICE_IDX = 60


# ===============================
# PREPROCESS
# ===============================

def preprocess_scan(scan):

    X = np.empty((100, IMG_SIZE, IMG_SIZE, 2))

    for j in range(100):

        slice_data = scan[:, :, j + VOLUME_START_AT]

        resized = cv2.resize(slice_data, (IMG_SIZE, IMG_SIZE))

        normalized = resized / (resized.max() + 1e-6)

        X[j, :, :, 0] = normalized
        X[j, :, :, 1] = normalized

    return X


# ===============================
# VOLUME RATIO
# ===============================

def calculate_volume_ratios(predictions, scan):

    pred_classes = np.argmax(predictions, axis=-1)

    brain_mask = scan > (scan.max() * 0.1)
    brain_voxels = np.sum(brain_mask)

    necrotic_voxels = np.sum(pred_classes == 1)
    edema_voxels = np.sum(pred_classes == 2)
    enhancing_voxels = np.sum(pred_classes == 3)

    total_tumor_voxels = necrotic_voxels + edema_voxels + enhancing_voxels

    ratios = {
        "Total Tumor": (total_tumor_voxels / brain_voxels) * 100,
        "Necrotic": (necrotic_voxels / brain_voxels) * 100,
        "Edema": (edema_voxels / brain_voxels) * 100,
        "Enhancing": (enhancing_voxels / brain_voxels) * 100
    }

    return ratios


def visualize(scan, predictions, output_path="result.png"):

    original = cv2.resize(
        scan[:, :, SLICE_IDX + VOLUME_START_AT],
        (IMG_SIZE, IMG_SIZE),
    )

    pred = predictions[SLICE_IDX]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(pred[:, :, 1], cmap="Reds")
    axes[1].set_title("Necrotic")

    axes[2].imshow(pred[:, :, 2], cmap="Reds")
    axes[2].set_title("Edema")

    axes[3].imshow(pred[:, :, 3], cmap="Reds")
    axes[3].set_title("Enhancing")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_inference(file_path, output_path=None):

    # Build model
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 2))

    # Load weights
    model.load_weights(WEIGHTS_PATH)

    # Load MRI scan
    scan = nib.load(file_path).get_fdata()

    # Preprocess
    X = preprocess_scan(scan)

    # Run prediction
    predictions = model.predict(X)

    # Calculate ratios
    ratios = calculate_volume_ratios(predictions, scan)

    # Optional visualization
    if output_path:
        visualize(scan, predictions, output_path=output_path)

    return ratios