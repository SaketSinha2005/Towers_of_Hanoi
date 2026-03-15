
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Configuration
IMG_SIZE = 128
VOLUME_START_AT = 22
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}


def preprocess_single_modality(scan_data):
    X = np.empty((100, IMG_SIZE, IMG_SIZE, 2))

    for j in range(100):
        slice_data = scan_data[:, :, j + VOLUME_START_AT]
        resized = cv2.resize(slice_data, (IMG_SIZE, IMG_SIZE))

        # Normalize
        if resized.max() > 0:
            normalized = resized / resized.max()
        else:
            normalized = resized

        # Duplicate to create 2 channels
        X[j, :, :, 0] = normalized
        X[j, :, :, 1] = normalized

    return X


def calculate_tumor_ratios(predictions, slice_idx, original_scan):
    """Calculate tumor ratios."""
    pred_classes = np.argmax(predictions[slice_idx], axis=-1)

    brain_slice = cv2.resize(
        original_scan[:, :, slice_idx + VOLUME_START_AT],
        (IMG_SIZE, IMG_SIZE)
    )
    brain_mask = brain_slice > (brain_slice.max() * 0.1)
    brain_pixels = np.sum(brain_mask)

    if brain_pixels == 0:
        return {'Total Tumor Ratio': 0.0, 'Enhancing Tumor Ratio': 0.0,
                'Tumor Core Ratio': 0.0, 'Edema Ratio': 0.0,
                'Brain Pixels': 0, 'Total Tumor Pixels': 0}

    necrotic_pixels = np.sum(pred_classes == 1)
    edema_pixels = np.sum(pred_classes == 2)
    enhancing_pixels = np.sum(pred_classes == 3)

    tumor_core_pixels = necrotic_pixels + enhancing_pixels
    total_tumor_pixels = necrotic_pixels + edema_pixels + enhancing_pixels

    return {
        'Total Tumor Ratio': (total_tumor_pixels / brain_pixels) * 100,
        'Enhancing Tumor Ratio': (enhancing_pixels / brain_pixels) * 100,
        'Tumor Core Ratio': (tumor_core_pixels / brain_pixels) * 100,
        'Edema Ratio': (edema_pixels / brain_pixels) * 100,
        'Brain Pixels': int(brain_pixels),
        'Total Tumor Pixels': int(total_tumor_pixels)
    }


def calculate_confidence_scores(predictions, slice_idx):
    """Calculate confidence scores."""
    slice_probs = predictions[slice_idx]
    pred_classes = np.argmax(slice_probs, axis=-1)

    confidence_scores = {}

    for class_id in [1, 2, 3]:
        class_mask = (pred_classes == class_id)

        if np.sum(class_mask) > 0:
            class_confidences = slice_probs[:, :, class_id][class_mask]
            confidence_scores[SEGMENT_CLASSES[class_id]] = {
                'mean': float(np.mean(class_confidences) * 100),
                'max': float(np.max(class_confidences) * 100),
                'pixels': int(np.sum(class_mask))
            }
        else:
            confidence_scores[SEGMENT_CLASSES[class_id]] = {
                'mean': 0.0, 'max': 0.0, 'pixels': 0
            }

    return confidence_scores


def predict_single_file(model, scan_data, slice_idx=60, save_path=None, filename="scan"):
    """Run inference on a single scan."""
    print("\n" + "="*80)
    print("SINGLE-FILE BRAIN TUMOR SEGMENTATION")
    print("="*80)

    # Preprocess
    print(f"Preprocessing...")
    X = preprocess_single_modality(scan_data)

    # Predict
    print("Running inference...")
    predictions = model.predict(X, verbose=0)
    print("✓ Complete!")

    # Get results
    pred_probs = predictions[slice_idx]

    # Metrics
    confidence_scores = calculate_confidence_scores(predictions, slice_idx)
    tumor_ratios = calculate_tumor_ratios(predictions, slice_idx, scan_data)

    # Visualization
    original_slice = cv2.resize(
        scan_data[:, :, slice_idx + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
    )
    original_norm = original_slice / (original_slice.max() + 1e-6)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    for ax in axes:
        ax.imshow(original_norm, cmap='gray')
        ax.axis('off')

    axes[0].set_title('Original MRI', fontsize=12, fontweight='bold')

    axes[1].text(64, 64, 'Ground Truth\nNot Available',
                 ha='center', va='center', fontsize=10, color='white',
                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    axes[1].set_title('Ground truth (N/A)', fontsize=12, fontweight='bold')

    axes[2].imshow(pred_probs[:, :, 1:4], cmap='Reds', alpha=0.3)
    axes[2].set_title(f'All classes\nTumor: {tumor_ratios["Total Tumor Ratio"]:.1f}%',
                      fontsize=11, fontweight='bold')

    axes[3].imshow(pred_probs[:, :, 1], cmap='OrRd', alpha=0.3)
    axes[3].set_title(f'{SEGMENT_CLASSES[1]}\nConf: {confidence_scores[SEGMENT_CLASSES[1]]["mean"]:.1f}%',
                      fontsize=11, fontweight='bold')

    axes[4].imshow(pred_probs[:, :, 2], cmap='OrRd', alpha=0.3)
    axes[4].set_title(f'{SEGMENT_CLASSES[2]}\nRatio: {tumor_ratios["Edema Ratio"]:.1f}%',
                      fontsize=11, fontweight='bold')

    axes[5].imshow(pred_probs[:, :, 3], cmap='OrRd', alpha=0.3)
    axes[5].set_title(f'{SEGMENT_CLASSES[3]}\nRatio: {tumor_ratios["Enhancing Tumor Ratio"]:.1f}%',
                      fontsize=11, fontweight='bold')

    fig.suptitle(f'{filename} | Slice: {slice_idx + VOLUME_START_AT}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")

    plt.close()

    # Print metrics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nTumor Ratios:")
    print(f"  Total: {tumor_ratios['Total Tumor Ratio']:.2f}%")
    print(f"  Enhancing: {tumor_ratios['Enhancing Tumor Ratio']:.2f}%")
    print(f"  Core: {tumor_ratios['Tumor Core Ratio']:.2f}%")
    print(f"  Edema: {tumor_ratios['Edema Ratio']:.2f}%")

    print(f"\nConfidence:")
    for name, scores in confidence_scores.items():
        print(f"  {name}: {scores['mean']:.1f}% (pixels: {scores['pixels']})")

    print("="*80)

    return predictions, confidence_scores, tumor_ratios


# For use as a module
def run_inference(model, nii_data, slice_idx=60):
    """Simple function for web integration."""
    X = preprocess_single_modality(nii_data)
    predictions = model.predict(X, verbose=0)
    confidence = calculate_confidence_scores(predictions, slice_idx)
    ratios = calculate_tumor_ratios(predictions, slice_idx, nii_data)
    return predictions, confidence, ratios