import os
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Import our custom modules
from preprocessing import (
    load_single_case,
    TRAIN_DATASET_PATH,
    VOLUME_START_AT,
    VOLUME_SLICES,
    IMG_SIZE,
    SEGMENT_CLASSES
)
from train import load_best_model


def predict_case(model, case_path, case_id):
    # Load and preprocess the case
    X = load_single_case(case_path, case_id)

    # Predict
    predictions = model.predict(X, verbose=1)

    return predictions


def visualize_prediction(case_path, case_id, predictions, slice_idx=60):
    # Load ground truth and original images
    gt_path = os.path.join(case_path, f'BraTS20_Training_{case_id}_seg.nii')
    flair_path = os.path.join(case_path, f'BraTS20_Training_{case_id}_flair.nii')

    gt = nib.load(gt_path).get_fdata()
    orig_image = nib.load(flair_path).get_fdata()

    # Extract predictions for each class
    core = predictions[:, :, :, 1]
    edema = predictions[:, :, :, 2]
    enhancing = predictions[:, :, :, 3]

    # Create visualization
    fig, axarr = plt.subplots(1, 6, figsize=(18, 5))

    # Resize original image for background
    background = cv2.resize(
        orig_image[:, :, slice_idx + VOLUME_START_AT],
        (IMG_SIZE, IMG_SIZE)
    )

    # Display background on all subplots
    for i in range(6):
        axarr[i].imshow(background, cmap="gray", interpolation='none')

    # Original FLAIR image
    axarr[0].imshow(background, cmap="gray")
    axarr[0].set_title('Original FLAIR')
    axarr[0].axis('off')

    # Ground truth
    curr_gt = cv2.resize(
        gt[:, :, slice_idx + VOLUME_START_AT],
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
    axarr[1].set_title('Ground Truth')
    axarr[1].axis('off')

    # All classes predicted
    axarr[2].imshow(predictions[slice_idx, :, :, 1:4], cmap="Reds",
                    interpolation='none', alpha=0.3)
    axarr[2].set_title('All Classes Predicted')
    axarr[2].axis('off')

    # Necrotic/Core (class 1)
    axarr[3].imshow(core[slice_idx, :, :], cmap="OrRd",
                    interpolation='none', alpha=0.3)
    axarr[3].set_title(f'{SEGMENT_CLASSES[1]} Predicted')
    axarr[3].axis('off')

    # Edema (class 2)
    axarr[4].imshow(edema[slice_idx, :, :], cmap="OrRd",
                    interpolation='none', alpha=0.3)
    axarr[4].set_title(f'{SEGMENT_CLASSES[2]} Predicted')
    axarr[4].axis('off')

    # Enhancing (class 3)
    axarr[5].imshow(enhancing[slice_idx, :, :], cmap="OrRd",
                    interpolation='none', alpha=0.3)
    axarr[5].set_title(f'{SEGMENT_CLASSES[3]} Predicted')
    axarr[5].axis('off')

    plt.tight_layout()
    plt.show()


def predict_and_visualize(model, case_path, case_id, slice_idx=60):
    """
    Predict and visualize results for a case.

    Args:
        model: Trained U-Net model
        case_path: Path to the patient directory
        case_id: Patient ID
        slice_idx: Index of slice to visualize
    """
    print(f"Predicting for case: {case_id}")

    # Predict
    predictions = predict_case(model, case_path, case_id)

    print(f"Prediction shape: {predictions.shape}")

    # Visualize
    visualize_prediction(case_path, case_id, predictions, slice_idx)


def batch_predict(model, case_ids, dataset_path=TRAIN_DATASET_PATH):
    """
    Predict segmentation for multiple cases.

    Args:
        model: Trained U-Net model
        case_ids: List of patient IDs
        dataset_path: Path to the BraTS dataset

    Returns:
        predictions_dict: Dictionary mapping case_id to predictions
    """
    predictions_dict = {}

    for case_id in case_ids:
        print(f"\nProcessing case: {case_id}")
        case_path = os.path.join(dataset_path, case_id)

        # Predict
        predictions = predict_case(model, case_path, case_id.split('_')[-1])
        predictions_dict[case_id] = predictions

        print(f"Completed: {case_id}")

    return predictions_dict


def calculate_dice_score(pred_mask, true_mask, class_id):
    """
    Calculate Dice score for a specific class.

    Args:
        pred_mask: Predicted segmentation mask (categorical)
        true_mask: Ground truth segmentation mask (categorical)
        class_id: Class ID to evaluate

    Returns:
        Dice score
    """
    pred_binary = (pred_mask == class_id).astype(float)
    true_binary = (true_mask == class_id).astype(float)

    intersection = np.sum(pred_binary * true_binary)
    denominator = np.sum(pred_binary) + np.sum(true_binary)

    if denominator == 0:
        return 1.0  # Both masks are empty for this class

    dice = (2.0 * intersection) / denominator
    return dice


def evaluate_predictions(predictions, case_path, case_id):
    """
    Evaluate predictions against ground truth.

    Args:
        predictions: Predicted segmentation masks
        case_path: Path to the patient directory
        case_id: Patient ID

    Returns:
        scores: Dictionary of Dice scores per class
    """
    # Load ground truth
    gt_path = os.path.join(case_path, f'BraTS20_Training_{case_id}_seg.nii')
    gt = nib.load(gt_path).get_fdata()

    # Get predicted classes (argmax over channel dimension)
    pred_classes = np.argmax(predictions, axis=-1)

    # Calculate Dice scores for each class
    scores = {}
    for class_id in [1, 2, 3]:  # Skip background (class 0)
        dice_scores = []

        for i in range(VOLUME_SLICES):
            # Get true mask for this slice
            true_slice = gt[:, :, i + VOLUME_START_AT]
            true_resized = cv2.resize(
                true_slice,
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

            # Convert class 4 to class 3
            true_resized[true_resized == 4] = 3

            # Calculate Dice for this slice
            dice = calculate_dice_score(pred_classes[i], true_resized, class_id)
            dice_scores.append(dice)

        # Average Dice across all slices
        scores[SEGMENT_CLASSES[class_id]] = np.mean(dice_scores)

    return scores


def main():
    """Example usage of prediction functions."""

    # Load the trained model
    print("Loading model...")
    model = load_best_model('best_model.weights.h5')

    # Example 1: Predict and visualize a single case
    print("\n" + "=" * 80)
    print("Example 1: Single case prediction and visualization")
    print("=" * 80)

    case_id = "355"  # Change this to your case ID
    case_path = os.path.join(TRAIN_DATASET_PATH, f'BraTS20_Training_{case_id}')

    if os.path.exists(case_path):
        predict_and_visualize(model, case_path, case_id, slice_idx=60)

        # Evaluate predictions
        predictions = predict_case(model, case_path, case_id)
        scores = evaluate_predictions(predictions, case_path, case_id)

        print("\nDice Scores:")
        for class_name, score in scores.items():
            print(f"  {class_name}: {score:.4f}")
    else:
        print(f"Case path not found: {case_path}")

    # Example 2: Batch prediction
    print("\n" + "=" * 80)
    print("Example 2: Batch prediction")
    print("=" * 80)

    # Get a few test case IDs (you would get these from your test split)
    from preprocessing import get_data_paths
    _, _, test_ids = get_data_paths()

    # Predict on first 3 test cases
    sample_cases = test_ids[:3]
    print(f"Predicting on {len(sample_cases)} cases...")

    predictions_dict = batch_predict(model, sample_cases)

    print(f"\nCompleted predictions for {len(predictions_dict)} cases")


if __name__ == "__main__":
    main()