"""
xai_seg/visualizer.py — XAI Visualization Utilities
Creates comprehensive visual reports:
  - Segmentation overlays
  - Grad-CAM heatmap overlays
  - Per-class comparison panels
  - Modality importance bar charts
  - XAI quality metrics (IoU between heatmap and ground truth)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import cv2
from typing import Dict, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Color Palette ────────────────────────────────────────────────────────────
CLASS_COLORS = {
    0: np.array([0,   0,   0  ]),  # Background — black
    1: np.array([255, 0,   0  ]),  # Necrotic   — red
    2: np.array([0,   255, 0  ]),  # Edema      — green
    3: np.array([255, 255, 0  ]),  # Enhancing  — yellow
}
CLASS_NAMES = {0: "Background", 1: "Necrotic Core", 2: "Edema", 3: "Enhancing"}
MODALITY_NAMES = {"t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR"}
COLORMAPS = {"gradcam": "jet", "gradcam++": "hot", "scorecam": "plasma", "ig": "viridis", "shap": "RdBu"}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert class index mask [H,W] → RGB [H,W,3]."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay segmentation mask on grayscale MRI slice."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    # Normalize to [0, 255]
    img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    mask_rgb = mask_to_rgb(mask)
    blend = (img_norm * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return blend


def heatmap_to_rgb(heatmap: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """Convert [H,W] float heatmap → RGB [H,W,3]."""
    colormap = plt.get_cmap(cmap)
    rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = config.OVERLAY_ALPHA, cmap: str = "jet") -> np.ndarray:
    """Overlay Grad-CAM heatmap on MRI slice."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    heat_rgb = heatmap_to_rgb(heatmap, cmap)
    blend    = (img_norm * (1 - alpha) + heat_rgb * alpha).astype(np.uint8)
    return blend


# ─── XAI Quality: Pointing Game / Heatmap-Mask IoU ───────────────────────────
def heatmap_mask_iou(heatmap: np.ndarray, mask: np.ndarray,
                     labels: list, threshold: float = 0.5) -> float:
    """
    Measures how well the heatmap focuses on the actual tumor region.
    Higher IoU = model is looking at the right area.
    """
    binary_mask = np.isin(mask, labels).astype(bool)
    binary_heat = (heatmap >= threshold).astype(bool)
    intersection = (binary_mask & binary_heat).sum()
    union        = (binary_mask | binary_heat).sum()
    return float(intersection / (union + 1e-8))


def pointing_game_score(heatmap: np.ndarray, mask: np.ndarray, labels: list) -> float:
    """Check if the maximum heatmap activation falls within the tumor mask."""
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    binary_mask = np.isin(mask, labels).astype(bool)
    return float(binary_mask[peak_y, peak_x])


# ─── Main Visualization Functions ─────────────────────────────────────────────
def plot_segmentation_result(
        image: np.ndarray,        # [4, H, W]
        pred_mask: np.ndarray,    # [H, W]
        gt_mask: np.ndarray,      # [H, W]
        patient_id: str = "",
        save_path: str = None
):
    """4-panel: 4 MRI modalities + pred + GT overlay."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor("#0d0d0d")
    plt.suptitle(f"Segmentation Result — {patient_id}", color="white", fontsize=14, y=1.01)

    for i, mod in enumerate(config.MODALITIES):
        ax = axes[0, i]
        ax.imshow(image[i], cmap="gray")
        ax.set_title(MODALITY_NAMES.get(mod, mod), color="white", fontsize=11)
        ax.axis("off")

    # GT overlay
    axes[1, 0].imshow(overlay_mask(image[1], gt_mask))   # T1ce background
    axes[1, 0].set_title("Ground Truth", color="white", fontsize=11)
    axes[1, 0].axis("off")

    # Pred overlay
    axes[1, 1].imshow(overlay_mask(image[1], pred_mask))
    axes[1, 1].set_title("Prediction", color="white", fontsize=11)
    axes[1, 1].axis("off")

    # Error map
    error = (pred_mask != gt_mask).astype(np.uint8)
    axes[1, 2].imshow(image[1], cmap="gray")
    axes[1, 2].imshow(error, cmap="Reds", alpha=0.6)
    axes[1, 2].set_title("Error Map", color="white", fontsize=11)
    axes[1, 2].axis("off")

    # Legend
    axes[1, 3].axis("off")
    patches = [mpatches.Patch(color=np.array(c)/255, label=n)
               for c, n in zip(CLASS_COLORS.values(), CLASS_NAMES.values())]
    axes[1, 3].legend(handles=patches, loc="center", fontsize=12,
                      facecolor="#1a1a1a", labelcolor="white")
    axes[1, 3].set_title("Legend", color="white", fontsize=11)

    for ax_row in axes:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [Viz] Saved segmentation: {save_path}")
    plt.close()


def plot_gradcam_report(
        image: np.ndarray,          # [4, H, W]
        gt_mask: np.ndarray,        # [H, W]
        pred_mask: np.ndarray,      # [H, W]
        heatmaps: Dict[str, np.ndarray],   # {"gradcam": [H,W], "gradcam++": [H,W], ...}
        target_class: int = config.GRADCAM_CLASS,
        patient_id: str = "",
        save_path: str = None
):
    """Comprehensive Grad-CAM report with multiple methods."""
    n_methods = len(heatmaps)
    fig = plt.figure(figsize=(6 * (n_methods + 2), 12))
    fig.patch.set_facecolor("#0d0d0d")
    gs = gridspec.GridSpec(2, n_methods + 2, figure=fig, hspace=0.3, wspace=0.1)

    class_name = CLASS_NAMES.get(target_class, str(target_class))
    plt.suptitle(f"XAI Report — {patient_id} | Target: {class_name}",
                 color="white", fontsize=15)

    # Row 0: Original T1ce + GT + Pred
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image[1], cmap="gray")   # T1ce
    ax.set_title("T1ce (Input)", color="white"); ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(overlay_mask(image[1], gt_mask))
    ax.set_title("Ground Truth", color="white"); ax.axis("off")

    for i, (method, cam) in enumerate(heatmaps.items()):
        ax = fig.add_subplot(gs[0, i + 2])
        cmap = COLORMAPS.get(method, "jet")
        ax.imshow(overlay_heatmap(image[1], cam, cmap=cmap))
        # Compute quality score
        iou = heatmap_mask_iou(cam, gt_mask,
                               labels=[1, 2, 3] if target_class == 0 else [target_class])
        pg  = pointing_game_score(cam, gt_mask,
                                  labels=[1, 2, 3] if target_class == 0 else [target_class])
        ax.set_title(f"{method.upper()}\nHeatmap-IoU: {iou:.3f} | PG: {pg:.0f}",
                     color="white", fontsize=9)
        ax.axis("off")

    # Row 1: Contour overlay showing overlap with GT
    for i, (method, cam) in enumerate(heatmaps.items()):
        ax = fig.add_subplot(gs[1, i + 2])
        base = overlay_heatmap(image[1], cam)
        # Draw GT contour
        gt_binary = np.isin(gt_mask, [1, 2, 3]).astype(np.uint8) * 255
        contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        cv2.drawContours(base_bgr, contours, -1, (0, 255, 0), 2)
        ax.imshow(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{method} + GT Contour", color="white", fontsize=9)
        ax.axis("off")

    # Colorbar
    ax_cb = fig.add_subplot(gs[1, 0])
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax_cb.imshow(gradient, aspect="auto", cmap="jet", origin="lower")
    ax_cb.set_title("Heatmap\nIntensity", color="white", fontsize=9)
    ax_cb.set_xticks([])
    ax_cb.set_yticks([0, 128, 255])
    ax_cb.set_yticklabels(["Low", "Mid", "High"], color="white")

    for ax in fig.axes:
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [Viz] Saved Grad-CAM report: {save_path}")
    plt.close()


def plot_modality_importance(importance: dict, patient_id: str = "", save_path: str = None):
    """Bar chart of SHAP modality importance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a1a")

    mods = list(importance.keys())
    vals = [importance[m] * 100 for m in mods]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    bars = ax.bar([MODALITY_NAMES.get(m, m) for m in mods], vals, color=colors, width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", color="white", fontsize=11)

    ax.set_title(f"Modality Importance (SHAP) — {patient_id}", color="white", fontsize=13)
    ax.set_ylabel("Relative Importance (%)", color="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#555")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [Viz] Saved modality importance: {save_path}")
    plt.close()


def save_full_xai_panel(
        image, gt_mask, pred_mask,
        heatmaps, shap_importance,
        patient_id, slice_idx, save_dir
):
    """Save all XAI outputs for one patient/slice."""
    os.makedirs(save_dir, exist_ok=True)
    base = f"{patient_id}_z{slice_idx:03d}"

    plot_segmentation_result(
        image, pred_mask, gt_mask, patient_id,
        save_path=os.path.join(save_dir, f"{base}_segmentation.png"))

    plot_gradcam_report(
        image, gt_mask, pred_mask, heatmaps, patient_id=patient_id,
        save_path=os.path.join(save_dir, f"{base}_gradcam_report.png"))

    if shap_importance:
        plot_modality_importance(
            shap_importance, patient_id,
            save_path=os.path.join(save_dir, f"{base}_modality_importance.png"))