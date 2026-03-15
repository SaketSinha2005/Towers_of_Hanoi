"""
inference.py — Run inference on a NEW MRI scan

Output saved to:  outputs/inference/<scan_name>/
  - segmentation.png     → predicted tumor mask overlaid on MRI
  - gradcam_report.png   → Grad-CAM heatmaps showing model focus
  - prediction_mask.npy  → raw predicted mask array [H, W]
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from models.unet import build_model
from models.train import load_checkpoint
from xai.gradcam import GradCAMExplainer
from xai.visualizer import (
    overlay_mask, overlay_heatmap,
    mask_to_rgb, CLASS_COLORS, CLASS_NAMES,
    MODALITY_NAMES, COLORMAPS,
    heatmap_mask_iou, pointing_game_score
)


# ─── Input Loaders ────────────────────────────────────────────────────────────

def load_h5(path: str, slice_idx: int = None) -> np.ndarray:
    """Load image from H5 file → [4, H, W] float32."""
    import h5py
    with h5py.File(path, "r") as f:
        image = f["image"][:]
    image = image.astype(np.float32)
    # Fix layout: H5 may store as [H, W, 4] or [4, H, W]
    if image.ndim == 3 and image.shape[-1] == 4:
        image = np.transpose(image, (2, 0, 1))
    return image   # [4, H, W]


def load_nifti_slice(t1: str, t1ce: str, t2: str, flair: str,
                     slice_idx: int = None) -> np.ndarray:
    """
    Load 4 NIfTI volumes and extract one axial slice.
    If slice_idx is None, picks the middle slice automatically.
    Returns [4, H, W] float32.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("Install nibabel: pip install nibabel")

    modalities = []
    for path in [t1, t1ce, t2, flair]:
        vol = nib.load(path).get_fdata().astype(np.float32)  # [H, W, D]
        if slice_idx is None:
            slice_idx = vol.shape[2] // 2   # middle slice
        slc = vol[:, :, slice_idx]           # [H, W]
        modalities.append(slc)

    image = np.stack(modalities, axis=0)   # [4, H, W]
    print(f"[Inference] Loaded NIfTI slice {slice_idx} | shape={image.shape}")
    return image


def load_npy(path: str) -> np.ndarray:
    """Load pre-stacked numpy array [4, H, W]."""
    image = np.load(path).astype(np.float32)
    if image.ndim == 2:
        # Single modality — stack 4 copies
        image = np.stack([image] * 4, axis=0)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = np.transpose(image, (2, 0, 1))
    assert image.ndim == 3, f"Expected [4, H, W], got {image.shape}"
    return image


def normalize_slice(image: np.ndarray) -> np.ndarray:
    """Z-score normalize each modality channel independently."""
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        ch   = image[c]
        mask = ch > 0
        if mask.sum() == 0:
            continue
        mu    = ch[mask].mean()
        sigma = ch[mask].std()
        sigma = sigma if sigma > 1e-8 else 1.0
        out[c] = np.where(mask, (ch - mu) / sigma, 0.0)
    return out


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize [4, H, W] to [4, size[0], size[1]]."""
    from skimage.transform import resize
    return np.stack([
        resize(image[c], size, order=1, preserve_range=True,
               anti_aliasing=True).astype(np.float32)
        for c in range(image.shape[0])
    ], axis=0)


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(image_np: np.ndarray, model, device) -> np.ndarray:
    """
    Run model on [4, H, W] numpy array.
    Returns predicted mask [H, W] with class indices 0-3.
    """
    image_t = torch.from_numpy(image_np).unsqueeze(0).float().to(device)  # [1,4,H,W]
    with torch.no_grad():
        logits = model(image_t)                      # [1, 4, H, W]
        pred   = logits.argmax(dim=1)[0].cpu().numpy()  # [H, W]
    return pred


# ─── Visualization ────────────────────────────────────────────────────────────

def save_segmentation_output(image_np, pred_mask, save_path, scan_name):
    """Save 2-row panel: 4 modalities + prediction overlay + legend."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor("#0d0d0d")
    plt.suptitle(f"Tumor Segmentation — {scan_name}",
                 color="white", fontsize=14, y=1.01)

    # Row 1: 4 MRI modalities
    mod_names = ["T1", "T1ce", "T2", "FLAIR"]
    for i in range(4):
        axes[0, i].imshow(image_np[i], cmap="gray")
        axes[0, i].set_title(mod_names[i], color="white", fontsize=11)
        axes[0, i].axis("off")

    # Row 2: prediction overlay on T1ce, blank, blank, legend
    axes[1, 0].imshow(overlay_mask(image_np[1], pred_mask))
    axes[1, 0].set_title("Predicted Segmentation", color="white", fontsize=11)
    axes[1, 0].axis("off")

    # Class distribution bar
    classes, counts = np.unique(pred_mask, return_counts=True)
    total = pred_mask.size
    colors_list = [np.array(CLASS_COLORS[c]) / 255 for c in classes]
    axes[1, 1].bar(
        [CLASS_NAMES.get(c, str(c)) for c in classes],
        [cnt / total * 100 for cnt in counts],
        color=colors_list
    )
    axes[1, 1].set_title("Class Distribution (%)", color="white", fontsize=11)
    axes[1, 1].tick_params(colors="white")
    axes[1, 1].set_facecolor("#1a1a1a")
    for sp in axes[1, 1].spines.values():
        sp.set_color("#555")

    # RGB mask
    axes[1, 2].imshow(mask_to_rgb(pred_mask))
    axes[1, 2].set_title("Segmentation Mask", color="white", fontsize=11)
    axes[1, 2].axis("off")

    # Legend
    axes[1, 3].axis("off")
    patches = [
        mpatches.Patch(color=np.array(c) / 255, label=n)
        for c, n in zip(CLASS_COLORS.values(), CLASS_NAMES.values())
    ]
    axes[1, 3].legend(handles=patches, loc="center", fontsize=12,
                      facecolor="#1a1a1a", labelcolor="white")
    axes[1, 3].set_title("Legend", color="white", fontsize=11)

    for ax_row in axes:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Inference] Saved segmentation → {save_path}")


def save_gradcam_output(image_np, pred_mask, heatmaps, save_path, scan_name):
    """Save Grad-CAM heatmap panel."""
    import cv2
    import matplotlib.gridspec as gridspec

    n = len(heatmaps)
    if n == 0:
        return

    fig = plt.figure(figsize=(6 * (n + 2), 12))
    fig.patch.set_facecolor("#0d0d0d")
    gs  = gridspec.GridSpec(2, n + 2, figure=fig, hspace=0.3, wspace=0.1)
    plt.suptitle(f"XAI Grad-CAM Report — {scan_name}",
                 color="white", fontsize=14)

    # T1ce input
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image_np[1], cmap="gray")
    ax.set_title("T1ce (Input)", color="white")
    ax.axis("off")

    # Prediction overlay
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(overlay_mask(image_np[1], pred_mask))
    ax.set_title("Prediction", color="white")
    ax.axis("off")

    for i, (method, cam) in enumerate(heatmaps.items()):
        ax = fig.add_subplot(gs[0, i + 2])
        cmap = COLORMAPS.get(method, "jet")
        ax.imshow(overlay_heatmap(image_np[1], cam, cmap=cmap))
        ax.set_title(f"{method.upper()}", color="white", fontsize=10)
        ax.axis("off")

        # Row 2: raw heatmap
        ax2 = fig.add_subplot(gs[1, i + 2])
        ax2.imshow(cam, cmap=cmap)
        ax2.set_title(f"{method} (raw)", color="white", fontsize=9)
        ax2.axis("off")

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

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Inference] Saved Grad-CAM   → {save_path}")


def print_tumor_summary(pred_mask: np.ndarray):
    """Print a readable summary of predicted tumor regions."""
    total = pred_mask.size
    print("\n" + "─" * 45)
    print("  TUMOR SEGMENTATION SUMMARY")
    print("─" * 45)

    region_pixels = {
        "Background":    (pred_mask == 0).sum(),
        "Necrotic Core": (pred_mask == 1).sum(),
        "Edema":         (pred_mask == 2).sum(),
        "Enhancing":     (pred_mask == 3).sum(),
    }

    for name, count in region_pixels.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<16} {pct:5.1f}%  {bar}")

    tumor_pixels = total - region_pixels["Background"]
    tumor_pct    = tumor_pixels / total * 100
    print("─" * 45)
    print(f"  Total tumor area: {tumor_pct:.2f}% of slice")

    if tumor_pixels == 0:
        print("  ⚠  No tumor detected in this slice.")
        print("     Try a different slice (--slice N) closer to the tumor center.")
    print("─" * 45 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run brain tumor segmentation on a new MRI scan",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Input options
    parser.add_argument("--input", type=str, default=None,
                        help="Path to .h5 or .npy file")
    parser.add_argument("--t1",    type=str, default=None,
                        help="Path to T1 NIfTI file (.nii/.nii.gz)")
    parser.add_argument("--t1ce",  type=str, default=None,
                        help="Path to T1ce NIfTI file")
    parser.add_argument("--t2",    type=str, default=None,
                        help="Path to T2 NIfTI file")
    parser.add_argument("--flair", type=str, default=None,
                        help="Path to FLAIR NIfTI file")
    parser.add_argument("--slice", type=int, default=None,
                        help="Axial slice index to use (NIfTI only, default=middle)")

    # Model
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to trained model checkpoint")

    # XAI
    parser.add_argument("--no-gradcam", action="store_true",
                        help="Skip Grad-CAM generation")
    parser.add_argument("--no-ig", action="store_true",
                        help="Skip Integrated Gradients")
    parser.add_argument("--gradcam-class", type=int,
                        default=config.GRADCAM_CLASS,
                        help="Target class for Grad-CAM (0=bg,1=necrotic,2=edema,3=enhancing)")

    # Output
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(config.OUTPUT_DIR, "inference"),
                        help="Output directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Name for this scan (used in filenames)")

    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(config.DEVICE)

    print("\n" + "=" * 55)
    print("  Brain Tumor Segmentation — New MRI Inference")
    print(f"  Device: {config.DEVICE}  |  Model: {config.MODEL_NAME}")
    print("=" * 55 + "\n")

    # ── Load image ────────────────────────────────────────────────────────────
    if args.input:
        ext = os.path.splitext(args.input)[1].lower()
        if ext == ".h5":
            print(f"[Inference] Loading H5 file: {args.input}")
            image_np  = load_h5(args.input)
            scan_name = args.name or os.path.splitext(
                            os.path.basename(args.input))[0]
        elif ext == ".npy":
            print(f"[Inference] Loading NumPy file: {args.input}")
            image_np  = load_npy(args.input)
            scan_name = args.name or os.path.splitext(
                            os.path.basename(args.input))[0]
        else:
            print(f"[ERROR] Unsupported file type: {ext}")
            print("  Supported: .h5, .npy, or use --t1/--t1ce/--t2/--flair")
            sys.exit(1)

    elif all([args.t1, args.t1ce, args.t2, args.flair]):
        print("[Inference] Loading NIfTI files...")
        image_np  = load_nifti_slice(args.t1, args.t1ce, args.t2, args.flair,
                                     slice_idx=args.slice)
        scan_name = args.name or "nifti_scan"
    else:
        print("[ERROR] Provide either --input (h5/npy) or all four of "
              "--t1 --t1ce --t2 --flair")
        sys.exit(1)

    print(f"[Inference] Image shape: {image_np.shape}  "
          f"dtype={image_np.dtype}")

    # ── Normalize + resize ────────────────────────────────────────────────────
    image_np = normalize_slice(image_np)
    if image_np.shape[1:] != tuple(config.SLICE_SIZE):
        print(f"[Inference] Resizing {image_np.shape[1:]} → {config.SLICE_SIZE}")
        image_np = resize_image(image_np, config.SLICE_SIZE)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("  Train first: python3 main.py --mode train")
        sys.exit(1)

    model = build_model(config.MODEL_NAME).to(device)
    load_checkpoint(model, None, args.checkpoint)
    model.eval()
    print(f"[Inference] Loaded model from: {args.checkpoint}")

    # ── Run inference ─────────────────────────────────────────────────────────
    pred_mask = run_inference(image_np, model, device)
    print(f"[Inference] Prediction complete | unique classes: "
          f"{np.unique(pred_mask).tolist()}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print_tumor_summary(pred_mask)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(args.out_dir, scan_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Save prediction mask ──────────────────────────────────────────────────
    mask_path = os.path.join(out_dir, f"{scan_name}_prediction_mask.npy")
    np.save(mask_path, pred_mask)
    print(f"  [Inference] Saved mask array → {mask_path}")

    # ── Save segmentation PNG ─────────────────────────────────────────────────
    seg_path = os.path.join(out_dir, f"{scan_name}_segmentation.png")
    save_segmentation_output(image_np, pred_mask, seg_path, scan_name)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if not args.no_gradcam:
        heatmaps = {}
        H, W     = image_np.shape[1], image_np.shape[2]
        image_t  = torch.from_numpy(image_np).unsqueeze(0).float().to(device)

        for method in config.XAI_METHODS:
            try:
                explainer = GradCAMExplainer(model,
                                             target_layer=config.GRADCAM_LAYER,
                                             method=method)
                heatmaps[method] = explainer.explain(
                    image_t,
                    target_class=args.gradcam_class,
                    target_size=(H, W)
                )
                explainer.cleanup()
                print(f"  [Inference] {method} heatmap generated")
            except Exception as e:
                print(f"  [Inference] {method} failed: {e}")

        # IG heatmap
        if not args.no_ig:
            try:
                from xai.shap_explainer import IntegratedGradientsExplainer
                ig = IntegratedGradientsExplainer(model,
                                                  target_class=args.gradcam_class)
                heatmaps["ig"] = ig.spatial_heatmap(image_t)
                print("  [Inference] IG heatmap generated")
            except Exception as e:
                print(f"  [Inference] IG failed: {e}")

        if heatmaps:
            cam_path = os.path.join(out_dir, f"{scan_name}_gradcam_report.png")
            save_gradcam_output(image_np, pred_mask, heatmaps, cam_path, scan_name)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  All outputs saved to: {out_dir}/")
    print(f"    {scan_name}_segmentation.png")
    print(f"    {scan_name}_gradcam_report.png")
    print(f"    {scan_name}_prediction_mask.npy")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
