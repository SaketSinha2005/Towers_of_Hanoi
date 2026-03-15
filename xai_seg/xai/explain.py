"""
Loads a trained model checkpoint, runs it on test samples,
and generates Grad-CAM, SHAP, and IG explanations + visual reports.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.unet import build_model
from models.train import load_checkpoint
from utils.dataloader import get_dataloaders
from utils.metrics import evaluate_brats
from xai_seg.gradcam import GradCAMExplainer
from xai_seg.visualizer import save_full_xai_panel


def explain_model(
    checkpoint_path: str,
    n_samples: int = 20,
    methods: list = None,
    use_shap: bool = True,
    use_ig: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame of per-sample metrics.

    Args:
        checkpoint_path : path to best_model.pth
        n_samples       : how many test slices to explain
        methods         : list of gradcam methods e.g. ["gradcam", "gradcam++"]
        use_shap        : whether to run SHAP modality importance
        use_ig          : whether to run Integrated Gradients
    """
    methods = methods or config.XAI_METHODS
    device  = torch.device(config.DEVICE)

    # ── Load model ───────────────────────────────────────────────────────────
    model = build_model(config.MODEL_NAME).to(device)
    load_checkpoint(model, None, checkpoint_path)
    model.eval()
    print(f"[Explain] Loaded checkpoint: {checkpoint_path}")

    # ── Data ─────────────────────────────────────────────────────────────────
    _, _, test_loader, _ = get_dataloaders()

    # ── Grad-CAM explainers ───────────────────────────────────────────────────
    cam_explainers = {}
    for method in methods:
        try:
            cam_explainers[method] = GradCAMExplainer(
                model,
                target_layer=config.GRADCAM_LAYER,
                method=method
            )
            print(f"[Explain] Initialized {method} explainer")
        except Exception as e:
            print(f"[Explain] Could not init {method}: {e}")

    # ── Optional: SHAP ───────────────────────────────────────────────────────
    shap_explainer = None
    if use_shap:
        try:
            from xai_seg.shap_explainer import SHAPExplainer

            # Collect background samples robustly (loop until we have enough)
            bg_list = []
            n_collected = 0
            for bg_batch in test_loader:
                imgs = bg_batch["image"]
                if not isinstance(imgs, torch.Tensor):
                    continue
                imgs = imgs.float()
                if imgs.ndim != 4:
                    continue
                bg_list.append(imgs)
                n_collected += imgs.shape[0]
                if n_collected >= config.SHAP_SAMPLES:
                    break

            if len(bg_list) == 0:
                raise RuntimeError("Could not collect any background samples for SHAP")

            background = torch.cat(bg_list, dim=0)[:config.SHAP_SAMPLES].float()
            print(f"[Explain] SHAP background: shape={tuple(background.shape)}, "
                  f"dtype={background.dtype}")

            shap_explainer = SHAPExplainer(
                model, background,
                target_class=config.GRADCAM_CLASS
            )
        except Exception as e:
            print(f"[Explain] SHAP init skipped: {e}")
            shap_explainer = None

    # ── Optional: Integrated Gradients (manual, MPS-safe) ────────────────────
    ig_explainer = None
    if use_ig:
        try:
            from xai_seg.shap_explainer import IntegratedGradientsExplainer
            ig_explainer = IntegratedGradientsExplainer(
                model,
                target_class=config.GRADCAM_CLASS
            )
            print("[Explain] Initialized IG explainer (manual float32 implementation)")
        except Exception as e:
            print(f"[Explain] IG init skipped: {e}")
            ig_explainer = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    records   = []
    processed = 0

    for batch in tqdm(test_loader, desc="[Explain] Generating XAI"):
        if processed >= n_samples:
            break

        # Always float32
        image_t   = batch["image"].float()      # [1, 4, H, W]
        mask_t    = batch["mask"]               # [1, H, W]
        patient   = batch["patient"][0]
        slice_idx = int(batch["slice"][0])

        image_np = image_t[0].cpu().numpy()     # [4, H, W]
        gt_np    = mask_t[0].cpu().numpy()      # [H, W]

        # ── Prediction — inside no_grad for efficiency ───────────────────────
        with torch.no_grad():
            logits  = model(image_t.to(device).float())
            pred_np = logits.argmax(dim=1)[0].cpu().numpy()

        brats_m = evaluate_brats(pred_np, gt_np)

        # ── Grad-CAM — MUST be outside no_grad (needs gradients) ─────────────
        H, W     = image_np.shape[1], image_np.shape[2]
        heatmaps = {}

        for method, explainer in cam_explainers.items():
            try:
                heatmap = explainer.explain(
                    image_t.to(device).float(),
                    target_class=config.GRADCAM_CLASS,
                    target_size=(H, W)
                )
                heatmaps[method] = heatmap
            except Exception as e:
                print(f"[Explain] {method} failed for {patient}: {e}")
                heatmaps[method] = np.zeros((H, W), dtype=np.float32)

        # ── SHAP modality importance ──────────────────────────────────────────
        shap_importance = None
        if shap_explainer is not None:
            try:
                shap_importance = shap_explainer.modality_importance(
                    image_t.to(device).float()
                )
            except Exception as e:
                print(f"[Explain] SHAP failed for {patient}: {e}")

        # ── Integrated Gradients heatmap ──────────────────────────────────────
        if ig_explainer is not None:
            try:
                ig_map = ig_explainer.spatial_heatmap(
                    image_t.to(device).float()
                )
                heatmaps["ig"] = ig_map
            except Exception as e:
                print(f"[Explain] IG heatmap failed for {patient}: {e}")

        # ── Save visual outputs ───────────────────────────────────────────────
        save_dir = os.path.join(config.HEATMAP_DIR, patient)
        save_full_xai_panel(
            image_np, gt_np, pred_np,
            heatmaps, shap_importance,
            patient, slice_idx, save_dir
        )

        # ── Record metrics ────────────────────────────────────────────────────
        row = {"patient": patient, "slice": slice_idx}
        for region, mvals in brats_m.items():
            for metric, val in mvals.items():
                row[f"{region}_{metric}"] = val
        if shap_importance:
            for mod, imp in shap_importance.items():
                row[f"shap_{mod}"] = imp
        records.append(row)
        processed += 1

    # ── Cleanup hooks ─────────────────────────────────────────────────────────
    for explainer in cam_explainers.values():
        try:
            explainer.cleanup()
        except Exception:
            pass

    # ── Save summary CSV ──────────────────────────────────────────────────────
    df       = pd.DataFrame(records)
    csv_path = os.path.join(config.REPORT_DIR, "xai_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Explain] Saved metrics  → {csv_path}")
    print(f"[Explain] XAI panels     → {config.HEATMAP_DIR}")
    print(f"[Explain] Processed {processed} samples")

    # ── Print summary table ───────────────────────────────────────────────────
    if not df.empty:
        print("\n[Explain] Mean metrics across samples:")
        print(f"  {'Region':<20} {'Dice':>8} {'IoU':>8}")
        print("  " + "-" * 38)
        for region in ["whole_tumor", "tumor_core", "enhancing"]:
            dice = df.get(f"{region}_dice", pd.Series([0])).mean()
            iou  = df.get(f"{region}_iou",  pd.Series([0])).mean()
            print(f"  {region:<20} {dice:>8.4f} {iou:>8.4f}")

    return df
