"""
Modes:
  preprocess : Preprocess BraTS2020 volumes → .npy
  train      : Train U-Net model
  explain    : Generate XAI explanations
  all        : Run full pipeline
  evaluate   : Evaluate a checkpoint on the test set
"""

import argparse
import os
import sys
import config


def parse_args():
    parser = argparse.ArgumentParser(description="XAI Brain Tumor Segmentation")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["preprocess", "train", "explain", "evaluate", "all"],
                        help="Pipeline stage to run")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to model checkpoint (for explain/evaluate)")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME,
                        choices=["unet", "attention_unet"],
                        help="Model architecture")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of samples to explain (explain mode)")
    parser.add_argument("--xai_seg-methods", nargs="+",
                        default=["gradcam", "gradcam++"],
                        choices=["gradcam", "gradcam++", "scorecam"],
                        help="Grad-CAM methods to use")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP (faster)")
    parser.add_argument("--no-ig", action="store_true",
                        help="Skip Integrated Gradients")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root directory")
    return parser.parse_args()


def run_preprocess(args):
    from utils.preprocessing import preprocess_all
    root = args.data_root or config.DATA_RAW
    print(f"\n{'='*60}")
    print("STEP 1: PREPROCESSING")
    print(f"{'='*60}")
    preprocess_all(data_root=root, output_dir=config.DATA_PROCESSED)


def run_train(args):
    from models.train import train
    # Override model if specified
    if args.model:
        config.MODEL_NAME = args.model
    print(f"\n{'='*60}")
    print("STEP 2: TRAINING")
    print(f"{'='*60}")
    ckpt = train(resume=args.resume)
    return ckpt


def run_explain(args, checkpoint=None):
    from xai.explain import explain_model
    ckpt = checkpoint or args.checkpoint
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print("Please train first: python main.py --mode train")
        sys.exit(1)
    print(f"\n{'='*60}")
    print("STEP 3: XAI EXPLANATIONS")
    print(f"{'='*60}")
    df = explain_model(
        checkpoint_path = ckpt,
        n_samples       = args.n_samples,
        methods         = args.xai_methods,
        use_shap        = not args.no_shap,
        use_ig          = not args.no_ig
    )
    return df


def run_evaluate(args, checkpoint=None):
    """Evaluate model on test set and print detailed metrics."""
    import torch
    import numpy as np
    from tqdm import tqdm
    from models.unet import build_model
    from models.train import load_checkpoint
    from utils.dataloader import get_dataloaders
    from utils.metrics import evaluate_brats, MetricMeter

    ckpt = checkpoint or args.checkpoint
    device = torch.device(config.DEVICE)

    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    model = build_model(config.MODEL_NAME).to(device)
    load_checkpoint(model, None, ckpt)
    model.eval()

    _, _, test_loader, _ = get_dataloaders()
    meter = MetricMeter()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks  = batch["mask"].numpy()
            preds  = model(images).argmax(dim=1).cpu().numpy()

            for p, g in zip(preds, masks):
                bm = evaluate_brats(p, g)
                flat = {}
                for region, mvals in bm.items():
                    for metric, val in mvals.items():
                        flat[f"{region}_{metric}"] = val
                meter.update(flat)

    results = meter.mean()
    print("\nTest Set Results:")
    print(f"  {'Region':<20} {'Dice':>8} {'IoU':>8} {'HD95':>8} {'Sens':>8} {'Spec':>8}")
    print("  " + "-" * 64)
    for region in ["whole_tumor", "tumor_core", "enhancing"]:
        dice = results.get(f"{region}_dice", 0)
        iou  = results.get(f"{region}_iou", 0)
        hd   = results.get(f"{region}_hausdorff95", 0)
        sens = results.get(f"{region}_sensitivity", 0)
        spec = results.get(f"{region}_specificity", 0)
        print(f"  {region:<20} {dice:>8.4f} {iou:>8.4f} {hd:>8.2f} {sens:>8.4f} {spec:>8.4f}")

    return results


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  XAI Brain Tumor Segmentation — BraTS2020")
    print(f"  Device: {config.DEVICE}  |  Model: {config.MODEL_NAME}")
    print("=" * 60)

    if args.mode == "preprocess":
        run_preprocess(args)

    elif args.mode == "train":
        run_train(args)

    elif args.mode == "explain":
        run_explain(args)

    elif args.mode == "evaluate":
        run_evaluate(args)

    elif args.mode == "all":
        run_preprocess(args)
        ckpt = run_train(args)
        run_evaluate(args, checkpoint=ckpt)
        run_explain(args, checkpoint=ckpt)

    print("\n[Done] ✓")


if __name__ == "__main__":
    main()
