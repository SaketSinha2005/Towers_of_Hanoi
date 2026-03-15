"""
config.py — Central configuration for XAI Tumor Segmentation
Optimized for Apple Silicon (Mac M3) with MPS backend.

⚡ DEADLINE MODE (trains done by 6 PM):
   - NUM_EPOCHS      = 100   (hard ceiling)
   - EARLY_STOP_PAT  = 10    (stops at ~60-80 if converged — saves 30-60 min)
   - BATCH_SIZE      = 16    (safe for M3 unified memory)
   - GRAD_ACCUM      = 2     (effective BS = 32, stable gradients)
   Estimated time: ~4–5 hrs worst case | ~3–3.5 hrs with early stopping
   → Finishes comfortably before 6 PM IST
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw", "BraTS2020_training_data", "data", "data")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
PRED_DIR        = os.path.join(OUTPUT_DIR, "predictions")
HEATMAP_DIR     = os.path.join(OUTPUT_DIR, "heatmaps")
REPORT_DIR      = os.path.join(OUTPUT_DIR, "reports")
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")

for d in [DATA_PROCESSED, PRED_DIR, HEATMAP_DIR, REPORT_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── MRI Modalities ───────────────────────────────────────────────────────────
MODALITIES     = ["t1", "t1ce", "t2", "flair"]
NUM_MODALITIES = len(MODALITIES)   # 4

# ─── Segmentation Classes ─────────────────────────────────────────────────────
# BraTS labels: 0=Background, 1=Necrotic core, 2=Edema, 4=Enhancing tumor
LABEL_MAP = {0: "Background", 1: "Necrotic Core", 2: "Edema", 4: "Enhancing Tumor"}
NUM_CLASSES = 4   # remapped to 0,1,2,3

# Tumor sub-regions for evaluation (BraTS convention)
WHOLE_TUMOR_LABELS = [1, 2, 4]   # all non-background
TUMOR_CORE_LABELS  = [1, 4]      # necrotic + enhancing
ENHANCING_LABELS   = [4]         # enhancing only

# ─── Image Dimensions ─────────────────────────────────────────────────────────
ORIGINAL_SIZE = (240, 240, 155)
CROP_SIZE     = (128, 128, 128)
SLICE_SIZE    = (128, 128)        # keep at 128 for M3 speed

# ─── Training ─────────────────────────────────────────────────────────────────
#
# ⚡ DEADLINE SETTINGS — tuned to finish by 6 PM on M3
#
#   Epoch time estimate  : ~2.5–3 min/epoch  (42,780 slices, BS=16, MPS)
#   100 epochs worst case: ~250–300 min = 4–5 hrs  → done by ~5:30–6 PM ✅
#   Early stop at ep ~65 : ~160–195 min = ~3 hrs   → done by ~4 PM      ✅
#
BATCH_SIZE       = 64       # safe for M3 unified memory (~4 MB/batch)
GRAD_ACCUM_STEPS = 1        # effective batch size = 32
NUM_EPOCHS       = 50      # hard ceiling — early stopping fires first
LEARNING_RATE    = 4e-4     # linear scaling: 1e-4 * (32/8)
WEIGHT_DECAY     = 1e-5
SCHEDULER        = "onecycle"
WARMUP_EPOCHS    = 5        # reduced from 10 — faster initial convergence
VAL_SPLIT        = 0.15
TEST_SPLIT       = 0.10
RANDOM_SEED      = 42
NUM_WORKERS      = 0        # MPS unstable with multiprocessing — keep 0
PIN_MEMORY       = False    # CUDA-only feature
EARLY_STOP_PAT   = 10       # ⚡ reduced from 20 → exits ~10 epochs after plateau
MIXED_PRECISION  = False    # AMP not supported on MPS
GRADIENT_CLIP    = 1.0

# ─── OneCycleLR ───────────────────────────────────────────────────────────────
ONECYCLE_MAX_LR    = LEARNING_RATE   # peak LR  = 4e-4
ONECYCLE_DIV       = 10.0            # start LR = 4e-5
ONECYCLE_FINAL_DIV = 1e4             # end LR   = 4e-8
ONECYCLE_PCT_START = 0.3             # 30% of steps = warmup

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "attention_unet"
IN_CHANNELS  = NUM_MODALITIES    # 4
OUT_CHANNELS = NUM_CLASSES       # 4
FEATURES     = [32, 64, 128, 256, 512]
DROPOUT      = 0.3
BILINEAR     = False

# ─── Loss Function ────────────────────────────────────────────────────────────
LOSS_FN       = "combined"
DICE_WEIGHT   = 0.6
CE_WEIGHT     = 0.4
FOCAL_GAMMA   = 2.0
CLASS_WEIGHTS = [0.1, 1.0, 1.0, 1.5]   # down-weight background

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN  = True
FLIP_PROB      = 0.5
ROTATE_PROB    = 0.4
ROTATE_DEGREES = 20
BRIGHTNESS_PROB= 0.4
GAUSSIAN_PROB  = 0.3
ELASTIC_PROB   = 0.2

# ─── XAI Settings ─────────────────────────────────────────────────────────────
GRADCAM_LAYER    = "encoder4"
GRADCAM_CLASS    = 3                        # Enhancing tumor (after remap)
XAI_METHODS      = ["gradcam", "gradcam++"] # scorecam skipped — too slow on M3
OVERLAY_ALPHA    = 0.5

SHAP_SAMPLES = 20    # 50 is too slow on M3
IG_STEPS     = 25    # 50 steps slower; 25 accurate enough

# ─── Evaluation ───────────────────────────────────────────────────────────────
METRICS          = ["dice", "iou", "hausdorff95", "sensitivity", "specificity"]
SAVE_BEST_METRIC = "dice"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_INTERVAL = 10
TENSORBOARD  = True
SAVE_FREQ    = 5    # save checkpoint every N epochs

# ─── Device — Apple Silicon MPS ───────────────────────────────────────────────
import torch

def _get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = _get_device()
print(f"[Config] Using device: {DEVICE}")

if DEVICE == "mps":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"