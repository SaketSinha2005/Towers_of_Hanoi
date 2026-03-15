"""
utils/metrics.py — Segmentation Metrics
Dice, IoU, Hausdorff95, Sensitivity, Specificity
Evaluated for Whole Tumor / Tumor Core / Enhancing Tumor (BraTS convention).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from typing import Dict, Tuple


# ─── Helper ───────────────────────────────────────────────────────────────────
def to_binary(pred: np.ndarray, labels: list) -> np.ndarray:
    """Collapse multi-class mask to binary for given label set."""
    out = np.zeros_like(pred, dtype=bool)
    for l in labels:
        out |= (pred == l)
    return out


# ─── Dice ─────────────────────────────────────────────────────────────────────
def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)


# ─── IoU ──────────────────────────────────────────────────────────────────────
def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return (intersection + smooth) / (union + smooth)


# ─── Hausdorff 95th percentile ────────────────────────────────────────────────
def hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")
    # Distance from each pred surface point to gt and vice versa
    dist_pred_to_gt = distance_transform_edt(~gt)[pred]
    dist_gt_to_pred = distance_transform_edt(~pred)[gt]
    return float(max(np.percentile(dist_pred_to_gt, 95),
                     np.percentile(dist_gt_to_pred, 95)))


# ─── Sensitivity / Specificity ────────────────────────────────────────────────
def sensitivity(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    tp = (pred.astype(bool) & gt.astype(bool)).sum()
    fn = (~pred.astype(bool) & gt.astype(bool)).sum()
    return (tp + smooth) / (tp + fn + smooth)


def specificity(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    tn = (~pred.astype(bool) & ~gt.astype(bool)).sum()
    fp = (pred.astype(bool) & ~gt.astype(bool)).sum()
    return (tn + smooth) / (tn + fp + smooth)


# ─── BraTS Evaluation (WT / TC / ET) ─────────────────────────────────────────
def evaluate_brats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all three BraTS tumor regions.
    pred, gt: 2D or 3D arrays with class indices {0,1,2,3}
    Returns nested dict: region → metric → value
    """
    import config
    regions = {
        "whole_tumor":   config.WHOLE_TUMOR_LABELS,
        "tumor_core":    config.TUMOR_CORE_LABELS,
        "enhancing":     config.ENHANCING_LABELS,
    }
    results = {}
    for name, labels in regions.items():
        p = to_binary(pred, [l if l < 4 else 3 for l in labels])
        g = to_binary(gt,   [l if l < 4 else 3 for l in labels])
        results[name] = {
            "dice":        dice_score(p, g),
            "iou":         iou_score(p, g),
            "hausdorff95": hausdorff95(p, g),
            "sensitivity": sensitivity(p, g),
            "specificity": specificity(p, g),
        }
    return results


# ─── Batch-level (PyTorch) Dice for training ──────────────────────────────────
def batch_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                    num_classes: int, smooth: float = 1e-6) -> torch.Tensor:
    """
    Differentiable Dice loss for training.
    pred   : [B, C, H, W]  logits
    target : [B, H, W]     long
    """
    pred_soft = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (pred_soft * target_oh).sum(dim=dims)
    denom = (pred_soft + target_oh).sum(dim=dims)
    dice  = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


class CombinedLoss(torch.nn.Module):
    """
    Dice + Cross-Entropy combined loss.
    MPS note: class_weights tensor is registered as a buffer so it
    automatically moves to the correct device with .to(device).
    """
    def __init__(self, num_classes: int, class_weights=None,
                 dice_w: float = 0.6, ce_w: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.dice_w      = dice_w
        self.ce_w        = ce_w

        if class_weights is not None:
            # Register as buffer — moves automatically with model.to(device)
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float)
            )
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # CrossEntropyLoss with weights living on the same device as pred
        ce_loss = torch.nn.functional.cross_entropy(
            pred, target, weight=self.class_weights
        )
        dice_loss = batch_dice_loss(pred, target, self.num_classes)
        return self.dice_w * dice_loss + self.ce_w * ce_loss


# ─── Aggregator for epoch-level logging ───────────────────────────────────────
class MetricMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k not in self.values:
                self.values[k] = []
            self.values[k].append(v)

    def mean(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.values.items()}

    def __str__(self) -> str:
        m = self.mean()
        return " | ".join(f"{k}: {v:.4f}" for k, v in m.items())