"""
xai_seg/gradcam.py — Grad-CAM & Grad-CAM++ for U-Net Segmentation

Fixes applied (MPS / Apple Silicon):
  1. Input always cast to .float() BEFORE .requires_grad_(True)
     — MPS only allows autograd on float32 leaf tensors
  2. All intermediate tensors .detach().cpu().float() before .numpy()
  3. Null checks on hooks.gradients / hooks.activations
  4. GradCAMExplainer.explain() always passes .float() to inner CAM

Usage:
    explainer = GradCAMExplainer(model, target_layer="encoder4")
    heatmap   = explainer.explain(image_tensor, target_class=3)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Hook Manager ─────────────────────────────────────────────────────────────
class HookManager:
    """Registers forward and backward hooks on a target layer."""
    def __init__(self):
        self.activations = None
        self.gradients   = None
        self._handles    = []

    def register(self, layer: torch.nn.Module):
        self._handles.append(
            layer.register_forward_hook(self._save_activation))
        self._handles.append(
            layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for segmentation models.
    Returns a spatial heatmap [H, W] in [0, 1] showing which regions
    most influenced the prediction of `target_class`.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.hooks = HookManager()
        self.hooks.register(target_layer)

    def explain(self, image: torch.Tensor, target_class: int,
                target_size: tuple = None) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device

        # FIX: float32 BEFORE requires_grad — MPS only allows grad on float32
        image = image.to(device).float().detach().clone().requires_grad_(True)

        output = self.model(image)                        # [1, C, H, W]
        score  = output[:, target_class, :, :].sum()

        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients   = self.hooks.gradients    # [1, C, h, w]
        activations = self.hooks.activations  # [1, C, h, w]

        if gradients is None or activations is None:
            h = target_size[0] if target_size else image.shape[2]
            w = target_size[1] if target_size else image.shape[3]
            print("[GradCAM] Warning: hooks returned None")
            return np.zeros((h, w), dtype=np.float32)

        weights = gradients.mean(dim=(2, 3), keepdim=True)           # [1, C, 1, 1]
        cam     = (weights * activations).sum(dim=1, keepdim=True)   # [1, 1, h, w]
        cam     = F.relu(cam)

        # FIX: detach + cpu + float() before numpy
        cam = cam.squeeze().detach().cpu().float().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        if target_size:
            cam = cv2.resize(cam, (target_size[1], target_size[0]))

        return cam.astype(np.float32)

    def cleanup(self):
        self.hooks.remove()


# ─── Grad-CAM++ ───────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    """
    Grad-CAM++ — better localization for multiple instances.
    Uses second-order gradient weighting.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.hooks = HookManager()
        self.hooks.register(target_layer)

    def explain(self, image: torch.Tensor, target_class: int,
                target_size: tuple = None) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device

        # FIX: float32 BEFORE requires_grad
        image = image.to(device).float().detach().clone().requires_grad_(True)

        output = self.model(image)
        score  = output[:, target_class, :, :].sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.hooks.gradients    # [1, C, h, w]
        acts  = self.hooks.activations  # [1, C, h, w]

        if grads is None or acts is None:
            h = target_size[0] if target_size else image.shape[2]
            w = target_size[1] if target_size else image.shape[3]
            print("[GradCAM++] Warning: hooks returned None")
            return np.zeros((h, w), dtype=np.float32)

        alpha_numer = grads ** 2
        alpha_denom = (2 * grads ** 2
                       + acts * (grads ** 3).sum(dim=(2, 3), keepdim=True)
                       + 1e-7)
        alpha   = alpha_numer / alpha_denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)

        # FIX: detach + cpu + float() before numpy
        cam = F.relu(cam).squeeze().detach().cpu().float().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        if target_size:
            cam = cv2.resize(cam, (target_size[1], target_size[0]))

        return cam.astype(np.float32)

    def cleanup(self):
        self.hooks.remove()


# ─── Score-CAM ────────────────────────────────────────────────────────────────
class ScoreCAM:
    """
    Score-CAM — gradient-free, uses activation maps as perturbation masks.
    Slower but more faithful.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model   = model
        self._act    = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self._act = out.detach()

    @torch.no_grad()
    def explain(self, image: torch.Tensor, target_class: int,
                target_size: tuple = None, top_k: int = 64) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device

        # FIX: float32 for MPS
        image = image.to(device).float()

        _ = self.model(image)
        activations    = self._act
        baseline       = self.model(torch.zeros_like(image))
        baseline_score = baseline[0, target_class].sum().item()

        H, W = image.shape[2], image.shape[3]
        k    = min(top_k, activations.shape[1])

        channel_scores = activations[0].abs().sum(dim=(1, 2))
        top_idx        = channel_scores.topk(k).indices

        scores = []
        for i in top_idx:
            mask = F.interpolate(
                activations[:, i:i+1, :, :],
                size=(H, W), mode="bilinear", align_corners=False
            )
            mask      = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            perturbed = image * mask
            out       = self.model(perturbed)
            scores.append(out[0, target_class].sum().item() - baseline_score)

        weights = F.softmax(
            torch.tensor(scores, dtype=torch.float32), dim=0
        ).numpy()

        cam = np.zeros((H, W), dtype=np.float32)
        for w_i, idx in zip(weights, top_idx.cpu().numpy()):
            act_i = F.interpolate(
                activations[:, idx:idx+1, :, :],
                size=(H, W), mode="bilinear", align_corners=False
            )
            # FIX: detach + cpu + float() before numpy
            act_i = act_i.squeeze().detach().cpu().float().numpy()
            act_i = (act_i - act_i.min()) / (act_i.max() - act_i.min() + 1e-8)
            cam  += w_i * act_i

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        if target_size:
            cam = cv2.resize(cam, (target_size[1], target_size[0]))

        return cam.astype(np.float32)

    def cleanup(self):
        self._handle.remove()


# ─── Unified Explainer Wrapper ────────────────────────────────────────────────
class GradCAMExplainer:
    """
    Unified Grad-CAM explainer.
    Resolves the target layer by dotted attribute name from the model.
    """
    def __init__(self, model: torch.nn.Module,
                 target_layer: str = config.GRADCAM_LAYER,
                 method: str = "gradcam"):
        self.model  = model
        self.method = method

        layer = self._get_layer(model, target_layer)
        print(f"[GradCAM] Target layer: {target_layer} | Method: {method}")

        if method == "gradcam":
            self._cam = GradCAM(model, layer)
        elif method == "gradcam++":
            self._cam = GradCAMPlusPlus(model, layer)
        elif method == "scorecam":
            self._cam = ScoreCAM(model, layer)
        else:
            raise ValueError(f"Unknown method: {method}. "
                             f"Choose: gradcam | gradcam++ | scorecam")

    @staticmethod
    def _get_layer(model, name: str):
        obj = model
        for p in name.split("."):
            if not hasattr(obj, p):
                raise AttributeError(
                    f"Model has no attribute '{p}' in layer path '{name}'"
                )
            obj = getattr(obj, p)
        return obj

    def explain(self, image: torch.Tensor,
                target_class: int = config.GRADCAM_CLASS,
                target_size: tuple = None) -> np.ndarray:
        # FIX: Always float32 into CAM
        return self._cam.explain(image.float(), target_class, target_size)

    def explain_all_classes(self, image: torch.Tensor,
                            target_size: tuple = None) -> dict:
        """Return heatmaps for all 4 tumor classes."""
        return {
            name: self.explain(image, cls_idx, target_size)
            for cls_idx, name in {
                0: "background", 1: "necrotic",
                2: "edema",      3: "enhancing"
            }.items()
        }

    def cleanup(self):
        self._cam.cleanup()