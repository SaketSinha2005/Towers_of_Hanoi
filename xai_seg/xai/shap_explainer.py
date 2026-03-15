"""
xai_seg/shap_explainer.py — SHAP-based Feature Importance + Integrated Gradients

Fixes applied (MPS / Apple Silicon):
  1. All tensors explicitly float32 everywhere — MPS rejects float64
  2. SHAP background: robust collection, shape validated, float32 cast
  3. SHAP DeepExplainer: wrapper returns float32 scalar
  4. IG: Captum's IntegratedGradients uses torch.linspace which returns float64
     on MPS and crashes. REPLACED with a manual float32-safe IG implementation
     that does NOT use Captum at all — same math, fully MPS-compatible.
  5. All .numpy() calls guarded with .detach().cpu().float()
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[SHAP] shap not installed. Run: pip install shap")


# ─── SHAP Explainer ───────────────────────────────────────────────────────────
class SHAPExplainer:
    """
    SHAP Deep Explainer for U-Net segmentation.
    Explains WHICH modality channels contribute most to the tumor prediction.
    """

    def __init__(self, model: torch.nn.Module, background: torch.Tensor,
                 target_class: int = config.GRADCAM_CLASS):
        if not SHAP_AVAILABLE:
            raise ImportError("Install shap: pip install shap")

        self.model        = model
        self.target_class = target_class
        self.device       = next(model.parameters()).device

        # ── Validate + cast background ───────────────────────────────────────
        if not isinstance(background, torch.Tensor):
            background = torch.tensor(np.array(background), dtype=torch.float32)

        # Ensure [N, C, H, W] layout
        background = background.float()
        if background.ndim == 3:
            background = background.unsqueeze(0)   # [1, C, H, W]
        if background.ndim != 4:
            raise ValueError(
                f"[SHAP] Background must be [N, C, H, W], got {background.shape}"
            )

        background = background.to(self.device)

        self._wrapper = self._build_wrapper(target_class)

        print(f"[SHAP] Initializing DeepExplainer with "
              f"{background.shape[0]} background samples "
              f"shape={tuple(background.shape)} dtype={background.dtype}...")
        self.explainer = shap.DeepExplainer(self._wrapper, background)
        print("[SHAP] Ready.")

    def _build_wrapper(self, target_class: int):
        """Scalar-output wrapper so SHAP can compute attributions."""
        model = self.model

        class Wrapper(torch.nn.Module):
            def forward(self, x):
                x   = x.float()            # guard against float64
                out = model(x)             # [B, C, H, W]
                return out[:, target_class, :, :].mean(dim=(1, 2))

        return Wrapper().to(self.device)

    def explain(self, image: torch.Tensor) -> np.ndarray:
        """
        Returns per-modality SHAP values [4, H, W].
        """
        image = image.float().to(self.device)
        vals  = self.explainer.shap_values(image)   # list or array

        if isinstance(vals, list):
            vals = vals[0]
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().float().numpy()

        vals = np.array(vals, dtype=np.float32)
        if vals.ndim == 4:
            vals = vals.squeeze(0)   # [4, H, W]
        return vals

    def modality_importance(self, image: torch.Tensor) -> dict:
        """Mean absolute SHAP per modality, normalised to sum=1."""
        vals       = np.abs(self.explain(image))   # [4, H, W]
        importance = {mod: float(vals[i].mean())
                      for i, mod in enumerate(config.MODALITIES)}
        total = sum(importance.values()) + 1e-8
        return {k: v / total for k, v in importance.items()}

    def spatial_importance_map(self, image: torch.Tensor) -> np.ndarray:
        """Aggregate across modalities → spatial heatmap [H, W]."""
        vals    = self.explain(image)
        heatmap = np.abs(vals).mean(axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.astype(np.float32)


# ─── Manual Integrated Gradients (MPS-safe, no Captum) ───────────────────────
class IntegratedGradientsExplainer:
    """
    Integrated Gradients — manually implemented to avoid Captum's internal
    torch.linspace call which returns float64 on MPS and crashes.

    Math is identical to the original IG paper (Sundararajan et al. 2017):
        IG(x) = (x - x') * (1/n) * SUM_k grad_F(x' + k/n * (x - x'))

    This implementation stays entirely in float32 on any device.
    No Captum dependency.
    """

    def __init__(self, model: torch.nn.Module,
                 target_class: int = config.GRADCAM_CLASS,
                 n_steps: int = config.IG_STEPS):
        self.model        = model
        self.target_class = target_class
        self.n_steps      = n_steps
        self.device       = next(model.parameters()).device

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar score for target class given input x [B, C, H, W]."""
        out = self.model(x.float())
        return out[:, self.target_class, :, :].sum(dim=(1, 2))

    def explain(self, image: torch.Tensor,
                baseline: torch.Tensor = None) -> np.ndarray:
        """
        Args:
            image    : [1, 4, H, W]  float32
            baseline : [1, 4, H, W]  float32  (default: zeros)
        Returns:
            attributions : [4, H, W]  float32
        """
        # ── Always float32 ────────────────────────────────────────────────────
        image = image.float().to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(image, dtype=torch.float32)
        else:
            baseline = baseline.float().to(self.device)

        delta = image - baseline   # [1, 4, H, W]

        # ── Build alphas as explicit float32 (avoids Captum linspace bug) ────
        alphas = torch.linspace(0.0, 1.0, self.n_steps,
                                dtype=torch.float32,
                                device=self.device)

        integrated_grads = torch.zeros_like(image, dtype=torch.float32)

        self.model.eval()
        for alpha in alphas:
            interp = (baseline + alpha * delta).float().detach().requires_grad_(True)

            score = self._score(interp).sum()
            self.model.zero_grad()
            score.backward()

            if interp.grad is not None:
                integrated_grads = integrated_grads + interp.grad.float().detach()
            else:
                print(f"  [IG] Warning: grad is None at alpha={alpha:.3f}")

        # Average over steps, scale by (x - x')
        integrated_grads = (integrated_grads / float(self.n_steps)) * delta.float()

        # Convergence diagnostic
        with torch.no_grad():
            f_x    = self._score(image.float()).sum().item()
            f_base = self._score(baseline.float()).sum().item()
            err    = abs(integrated_grads.sum().item() - (f_x - f_base))
        print(f"  [IG] Convergence delta: {err:.6f}")

        # Safe MPS → numpy: detach + cpu + float() before .numpy()
        return integrated_grads.squeeze(0).detach().cpu().float().numpy()  # [4,H,W]

    def spatial_heatmap(self, image: torch.Tensor) -> np.ndarray:
        """Aggregated spatial attribution [H, W] in [0, 1]."""
        attrs   = np.abs(self.explain(image)).mean(axis=0)
        heatmap = (attrs - attrs.min()) / (attrs.max() - attrs.min() + 1e-8)
        return heatmap.astype(np.float32)