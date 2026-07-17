"""Model inference and SmoothGrad sensitivity maps.

The original 2021 project shipped a TensorFlow 1.x frozen DenseNet-121
graph. Those exact trained weights were converted to ONNX
(``scripts/convert_model.py``) and are loaded here as a native PyTorch
module via ``onnx2torch``, which restores autograd support so the
SmoothGrad visualization (https://arxiv.org/abs/1706.03825) works the
same way it did in the original app.
"""

from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .preprocessing import prepare_input

MODEL_PATH = Path(os.environ.get(
    "XRV_MODEL_PATH",
    Path(__file__).resolve().parent.parent / "model" / "densenet121_mura.onnx",
))

# SmoothGrad settings (original app: 64 samples, sigma 0.0125).
SMOOTHGRAD_SAMPLES = int(os.environ.get("XRV_SMOOTHGRAD_SAMPLES", "48"))
SMOOTHGRAD_SIGMA = float(os.environ.get("XRV_SMOOTHGRAD_SIGMA", "0.0125"))
SMOOTHGRAD_BATCH = int(os.environ.get("XRV_SMOOTHGRAD_BATCH", "16"))
DISPLAY_SIZE = 600
ABNORMAL_CLASS = 1

# Test-time augmentation. Averaging the model's prediction over a few
# label-preserving views (horizontal flip, small rotations, mild zoom) — the
# "Test Time Multi-View" stage from the original project design. The model was
# trained with exactly these augmentations (flip, rotate up to 30 deg, zoom),
# so every view stays inside its training distribution.
TTA_ENABLED = os.environ.get("XRV_TTA", "1") != "0"

# Compact viridis colormap (anchor points, linearly interpolated) so we can
# reproduce the original matplotlib overlay without a matplotlib dependency.
_VIRIDIS_ANCHORS = np.array([
    (68, 1, 84), (71, 24, 106), (72, 40, 120), (71, 57, 132),
    (66, 73, 140), (60, 88, 145), (54, 103, 148), (47, 117, 149),
    (42, 130, 150), (37, 144, 150), (33, 158, 148), (32, 171, 144),
    (41, 184, 135), (61, 195, 122), (91, 206, 101), (127, 214, 75),
    (167, 219, 51), (208, 222, 39), (253, 231, 37),
], dtype=np.float32)


def _viridis(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to viridis RGB (uint8)."""
    v = np.clip(values, 0.0, 1.0) * (len(_VIRIDIS_ANCHORS) - 1)
    lo = np.floor(v).astype(int)
    hi = np.minimum(lo + 1, len(_VIRIDIS_ANCHORS) - 1)
    frac = (v - lo)[..., None]
    rgb = _VIRIDIS_ANCHORS[lo] * (1 - frac) + _VIRIDIS_ANCHORS[hi] * frac
    return rgb.astype(np.uint8)


@dataclass
class AnalysisResult:
    normal_probability: float
    abnormal_probability: float
    input_png: bytes
    heatmap_png: bytes


class XRayModel:
    """DenseNet-121 MURA abnormality model with SmoothGrad visualization."""

    def __init__(self, model_path: Path = MODEL_PATH):
        import onnx
        from onnx2torch import convert

        # Load + shape-infer in memory (avoids temp files next to the model,
        # which sync clients like OneDrive can lock mid-write).
        model = onnx.shape_inference.infer_shapes(onnx.load(str(model_path)))
        self.module = convert(model).eval()
        self._lock = threading.Lock()

    def _forward(self, images: torch.Tensor):
        """Run the graph. Returns (logits, probability)."""
        return self.module(images)

    @torch.no_grad()
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Probabilities [normal, abnormal] for an NHWC [0,1] float batch."""
        images = torch.from_numpy(batch).float()
        _, prob = self._forward(images)
        return prob.numpy()

    @torch.no_grad()
    def predict_tta(self, batch: np.ndarray) -> np.ndarray:
        """Multi-view test-time-augmented probability for a single image.

        Runs the model over several label-preserving views and averages the
        softmax outputs. The average of probability vectors is itself a valid
        distribution, so the result still sums to 1. Falls back to a plain
        single-view prediction when TTA is disabled.
        """
        if not TTA_ENABLED:
            return self.predict(batch)[0]
        views = _tta_views(batch[0])
        probs = self.predict(views)  # [n_views, 2]
        return probs.mean(axis=0)

    def smoothgrad(self, batch: np.ndarray) -> np.ndarray:
        """SmoothGrad sensitivity map for the abnormal class.

        Averages input gradients of the abnormal-class logit over
        noise-perturbed copies of the image, exactly like the original app.
        """
        base = torch.from_numpy(batch).float()
        grads = []
        remaining = SMOOTHGRAD_SAMPLES
        while remaining > 0:
            n = min(SMOOTHGRAD_BATCH, remaining)
            remaining -= n
            noise = torch.randn(n, *base.shape[1:]) * SMOOTHGRAD_SIGMA
            noisy = (base.repeat(n, 1, 1, 1) + noise).requires_grad_(True)
            logits, _ = self._forward(noisy)
            # Same as the original's masked_logits (logits * one_hot(label)):
            # differentiate only the abnormal-class logit.
            logits[:, ABNORMAL_CLASS].sum().backward()
            grads.append(noisy.grad.detach())
        mean_grad = torch.cat(grads).mean(dim=0).numpy()  # HWC

        # Same post-processing as the original: |grad| summed over channels,
        # thresholded at the 99th percentile, then smoothed with a 5x5 box blur.
        mask = np.abs(mean_grad).sum(axis=-1)
        threshold = np.percentile(mask.ravel(), 99)
        mask[mask < threshold] = 0.0
        return _box_blur(mask, kernel=5)

    def analyze(self, image: Image.Image) -> AnalysisResult:
        batch = prepare_input(image)

        with self._lock:
            prob = self.predict_tta(batch)
            mask = self.smoothgrad(batch)

        display = np.clip(batch[0] * 255.0, 0, 255).astype(np.uint8)
        input_img = Image.fromarray(display).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR
        )

        # Overlay: viridis-colored sensitivity map at 35% alpha, like the
        # original matplotlib rendering.
        mask_norm = mask / mask.max() if mask.max() > 0 else mask
        mask_img = Image.fromarray(_viridis(mask_norm)).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR
        )
        heatmap = Image.blend(input_img, mask_img, alpha=0.35)

        return AnalysisResult(
            normal_probability=round(float(prob[0]) * 100, 1),
            abnormal_probability=round(float(prob[1]) * 100, 1),
            input_png=_to_png(input_img),
            heatmap_png=_to_png(heatmap),
        )


def _tta_views(img: np.ndarray) -> np.ndarray:
    """Label-preserving views of an HWC [0,1] image, stacked as an NHWC batch.

    View 0 is always the original image, so callers can recover the canonical
    prediction from ``views[0]`` if needed.
    """
    views = [img, np.ascontiguousarray(img[:, ::-1, :])]  # original + h-flip

    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    for angle in (-8, 8):  # mild rotations, well within the +/-30 deg used in training
        rot = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        views.append(np.asarray(rot).astype(np.float32) / 255.0)

    # Mild center zoom (~1.11x): crop the central 90% and resize back.
    h, w = img.shape[:2]
    ch, cw = int(h * 0.9), int(w * 0.9)
    top, left = (h - ch) // 2, (w - cw) // 2
    crop = Image.fromarray((np.clip(img[top:top + ch, left:left + cw], 0, 1) * 255).astype(np.uint8))
    zoom = crop.resize((w, h), Image.BILINEAR)
    views.append(np.asarray(zoom).astype(np.float32) / 255.0)

    return np.stack(views).astype(np.float32)


def _box_blur(mask: np.ndarray, kernel: int = 5) -> np.ndarray:
    """5x5 mean filter, equivalent to the original cv2.filter2D smoothing."""
    pad = kernel // 2
    padded = np.pad(mask, pad, mode="edge")
    out = np.zeros_like(mask)
    for dy in range(kernel):
        for dx in range(kernel):
            out += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out / (kernel * kernel)


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
