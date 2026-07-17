"""Image preprocessing that mirrors the original training pipeline.

The model was trained on MURA images that were resized to 256x256 with
aspect ratio preserved (padded to square), then center-cropped to 224x224
and scaled to [0, 1]. ImageNet BGR mean subtraction happens *inside* the
exported model graph, so it is intentionally not repeated here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

RESIZE_SIZE = 256
CROP_SIZE = 224


def resize_keep_aspect(img: Image.Image, desired_size: int = RESIZE_SIZE) -> np.ndarray:
    """Resize so the longest side equals ``desired_size``, padding to square."""
    old_size = img.size  # (width, height)
    ratio = desired_size / max(old_size)
    new_size = tuple(int(x * ratio) for x in old_size)

    img = img.convert("RGB").resize(new_size, Image.LANCZOS)

    canvas = Image.new("RGB", (desired_size, desired_size))
    canvas.paste(
        img,
        ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2),
    )
    return np.asarray(canvas)


def center_crop(img: np.ndarray, height: int = CROP_SIZE, width: int = CROP_SIZE) -> np.ndarray:
    """Central crop of an HWC image array."""
    h, w = img.shape[:2]
    top = int(np.ceil((h - height) / 2))
    left = int(np.ceil((w - width) / 2))
    return img[top : top + height, left : left + width]


def prepare_input(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 NHWC batch of one, scaled to [0, 1]."""
    arr = resize_keep_aspect(img)
    arr = center_crop(arr)
    arr = arr.astype(np.float32) / 255.0
    return arr[np.newaxis, ...]
