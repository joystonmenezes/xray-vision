"""Measure the robustness benefit of test-time augmentation (TTA).

Accuracy on MURA can only be measured against Stanford's labeled test set
(their terms, their account). What we *can* measure here, with no labels, is
**stability**: a good abnormality model should give the same answer when the
same X-ray is captured at a slightly different angle. This sweeps the input
through a range of rotations and reports how much the abnormal-probability
swings — single-view (the old behavior) vs. TTA (the new behavior).

    python scripts/benchmark_tta.py [image.png]

Lower swing = more robust. TTA smooths orientation sensitivity because each
prediction is itself averaged over several views.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.inference import XRayModel  # noqa: E402
from app.preprocessing import prepare_input  # noqa: E402

SWEEP = range(-20, 21, 4)  # rotations, -20 to +20 degrees


def main() -> None:
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "docs" / "example_input.png"
    )
    print(f"Image: {img_path.name}\n")

    model = XRayModel()
    original = Image.open(img_path).convert("RGB")

    single, tta = [], []
    print(f"{'angle':>6} | {'single-view':>12} | {'TTA':>8}")
    print("-" * 34)
    for angle in SWEEP:
        rotated = original.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        batch = prepare_input(rotated)
        s = float(model.predict(batch)[0][1]) * 100
        t = float(model.predict_tta(batch)[1]) * 100
        single.append(s)
        tta.append(t)
        print(f"{angle:>5} deg | {s:>11.2f}% | {t:>6.2f}%")

    s_std, t_std = statistics.pstdev(single), statistics.pstdev(tta)
    s_span, t_span = max(single) - min(single), max(tta) - min(tta)
    print("-" * 34)
    print(f"\nAcross {len(single)} orientations of the same study:")
    print(f"  std dev   single-view {s_std:5.2f}%   ->   TTA {t_std:5.2f}%   "
          f"({_delta(s_std, t_std)})")
    print(f"  max swing single-view {s_span:5.2f}%   ->   TTA {t_span:5.2f}%   "
          f"({_delta(s_span, t_span)})")
    print("\nLower = the app's answer changes less when the X-ray is tilted.")


def _delta(before: float, after: float) -> str:
    if before == 0:
        return "n/a"
    pct = (after - before) / before * 100
    return f"{pct:+.0f}%"


if __name__ == "__main__":
    main()
