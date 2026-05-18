#!/usr/bin/env python3
"""Create minimal normal training images for first-time setup."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def _render_or_low_memory() -> bool:
    r = os.getenv("RENDER", "").strip().lower()
    l = os.getenv("LOW_MEMORY", "").strip().lower()
    return r in ("1", "true", "yes") or l in ("1", "true", "yes")


def _image_size() -> int:
    default = "128" if _render_or_low_memory() else "224"
    raw = os.getenv("IMAGE_SIZE", default) or default
    return int(raw)


def _train_count() -> int:
    default = "6" if _render_or_low_memory() else "12"
    raw = os.getenv("BOOTSTRAP_TRAIN_IMAGES", default) or default
    return int(raw)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data" / "custom" / "train" / "good"
    test_good = root / "data" / "custom" / "test" / "good"
    test_defect = root / "data" / "custom" / "test" / "defective"

    for directory in (train_dir, test_good, test_defect):
        directory.mkdir(parents=True, exist_ok=True)

    size = _image_size()
    train_n = _train_count()
    rng = np.random.default_rng(42)

    for i in range(train_n):
        base = rng.integers(180, 220, size=(size, size, 3), dtype=np.uint8)
        noise = rng.integers(0, 25, size=base.shape, dtype=np.uint8)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(train_dir / f"normal_{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    for i in range(max(2, train_n // 3)):
        img = rng.integers(160, 210, size=(size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(test_good / f"test_good_{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Wrote {train_n} training images ({size}x{size}) to {train_dir}")
    print("Run: python -m training.train_patchcore")


if __name__ == "__main__":
    main()
