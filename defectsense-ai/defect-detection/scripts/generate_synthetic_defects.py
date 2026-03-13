from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def add_scratches(image: np.ndarray, num_scratches: int = 5) -> np.ndarray:
    """Add random scratch-like lines to the image."""

    h, w, _ = image.shape
    img = image.copy()
    for _ in range(num_scratches):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
        thickness = random.randint(1, 3)
        color = (random.randint(150, 255),) * 3
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=thickness)
    return img


def add_blobs(image: np.ndarray, num_blobs: int = 8) -> np.ndarray:
    """Add circular or elliptical blob defects."""

    h, w, _ = image.shape
    img = image.copy()
    for _ in range(num_blobs):
        center = (random.randint(0, w - 1), random.randint(0, h - 1))
        axes = (random.randint(5, 20), random.randint(5, 20))
        angle = random.randint(0, 180)
        color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness=-1)
    return img


def add_cracks(image: np.ndarray, num_cracks: int = 3) -> np.ndarray:
    """Add crack-like polylines."""

    h, w, _ = image.shape
    img = image.copy()
    for _ in range(num_cracks):
        points = []
        num_points = random.randint(3, 8)
        for _ in range(num_points):
            points.append([random.randint(0, w - 1), random.randint(0, h - 1)])
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        color = (random.randint(0, 50),) * 3
        thickness = random.randint(1, 3)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)
    return img


def add_discoloration(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Apply localized discoloration using HSV manipulation."""

    h, w, _ = image.shape
    img = image.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
    radius = random.randint(min(h, w) // 10, min(h, w) // 4)

    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

    hsv[..., 1][mask] *= 1 + strength * (random.random() - 0.5)
    hsv[..., 2][mask] *= 1 + strength * (random.random() - 0.5)

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img_discolored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_discolored


def generate_defects_for_image(image: np.ndarray) -> np.ndarray:
    """Apply a random combination of defect augmentations."""

    img = image.copy()
    if random.random() < 0.8:
        img = add_scratches(img)
    if random.random() < 0.8:
        img = add_blobs(img)
    if random.random() < 0.6:
        img = add_cracks(img)
    if random.random() < 0.5:
        img = add_discoloration(img)
    return img


def main() -> None:
    """Generate synthetic defects from normal images."""

    source_dir = Path("./data/custom/test/good")
    target_dir = Path("./data/custom/test/defective")
    target_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_paths.extend(source_dir.rglob(ext))

    if not image_paths:
        raise SystemExit(f"No source images found in {source_dir}")

    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        defective = generate_defects_for_image(bgr)
        out_path = target_dir / path.name
        cv2.imwrite(str(out_path), defective)
        print(f"Wrote synthetic defective image to {out_path}")


if __name__ == "__main__":
    main()

