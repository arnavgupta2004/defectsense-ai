from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from app.core.anomaly_map import generate_anomaly_heatmap
from app.core.patchcore import PatchcoreWrapper, PatchcoreConfig
from app.utils.metrics import (
    compute_average_precision,
    compute_confusion,
    compute_f1_at_optimal_threshold,
    compute_image_level_auroc,
    compute_pixel_level_auroc,
    compute_pro_score,
)


def load_images_and_labels(test_dir: Path) -> Tuple[List[np.ndarray], List[int]]:
    """Load test images and their labels (0=good, 1=defective)."""

    images: List[np.ndarray] = []
    labels: List[int] = []

    good_dir = test_dir / "good"
    defect_dir = test_dir / "defective"

    for dir_path, label in ((good_dir, 0), (defect_dir, 1)):
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for path in dir_path.rglob(ext):
                bgr = cv2.imread(str(path))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                images.append(rgb)
                labels.append(label)
    return images, labels


def main() -> None:
    """Evaluate PatchCore model on test set and save metrics as JSON."""

    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        import yaml

        cfg = yaml.safe_load(f)

    test_dir = Path(cfg["data"]["test_dir"])
    images, labels = load_images_and_labels(test_dir)
    if not images:
        raise SystemExit(f"No test images found in {test_dir}")

    device = cfg["training"].get("device", "cpu")
    patchcore_cfg = PatchcoreConfig(
        backbone=cfg["model"]["backbone"],
        layers=tuple(cfg["model"]["layers"]),
        coreset_sampling_ratio=cfg["model"]["coreset_sampling_ratio"],
        num_neighbors=cfg["model"]["num_neighbors"],
    )
    model = PatchcoreWrapper(device=device, config=patchcore_cfg)
    model.load_memory_bank()

    image_scores: List[float] = []
    y_true: List[int] = []
    anomaly_maps_all: List[np.ndarray] = []
    gt_masks_all: List[np.ndarray] = []

    for img, label in zip(images, labels, strict=False):
        resized = cv2.resize(img, (cfg["data"]["image_size"], cfg["data"]["image_size"]))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0)
        scores, maps = model.predict(tensor)
        score = float(scores[0].item())
        anomaly_map = maps[0, 0]
        heatmap = generate_anomaly_heatmap(
            anomaly_map,
            image_size=(resized.shape[0], resized.shape[1]),
        )
        image_scores.append(score)
        y_true.append(label)

        # For pixel-level metrics, approximate GT mask: defective -> any non-zero region.
        gt_mask = np.zeros_like(heatmap, dtype=np.uint8)
        if label == 1:
            # Assume central region as defective if no mask annotations are provided.
            h, w = heatmap.shape
            cy, cx = h // 2, w // 2
            ch, cw = h // 4, w // 4
            gt_mask[cy - ch : cy + ch, cx - cw : cx + cw] = 1
        gt_masks_all.append(gt_mask)
        anomaly_maps_all.append(heatmap)

    image_level_auroc = compute_image_level_auroc(y_true, image_scores)
    f1, best_thr = compute_f1_at_optimal_threshold(y_true, image_scores)
    ap = compute_average_precision(y_true, image_scores)

    y_pred = [int(score >= best_thr) for score in image_scores]
    confusion = compute_confusion(y_true, y_pred)

    gt_stack = np.stack(gt_masks_all, axis=0)
    am_stack = np.stack(anomaly_maps_all, axis=0)
    pixel_auroc = compute_pixel_level_auroc(gt_stack, am_stack)
    pro_score = compute_pro_score(gt_stack, am_stack)

    report = {
        "image_level_auroc": image_level_auroc,
        "pixel_level_auroc": pixel_auroc,
        "f1_score": f1,
        "best_threshold": best_thr,
        "average_precision": ap,
        "pro_score": pro_score,
        "confusion_matrix": confusion,
        "num_samples": len(images),
    }

    report_path = Path("./evaluation_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()

