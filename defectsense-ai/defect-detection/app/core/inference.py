from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from app.core.anomaly_map import generate_anomaly_heatmap
from app.core.config import settings
from app.runtime_settings import get_threshold
from app.core.postprocessor import postprocess_anomaly_map
from app.core.preprocessor import preprocess_image
from app.models.schemas import DetectionResultCreate
from app.services.model_service import require_ready_model
from app.utils.visualizer import build_annotated_image


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an image from disk as RGB uint8."""

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_inference_on_rgb(
    rgb: np.ndarray,
    *,
    image_id: str,
    filename: str,
) -> Tuple[DetectionResultCreate, np.ndarray, np.ndarray]:
    """Run PatchCore inference on an RGB image and build a detection payload."""

    model = require_ready_model()
    tensor = preprocess_image(rgb, size=settings.image_size).unsqueeze(0)

    start = time.perf_counter()
    image_scores, anomaly_maps = model.predict(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    anomaly_map = anomaly_maps[0, 0]
    display_rgb = cv2.resize(rgb, (settings.image_size, settings.image_size))
    heatmap = generate_anomaly_heatmap(
        anomaly_map,
        image_size=(display_rgb.shape[0], display_rgb.shape[1]),
    )

    threshold = get_threshold()
    status_label, anomaly_score, regions = postprocess_anomaly_map(
        heatmap,
        threshold=threshold,
    )

    _, annotated_b64 = build_annotated_image(
        display_rgb,
        heatmap,
        regions,
        status_label,
        anomaly_score,
        threshold,
    )

    result = DetectionResultCreate(
        image_id=image_id,
        filename=filename,
        status=status_label,
        anomaly_score=anomaly_score,
        threshold=threshold,
        defect_regions=regions,
        annotated_image=annotated_b64,
        inference_time_ms=elapsed_ms,
        timestamp=datetime.now(timezone.utc),
    )
    return result, display_rgb, heatmap


def run_inference_on_path(path: Path, *, image_id: str) -> Tuple[DetectionResultCreate, np.ndarray, np.ndarray]:
    """Load an uploaded image and run inference."""

    rgb = load_rgb_image(path)
    return run_inference_on_rgb(rgb, image_id=image_id, filename=path.name)
