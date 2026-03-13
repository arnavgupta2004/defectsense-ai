from __future__ import annotations

import base64
from io import BytesIO
from typing import Iterable, Tuple

import cv2
import numpy as np
from PIL import Image

from app.models.schemas import DefectRegion, StatusType


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a heatmap onto an RGB image using the JET colormap."""

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heatmap_color, alpha, image, 1 - alpha, 0)
    return overlay


def draw_bounding_boxes(
    image: np.ndarray,
    regions: Iterable[DefectRegion],
    status: StatusType,
) -> np.ndarray:
    """Draw defect bounding boxes and status color code onto the image."""

    color = (0, 255, 0) if status == "NORMAL" else (255, 0, 0)
    img = image.copy()
    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
    return img


def add_text(
    image: np.ndarray,
    status: StatusType,
    anomaly_score: float,
    threshold: float,
) -> np.ndarray:
    """Add anomaly score and status text to the image."""

    img = image.copy()
    text = f"{status} | score={anomaly_score:.3f} | thr={threshold:.2f}"
    cv2.putText(
        img,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def image_to_base64(image: np.ndarray, format: str = "PNG") -> str:
    """Encode an RGB numpy image as a base64 string."""

    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def build_annotated_image(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    regions: Iterable[DefectRegion],
    status: StatusType,
    anomaly_score: float,
    threshold: float,
) -> Tuple[np.ndarray, str]:
    """Create an annotated image overlay and return both array and base64."""

    overlay = overlay_heatmap(original_rgb, heatmap)
    overlay = draw_bounding_boxes(overlay, regions, status)
    overlay = add_text(overlay, status, anomaly_score, threshold)
    encoded = image_to_base64(overlay)
    return overlay, encoded

