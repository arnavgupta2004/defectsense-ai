from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from app.models.schemas import DefectRegion, SeverityType, StatusType


def _severity_from_score(score: float) -> SeverityType:
    """Map a normalized anomaly score to a discrete severity label."""

    if score < 0.33:
        return "LOW"
    if score < 0.66:
        return "MEDIUM"
    return "HIGH"


def postprocess_anomaly_map(
    heatmap: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[StatusType, float, List[DefectRegion]]:
    """Classify image and localize defects from an anomaly heatmap.

    Parameters
    ----------
    heatmap:
        Normalized anomaly heatmap of shape (H, W) in ``[0, 1]``.
    threshold:
        Pixel-level decision threshold.

    Returns
    -------
    status:
        ``NORMAL`` or ``DEFECTIVE`` based on global anomaly score.
    anomaly_score:
        Max anomaly score in the heatmap.
    regions:
        List of detected defect regions with bounding boxes and severity.
    """

    h, w = heatmap.shape
    anomaly_score: float = float(heatmap.max())
    status: StatusType = "DEFECTIVE" if anomaly_score >= threshold else "NORMAL"

    binary = (heatmap >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[DefectRegion] = []
    image_area = float(h * w)

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        region_values = heatmap[mask == 255]
        if region_values.size == 0:
            continue
        region_score = float(region_values.mean())
        area_percent = 100.0 * float(region_values.size) / image_area
        severity = _severity_from_score(region_score)
        regions.append(
            DefectRegion(
                bbox=(int(x), int(y), int(bw), int(bh)),
                severity=severity,
                area_percent=area_percent,
            )
        )

    return status, anomaly_score, regions

