from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


def generate_anomaly_heatmap(
    anomaly_map: torch.Tensor,
    image_size: Tuple[int, int],
    sigma: int = 4,
) -> np.ndarray:
    """Generate a smoothed, normalized anomaly heatmap.

    Parameters
    ----------
    anomaly_map:
        Tensor of shape (H, W) or (1, H, W) with patch-level anomaly scores.
    image_size:
        Target ``(height, width)`` for the upsampled map.
    sigma:
        Standard deviation for Gaussian smoothing.
    """

    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.squeeze(0)
    anomaly_np = anomaly_map.detach().cpu().numpy()
    anomaly_np = cv2.resize(anomaly_np, (image_size[1], image_size[0]), interpolation=cv2.INTER_CUBIC)
    if sigma > 0:
        anomaly_np = cv2.GaussianBlur(anomaly_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    anomaly_np -= anomaly_np.min()
    denom = anomaly_np.max() - anomaly_np.min() + 1e-8
    anomaly_np = anomaly_np / denom
    return anomaly_np.astype(np.float32)

