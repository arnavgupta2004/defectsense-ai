from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_image_level_auroc(
    y_true: Iterable[int],
    y_scores: Iterable[float],
) -> float:
    """Compute image-level AUROC."""

    y_true_arr = np.asarray(list(y_true))
    y_scores_arr = np.asarray(list(y_scores))
    return float(roc_auc_score(y_true_arr, y_scores_arr))


def compute_pixel_level_auroc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """Compute pixel-level AUROC given flattened GT and score maps."""

    y_true_flat = y_true.astype(np.uint8).ravel()
    y_scores_flat = y_scores.astype(np.float32).ravel()
    return float(roc_auc_score(y_true_flat, y_scores_flat))


def compute_f1_at_optimal_threshold(
    y_true: Iterable[int],
    y_scores: Iterable[float],
) -> Tuple[float, float]:
    """Compute F1 score and threshold that maximizes it."""

    y_true_arr = np.asarray(list(y_true))
    y_scores_arr = np.asarray(list(y_scores))
    thresholds = np.linspace(0, 1, num=101)
    best_f1 = 0.0
    best_thr = 0.5
    for thr in thresholds:
        y_pred = (y_scores_arr >= thr).astype(int)
        f1 = f1_score(y_true_arr, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return float(best_f1), best_thr


def compute_average_precision(
    y_true: Iterable[int],
    y_scores: Iterable[float],
) -> float:
    """Compute Average Precision score."""

    y_true_arr = np.asarray(list(y_true))
    y_scores_arr = np.asarray(list(y_scores))
    return float(average_precision_score(y_true_arr, y_scores_arr))


def compute_confusion(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> Dict[str, int]:
    """Compute confusion matrix as a dictionary."""

    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_pro_score(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
    num_thresholds: int = 50,
) -> float:
    """Approximate Per-Region Overlap (PRO) score.

    This implementation sweeps thresholds and computes mean IoU between predicted
    and ground-truth regions, then averages across thresholds.
    """

    thresholds = np.linspace(0, 1, num=num_thresholds)
    ious = []
    for thr in thresholds:
        pred = (anomaly_maps >= thr).astype(np.uint8)
        intersection = (gt_masks & pred).sum()
        union = (gt_masks | pred).sum()
        if union == 0:
            continue
        ious.append(intersection / float(union))
    if not ious:
        return 0.0
    return float(np.mean(ious))

