from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import cv2
import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.core.config import settings
from app.core.patchcore import build_patchcore
from app.core.preprocessor import preprocess_batch
from app.state import TRAINING_STATE
from app.models.schemas import TrainRequest, TrainStatus
from app.services.model_service import memory_bank_size, reload_patchcore


router = APIRouter(tags=["train"])


def _collect_training_batches(dataset_dir: Path) -> tuple[List[torch.Tensor], int]:
    """Load images from ``dataset_dir`` and return preprocessed batches."""

    image_paths: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_paths.extend(dataset_dir.rglob(ext))
    if not image_paths:
        raise RuntimeError(f"No training images found in {dataset_dir}")

    batches: List[torch.Tensor] = []
    current_rgb: List[object] = []
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        current_rgb.append(rgb)
        if len(current_rgb) >= settings.train_batch_size:
            batches.append(preprocess_batch(current_rgb, size=settings.image_size))
            current_rgb = []
    if current_rgb:
        batches.append(preprocess_batch(current_rgb, size=settings.image_size))

    if not batches:
        raise RuntimeError(f"No readable training images in {dataset_dir}")

    return batches, len(image_paths)


def _train_patchcore_background(dataset_dir: Path) -> None:
    """Background task to train PatchCore on normal images."""

    try:
        TRAINING_STATE.update(
            {
                "status": "TRAINING",
                "message": f"Training on {dataset_dir}",
                "last_trained_at": None,
            }
        )
        batches, num_images = _collect_training_batches(dataset_dir)
        model = build_patchcore(device=settings.model_device)
        model.fit(batches)
        model.save_memory_bank()
        reload_patchcore()

        TRAINING_STATE.update(
            {
                "status": "READY",
                "last_trained_at": datetime.now(timezone.utc),
                "message": f"Trained on {num_images} images.",
                "memory_bank_size": memory_bank_size(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        TRAINING_STATE.update({"status": "ERROR", "message": str(exc)})


@router.post("/train", response_model=TrainStatus, status_code=status.HTTP_202_ACCEPTED)
def trigger_training(
    payload: TrainRequest,
    background_tasks: BackgroundTasks,
) -> TrainStatus:
    """Trigger background training on normal (good) images."""

    dataset_dir = Path(payload.dataset_path) if payload.dataset_path else settings.train_dir
    if not dataset_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Training directory {dataset_dir} does not exist.",
        )

    if TRAINING_STATE.get("status") == "TRAINING":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is already in progress.",
        )

    TRAINING_STATE.update(
        {
            "status": "TRAINING",
            "message": f"Queued training on {dataset_dir}",
            "last_trained_at": None,
        }
    )
    background_tasks.add_task(_train_patchcore_background, dataset_dir)
    return TrainStatus(**TRAINING_STATE)  # type: ignore[arg-type]
