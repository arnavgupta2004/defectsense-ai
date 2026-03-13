from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import torch
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.core.config import settings
from app.core.patchcore import build_patchcore
from app.main import TRAINING_STATE
from app.models.schemas import TrainRequest, TrainStatus


router = APIRouter(tags=["train"])


def _train_patchcore_background(dataset_dir: Path) -> None:
    """Background task to train PatchCore on normal images."""

    try:
        TRAINING_STATE.update({"status": "TRAINING", "message": f"Training on {dataset_dir}"})
        image_paths: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_paths.extend(Path(dataset_dir).rglob(ext))
        if not image_paths:
            raise RuntimeError(f"No training images found in {dataset_dir}")

        model = build_patchcore(device="cpu")
        batches: List[torch.Tensor] = []
        current: List[torch.Tensor] = []
        for path in image_paths:
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224))
            tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
            current.append(tensor)
            if len(current) >= 16:
                batches.append(torch.stack(current, dim=0))
                current = []
        if current:
            batches.append(torch.stack(current, dim=0))

        model.fit(batches)
        model.save_memory_bank()

        TRAINING_STATE.update(
            {
                "status": "READY",
                "last_trained_at": datetime.utcnow(),
                "message": f"Trained on {len(image_paths)} images.",
                "memory_bank_size": None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        TRAINING_STATE.update({"status": "ERROR", "message": str(exc)})


@router.post("/train", response_model=TrainStatus, status_code=status.HTTP_202_ACCEPTED)
def trigger_training(
    payload: TrainRequest,
    background_tasks: BackgroundTasks,
) -> TrainStatus:
    """Trigger background training on ``data/custom/train/good`` or provided dataset."""

    dataset_dir = Path(payload.dataset_path) if payload.dataset_path else Path(
        "./data/custom/train/good"
    )
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

    background_tasks.add_task(_train_patchcore_background, dataset_dir)
    return TrainStatus(**TRAINING_STATE)  # type: ignore[arg-type]

