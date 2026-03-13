from __future__ import annotations

import time
from datetime import datetime
from typing import List

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from app.core.anomaly_map import generate_anomaly_heatmap
from app.core.config import settings
from app.core.patchcore import build_patchcore
from app.core.postprocessor import postprocess_anomaly_map
from app.database import get_db
from app.models.database import DetectionResult as DetectionResultORM
from app.models.schemas import DetectionResultCreate, DetectionResultRead
from app.utils.file_handler import get_uploaded_path
from app.utils.visualizer import build_annotated_image


router = APIRouter(tags=["detect"])


@router.post(
    "/detect/{image_id}",
    response_model=DetectionResultRead,
    status_code=status.HTTP_200_OK,
)
def run_detection(
    image_id: str = Path(..., description="UUID of the uploaded image."),
    db: Session = Depends(get_db),
) -> DetectionResultRead:
    """Run PatchCore-based defect detection on a previously uploaded image."""

    path = get_uploaded_path(image_id)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found for id {image_id}.",
        )
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original_rgb, (224, 224))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0)

    model = build_patchcore(device="cpu")

    start = time.perf_counter()
    # This assumes the memory bank has been pre-trained and saved.
    image_scores, anomaly_maps = model.predict(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    image_score = float(image_scores[0].item())
    anomaly_map = anomaly_maps[0, 0]
    heatmap = generate_anomaly_heatmap(anomaly_map, image_size=(resized.shape[0], resized.shape[1]))

    status_label, anomaly_score, regions = postprocess_anomaly_map(
        heatmap,
        threshold=settings.model_threshold,
    )

    _, annotated_b64 = build_annotated_image(
        resized,
        heatmap,
        regions,
        status_label,
        anomaly_score,
        settings.model_threshold,
    )

    timestamp = datetime.utcnow()
    result_create = DetectionResultCreate(
        image_id=image_id,
        filename=path.name,
        status=status_label,
        anomaly_score=anomaly_score,
        threshold=settings.model_threshold,
        defect_regions=regions,
        annotated_image=annotated_b64,
        inference_time_ms=elapsed_ms,
        timestamp=timestamp,
    )

    orm = DetectionResultORM(
        image_id=result_create.image_id,
        filename=result_create.filename,
        status=result_create.status,
        anomaly_score=result_create.anomaly_score,
        threshold=result_create.threshold,
        defect_regions=[region.model_dump() for region in result_create.defect_regions],
        annotated_image_base64=result_create.annotated_image,
        inference_time_ms=result_create.inference_time_ms,
        timestamp=result_create.timestamp,
    )
    db.add(orm)
    db.commit()
    db.refresh(orm)

    response = DetectionResultRead(
        id=orm.id,
        image_id=orm.image_id,
        filename=orm.filename,
        status=orm.status,
        anomaly_score=orm.anomaly_score,
        threshold=orm.threshold,
        defect_regions=[
            type(result_create.defect_regions[0])(**region)  # DefectRegion
            for region in (orm.defect_regions or [])
        ]
        if orm.defect_regions
        else [],
        annotated_image=orm.annotated_image_base64 or "",
        inference_time_ms=orm.inference_time_ms,
        timestamp=orm.timestamp,
    )
    return response

