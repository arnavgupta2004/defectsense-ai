from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database import DetectionResult as DetectionResultORM
from app.models.schemas import DetectionResultRead, ResultsFilter


router = APIRouter(tags=["results"])


@router.get("/results/{image_id}", response_model=DetectionResultRead)
def get_result(
    image_id: str = Path(..., description="UUID of the inspected image."),
    db: Session = Depends(get_db),
) -> DetectionResultRead:
    """Fetch a stored detection result for a given image id."""

    stmt = select(DetectionResultORM).where(DetectionResultORM.image_id == image_id)
    orm = db.scalars(stmt).first()
    if orm is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No result found for image id {image_id}.",
        )
    return DetectionResultRead(
        id=orm.id,
        image_id=orm.image_id,
        filename=orm.filename,
        status=orm.status,
        anomaly_score=orm.anomaly_score,
        threshold=orm.threshold,
        defect_regions=orm.defect_regions or [],
        annotated_image=orm.annotated_image_base64 or "",
        inference_time_ms=orm.inference_time_ms,
        timestamp=orm.timestamp,
    )


@router.get("/results", response_model=List[DetectionResultRead])
def list_results(
    status_filter: str | None = Query(default=None, alias="status"),
    filename_contains: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> List[DetectionResultRead]:
    """List past detection results with optional filters."""

    stmt = select(DetectionResultORM)
    if status_filter:
        stmt = stmt.where(DetectionResultORM.status == status_filter)
    if filename_contains:
        stmt = stmt.where(DetectionResultORM.filename.contains(filename_contains))
    stmt = stmt.order_by(DetectionResultORM.timestamp.desc()).limit(limit)

    results: List[DetectionResultRead] = []
    for orm in db.scalars(stmt).all():
        results.append(
            DetectionResultRead(
                id=orm.id,
                image_id=orm.image_id,
                filename=orm.filename,
                status=orm.status,
                anomaly_score=orm.anomaly_score,
                threshold=orm.threshold,
                defect_regions=orm.defect_regions or [],
                annotated_image=orm.annotated_image_base64 or "",
                inference_time_ms=orm.inference_time_ms,
                timestamp=orm.timestamp,
            )
        )
    return results

