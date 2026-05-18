from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.database import DetectionResult as DetectionResultORM
from app.models.schemas import DefectRegion, DetectionResultCreate, DetectionResultRead


def upsert_detection_result(db: Session, payload: DetectionResultCreate) -> DetectionResultRead:
    """Insert or update a detection result keyed by ``image_id``."""

    stmt = select(DetectionResultORM).where(DetectionResultORM.image_id == payload.image_id)
    orm = db.scalars(stmt).first()

    regions_json = [region.model_dump() for region in payload.defect_regions]

    if orm is None:
        orm = DetectionResultORM(
            image_id=payload.image_id,
            filename=payload.filename,
            status=payload.status,
            anomaly_score=payload.anomaly_score,
            threshold=payload.threshold,
            defect_regions=regions_json,
            annotated_image_base64=payload.annotated_image,
            inference_time_ms=payload.inference_time_ms,
            timestamp=payload.timestamp,
        )
        db.add(orm)
    else:
        orm.filename = payload.filename
        orm.status = payload.status
        orm.anomaly_score = payload.anomaly_score
        orm.threshold = payload.threshold
        orm.defect_regions = regions_json
        orm.annotated_image_base64 = payload.annotated_image
        orm.inference_time_ms = payload.inference_time_ms
        orm.timestamp = payload.timestamp

    db.commit()
    db.refresh(orm)
    return _orm_to_read(orm)


def _orm_to_read(orm: DetectionResultORM) -> DetectionResultRead:
    regions = [
        DefectRegion(**region) if isinstance(region, dict) else region
        for region in (orm.defect_regions or [])
    ]
    return DetectionResultRead(
        id=orm.id,
        image_id=orm.image_id,
        filename=orm.filename,
        status=orm.status,
        anomaly_score=orm.anomaly_score,
        threshold=orm.threshold,
        defect_regions=regions,
        annotated_image=orm.annotated_image_base64 or "",
        inference_time_ms=orm.inference_time_ms,
        timestamp=orm.timestamp,
    )
