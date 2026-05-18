from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from app.core.inference import run_inference_on_path
from app.database import get_db
from app.models.schemas import DetectionResultRead
from app.repositories.detection import upsert_detection_result
from app.utils.file_handler import get_uploaded_path


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

    try:
        path = get_uploaded_path(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found for id {image_id}.",
        ) from exc

    result_create, _, _ = run_inference_on_path(path, image_id=image_id)
    return upsert_detection_result(db, result_create)
