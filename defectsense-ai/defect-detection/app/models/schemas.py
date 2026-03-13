from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


StatusType = Literal["NORMAL", "DEFECTIVE"]
SeverityType = Literal["LOW", "MEDIUM", "HIGH"]


class DefectRegion(BaseModel):
    """Region in the image predicted as defective."""

    bbox: Tuple[int, int, int, int] = Field(
        ...,
        description="Bounding box in [x, y, w, h] format.",
    )
    severity: SeverityType
    area_percent: float


class DetectionResultBase(BaseModel):
    """Shared attributes for detection responses."""

    image_id: str
    filename: str
    status: StatusType
    anomaly_score: float
    threshold: float
    defect_regions: List[DefectRegion]
    annotated_image: str
    inference_time_ms: float
    timestamp: datetime


class DetectionResultCreate(BaseModel):
    """Payload for creating a detection result in the database."""

    image_id: str
    filename: str
    status: StatusType
    anomaly_score: float
    threshold: float
    defect_regions: Sequence[DefectRegion]
    annotated_image: str
    inference_time_ms: float
    timestamp: datetime


class DetectionResultRead(DetectionResultBase):
    """Detection result returned from the API."""

    id: int

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Response returned after uploading an image."""

    image_id: str
    filename: str


class TrainRequest(BaseModel):
    """Request to trigger PatchCore training."""

    dataset_path: Optional[str] = Field(
        default=None,
        description="Optional custom training directory; defaults to config.train_dir.",
    )


class TrainStatus(BaseModel):
    """Model training status and recent metrics."""

    status: Literal["IDLE", "TRAINING", "READY", "ERROR"]
    last_trained_at: Optional[datetime] = None
    message: Optional[str] = None
    image_level_auroc: Optional[float] = None
    pixel_level_auroc: Optional[float] = None
    f1_score: Optional[float] = None
    memory_bank_size: Optional[int] = None


class ResultsFilter(BaseModel):
    """Filter for listing detection results."""

    status: Optional[StatusType] = None
    filename_contains: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=500)

