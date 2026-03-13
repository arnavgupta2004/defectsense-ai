from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text

from app.database import Base


class DetectionResult(Base):
    """ORM model storing detection results for an inspected image."""

    __tablename__ = "detection_results"

    id: Any = Column(Integer, primary_key=True, index=True, autoincrement=True)
    image_id: str = Column(String(64), unique=True, index=True, nullable=False)
    filename: str = Column(String(255), nullable=False)
    status: str = Column(String(16), nullable=False)
    anomaly_score: float = Column(Float, nullable=False)
    threshold: float = Column(Float, nullable=False)
    defect_regions: Any = Column(JSON, nullable=True)
    annotated_image_base64: str = Column(Text, nullable=True)
    inference_time_ms: float = Column(Float, nullable=False)
    timestamp: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

