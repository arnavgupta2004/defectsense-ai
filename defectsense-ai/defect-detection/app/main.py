from __future__ import annotations

from datetime import datetime
from typing import Dict

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import get_db, init_db
from app.models.schemas import TrainStatus

from app.api.routes import upload, detect, train, results  # type: ignore[import-not-found]


app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0.0",
    description="PatchCore-based anomaly detection for industrial surface defects.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TRAINING_STATE: Dict[str, object] = {
    "status": "IDLE",
    "last_trained_at": None,
    "message": None,
    "image_level_auroc": None,
    "pixel_level_auroc": None,
    "f1_score": None,
    "memory_bank_size": None,
}


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database and any required resources."""

    init_db()


@app.get("/api/health", tags=["system"])
def health_check(db: Session = Depends(get_db)) -> Dict[str, str]:
    """Simple health check endpoint."""

    _ = db  # ensure DB dependency is valid
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/model/status", response_model=TrainStatus, tags=["model"])
def model_status() -> TrainStatus:
    """Return current model training status and latest metrics."""

    return TrainStatus(**TRAINING_STATE)  # type: ignore[arg-type]


app.include_router(upload.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
app.include_router(train.router, prefix="/api")
app.include_router(results.router, prefix="/api")


__all__ = ["app", "TRAINING_STATE"]

