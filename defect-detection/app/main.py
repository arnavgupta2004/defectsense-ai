from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.runtime_tuning import configure_runtime_memory
from app.database import get_db, init_db
from app.models.schemas import ThresholdUpdate, TrainStatus
from app.runtime_settings import get_threshold, set_threshold
from app.services.model_service import memory_bank_size, reload_patchcore
from app.state import TRAINING_STATE

from app.api.routes import upload, detect, train, results, dashboard  # type: ignore[import-not-found]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize database and attempt to load the PatchCore memory bank."""

    configure_runtime_memory()
    init_db()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.model_memory_bank_path.parent.mkdir(parents=True, exist_ok=True)
    settings.train_dir.mkdir(parents=True, exist_ok=True)

    try:
        reload_patchcore()
        if memory_bank_size() > 0:
            TRAINING_STATE.update(
                {
                    "status": "READY",
                    "memory_bank_size": memory_bank_size(),
                    "message": "Loaded memory bank from disk.",
                }
            )
    except Exception:  # noqa: BLE001
        TRAINING_STATE.update({"status": "IDLE", "message": "No trained model on disk."})

    yield


app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0.0",
    description="PatchCore-based anomaly detection for industrial surface defects.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["system"])
def health_check(db: Session = Depends(get_db)) -> Dict[str, str]:
    """Simple health check endpoint."""

    _ = db
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/model/status", response_model=TrainStatus, tags=["model"])
def model_status() -> TrainStatus:
    """Return current model training status and latest metrics."""

    state = dict(TRAINING_STATE)
    if state.get("memory_bank_size") is None:
        state["memory_bank_size"] = memory_bank_size() or None
    state["threshold"] = get_threshold()
    return TrainStatus(**state)  # type: ignore[arg-type]


@app.patch("/api/model/threshold", response_model=TrainStatus, tags=["model"])
def update_threshold(payload: ThresholdUpdate) -> TrainStatus:
    """Set the anomaly score threshold used during detection."""

    if TRAINING_STATE.get("status") not in ("READY", "IDLE"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot change threshold while training is in progress.",
        )
    set_threshold(payload.threshold)
    return model_status()


app.include_router(upload.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
app.include_router(train.router, prefix="/api")
app.include_router(results.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")


__all__ = ["app"]
