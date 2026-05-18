from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import get_db
from app.state import TRAINING_STATE
from app.models.database import DetectionResult as DetectionResultORM
from app.models.schemas import DashboardStats, DetectionResultRead
from app.repositories.detection import _orm_to_read


router = APIRouter(tags=["dashboard"])

DEFECT_COLORS = {
    "LOW": "#22C55E",
    "MEDIUM": "#F59E0B",
    "HIGH": "#EF4444",
}


@router.get("/dashboard", response_model=DashboardStats)
def get_dashboard(db: Session = Depends(get_db)) -> DashboardStats:
    """Aggregate inspection metrics for the dashboard UI."""

    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    today_stmt = select(DetectionResultORM).where(DetectionResultORM.timestamp >= start_of_day)
    today_rows = list(db.scalars(today_stmt).all())
    total_today = len(today_rows)

    pass_count = sum(1 for row in today_rows if row.status == "NORMAL")
    fail_count = total_today - pass_count
    pass_rate = (100.0 * pass_count / total_today) if total_today else 0.0
    avg_score = (
        sum(row.anomaly_score for row in today_rows) / total_today if total_today else 0.0
    )

    severity_counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for row in today_rows:
        if row.status != "DEFECTIVE":
            continue
        for region in row.defect_regions or []:
            sev = region.get("severity", "LOW") if isinstance(region, dict) else "LOW"
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

    defect_distribution = [
        {"name": name, "value": count, "color": DEFECT_COLORS.get(name, "#6366F1")}
        for name, count in severity_counts.items()
        if count > 0
    ]

    recent_stmt = (
        select(DetectionResultORM)
        .order_by(DetectionResultORM.timestamp.desc())
        .limit(8)
    )
    recent_results = [_orm_to_read(orm) for orm in db.scalars(recent_stmt).all()]

    auroc = float(TRAINING_STATE.get("image_level_auroc") or 0.0)

    return DashboardStats(
        total_inspected_today=total_today,
        pass_rate=round(pass_rate, 1),
        defects_detected=fail_count,
        avg_anomaly_score=round(avg_score, 4),
        auroc=auroc,
        defect_distribution=defect_distribution,
        recent_results=recent_results,
    )
