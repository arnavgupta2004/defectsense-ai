from __future__ import annotations

from app.core.config import settings

_threshold_override: float | None = None


def get_threshold() -> float:
    if _threshold_override is not None:
        return _threshold_override
    return settings.model_threshold


def set_threshold(value: float) -> float:
    global _threshold_override
    _threshold_override = value
    return _threshold_override
