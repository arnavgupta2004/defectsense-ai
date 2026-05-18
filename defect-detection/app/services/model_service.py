from __future__ import annotations

import threading
from typing import Optional

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.patchcore import PatchcoreWrapper, build_patchcore

_lock = threading.Lock()
_model: Optional[PatchcoreWrapper] = None
_deferred_load_thread: Optional[threading.Thread] = None


def register_deferred_model_load(thread: threading.Thread) -> None:
    """Track the background PatchCore load so get_patchcore can wait instead of duplicating work."""

    global _deferred_load_thread
    _deferred_load_thread = thread


def get_patchcore() -> PatchcoreWrapper:
    """Return the process-wide PatchCore wrapper (lazy init)."""

    global _model
    if _model is not None:
        return _model

    t = _deferred_load_thread
    if t is not None and t.is_alive():
        t.join(timeout=180.0)

    if _model is not None:
        return _model

    with _lock:
        if _model is None:
            _model = build_patchcore(device=settings.model_device)
    return _model


def reload_patchcore() -> PatchcoreWrapper:
    """Rebuild the wrapper and load the memory bank from disk."""

    global _model
    with _lock:
        wrapper = build_patchcore(device=settings.model_device)
        try:
            wrapper.load_memory_bank()
        except FileNotFoundError:
            pass
        _model = wrapper
    return _model


def require_ready_model() -> PatchcoreWrapper:
    """Return a model with a loaded memory bank or raise HTTP 503."""

    model = get_patchcore()
    if not model.is_ready:
        try:
            model.load_memory_bank()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Model is not trained yet. Place normal images in "
                    "data/custom/train/good and call POST /api/train, or run "
                    "python -m training.train_patchcore."
                ),
            ) from exc
    return model


def memory_bank_size() -> int:
    """Return the number of patch embeddings in the memory bank."""

    model = get_patchcore()
    if model.memory_bank is None:
        return 0
    return int(model.memory_bank.shape[0])
