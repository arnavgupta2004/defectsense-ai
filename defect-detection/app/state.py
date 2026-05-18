from __future__ import annotations

from typing import Any, Dict

TRAINING_STATE: Dict[str, Any] = {
    "status": "IDLE",
    "last_trained_at": None,
    "message": None,
    "image_level_auroc": None,
    "pixel_level_auroc": None,
    "f1_score": None,
    "memory_bank_size": None,
}
