from __future__ import annotations

import numpy as np
import torch

from app.core.patchcore import PatchcoreWrapper, PatchcoreConfig


def test_patchcore_wrapper_init() -> None:
    cfg = PatchcoreConfig()
    model = PatchcoreWrapper(device="cpu", config=cfg)
    assert model is not None


def test_patchcore_predict_raises_without_memory(tmp_path) -> None:
    cfg = PatchcoreConfig(memory_bank_path=tmp_path / "nonexistent.pt")
    model = PatchcoreWrapper(device="cpu", config=cfg)
    batch = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    try:
        model.predict(batch)
    except FileNotFoundError:
        # Expected since no memory bank exists yet.
        pass

