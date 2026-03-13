from __future__ import annotations

import numpy as np

from app.core.preprocessor import preprocess_image, preprocess_batch


def test_preprocess_single_image() -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    tensor = preprocess_image(img, size=224)
    assert tensor.shape == (3, 224, 224)


def test_preprocess_batch() -> None:
    imgs = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(4)]
    batch = preprocess_batch(imgs, size=224)
    assert batch.shape == (4, 3, 224, 224)

