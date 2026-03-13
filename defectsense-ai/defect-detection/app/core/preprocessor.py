from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

ImageLike = Union[np.ndarray, Image.Image]


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def _to_numpy(image: ImageLike) -> np.ndarray:
    """Convert a PIL or numpy image to a BGR numpy array."""

    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def preprocess_image(
    image: ImageLike,
    size: int = 224,
    gaussian_ksize: int = 3,
) -> torch.Tensor:
    """Preprocess a single image for model input.

    Steps:
    - Resize to ``size x size``
    - Denoise with Gaussian blur
    - Convert to tensor
    - Apply ImageNet normalization
    """

    bgr = _to_numpy(image)
    resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    if gaussian_ksize > 1:
        resized = cv2.GaussianBlur(resized, (gaussian_ksize, gaussian_ksize), 0)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def preprocess_batch(
    images: Iterable[ImageLike],
    size: int = 224,
    gaussian_ksize: int = 3,
) -> torch.Tensor:
    """Preprocess a batch of images and stack into a single tensor."""

    tensors: List[torch.Tensor] = [
        preprocess_image(img, size=size, gaussian_ksize=gaussian_ksize) for img in images
    ]
    if not tensors:
        raise ValueError("No images provided for preprocessing.")
    batch = torch.stack(tensors, dim=0)
    return batch

