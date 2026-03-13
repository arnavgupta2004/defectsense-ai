from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Tuple

from fastapi import UploadFile

from app.core.config import settings


def save_upload(file: UploadFile) -> Tuple[str, Path]:
    """Persist an uploaded image and return its UUID and path."""

    image_id = str(uuid.uuid4())
    extension = Path(file.filename or "").suffix or ".png"
    filename = f"{image_id}{extension}"
    path = settings.upload_dir / filename
    with path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return image_id, path


def get_uploaded_path(image_id: str) -> Path:
    """Return the path of a previously uploaded file by its UUID."""

    for file in settings.upload_dir.iterdir():
        if file.stem == image_id:
            return file
    raise FileNotFoundError(f"Uploaded image with id {image_id} not found.")

