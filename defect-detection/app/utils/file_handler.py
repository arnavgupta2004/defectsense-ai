from __future__ import annotations

import uuid
from pathlib import Path
from typing import Tuple

import aiofiles
from fastapi import UploadFile

from app.core.config import settings


def get_uploaded_path(image_id: str) -> Path:
    """Return the path of a previously uploaded file by its UUID."""

    for file in settings.upload_dir.iterdir():
        if file.stem == image_id:
            return file
    raise FileNotFoundError(f"Uploaded image with id {image_id} not found.")


async def save_upload(file: UploadFile) -> Tuple[str, Path]:
    """Persist an uploaded image asynchronously and return its UUID and path."""

    image_id = str(uuid.uuid4())
    extension = Path(file.filename or "").suffix.lower() or ".png"
    if extension not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        extension = ".png"
    filename = f"{image_id}{extension}"
    path = settings.upload_dir / filename
    async with aiofiles.open(path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            await buffer.write(chunk)
    return image_id, path


def save_upload_sync(file_path: Path, data: bytes) -> Tuple[str, Path]:
    """Synchronous helper used in tests and scripts."""

    image_id = str(uuid.uuid4())
    dest = settings.upload_dir / f"{image_id}{file_path.suffix or '.png'}"
    dest.write_bytes(data)
    return image_id, dest
