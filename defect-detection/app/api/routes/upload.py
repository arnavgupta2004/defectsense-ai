from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.schemas import UploadResponse
from app.utils.file_handler import save_upload


router = APIRouter(tags=["upload"])

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
}


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a product image and return its generated image ID."""

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type: {content_type}",
        )
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    image_id, path = await save_upload(file)
    return UploadResponse(image_id=image_id, filename=path.name)
