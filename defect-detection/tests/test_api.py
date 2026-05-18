from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _create_temp_image(tmp_path: Path) -> Path:
    path = tmp_path / "test.png"
    img = Image.new("RGB", (64, 64), color="white")
    img.save(path)
    return path


def test_health() -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_endpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    img_path = _create_temp_image(tmp_path)
    with img_path.open("rb") as f:
        files = {"file": ("test.png", f, "image/png")}
        resp = client.post("/api/upload", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert "image_id" in data
    assert data["filename"].endswith(".png")

