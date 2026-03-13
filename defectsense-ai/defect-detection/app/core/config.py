from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    app_env: str = Field(default="development", alias="APP_ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    db_url: str = Field(default="sqlite:///./defects.db", alias="DB_URL")

    model_memory_bank_path: Path = Field(
        default=Path("./artifacts/patchcore_memory_bank.pt"),
        alias="MODEL_MEMORY_BANK_PATH",
    )
    model_backbone: str = Field(default="wide_resnet50_2", alias="MODEL_BACKBONE")
    model_threshold: float = Field(default=0.5, alias="MODEL_THRESHOLD")

    upload_dir: Path = Field(default=Path("./uploads"), alias="UPLOAD_DIR")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.model_memory_bank_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()

