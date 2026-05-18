from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        protected_namespaces=(),
    )

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
    model_device: str = Field(default="cpu", alias="MODEL_DEVICE")
    image_size: int = Field(default=224, alias="IMAGE_SIZE")
    low_memory: bool = Field(default=False, alias="LOW_MEMORY")
    coreset_sampling_ratio: float = Field(default=0.1, alias="CORESET_SAMPLING_RATIO")
    inference_chunk_size: int = Field(default=8192, alias="INFERENCE_CHUNK_SIZE")
    train_batch_size: int = Field(default=16, alias="TRAIN_BATCH_SIZE")
    bootstrap_train_images: int = Field(default=12, alias="BOOTSTRAP_TRAIN_IMAGES")
    feature_layers: str = Field(default="layer2,layer3", alias="FEATURE_LAYERS")
    num_neighbors: int = Field(default=9, alias="NUM_NEIGHBORS")

    upload_dir: Path = Field(default=Path("./uploads"), alias="UPLOAD_DIR")
    train_dir: Path = Field(default=Path("./data/custom/train/good"), alias="TRAIN_DIR")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    cors_origins: str = Field(
        default="http://localhost:8080,http://localhost:5173,http://localhost:3000",
        alias="CORS_ORIGINS",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> str:
        if isinstance(value, list):
            return ",".join(str(v) for v in value)
        return str(value) if value is not None else ""

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def feature_layer_tuple(self) -> Tuple[str, ...]:
        return tuple(layer.strip() for layer in self.feature_layers.split(",") if layer.strip())

    @model_validator(mode="after")
    def apply_low_memory_profile(self) -> "Settings":
        """Tune defaults for Render free tier (~512 MB RAM)."""

        on_render = os.getenv("RENDER") == "true"
        if not (self.low_memory or on_render):
            return self

        object.__setattr__(self, "low_memory", True)
        if not os.getenv("IMAGE_SIZE"):
            object.__setattr__(self, "image_size", 128)
        if not os.getenv("CORESET_SAMPLING_RATIO"):
            object.__setattr__(self, "coreset_sampling_ratio", 0.04)
        if not os.getenv("INFERENCE_CHUNK_SIZE"):
            object.__setattr__(self, "inference_chunk_size", 256)
        if not os.getenv("TRAIN_BATCH_SIZE"):
            object.__setattr__(self, "train_batch_size", 2)
        if not os.getenv("BOOTSTRAP_TRAIN_IMAGES"):
            object.__setattr__(self, "bootstrap_train_images", 6)
        if not os.getenv("FEATURE_LAYERS"):
            object.__setattr__(self, "feature_layers", "layer3")
        if not os.getenv("NUM_NEIGHBORS"):
            object.__setattr__(self, "num_neighbors", 3)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.model_memory_bank_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()

