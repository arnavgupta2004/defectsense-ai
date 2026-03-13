from __future__ import annotations

import time
from pathlib import Path
from typing import List

import cv2
import torch
import yaml

from app.core.patchcore import PatchcoreConfig, PatchcoreWrapper
from app.core.preprocessor import preprocess_batch


def load_config(path: Path) -> dict:
    """Load training configuration from YAML."""

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_image_paths(train_dir: Path, extensions: list[str]) -> List[Path]:
    """Collect all training image paths from the directory."""

    paths: List[Path] = []
    for ext in extensions:
        paths.extend(train_dir.rglob(f"*{ext}"))
    return paths


def main() -> None:
    """Train PatchCore memory bank on normal images."""

    config_path = Path(__file__).with_name("config.yaml")
    cfg = load_config(config_path)

    requested_device = str(cfg["training"].get("device", "cpu")).lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    else:
        device = requested_device

    patchcore_cfg = PatchcoreConfig(
        backbone=cfg["model"]["backbone"],
        layers=tuple(cfg["model"]["layers"]),
        coreset_sampling_ratio=cfg["model"]["coreset_sampling_ratio"],
        num_neighbors=cfg["model"]["num_neighbors"],
    )
    wrapper = PatchcoreWrapper(device=device, config=patchcore_cfg)

    # anomalib expects a folder datamodule root, with normal images in a subdir.
    # Our config points to .../train/good, so we convert to root=.../train and normal_dir=good.
    train_good_dir = Path(cfg["data"]["train_dir"])
    if not train_good_dir.exists():
        raise SystemExit(f"Training directory not found: {train_good_dir}")
    root_dir = train_good_dir.parent
    normal_dir_name = train_good_dir.name

    extensions = tuple(cfg["data"]["extensions"])
    batch_size = int(cfg["data"]["batch_size"])
    image_size = int(cfg["data"]["image_size"])

    # Validate there are images before starting the engine.
    image_paths = collect_image_paths(train_good_dir, list(extensions))
    if not image_paths:
        raise SystemExit(f"No training images found in {train_good_dir}")

    start_time = time.time()

    batches: List[torch.Tensor] = []
    current_images: List[object] = []
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        current_images.append(rgb)
        if len(current_images) >= batch_size:
            batch = preprocess_batch(current_images, size=image_size)
            batches.append(batch)
            current_images = []
    if current_images:
        batches.append(preprocess_batch(current_images, size=image_size))

    wrapper.fit(batches)
    wrapper.save_memory_bank()

    elapsed = time.time() - start_time
    print(f"Training complete. Images processed: {len(image_paths)}")
    print(f"Memory bank saved to: {patchcore_cfg.memory_bank_path}")
    print(f"Time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

