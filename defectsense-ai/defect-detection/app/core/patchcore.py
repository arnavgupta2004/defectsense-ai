from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from app.core.config import settings
from app.core.feature_extractor import build_feature_extractor


@dataclass
class PatchcoreConfig:
    """Configuration for the PatchCore wrapper."""

    backbone: str = settings.model_backbone
    layers: Tuple[str, ...] = ("layer2", "layer3")
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    memory_bank_path: Path = settings.model_memory_bank_path
    embedding_dim: int = 0


class PatchcoreWrapper:
    """PatchCore-style memory bank + nearest neighbor inference (no labels required).

    This implementation:
    - extracts WideResNet intermediate feature maps (layer2/layer3)
    - aligns spatial sizes and concatenates channels
    - stores patch embeddings from normal images in a memory bank
    - applies coreset subsampling (random) to keep the memory bank compact
    - scores test patches by nearest-neighbor distance to the memory bank
    """

    def __init__(self, device: str | torch.device = "cpu", config: PatchcoreConfig | None = None):
        self.device = torch.device(device)
        self.config = config or PatchcoreConfig()
        self.extractor = build_feature_extractor(layers=self.config.layers, pretrained=True, device=self.device)
        self._memory_bank_built: bool = False
        self.memory_bank: torch.Tensor | None = None  # (M, C)

    @property
    def is_ready(self) -> bool:
        """Return ``True`` if the memory bank has been built or loaded."""

        return self._memory_bank_built

    def _extract_patch_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract per-patch embeddings from a preprocessed batch.

        Parameters
        ----------
        batch:
            Tensor of shape (N, 3, H, W), ImageNet-normalized.

        Returns
        -------
        embeddings:
            Tensor of shape (N, C, h, w)
        """

        with torch.no_grad():
            feats = self.extractor(batch.to(self.device))
        # Align spatial dims (use smallest H/W) and concatenate channels
        h = min(f.shape[-2] for f in feats)
        w = min(f.shape[-1] for f in feats)
        aligned = []
        for f in feats:
            if f.shape[-2:] != (h, w):
                f = torch.nn.functional.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
            aligned.append(f)
        emb = torch.cat(aligned, dim=1)  # (N, C, h, w)
        return emb

    def fit(self, batches: Iterable[torch.Tensor]) -> None:
        """Build the memory bank from batches of *preprocessed* normal images."""

        all_patches: List[torch.Tensor] = []
        for batch in batches:
            emb = self._extract_patch_embeddings(batch)
            n, c, h, w = emb.shape
            patches = emb.permute(0, 2, 3, 1).reshape(n * h * w, c)
            all_patches.append(patches.cpu())

        if not all_patches:
            raise RuntimeError("No training data produced any embeddings.")

        bank = torch.cat(all_patches, dim=0)  # (T, C)
        self.config.embedding_dim = int(bank.shape[1])

        # Coreset subsampling (random).
        ratio = float(self.config.coreset_sampling_ratio)
        ratio = max(0.0, min(1.0, ratio))
        if 0.0 < ratio < 1.0:
            m = max(1, int(bank.shape[0] * ratio))
            idx = torch.randperm(bank.shape[0])[:m]
            bank = bank[idx]

        self.memory_bank = bank.to(self.device)
        self._memory_bank_built = True

    def save_memory_bank(self, path: Path | None = None) -> None:
        """Persist the memory bank to disk."""

        if not self._memory_bank_built:
            raise RuntimeError("Memory bank has not been built; train the model first.")
        path = path or self.config.memory_bank_path
        if self.memory_bank is None:
            raise RuntimeError("Memory bank tensor missing despite ready state.")
        state = {"memory_bank": self.memory_bank.detach().cpu(), "config": self.config.__dict__}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load_memory_bank(self, path: Path | None = None) -> None:
        """Load a memory bank from disk."""

        path = path or self.config.memory_bank_path
        if not path.exists():
            raise FileNotFoundError(f"Memory bank file not found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.memory_bank = checkpoint["memory_bank"].to(self.device)
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            cfg = checkpoint["config"]
            if "embedding_dim" in cfg:
                self.config.embedding_dim = int(cfg["embedding_dim"])
        self._memory_bank_built = True

    def predict(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference and return image-level scores and patch anomaly maps.

        Returns
        -------
        image_scores:
            Tensor of shape (N,) with per-image anomaly scores.
        anomaly_maps:
            Tensor of shape (N, 1, H, W) with patch-level anomaly scores.
        """

        if not self._memory_bank_built:
            self.load_memory_bank()
        if self.memory_bank is None or self.memory_bank.numel() == 0:
            raise RuntimeError("Memory bank is empty; train the model first.")

        emb = self._extract_patch_embeddings(batch.to(self.device))
        n, c, h, w = emb.shape
        patches = emb.permute(0, 2, 3, 1).reshape(n * h * w, c)  # (P, C)

        # Nearest neighbor distance to memory bank, chunked to reduce RAM usage.
        bank = self.memory_bank  # (M, C)
        chunk = 8192
        dists_min: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, patches.shape[0], chunk):
                p = patches[i : i + chunk]
                # (chunk, M)
                d = torch.cdist(p, bank, p=2)
                dmin = d.min(dim=1).values
                dists_min.append(dmin)
        dist = torch.cat(dists_min, dim=0)  # (P,)

        anomaly_map = dist.reshape(n, h, w)
        image_scores = anomaly_map.view(n, -1).max(dim=1).values

        # Normalize anomaly map per image.
        anomaly_maps = anomaly_map.unsqueeze(1)
        maxv = anomaly_maps.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        anomaly_maps = anomaly_maps / maxv
        return image_scores.detach().cpu(), anomaly_maps.detach().cpu()


def build_patchcore(device: str | torch.device = "cpu") -> PatchcoreWrapper:
    """Construct a PatchcoreWrapper using global settings."""

    return PatchcoreWrapper(device=device)

