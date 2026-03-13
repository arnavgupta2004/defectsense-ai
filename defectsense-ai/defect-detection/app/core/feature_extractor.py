from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class WideResNetFeatureExtractor(nn.Module):
    """WideResNet50_2 feature extractor returning intermediate layer maps.

    Extracts features from layer2 and layer3 and applies adaptive average pooling
    to obtain patch embeddings suitable for PatchCore-style methods.
    """

    def __init__(
        self,
        layers: Sequence[str] = ("layer2", "layer3"),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.layers: Tuple[str, ...] = tuple(layers)

        weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        backbone = wide_resnet50_2(weights=weights)

        # Keep all modules, we will tap into intermediate outputs via forward hooks.
        self.backbone = backbone
        self._feature_maps: Dict[str, torch.Tensor] = {}

        for name in self.layers:
            module = dict(self.backbone.named_modules())[name]
            module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _input, output: torch.Tensor) -> None:
            self._feature_maps[name] = output

        return hook

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning pooled feature maps for each configured layer."""

        self._feature_maps.clear()
        _ = self.backbone(x)
        features: List[torch.Tensor] = []
        for name in self.layers:
            fmap = self._feature_maps[name]
            # Each fmap: (N, C, H, W); we return as-is for PatchCore patch extraction.
            features.append(fmap)
        return features


def build_feature_extractor(
    layers: Sequence[str] = ("layer2", "layer3"),
    pretrained: bool = True,
    device: torch.device | None = None,
) -> WideResNetFeatureExtractor:
    """Construct and optionally move a WideResNet feature extractor to a device."""

    model = WideResNetFeatureExtractor(layers=layers, pretrained=pretrained)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model

