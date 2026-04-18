from __future__ import annotations

from typing import Iterable, Optional
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _resolve_vgg16_weights(weights_name: Optional[str]):
    if weights_name is None:
        return None

    from torchvision import models

    enum_cls = getattr(models, "VGG16_Weights", None)
    if enum_cls is None:
        return None
    return getattr(enum_cls, weights_name)


def _build_vgg16(weights):
    from torchvision import models

    try:
        return models.vgg16(weights=weights)
    except TypeError:
        return models.vgg16(pretrained=weights is not None)


class VGGPerceptualLoss(nn.Module):
    def __init__(
        self,
        *,
        layers: Optional[Iterable[int]] = None,
        layer_weights: Optional[Iterable[float]] = None,
        resize_to: int = 224,
        normalize_inputs: bool = True,
        weights: Optional[str] = "IMAGENET1K_V1",
        weights_path: Optional[str] = None,
        strict_weights: bool = False,
    ) -> None:
        super().__init__()

        from torchvision import models

        layer_ids = tuple(layers or (3, 8, 15, 22))
        if not layer_ids:
            raise ValueError("layers must contain at least one feature index")

        resolved_weights = _resolve_vgg16_weights(weights)
        try:
            backbone = _build_vgg16(resolved_weights)
        except Exception as exc:
            if strict_weights:
                raise
            warnings.warn(
                f"Falling back to randomly initialized VGG16 perceptual features: {exc}",
                stacklevel=2,
            )
            backbone = _build_vgg16(None)

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            backbone.load_state_dict(state_dict, strict=strict_weights)

        self.features = backbone.features.eval()
        for param in self.features.parameters():
            param.requires_grad_(False)

        self.layer_ids = layer_ids
        self.max_layer = max(layer_ids)
        self.resize_to = resize_to
        self.normalize_inputs = normalize_inputs
        if layer_weights is None:
            layer_weights = [1.0] * len(layer_ids)
        layer_weights = tuple(float(weight) for weight in layer_weights)
        if len(layer_weights) != len(layer_ids):
            raise ValueError("layer_weights must match layers length")
        self.layer_weights = layer_weights

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _preprocess(self, x: Tensor) -> Tensor:
        x = ((x + 1.0) * 0.5).clamp_(0.0, 1.0)
        if self.resize_to is not None and x.shape[-1] != self.resize_to:
            x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        if self.normalize_inputs:
            x = (x - self.mean) / self.std
        return x

    def _extract(self, x: Tensor) -> list[Tensor]:
        features = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer_ids:
                features.append(x)
            if idx >= self.max_layer:
                break
        return features

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = self._preprocess(pred)
        target = self._preprocess(target)

        pred_features = self._extract(pred)
        with torch.no_grad():
            target_features = [feature.detach() for feature in self._extract(target)]

        losses = pred.new_zeros(pred.shape[0])
        for weight, pred_feature, target_feature in zip(self.layer_weights, pred_features, target_features):
            diff = (pred_feature - target_feature).abs().flatten(1).mean(dim=1)
            losses = losses + weight * diff
        return losses


def build_perceptual_loss(cfg: Optional[dict]) -> Optional[nn.Module]:
    if not cfg or not cfg.get("enabled", False) or float(cfg.get("lambda", 0.0)) <= 0.0:
        return None

    loss_type = str(cfg.get("type", "vgg16")).lower()
    if loss_type != "vgg16":
        raise ValueError(f"Unsupported perceptual loss type: {cfg.get('type')}")

    return VGGPerceptualLoss(
        layers=cfg.get("layers"),
        layer_weights=cfg.get("layer_weights"),
        resize_to=int(cfg.get("resize_to", 224)),
        normalize_inputs=bool(cfg.get("normalize_inputs", True)),
        weights=cfg.get("weights", "IMAGENET1K_V1"),
        weights_path=cfg.get("weights_path"),
        strict_weights=bool(cfg.get("strict_weights", False)),
    )
