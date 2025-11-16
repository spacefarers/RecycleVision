"""Model factory for RecycleVision."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


def _replace_silu_with_relu6(module: nn.Module) -> None:
    """Swap all SiLU activations with ReLU6 to keep the network QAT-friendly."""
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU6(inplace=True))
        else:
            _replace_silu_with_relu6(child)


def _build_classifier(in_features: int, num_classes: int, drop_rate: float, hidden_dims: Sequence[int]) -> nn.Sequential:
    """Build a small quantization-friendly classifier head."""
    layers: list[nn.Module] = []
    prev = in_features
    for hidden_dim in hidden_dims:
        layers.append(nn.Dropout(p=drop_rate))
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        prev = hidden_dim

    layers.append(nn.Dropout(p=drop_rate))
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


def _init_linear_layers(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def create_model(
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    head_dims: Iterable[int] = (512, 256),
) -> nn.Module:
    """
    Create an EfficientNet-B0 backbone with a quantization-friendly classification head.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet weights.
        drop_rate: Dropout probability used in the head.
        head_dims: Hidden layer sizes for the classification head.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Swap SiLU with ReLU6 for better compatibility with int8 quantization.
    _replace_silu_with_relu6(model)

    in_features = model.classifier[1].in_features
    model.classifier = _build_classifier(in_features, num_classes, drop_rate, tuple(head_dims))
    _init_linear_layers(model.classifier)

    return model
