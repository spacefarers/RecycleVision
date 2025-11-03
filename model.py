"""Model factory for RecycleVision."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def create_model(num_classes: int, pretrained: bool = True, drop_rate: float = 0.3) -> nn.Module:
    """Create a MobileNetV3-Small classification model with enhanced classifier."""
    model = mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=drop_rate),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=drop_rate),
        nn.Linear(256, num_classes),
    )

    # Initialize classifier layers with proper weights
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    return model
