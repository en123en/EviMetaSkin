"""Pure-vision baseline constructors used in the comparison experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")
