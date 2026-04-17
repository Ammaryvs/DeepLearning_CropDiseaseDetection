"""
MobileNetV2 transfer learning model for crop disease classification.

Architecture:
    - Backbone: MobileNetV2 pretrained on ImageNet (DEFAULT weights)
    - Head: Dropout → Linear(1280, num_classes)

Notebook results (MobileNet.ipynb):
    LR=0.0001, Dropout=0.1, Adam, StepLR(step_size=7, gamma=0.1), 18 epochs
    → Epoch 1: Train 90.03%  |  Val 98.53%
"""
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int = 38, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v2(weights=weights)

        in_features = self.model.classifier[-1].in_features  # 1280
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def build_mobilenetv2(num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    return MobileNetV2Classifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
