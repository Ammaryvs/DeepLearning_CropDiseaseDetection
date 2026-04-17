"""
EfficientNet-B0 transfer learning model for crop disease classification.

Architecture:
    - Backbone: EfficientNet-B0 pretrained on ImageNet (DEFAULT weights)
    - Head: Dropout → Linear(1280, num_classes)

Notebook results (efficientnet.ipynb):
    LR=0.001, Dropout=0.2, Adam, StepLR(step_size=7, gamma=0.1), 12 epochs
    → Val acc: 99.78%  |  Test acc: 99.78%
    → Precision/Recall/F1 (weighted): 99.78%
"""
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes: int = 38, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        in_features = self.model.classifier[-1].in_features  # 1280
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def build_efficientnet_b0(num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
