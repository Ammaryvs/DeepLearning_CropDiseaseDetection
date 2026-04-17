"""
SimpleCNN model for crop disease classification.

Architecture:
    - 4 convolutional blocks: Conv2d(3p1) → ReLU → MaxPool2d(2)
      Channels: 3 → 32 → 64 → 128 → 256
    - Classifier head: AdaptiveAvgPool2d(1) → Flatten
                       → Linear(256, 512) → ReLU → Dropout(0.5)
                       → Linear(512, num_classes)

AdaptiveAvgPool2d(1) makes the model input-size agnostic (works with any image size).

Grid search findings (simplecnn.ipynb) — both tested at lr=0.1:
    Adam lr=0.1 → test_acc=0.0959  (LR too high, failed to learn)
    SGD  lr=0.1 → NaN loss         (diverged immediately)
Recommended: Adam lr=0.001
"""
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_simple_cnn(num_classes: int, dropout: float = 0.5) -> nn.Module:
    return SimpleCNN(num_classes=num_classes, dropout=dropout)
