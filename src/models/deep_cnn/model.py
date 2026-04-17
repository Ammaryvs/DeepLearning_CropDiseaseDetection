"""
DeepCNN model for crop disease classification.

Architecture:
    - 5 convolutional blocks: Conv2d → BatchNorm → ReLU → MaxPool2d(2)
      Channels: 3 → 32 → 64 → 128 → 256 → 512
    - Classifier head: Flatten → Linear(25088, 512) → ReLU → Dropout(0.5)
                       → Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, num_classes)

Input: 224×224 RGB images.
After 5 MaxPool2d(2) layers: spatial size reduces to 7×7 → 512×7×7 = 25088 features.
"""
import torch.nn as nn


class DeepCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_deep_cnn(num_classes: int, dropout: float = 0.5) -> nn.Module:
    return DeepCNN(num_classes=num_classes, dropout=dropout)
