import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=38, pretrained=True, dropout=0.2):
        super().__init__()

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v2(weights=weights)

        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def create_mobilenetv2(num_classes=38, pretrained=True, dropout=0.2):
    return MobileNetV2Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = create_mobilenetv2(num_classes=38)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
