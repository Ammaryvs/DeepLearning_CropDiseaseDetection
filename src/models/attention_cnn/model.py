import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet


class ChannelAttention(nn.Module):
    """Channel attention module used by CBAM."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module used by CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class CBAMBasicBlock(BasicBlock):
    """Basic ResNet block augmented with CBAM."""

    def __init__(self, *args, reduction_ratio: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv2.out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CBAMBottleneck(Bottleneck):
    """Bottleneck ResNet block augmented with CBAM."""

    def __init__(self, *args, reduction_ratio: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv3.out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CBAMResNet(nn.Module):
    """ResNet with CBAM inserted after each residual block."""

    def __init__(
        self,
        block,
        layers,
        num_classes: int = 26,
        pretrained: bool = False,
        dropout_rate: float = 0.3,
        reduction_ratio: int = 16,
    ):
        super().__init__()
        self.model = ResNet(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=True,
        )

        if dropout_rate > 0:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )

        if pretrained:
            self._load_pretrained_backbone()

    def _load_pretrained_backbone(self) -> None:
        try:
            weights = models.ResNet50_Weights.DEFAULT
            pretrained = models.resnet50(weights=weights)
        except AttributeError:
            pretrained = models.resnet50(pretrained=True)

        pretrained_state = pretrained.state_dict()
        model_state = self.model.state_dict()
        filtered_state = {
            key: value
            for key, value in pretrained_state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        model_state.update(filtered_state)
        self.model.load_state_dict(model_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_cbam_resnet50(
    num_classes: int = 26,
    pretrained: bool = False,
    dropout_rate: float = 0.3,
    reduction_ratio: int = 16,
) -> CBAMResNet:
    return CBAMResNet(
        CBAMBottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        reduction_ratio=reduction_ratio,
    )


__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "CBAMBasicBlock",
    "CBAMBottleneck",
    "CBAMResNet",
    "create_cbam_resnet50",
]
