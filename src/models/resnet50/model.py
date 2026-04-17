"""
ResNet50 transfer learning model for crop disease classification.

Architecture:
    - Backbone: ResNet50 pretrained on ImageNet (2048-d global avg-pool output)
    - Head: Dropout → Linear(2048, num_classes)

Two-phase fine-tuning strategy:
    Phase 1 – Feature extraction  : freeze backbone, train head only.
    Phase 2 – Fine-tuning         : unfreeze layer4 + head, use lower LR.
"""
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_resnet50(num_classes: int, pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    """
    Build a ResNet50 with a custom classification head.

    Args:
        num_classes: Number of output disease classes.
        pretrained:  Load ImageNet weights when True.
        dropout:     Dropout probability before the final linear layer.

    Returns:
        nn.Module ready for training.
    """
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the FC head (Phase 1)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('fc')


def unfreeze_top_layers(model: nn.Module) -> None:
    """Unfreeze layer4 and FC head for fine-tuning (Phase 2)."""
    for name, param in model.named_parameters():
        if name.startswith('layer4') or name.startswith('fc'):
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter in the model."""
    for param in model.parameters():
        param.requires_grad = True
