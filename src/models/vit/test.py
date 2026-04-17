import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from common.config import FullConfig
from common.dataloader import create_plantvillage_dataloaders
from model import VisionTransformer


def create_eval_dataloader(config: FullConfig) -> DataLoader:
    """Build the test dataloader using the same split settings as training."""
    data_bundle = create_plantvillage_dataloaders(
        root_dir=config.data.data_path,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        shuffle=config.data.shuffle,
        seed=config.data.random_seed,
        train_frac=config.data.train_split,
        val_frac=config.data.val_split,
        test_frac=config.data.test_split,
    )
    return data_bundle["loaders"]["test"]


def remap_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map legacy checkpoint keys to the current VisionTransformer module names."""
    remapped: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key
        if key == "cls_token":
            new_key = "patch_embed.cls_token"
        elif key == "pos_embed":
            new_key = "patch_embed.pos_embed"
        elif ".attn.qkv." in key:
            new_key = key.replace(".attn.qkv.", ".attn.qkv_proj.")
        elif ".attn.proj." in key:
            new_key = key.replace(".attn.proj.", ".attn.out_proj.")
        elif ".mlp.fc1." in key:
            new_key = key.replace(".mlp.fc1.", ".mlp.0.")
        elif ".mlp.fc2." in key:
            new_key = key.replace(".mlp.fc2.", ".mlp.3.")

        remapped[new_key] = value

    return remapped


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format in {path}")


def build_model(config: FullConfig, checkpoint: dict, device: torch.device) -> VisionTransformer:
    model = VisionTransformer(
        img_size=config.model.input_size,
        patch_size=16,
        in_channels=3,
        num_classes=config.model.num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=config.model.dropout_rate,
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = remap_checkpoint_keys(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model


def test_one_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def resolve_config(args, checkpoint: dict) -> FullConfig:
    if isinstance(checkpoint.get("config"), dict):
        return FullConfig.from_dict(checkpoint["config"])
    return FullConfig.from_yaml(args.config)


def resolve_checkpoint_path(args, config: FullConfig) -> Path:
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_absolute() or checkpoint_path.exists():
        return checkpoint_path
    return Path(config.checkpoint.checkpoint_dir) / checkpoint_path


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_checkpoint = Path(args.checkpoint)
    initial_path = default_checkpoint if default_checkpoint.is_absolute() else Path(args.checkpoint)
    if not initial_path.exists() and Path(args.config).exists():
        config = FullConfig.from_yaml(args.config)
        initial_path = resolve_checkpoint_path(args, config)

    checkpoint = load_checkpoint(initial_path, device)
    config = resolve_config(args, checkpoint)
    checkpoint_path = resolve_checkpoint_path(args, config)
    if checkpoint_path != initial_path:
        checkpoint = load_checkpoint(checkpoint_path, device)

    print(f"Using device: {device}")
    print(f"Loaded model from {checkpoint_path}")

    model = build_model(config, checkpoint, device)
    criterion = nn.CrossEntropyLoss()
    test_loader = create_eval_dataloader(config)

    test_loss, test_acc = test_one_epoch(model, test_loader, criterion, device)

    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Vision Transformer for Plant Disease Detection")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to the model config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename or full path",
    )

    args = parser.parse_args()
    main(args)
