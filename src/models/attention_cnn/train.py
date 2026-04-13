"""Training script for AttentionCNN using CBAM-ResNet."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from common.config import FullConfig, validate_config
from model import create_cbam_resnet50


def setup_logging(config: FullConfig):
    log_dir = Path(config.checkpoint.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"attention_cnn_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def setup_device(config: FullConfig) -> torch.device:
    if config.device.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA requested but not available. Using CPU.")
    elif config.device.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def create_model(config: FullConfig, device: torch.device) -> nn.Module:
    model = create_cbam_resnet50(
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout_rate=config.model.dropout_rate,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters - total: {total_params:,}, trainable: {trainable_params:,}")

    return model


def create_data_transforms(config: FullConfig, is_train: bool = True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train and config.data.augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.model.input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2 * config.data.augmentation_strength,
                contrast=0.2 * config.data.augmentation_strength,
                saturation=0.2 * config.data.augmentation_strength,
            ),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((config.model.input_size, config.model.input_size)),
        transforms.CenterCrop(config.model.input_size),
        transforms.ToTensor(),
        normalize,
    ])


def create_dataloaders(config: FullConfig):
    train_dataset = ImageFolder(config.data.train_path, transform=create_data_transforms(config, is_train=True))
    val_dataset = ImageFolder(config.data.val_path, transform=create_data_transforms(config, is_train=False))

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    return train_loader, val_loader


def create_optimizer(config: FullConfig, model: nn.Module) -> optim.Optimizer:
    optimizer_name = config.training.optimizer.lower()
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=config.training.momentum, weight_decay=config.training.weight_decay)

    raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def create_scheduler(config: FullConfig, optimizer: optim.Optimizer):
    scheduler_name = config.training.scheduler.lower()
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.training.epochs, eta_min=config.training.learning_rate * 0.01)
    if scheduler_name == "step":
        return StepLR(optimizer, step_size=config.training.scheduler_step_size, gamma=config.training.scheduler_gamma)
    if scheduler_name == "exponential":
        return ExponentialLR(optimizer, gamma=config.training.scheduler_gamma)

    logging.warning(f"Unknown scheduler: {config.training.scheduler}. No scheduler will be used.")
    return None


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: FullConfig,
    epoch: int,
    scaler: GradScaler = None,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if config.device.mixed_precision and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % config.checkpoint.log_frequency == 0:
            logging.info(
                f"Epoch [{epoch + 1}/{config.training.epochs}] "
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {running_loss / total:.4f} "
                f"Acc: {correct / total:.4f}"
            )

    return running_loss / total, correct / total


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, scaler: GradScaler = None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, metrics: dict, config: FullConfig, is_best: bool = False):
    checkpoint_dir = Path(config.checkpoint.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config.to_dict(),
    }

    filename = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, filename)
    logging.info(f"Saved checkpoint: {filename}")

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best model: {best_path}")


def train(config_path: str = "config.yaml"):
    config = FullConfig.from_yaml(config_path)
    if not validate_config(config):
        raise ValueError("Invalid configuration")

    setup_logging(config)
    device = setup_device(config)

    logging.info("Starting AttentionCNN CBAM-ResNet training")
    logging.info(str(config))

    model = create_model(config, device)
    train_loader, val_loader = create_dataloaders(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    scaler = GradScaler() if config.device.mixed_precision else None

    best_val_acc = 0.0
    for epoch in range(config.training.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config, epoch, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device, scaler)

        logging.info(
            f"Epoch {epoch + 1}/{config.training.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        if (epoch + 1) % config.checkpoint.save_frequency == 0 or is_best:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                config,
                is_best=is_best,
            )

    logging.info("Training finished")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    train(config_path)
