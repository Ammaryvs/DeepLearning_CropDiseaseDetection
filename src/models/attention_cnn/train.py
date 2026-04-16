"""Training script for AttentionCNN using CBAM-ResNet."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from common.config import FullConfig, validate_config
from common.dataloader import create_plantvillage_dataloaders
from common.metrics import MetricTracker, batch_accuracy, classification_metrics
from common.seed import set_seed
from common.utils import (
    EarlyStopping,
    History,
    load_checkpoint as load_training_checkpoint,
    load_json,
    save_checkpoint as save_training_checkpoint,
)
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
        force=True,
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


def create_dataloaders(config: FullConfig, seed: int = 42):
    data_bundle = create_plantvillage_dataloaders(
        root_dir=config.data.data_path,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        shuffle=config.data.shuffle,
        seed=seed,
        train_frac=config.data.train_split,
        val_frac=config.data.val_split,
        test_frac=config.data.test_split,
        train_tfms=create_data_transforms(config, is_train=True),
        val_tfms=create_data_transforms(config, is_train=False),
    )

    logging.info(f"Dataset root: {data_bundle['root_dir']}")
    logging.info(
        "Dataset split sizes - "
        f"train: {data_bundle['split_sizes']['train']} | "
        f"val: {data_bundle['split_sizes']['val']} | "
        f"test: {data_bundle['split_sizes']['test']}"
    )

    return data_bundle


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
    metrics = MetricTracker(("loss", "accuracy"))
    all_predictions = []
    all_targets = []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if config.device.mixed_precision and scaler is not None:
            with autocast('cuda'):
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

        batch_size = labels.size(0)
        detached_outputs = outputs.detach()
        predictions = detached_outputs.argmax(dim=1)

        metrics.update("loss", loss.item(), n=batch_size)
        metrics.update("accuracy", batch_accuracy(detached_outputs, labels), n=batch_size)
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

        if (batch_idx + 1) % config.checkpoint.log_frequency == 0:
            current_metrics = metrics.result()
            logging.info(
                f"Epoch [{epoch + 1}/{config.training.epochs}] "
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {current_metrics['loss']:.4f} "
                f"Acc: {current_metrics['accuracy']:.4f}"
            )

    epoch_metrics = metrics.result()
    classification_summary = classification_metrics(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=config.model.num_classes,
    )
    return {
        "loss": epoch_metrics["loss"],
        "accuracy": epoch_metrics["accuracy"],
        "precision": classification_summary["macro_precision"],
        "recall": classification_summary["macro_recall"],
        "f1": classification_summary["macro_f1"],
        "weighted_precision": classification_summary["weighted_precision"],
        "weighted_recall": classification_summary["weighted_recall"],
        "weighted_f1": classification_summary["weighted_f1"],
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: FullConfig,
    scaler: GradScaler = None,
):
    model.eval()
    metrics = MetricTracker(("loss", "accuracy"))
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            detached_outputs = outputs.detach()
            predictions = detached_outputs.argmax(dim=1)

            metrics.update("loss", loss.item(), n=batch_size)
            metrics.update("accuracy", batch_accuracy(detached_outputs, labels), n=batch_size)
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(labels.detach().cpu().tolist())

    val_metrics = metrics.result()
    classification_summary = classification_metrics(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=config.model.num_classes,
    )
    return {
        "loss": val_metrics["loss"],
        "accuracy": val_metrics["accuracy"],
        "precision": classification_summary["macro_precision"],
        "recall": classification_summary["macro_recall"],
        "f1": classification_summary["macro_f1"],
        "weighted_precision": classification_summary["weighted_precision"],
        "weighted_recall": classification_summary["weighted_recall"],
        "weighted_f1": classification_summary["weighted_f1"],
    }


def train(
    config_path: str = "config.yaml",
    config: FullConfig = None,
    run_name: str | None = None,
    resume_from: str | None = None,
    epochs_override: int | None = None,
):
    config = config or FullConfig.from_yaml(config_path)
    if epochs_override is not None:
        config.training.epochs = epochs_override
    config.data.random_seed = 42
    if not validate_config(config):
        raise ValueError("Invalid configuration")

    setup_logging(config)
    seed = 42
    set_seed(seed)
    logging.info(f"Random seed set to {seed}")
    device = setup_device(config)

    logging.info("Starting AttentionCNN CBAM-ResNet training")
    logging.info(str(config))

    data_bundle = create_dataloaders(config, seed=seed)
    inferred_num_classes = data_bundle["num_classes"]
    if config.model.num_classes != inferred_num_classes:
        logging.warning(
            "Config num_classes=%s does not match dataset classes=%s. Updating model config to match dataset.",
            config.model.num_classes,
            inferred_num_classes,
        )
        config.model.num_classes = inferred_num_classes

    train_loader = data_bundle["loaders"]["train"]
    val_loader = data_bundle["loaders"]["val"]
    model = create_model(config, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    scaler = GradScaler('cuda') if config.device.mixed_precision else None

    checkpoint_path = Path(resume_from).expanduser().resolve() if resume_from is not None else None
    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = 0
    best_metrics = None
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode="max" if config.training.early_stopping_metric == "val_accuracy" else "min",
    )
    training_history = History()

    if checkpoint_path is not None:
        checkpoint = load_training_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_acc = float(
            checkpoint.get(
                "best_val_acc",
                checkpoint.get("metrics", {}).get("val", {}).get("accuracy", 0.0),
            )
        )
        best_epoch = int(checkpoint.get("best_epoch", start_epoch if best_val_acc > 0 else 0))
        best_metrics = checkpoint.get("best_metrics")

        history_values = checkpoint.get("history")
        if isinstance(history_values, dict):
            training_history = History(values=history_values)
        else:
            history_file = Path(config.checkpoint.log_dir) / "training_history.json"
            if history_file.exists():
                training_history = History(values=load_json(history_file))

        history_dict = training_history.to_dict()
        val_acc_history = history_dict.get("val_acc", [])
        if val_acc_history:
            historical_best_val_acc = max(val_acc_history)
            if historical_best_val_acc > best_val_acc:
                best_val_acc = historical_best_val_acc
                best_epoch = val_acc_history.index(historical_best_val_acc) + 1

        early_stopping_state = checkpoint.get("early_stopping", {})
        if isinstance(early_stopping_state, dict):
            early_stopping.best_score = early_stopping_state.get("best_score")
            early_stopping.num_bad_epochs = int(early_stopping_state.get("num_bad_epochs", 0))
            early_stopping.should_stop = bool(early_stopping_state.get("should_stop", False))

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)

        logging.info(
            "Resumed training from checkpoint %s at epoch %s",
            checkpoint_path,
            start_epoch,
        )

    if start_epoch >= config.training.epochs:
        raise ValueError(
            "Checkpoint epoch is already at or beyond training.epochs. "
            "Increase training.epochs (or pass --epochs) to continue training."
        )

    for epoch in range(start_epoch, config.training.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config, epoch, scaler)
        val_metrics = validate(model, val_loader, criterion, device, config, scaler)

        logging.info(
            f"Epoch {epoch + 1}/{config.training.epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
            f"val_precision={val_metrics['precision']:.4f}, val_recall={val_metrics['recall']:.4f}"
        )

        training_history.update(
            train_loss=train_metrics["loss"],
            train_acc=train_metrics["accuracy"],
            train_precision=train_metrics["precision"],
            train_recall=train_metrics["recall"],
            train_f1=train_metrics["f1"],
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            val_f1=val_metrics["f1"],
        )

        if scheduler is not None:
            scheduler.step()

        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1
            best_metrics = {
                "train": dict(train_metrics),
                "val": dict(val_metrics),
            }

        should_save_regular = (epoch + 1) % config.checkpoint.save_frequency == 0
        should_update_best = config.checkpoint.save_best_only and is_best
        if should_save_regular or should_update_best:
            checkpoint_path = save_training_checkpoint(
                state={
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "metrics": {"train": train_metrics, "val": val_metrics},
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "history": training_history.to_dict(),
                    "early_stopping": {
                        "best_score": early_stopping.best_score,
                        "num_bad_epochs": early_stopping.num_bad_epochs,
                        "should_stop": early_stopping.should_stop,
                    },
                    "config": config.to_dict(),
                    "seed": seed,
                    "run_name": run_name,
                    "resume_from": str(checkpoint_path) if checkpoint_path is not None else None,
                },
                checkpoint_dir=config.checkpoint.checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch + 1:03d}.pt",
                save_primary=should_save_regular,
                is_best=should_update_best,
            )
            if should_save_regular:
                logging.info(f"Saved checkpoint: {checkpoint_path}")
            if should_update_best:
                logging.info(f"Saved best model: {Path(config.checkpoint.checkpoint_dir) / 'best_model.pt'}")

        monitored_score = val_metrics["accuracy"] if early_stopping.mode == "max" else val_metrics["loss"]
        early_stopping.step(monitored_score)
        if early_stopping.should_stop:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    history_file = Path(config.checkpoint.log_dir) / "training_history.json"
    training_history.save(history_file)
    logging.info(f"Training history saved: {history_file}")

    logging.info("Training finished")

    history_dict = training_history.to_dict()
    final_metrics = {
        "train": {
            "loss": history_dict["train_loss"][-1],
            "accuracy": history_dict["train_acc"][-1],
            "precision": history_dict["train_precision"][-1],
            "recall": history_dict["train_recall"][-1],
            "f1": history_dict["train_f1"][-1],
        },
        "val": {
            "loss": history_dict["val_loss"][-1],
            "accuracy": history_dict["val_acc"][-1],
            "precision": history_dict["val_precision"][-1],
            "recall": history_dict["val_recall"][-1],
            "f1": history_dict["val_f1"][-1],
        },
    }

    if best_metrics is None:
        best_epoch = len(history_dict["val_acc"])
        best_metrics = final_metrics

    return {
        "run_name": run_name or Path(config_path).stem,
        "seed": seed,
        "stopped_epoch": len(history_dict["train_loss"]),
        "best_epoch": best_epoch,
        "best_metric_name": config.checkpoint.best_model_metric,
        "parameters": {
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "batch_size": config.training.batch_size,
            "optimizer": config.training.optimizer,
            "scheduler": config.training.scheduler,
            "augmentation_strength": config.data.augmentation_strength,
            "train_split": config.data.train_split,
            "val_split": config.data.val_split,
            "test_split": config.data.test_split,
            "data_path": config.data.data_path,
        },
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "history": history_dict,
        "artifacts": {
            "checkpoint_dir": config.checkpoint.checkpoint_dir,
            "log_dir": config.checkpoint.log_dir,
            "history_file": str(history_file),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AttentionCNN using the provided config file.")
    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("--resume-from", dest="resume_from", default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the total number of epochs to train up to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path=args.config,
        resume_from=args.resume_from,
        epochs_override=args.epochs,
    )
