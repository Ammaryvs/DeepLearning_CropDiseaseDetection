"""
Vision Transformer Training Script using Configuration System

This script demonstrates how to train a Vision Transformer model using
the modular configuration system defined in src.common.config
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from torch.amp import autocast, GradScaler
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import sys
import os

# Import configuration system
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from common.config import FullConfig, validate_config
from common.dataloader import create_plantvillage_dataloaders
from common.metrics import MetricTracker, batch_accuracy, classification_metrics
from common.seed import set_seed
from common.utils import EarlyStopping, History, save_checkpoint as save_training_checkpoint


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(config: FullConfig):
    """Setup logging configuration."""
    log_dir = Path(config.checkpoint.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True,
    )
    
    logger = logging.getLogger(__name__)
    return logger


# ============================================================================
# DEVICE SETUP
# ============================================================================
def setup_device(config: FullConfig) -> torch.device:
    """Setup device based on configuration."""
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


# ============================================================================
# MODEL SETUP
# ============================================================================
def create_vit_model(config: FullConfig, device: torch.device) -> nn.Module:
    """
    Create Vision Transformer model.
    
    Note: This uses timm (pytorch-image-models) for better ViT implementations.
    Install with: pip install timm
    """
    try:
        import timm
        
        # Create ViT model from timm
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=config.model.pretrained,
            num_classes=config.model.num_classes,
            drop_rate=config.model.dropout_rate
        )
        
        logging.info(f"Loaded ViT model (pretrained={config.model.pretrained})")
        
    except ImportError:
        # Fallback: Use torchvision if timm not available
        logging.warning("timm not installed. Using standard PyTorch vision models. "
                       "Install timm for better ViT: pip install timm")
        raise ImportError("Please install timm: pip install timm")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters - Total: {total_params:,} | Trainable: {trainable_params:,}")
    
    return model


# ============================================================================
# DATA LOADING
# ============================================================================
def create_data_transforms(config: FullConfig, is_train: bool = True):
    """Create data transformation pipeline."""
    
    if is_train and config.data.augmentation:
        transforms_list = [
            transforms.RandomResizedCrop(
                config.model.input_size,
                scale=(0.8, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2 * config.data.augmentation_strength,
                contrast=0.2 * config.data.augmentation_strength,
                saturation=0.2 * config.data.augmentation_strength
            ),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:
        transforms_list = [
            transforms.Resize((config.model.input_size, config.model.input_size)),
            transforms.CenterCrop(config.model.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    
    return transforms.Compose(transforms_list)


def create_dataloaders(config: FullConfig, seed: int = 42):
    """Create train/validation dataloaders and return dataset metadata."""
    train_transform = create_data_transforms(config, is_train=True)
    val_transform = create_data_transforms(config, is_train=False)
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
        train_tfms=train_transform,
        val_tfms=val_transform,
    )

    logging.info(f"Dataset root: {data_bundle['root_dir']}")
    logging.info(
        "Dataset split sizes - "
        f"train: {data_bundle['split_sizes']['train']} | "
        f"val: {data_bundle['split_sizes']['val']} | "
        f"test: {data_bundle['split_sizes']['test']}"
    )

    return data_bundle


# ============================================================================
# OPTIMIZER & SCHEDULER
# ============================================================================
def create_optimizer(config: FullConfig, model: nn.Module) -> optim.Optimizer:
    """Create optimizer based on configuration."""
    
    if config.training.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        logging.info("Using Adam optimizer")
        
    elif config.training.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        logging.info("Using AdamW optimizer")
        
    elif config.training.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay
        )
        logging.info("Using SGD optimizer")
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
    
    return optimizer


def create_scheduler(config: FullConfig, optimizer: optim.Optimizer, steps_per_epoch: int):
    """Create learning rate scheduler based on configuration."""
    
    if config.training.scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.learning_rate * 0.01
        )
        logging.info("Using Cosine Annealing scheduler")
        
    elif config.training.scheduler.lower() == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.training.scheduler_step_size,
            gamma=config.training.scheduler_gamma
        )
        logging.info("Using Step scheduler")
        
    elif config.training.scheduler.lower() == "exponential":
        scheduler = ExponentialLR(
            optimizer,
            gamma=config.training.scheduler_gamma
        )
        logging.info("Using Exponential scheduler")
        
    else:
        logging.warning(f"Unknown scheduler: {config.training.scheduler}. Using no scheduler.")
        scheduler = None
    
    return scheduler


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: FullConfig,
    epoch: int,
    scaler: GradScaler = None
) -> dict:
    """Train for one epoch."""
    
    model.train()
    metrics = MetricTracker(("loss", "accuracy"))
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training if enabled
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
        
        # Log progress
        if (batch_idx + 1) % config.checkpoint.log_frequency == 0:
            current_metrics = metrics.result()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'acc': f"{current_metrics['accuracy']:.4f}",
            })
    
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
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: FullConfig,
    scaler: GradScaler = None
) -> dict:
    """Validate the model."""
    
    model.eval()
    metrics = MetricTracker(("loss", "accuracy"))
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
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


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train(config_path: str = "config.yaml", config: FullConfig = None, run_name: str | None = None):
    """Main training function."""
    
    # Load configuration
    config = config or FullConfig.from_yaml(config_path)
    config.data.random_seed = 42
    
    # Validate configuration
    if not validate_config(config):
        logging.error("Configuration validation failed!")
        return
    
    # Setup
    logger = setup_logging(config)
    seed = 42
    set_seed(seed)
    logging.info(f"Random seed set to {seed}")
    device = setup_device(config)
    
    logging.info("=" * 70)
    logging.info("Vision Transformer Training")
    logging.info("=" * 70)
    logging.info(str(config))
    
    # Create dataloaders
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

    # Create model after class count is confirmed
    model = create_vit_model(config, device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if config.device.mixed_precision else None
    
    # Training loop
    best_val_accuracy = 0.0
    best_epoch = 0
    best_metrics = None
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode="max" if config.training.early_stopping_metric == "val_accuracy" else "min",
    )
    training_history = History()
    
    for epoch in range(config.training.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch, scaler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config, scaler)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{config.training.epochs} - "
                    f"Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        
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
        
        # Determine if this is best epoch
        is_best = False
        if config.checkpoint.best_model_metric == "val_accuracy":
            is_best = val_metrics["accuracy"] > best_val_accuracy
            if is_best:
                best_val_accuracy = val_metrics["accuracy"]
        else:
            previous_best = early_stopping.best_score
            is_best = previous_best is None or val_metrics["loss"] < previous_best

        if is_best:
            best_epoch = epoch + 1
            best_metrics = {
                "train": dict(train_metrics),
                "val": dict(val_metrics),
            }
        
        # Save checkpoint
        should_save_regular = (epoch + 1) % config.checkpoint.save_frequency == 0
        should_update_best = config.checkpoint.save_best_only and is_best
        if should_save_regular or should_update_best:
            checkpoint_path = save_training_checkpoint(
                state={
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'metrics': {
                        'train': train_metrics,
                        'val': val_metrics,
                    },
                    'config': config.to_dict(),
                    'seed': seed,
                    'run_name': run_name,
                },
                checkpoint_dir=config.checkpoint.checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch + 1:03d}.pt",
                save_primary=should_save_regular,
                is_best=should_update_best,
            )
            if should_save_regular:
                logging.info(f"Checkpoint saved: {checkpoint_path}")
            if should_update_best:
                logging.info(
                    f"Best model updated: {Path(config.checkpoint.checkpoint_dir) / 'best_model.pt'}"
                )
        
        # Early stopping
        monitored_score = val_metrics["accuracy"] if early_stopping.mode == "max" else val_metrics["loss"]
        early_stopping.step(monitored_score)
        if early_stopping.should_stop:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save training history
    history_file = Path(config.checkpoint.log_dir) / "training_history.json"
    training_history.save(history_file)
    logging.info(f"Training history saved: {history_file}")
    
    logging.info("=" * 70)
    logging.info("Training completed!")
    logging.info("=" * 70)

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


if __name__ == "__main__":
    # Train with default config
    train("config.yaml")
