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
from torch.cuda.amp import autocast, GradScaler
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
sys.path.insert(0, '../../')
from common.config import FullConfig, validate_config


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
        ]
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


def create_dataloaders(config: FullConfig):
    """Create train and validation dataloaders."""
    from torchvision.datasets import ImageFolder
    
    train_transform = create_data_transforms(config, is_train=True)
    val_transform = create_data_transforms(config, is_train=False)
    
    # Load datasets
    train_dataset = ImageFolder(config.data.train_path, transform=train_transform)
    val_dataset = ImageFolder(config.data.val_path, transform=val_transform)
    
    logging.info(f"Train dataset: {len(train_dataset)} images")
    logging.info(f"Validation dataset: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    return train_loader, val_loader


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
) -> tuple:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training if enabled
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
        
        # Calculate metrics
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # Log progress
        if (batch_idx + 1) % config.checkpoint.log_frequency == 0:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.4f}'})
    
    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples
    
    return epoch_loss, epoch_accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler = None
) -> tuple:
    """Validate the model."""
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    val_loss = total_loss / total_samples
    val_accuracy = total_correct / total_samples
    
    return val_loss, val_accuracy


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================
def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: FullConfig,
    is_best: bool = False
):
    """Save model checkpoint."""
    
    checkpoint_dir = Path(config.checkpoint.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.to_dict()
    }
    
    # Regular checkpoint
    filename = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, filename)
    
    # Best checkpoint
    if is_best:
        best_filename = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_filename)
        logging.info(f"Best model saved: {best_filename}")
    else:
        logging.info(f"Checkpoint saved: {filename}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train(config_path: str = "config.yaml"):
    """Main training function."""
    
    # Load configuration
    config = FullConfig.from_yaml(config_path)
    
    # Validate configuration
    if not validate_config(config):
        logging.error("Configuration validation failed!")
        return
    
    # Setup
    logger = setup_logging(config)
    device = setup_device(config)
    
    logging.info("=" * 70)
    logging.info("Vision Transformer Training")
    logging.info("=" * 70)
    logging.info(str(config))
    
    # Create model
    model = create_vit_model(config, device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler() if config.device.mixed_precision else None
    
    # Training loop
    best_val_accuracy = 0.0
    patience_counter = 0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.training.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, scaler)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{config.training.epochs} - "
                    f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        # Determine if this is best epoch
        is_best = False
        if config.checkpoint.best_model_metric == "val_accuracy":
            is_best = val_acc > best_val_accuracy
            if is_best:
                best_val_accuracy = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            is_best = val_loss < training_history['val_loss'][0] if epoch > 0 else True
            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint.save_frequency == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch,
                {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                },
                config,
                is_best=is_best
            )
        
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save training history
    history_file = Path(config.checkpoint.log_dir) / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    logging.info(f"Training history saved: {history_file}")
    
    logging.info("=" * 70)
    logging.info("Training completed!")
    logging.info("=" * 70)


if __name__ == "__main__":
    # Train with default config
    train("config.yaml")