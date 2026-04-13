"""
Configuration management module for model training and inference.
Provides utilities to configure parameters for any model architecture.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
import yaml
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture and parameters."""
    model_name: str
    num_classes: int = 26
    input_size: int = 224
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.3
    
    def __str__(self) -> str:
        return f"{self.model_name} (classes: {self.num_classes}, input: {self.input_size})"


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    momentum: float = 0.9
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "cosine"  # cosine, step, exponential, linear
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"  # val_loss, val_accuracy
    
    def __str__(self) -> str:
        return (f"Training: {self.epochs} epochs, LR={self.learning_rate}, "
                f"BS={self.batch_size}, Optimizer={self.optimizer}")


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str = "./data"
    train_path: str = None
    val_path: str = None
    test_path: str = None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    augmentation: bool = True
    augmentation_strength: float = 0.5
    validation_split: float = 0.2
    random_seed: int = 42
    
    def __post_init__(self):
        if self.train_path is None:
            self.train_path = str(Path(self.data_path) / "train")
        if self.val_path is None:
            self.val_path = str(Path(self.data_path) / "val")
        if self.test_path is None:
            self.test_path = str(Path(self.data_path) / "test")


@dataclass
class DeviceConfig:
    """Configuration for device and distributed training."""
    device: str = "cuda"  # cuda, cpu, mps
    multi_gpu: bool = False
    gpu_ids: list = field(default_factory=lambda: [0])
    mixed_precision: bool = False
    
    def __str__(self) -> str:
        return f"Device: {self.device}, MultiGPU: {self.multi_gpu}"


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing and logging."""
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 5  # save every N epochs
    save_best_only: bool = True
    best_model_metric: str = "val_accuracy"
    log_dir: str = "./logs"
    log_frequency: int = 10  # log every N batches
    
    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class FullConfig:
    """Complete configuration combining all parameter types."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    device: DeviceConfig = field(default_factory=DeviceConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'FullConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        training_cfg = TrainingConfig(**config_dict.get('training', {}))
        data_cfg = DataConfig(**config_dict.get('data', {}))
        device_cfg = DeviceConfig(**config_dict.get('device', {}))
        checkpoint_cfg = CheckpointConfig(**config_dict.get('checkpoint', {}))
        
        return cls(
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            device=device_cfg,
            checkpoint=checkpoint_cfg
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FullConfig':
        """Create configuration from dictionary."""
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        training_cfg = TrainingConfig(**config_dict.get('training', {}))
        data_cfg = DataConfig(**config_dict.get('data', {}))
        device_cfg = DeviceConfig(**config_dict.get('device', {}))
        checkpoint_cfg = CheckpointConfig(**config_dict.get('checkpoint', {}))
        
        return cls(
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            device=device_cfg,
            checkpoint=checkpoint_cfg
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'device': asdict(self.device),
            'checkpoint': asdict(self.checkpoint)
        }
    
    def save_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, output_path: str) -> None:
        """Save configuration to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __str__(self) -> str:
        return (f"\n{'='*50}\n"
                f"Model Config:\n  {self.model}\n"
                f"Training Config:\n  {self.training}\n"
                f"Data Config:\n  Path: {self.data.data_path}\n"
                f"{self.device}\n"
                f"Checkpoint Dir: {self.checkpoint.checkpoint_dir}\n"
                f"{'='*50}")


# Helper functions for quick configuration setup

def create_default_config(
    model_name: str,
    num_classes: int = 26,
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    data_path: str = "./data",
    device: str = "cuda"
) -> FullConfig:
    """Create a configuration with default settings."""
    return FullConfig(
        model=ModelConfig(
            model_name=model_name,
            num_classes=num_classes
        ),
        training=TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        ),
        data=DataConfig(data_path=data_path),
        device=DeviceConfig(device=device)
    )


def update_config(
    config: FullConfig,
    **kwargs
) -> FullConfig:
    """Update specific configuration parameters."""
    config_dict = config.to_dict()
    
    for key, value in kwargs.items():
        # Handle nested updates like 'training.learning_rate'
        if '.' in key:
            section, param = key.split('.', 1)
            if section in config_dict:
                config_dict[section][param] = value
        else:
            # Try to update in main sections
            for section in ['model', 'training', 'data', 'device', 'checkpoint']:
                if key in config_dict.get(section, {}):
                    config_dict[section][key] = value
                    break
    
    return FullConfig.from_dict(config_dict)


def validate_config(config: FullConfig) -> bool:
    """Validate configuration parameters."""
    errors = []
    
    if config.training.epochs <= 0:
        errors.append("Epochs must be positive")
    
    if config.training.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.training.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.model.num_classes <= 0:
        errors.append("Number of classes must be positive")

    if min(config.data.train_split, config.data.val_split, config.data.test_split) < 0:
        errors.append("Data split fractions must be non-negative")

    split_sum = config.data.train_split + config.data.val_split + config.data.test_split
    if abs(split_sum - 1.0) > 1e-6:
        errors.append("train_split + val_split + test_split must equal 1.0")
    
    if config.training.optimizer not in ["adam", "sgd", "adamw"]:
        errors.append(f"Invalid optimizer: {config.training.optimizer}")
    
    if config.device.device not in ["cuda", "cpu", "mps"]:
        errors.append(f"Invalid device: {config.device.device}")
    
    if errors:
        print("Configuration Validation Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == "__main__":
    # Example usage
    config = create_default_config(
        model_name="ResNet50",
        num_classes=26,
        epochs=100,
        learning_rate=0.0001,
        batch_size=64
    )
    
    print(config)
    print("\nValidation:", validate_config(config))
    
    # Save configuration
    config.save_yaml("config.yaml")
    config.save_json("config.json")
    
    # Update configuration
    updated_config = update_config(
        config,
        **{
            "training.epochs": 50,
            "training.learning_rate": 0.001
        }
    )
    print("\nUpdated config:")
    print(updated_config)
