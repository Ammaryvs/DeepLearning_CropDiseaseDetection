# Configuration Helper Module

A flexible and comprehensive configuration system for managing model training parameters across all your deep learning models.

## Features

- **Modular Design**: Separate configuration classes for model, training, data, device, and checkpointing
- **Multiple Input Formats**: Load from YAML, JSON, dictionaries, or create programmatically
- **Easy Updates**: Simple API to modify specific parameters
- **Validation**: Built-in validation for configuration parameters
- **Export Options**: Save configurations to YAML or JSON for reproducibility
- **Type Safety**: Uses Python dataclasses with type hints

## Configuration Components

### 1. **ModelConfig**
Configure model architecture and parameters:
```python
model: ModelConfig(
    model_name="resnet50",
    num_classes=26,
    input_size=224,
    pretrained=True,
    freeze_backbone=False,
    dropout_rate=0.3
)
```

### 2. **TrainingConfig**
Configure training hyperparameters:
```python
training: TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    optimizer="adam",            # adam, sgd, adamw
    scheduler="cosine",          # cosine, step, exponential, linear
    early_stopping_patience=10
)
```

### 3. **DataConfig**
Configure data loading and preprocessing:
```python
data: DataConfig(
    data_path="./data",
    num_workers=4,
    augmentation=True,
    augmentation_strength=0.5,
    validation_split=0.2
)
```

### 4. **DeviceConfig**
Configure device and distributed training:
```python
device: DeviceConfig(
    device="cuda",           # cuda, cpu, mps
    multi_gpu=False,
    gpu_ids=[0],
    mixed_precision=False
)
```

### 5. **CheckpointConfig**
Configure model checkpointing and logging:
```python
checkpoint: CheckpointConfig(
    checkpoint_dir="./checkpoints",
    save_frequency=5,
    save_best_only=True,
    log_dir="./logs"
)
```

## Usage Examples

### Quick Start
```python
from src.common.config import create_default_config, validate_config

# Create configuration with custom parameters
config = create_default_config(
    model_name="resnet50",
    num_classes=26,
    epochs=100,
    learning_rate=0.0001,
    batch_size=64,
    device="cuda"
)

# Validate configuration
if validate_config(config):
    print("✓ Configuration is valid")

print(config)
```

### Load from YAML
```python
from src.common.config import FullConfig

# Load configuration
config = FullConfig.from_yaml("config.yaml")

# Access parameters
print(f"Training for {config.training.epochs} epochs")
print(f"Learning rate: {config.training.learning_rate}")
```

### Update Specific Parameters
```python
from src.common.config import update_config

# Modify specific parameters
config = update_config(
    config,
    **{
        "training.epochs": 150,
        "training.learning_rate": 0.0005,
        "training.batch_size": 128,
        "device.multi_gpu": True
    }
)
```

### Save Configuration
```python
# Save to YAML
config.save_yaml("my_config.yaml")

# Save to JSON
config.save_json("my_config.json")

# Load later
loaded_config = FullConfig.from_yaml("my_config.yaml")
```

### Use in Training Script
```python
config = FullConfig.from_yaml("config.yaml")

# Model setup
model = create_model(
    name=config.model.model_name,
    num_classes=config.model.num_classes,
    input_size=config.model.input_size,
    pretrained=config.model.pretrained
)

# Data loading
train_loader = create_dataloader(
    path=config.data.train_path,
    batch_size=config.training.batch_size,
    num_workers=config.data.num_workers,
    augmentation=config.data.augmentation
)

# Optimizer setup
if config.training.optimizer == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
elif config.training.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.training.learning_rate,
        momentum=config.training.momentum
    )

# Training loop
for epoch in range(config.training.epochs):
    for batch in train_loader:
        # training code
        pass
    
    # Save checkpoint
    if (epoch + 1) % config.checkpoint.save_frequency == 0:
        torch.save(
            model.state_dict(),
            f"{config.checkpoint.checkpoint_dir}/epoch_{epoch}.pt"
        )
```

## Supported Optimizers

- **adam**: Adaptive Moment Estimation
- **sgd**: Stochastic Gradient Descent
- **adamw**: AdamW (Adam with weight decay)

## Supported Schedulers

- **cosine**: Cosine annealing learning rate scheduler
- **step**: Step decay scheduler
- **exponential**: Exponential decay scheduler
- **linear**: Linear learning rate decay

## Model-Specific Recommendations

### ResNet50
```python
config = create_default_config(
    model_name="resnet50",
    epochs=100,
    learning_rate=0.001,
    batch_size=32
)
```

### Vision Transformer (ViT)
```python
config = create_default_config(
    model_name="vit",
    epochs=200,
    learning_rate=0.0001,      # Lower LR for transformer
    batch_size=16              # Larger memory requirement
)
```

### MobileNetV2 (Lightweight)
```python
config = create_default_config(
    model_name="mobilenetv2",
    epochs=50,
    learning_rate=0.001,
    batch_size=128              # Can use larger batch
)
```

### EfficientNet
```python
config = create_default_config(
    model_name="efficientnet_b0",
    epochs=100,
    learning_rate=0.001,
    batch_size=64
)
```

## Best Practices

1. **Always validate** configurations before training:
   ```python
   if not validate_config(config):
       print("Configuration has errors!")
       return
   ```

2. **Save configurations** for reproducibility:
   ```python
   config.save_yaml("experiments/resnet50_v1.yaml")
   ```

3. **Use sensible defaults** for different model sizes:
   - Smaller models (MobileNet): larger batch size, higher LR
   - Medium models (ResNet50): balanced settings
   - Large models (ViT): smaller batch size, lower LR

4. **Document your experiments** by saving configs:
   ```
   experiments/
   ├── resnet50_baseline.yaml
   ├── resnet50_highAugment.yaml
   └── vit_pretrained.yaml
   ```

5. **Use early stopping** with validation metrics:
   ```python
   training:
       early_stopping_patience: 15
       early_stopping_metric: val_loss
   ```

## API Reference

### FullConfig

#### Methods
- `from_yaml(yaml_path)`: Load configuration from YAML file
- `from_dict(config_dict)`: Create from dictionary
- `to_dict()`: Convert to dictionary
- `save_yaml(output_path)`: Save to YAML file
- `save_json(output_path)`: Save to JSON file

### Helper Functions

- `create_default_config(...)`: Quickly create configuration with defaults
- `update_config(config, **kwargs)`: Update specific parameters
- `validate_config(config)`: Validate configuration parameters

## File Structure

```
src/common/
├── config.py              # Main configuration module
├── config_examples.py     # Comprehensive usage examples
└── config_help.md        # This file

src/models/
├── config_example.yaml    # Example YAML configuration
├── simple_cnn/
│   ├── config.yaml        # Model-specific config
│   ├── model.py
│   └── train.py
└── resnet50/
    ├── config.yaml
    ├── model.py
    └── train.py
```

## Troubleshooting

### "Learning rate must be positive" error
```python
# ❌ Wrong
config.training.learning_rate = 0

# ✅ Correct
config.training.learning_rate = 0.001
```

### "Invalid optimizer" error
```python
# ❌ Wrong
config.training.optimizer = "adam_custom"

# ✅ Correct - use: adam, sgd, adamw
config.training.optimizer = "adam"
```

### Device not found
```python
# If CUDA not available, fall back to CPU
config.device.device = "cpu"  # or check torch.cuda.is_available()
```

## Contributing

To extend the configuration system:

1. Add new parameters to relevant dataclass
2. Update validation in `validate_config()`
3. Add examples to `config_examples.py`
4. Document in this file

## License

Same as the main project
