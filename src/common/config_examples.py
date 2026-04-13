"""
Usage Examples for the Configuration Module

This file demonstrates how to use the configuration helper functions
in your training and model scripts.
"""

from src.common.config import (
    FullConfig, ModelConfig, TrainingConfig, DataConfig, 
    DeviceConfig, CheckpointConfig,
    create_default_config, update_config, validate_config
)


# ============================================================================
# EXAMPLE 1: Create configuration from scratch with defaults
# ============================================================================
def example_basic_setup():
    """Most basic setup - use defaults and just customize key parameters."""
    config = create_default_config(
        model_name="resnet50",
        num_classes=26,
        epochs=100,
        learning_rate=0.0001,
        batch_size=64,
        data_path="./data",
        device="cuda"
    )
    
    print(config)
    return config


# ============================================================================
# EXAMPLE 2: Load configuration from YAML file
# ============================================================================
def example_load_from_yaml():
    """Load configuration from a YAML file."""
    config = FullConfig.from_yaml("src/models/config_example.yaml")
    
    print(config)
    
    # Validate before training
    if validate_config(config):
        print("✓ Configuration is valid")
    
    return config


# ============================================================================
# EXAMPLE 3: Create custom configuration from scratch
# ============================================================================
def example_custom_config():
    """Fully customize all configuration parameters."""
    model_cfg = ModelConfig(
        model_name="vit",
        num_classes=26,
        input_size=384,
        pretrained=True,
        dropout_rate=0.2
    )
    
    training_cfg = TrainingConfig(
        epochs=200,
        batch_size=16,
        learning_rate=0.0001,
        optimizer="adamw",
        scheduler="cosine",
        early_stopping_patience=20
    )
    
    data_cfg = DataConfig(
        data_path="./PlantVillage-Dataset/raw/segmented",
        num_workers=8,
        augmentation_strength=0.7
    )
    
    device_cfg = DeviceConfig(
        device="cuda",
        multi_gpu=True,
        gpu_ids=[0, 1]
    )
    
    checkpoint_cfg = CheckpointConfig(
        checkpoint_dir="./checkpoints/vit",
        save_frequency=1,
        best_model_metric="val_accuracy"
    )
    
    config = FullConfig(
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        device=device_cfg,
        checkpoint=checkpoint_cfg
    )
    
    print(config)
    return config


# ============================================================================
# EXAMPLE 4: Update configuration parameters after creation
# ============================================================================
def example_update_config():
    """Modify specific parameters in an existing configuration."""
    config = create_default_config(
        model_name="resnet50",
        num_classes=26,
        epochs=50,
        learning_rate=0.001
    )
    
    # Update specific nested parameters
    updated_config = update_config(
        config,
        **{
            "training.epochs": 100,
            "training.learning_rate": 0.0001,
            "training.batch_size": 64,
            "training.optimizer": "adamw",
            "data.augmentation": True,
            "device.multi_gpu": True
        }
    )
    
    print("Original:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  LR: {config.training.learning_rate}")
    
    print("\nUpdated:")
    print(f"  Epochs: {updated_config.training.epochs}")
    print(f"  LR: {updated_config.training.learning_rate}")
    
    return updated_config


# ============================================================================
# EXAMPLE 5: Use configuration in training script
# ============================================================================
def example_training_usage(config: FullConfig):
    """Show how to use configuration in training loop."""
    
    # Access model parameters
    print(f"Model: {config.model.model_name}")
    print(f"Classes: {config.model.num_classes}")
    print(f"Input size: {config.model.input_size}")
    
    # Access training parameters
    print(f"\nTraining for {config.training.epochs} epochs")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Optimizer: {config.training.optimizer}")
    
    # Access data parameters
    print(f"\nLoading data from: {config.data.data_path}")
    print(f"Number of workers: {config.data.num_workers}")
    print(f"Augmentation enabled: {config.data.augmentation}")
    
    # Access device parameters
    print(f"\nDevice: {config.device.device}")
    if config.device.multi_gpu:
        print(f"Using GPUs: {config.device.gpu_ids}")
    
    # Access checkpoint parameters
    print(f"\nCheckpoint dir: {config.checkpoint.checkpoint_dir}")
    print(f"Save every {config.checkpoint.save_frequency} epochs")


# ============================================================================
# EXAMPLE 6: Save and load configuration for reproducibility
# ============================================================================
def example_save_load_config():
    """Demonstrate saving and loading configurations."""
    # Create and configure
    config = create_default_config(
        model_name="mobilenetv2",
        epochs=50,
        learning_rate=0.001,
        batch_size=32
    )
    
    # Save to YAML
    config.save_yaml("my_training_config.yaml")
    print("✓ Saved to my_training_config.yaml")
    
    # Save to JSON
    config.save_json("my_training_config.json")
    print("✓ Saved to my_training_config.json")
    
    # Load back
    loaded_config = FullConfig.from_yaml("my_training_config.yaml")
    print("✓ Loaded from YAML")
    print(f"Loaded model: {loaded_config.model.model_name}")


# ============================================================================
# EXAMPLE 7: Different configurations for different models
# ============================================================================
def example_model_specific_configs():
    """Create optimized configurations for different model architectures."""
    
    configs = {}
    
    # ResNet50 - balanced configuration
    configs['resnet50'] = create_default_config(
        model_name='resnet50',
        epochs=100,
        learning_rate=0.001,
        batch_size=32
    )
    
    # ViT - requires lower learning rate
    configs['vit'] = create_default_config(
        model_name='vit',
        epochs=200,
        learning_rate=0.0001,
        batch_size=16
    )
    
    # MobileNetV2 - lighter model, can use larger batch size
    configs['mobilenetv2'] = create_default_config(
        model_name='mobilenetv2',
        epochs=50,
        learning_rate=0.001,
        batch_size=128
    )
    
    # EfficientNet - can use larger input size
    configs['efficientnet_b0'] = create_default_config(
        model_name='efficientnet_b0',
        epochs=100,
        learning_rate=0.001,
        batch_size=64
    )
    
    for model_name, config in configs.items():
        print(f"\n{model_name}:")
        print(f"  Epochs: {config.training.epochs}")
        print(f"  LR: {config.training.learning_rate}")
        print(f"  Batch: {config.training.batch_size}")
    
    return configs


# ============================================================================
# EXAMPLE 8: Integration with actual training loop
# ============================================================================
def example_training_loop_integration(config: FullConfig):
    """Shows how to integrate configuration with training loop."""
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed!")
        return
    
    # Setup from configuration
    num_epochs = config.training.epochs
    batch_size = config.training.batch_size
    lr = config.training.learning_rate
    optimizer_type = config.training.optimizer
    device_type = config.device.device
    
    # Create your dataloaders
    # train_loader = create_dataloader(
    #     config.data.train_path,
    #     batch_size=batch_size,
    #     num_workers=config.data.num_workers,
    #     augmentation=config.data.augmentation
    # )
    
    # Create your model
    # model = create_model(
    #     config.model.model_name,
    #     num_classes=config.model.num_classes,
    #     pretrained=config.model.pretrained
    # )
    
    # Create optimizer based on config
    # if optimizer_type == "adam":
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # elif optimizer_type == "sgd":
    #     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.training.momentum)
    
    # Training loop
    # for epoch in range(num_epochs):
    #     for batch in train_loader:
    #         # training code using config parameters
    #         if config.checkpoint.save_frequency > 0 and (epoch + 1) % config.checkpoint.save_frequency == 0:
    #             # save checkpoint to config.checkpoint.checkpoint_dir
    #             pass
    
    print("Training loop would use the configuration parameters above")


if __name__ == "__main__":
    print("=" * 70)
    print("EXAMPLE 1: Basic Setup")
    print("=" * 70)
    example_basic_setup()
    
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Update Configuration")
    print("=" * 70)
    example_update_config()
    
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Model-Specific Configs")
    print("=" * 70)
    example_model_specific_configs()
