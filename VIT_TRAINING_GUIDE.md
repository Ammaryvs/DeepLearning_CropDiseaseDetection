# Vision Transformer Training Guide

A step-by-step guide to training a Vision Transformer (ViT) for plant disease detection using the configuration system.

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
# Install required packages
pip install timm torch torchvision tqdm pyyaml
```

### Step 2: Configure Your Training
Edit `src/models/vit/config.yaml` with your desired parameters:
```yaml
training:
  epochs: 150
  batch_size: 16
  learning_rate: 0.0001
  optimizer: adamw
device:
  device: cuda
  mixed_precision: true
```

### Step 3: Run Training
```bash
cd src/models/vit
python train.py
```

---

## Complete Setup Guide

### Prerequisites

1. **Python 3.8+** and **PyTorch** with CUDA support
2. **Datasets**: PlantVillage or PlantDoc organized in train/val folders
3. **Required packages**:
   ```bash
   pip install torch torchvision timm tqdm pyyaml tensorboard
   ```

### Project Structure
```
src/models/vit/
├── config.yaml          # Configuration file (MODIFY THIS)
├── train.py            # Training script
├── test.py             # Evaluation script
└── model.py            # Model architecture (optional)
```

---

## Configuration Explained

### Model Parameters
```yaml
model:
  model_name: vit                 # Model type
  num_classes: 26                 # Number of disease classes
  input_size: 224                 # Input image size
  pretrained: true                # Use ImageNet pretrained weights
  dropout_rate: 0.1               # Dropout for regularization
```

**Key decisions:**
- `pretrained: true` - Recommended! Transfers knowledge from ImageNet
- `input_size: 224` - Standard for ViT. Can be 384 for higher accuracy (slower)
- `num_classes: 26` - 25 plant diseases + 1 healthy class

### Training Parameters
```yaml
training:
  epochs: 150                      # Number of training passes
  batch_size: 16                   # Samples per batch (lower = less GPU memory)
  learning_rate: 0.0001            # Step size (lower for transformers)
  optimizer: adamw                 # AdamW is best for ViT
  scheduler: cosine                # Learning rate schedule
  early_stopping_patience: 20      # Stop if no improvement for N epochs
```

**Tuning tips:**
- **Batch size**: Increase to 32-64 if you have enough GPU memory (faster training)
- **Learning rate**: Start with 0.0001, decrease if training is unstable
- **Epochs**: 150 is good for convergence, reduce to 50 for quick tests
- **Optimizer**: AdamW is recommended for Vision Transformers

### Data Parameters
```yaml
data:
  data_path: ../../../PlantVillage-Dataset/raw/segmented
  num_workers: 4                   # Parallel data loading threads
  augmentation: true               # Data augmentation (recommended)
  augmentation_strength: 0.5       # 0-1 scale for augmentation intensity
```

**Important:**
- Update `data_path` to point to your dataset
- Ensure dataset structure:
  ```
  dataset/
  ├── train/
  │   ├── Apple leaf/
  │   ├── Tomato early blight/
  │   └── ...
  ├── val/
  │   ├── Apple leaf/
  │   └── ...
  └── test/
  ```

### Device Parameters
```yaml
device:
  device: cuda                     # cuda, cpu, or mps
  multi_gpu: false                 # Set to true for multiple GPUs
  mixed_precision: true            # Uses AMP for faster training
```

**GPU settings:**
- `mixed_precision: true` - Makes training 2x faster with minimal quality loss
- `multi_gpu`: Use if you have multiple GPUs (requires setup)

### Checkpoint Parameters
```yaml
checkpoint:
  checkpoint_dir: ./checkpoints/vit
  save_frequency: 5                # Save every 5 epochs
  save_best_only: true             # Only keep best model
  best_model_metric: val_accuracy
  log_dir: ./logs/vit
```

---

## Training Examples

### Example 1: Quick Test (Fast, Low Accuracy)
```yaml
training:
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
data:
  augmentation: false
```
**Use when**: Testing the pipeline, debugging

### Example 2: Standard Training (Recommended)
```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adamw
  scheduler: cosine
data:
  augmentation: true
  augmentation_strength: 0.5
device:
  mixed_precision: true
```
**Use when**: Production training, best balance of speed/accuracy

### Example 3: High Accuracy (Slow, Best Results)
```yaml
model:
  input_size: 384              # Larger input
training:
  epochs: 200
  batch_size: 16               # Smaller for stability
  learning_rate: 0.00005       # Lower LR for fine details
  early_stopping_patience: 30
data:
  augmentation: true
  augmentation_strength: 0.7   # Stronger augmentation
device:
  mixed_precision: true
```
**Use when**: Final production model, accuracy is priority

---

## Running Training

### Basic Training
```bash
cd src/models/vit
python train.py
```

### Training with Custom Config
```bash
# Not needed if using config.yaml in the same directory
python train.py --config config.yaml
```

### Monitor Training
```bash
# In another terminal
tensorboard --logdir ./logs/vit
# Open http://localhost:6006
```

---

## Output Files

After training, you'll find:

```
vit/
├── checkpoints/vit/
│   ├── best_model.pt             # Best validation model
│   ├── checkpoint_epoch_050.pt    # Periodic checkpoints
│   └── checkpoint_epoch_100.pt
└── logs/vit/
    ├── training_20240101_120000.log
    └── training_history.json
```

**Files explained:**
- `best_model.pt` - Use this for inference! Contains model + optimizer state
- `checkpoint_epoch_*.pt` - For resuming training
- `training_history.json` - Loss/accuracy curves for analysis

---

## Common Issues & Solutions

### ❌ CUDA Out of Memory
**Solution**: Decrease `batch_size` in config.yaml
```yaml
training:
  batch_size: 8  # or 4, 2
```

### ❌ "timm not installed" Error
**Solution**: Install timm
```bash
pip install timm
```

### ❌ Dataset Not Found
**Solution**: Update `data_path` in config.yaml:
```yaml
data:
  data_path: C:/path/to/PlantVillage-Dataset/raw/segmented
```

### ❌ Poor Accuracy
**Solution**: Try these in order:
1. Increase epochs: `epochs: 200`
2. Reduce learning rate: `learning_rate: 0.00005`
3. Increase batch size: `batch_size: 32`
4. Enable stronger augmentation: `augmentation_strength: 0.7`

### ❌ Training Too Slow
**Solution**:
1. Enable mixed precision: `mixed_precision: true`
2. Increase batch size: `batch_size: 64`
3. Reduce input size: `input_size: 224` (from 384)
4. Use fewer workers: `num_workers: 2`

---

## Next Steps

### After Training

1. **Evaluate on Test Set**
   ```bash
   python test.py --model checkpoints/vit/best_model.pt
   ```

2. **Generate Predictions**
   ```python
   import torch
   from train import create_vit_model
   from src.common.config import FullConfig
   
   config = FullConfig.from_yaml("config.yaml")
   model = create_vit_model(config, torch.device("cuda"))
   model.load_state_dict(torch.load("checkpoints/vit/best_model.pt")['model_state_dict'])
   ```

3. **Visualize Results**
   - Plot training history from `logs/vit/training_history.json`
   - Create confusion matrix on test set
   - Analyze misclassified examples

---

## Performance Expectations

| Batch Size | Epochs | Time/Epoch | Final Accuracy |
|-----------|--------|-----------|-----------------|
| 64        | 100    | ~2 min    | ~88%            |
| 32        | 150    | ~4 min    | ~92%            |
| 16        | 200    | ~8 min    | ~94%            |

*Times approximate for single GPU (RTX 3090)*

---

## ViT Specifics

Why Vision Transformer for this task?

✅ **Advantages:**
- Better global context understanding (good for disease patterns)
- Pre-trained on huge ImageNet dataset
- Flexible input sizes (224, 384, 512, etc.)

⚠️ **Challenges:**
- Higher memory requirements than CNNs
- Slower training/inference than ResNet
- Needs data augmentation for good generalization

**Optimization Strategy:**
- Start with pretrained weights (`pretrained: true`)
- Use AdamW optimizer (better convergence)
- Cosine annealing scheduler (smooth LR decay)
- Early stopping patience of 20-30 epochs

---

## Advanced Configuration

### Multi-GPU Training
```yaml
device:
  device: cuda
  multi_gpu: true
  gpu_ids: [0, 1, 2]  # Use GPUs 0, 1, 2
```

### Resuming Training
```python
# Load checkpoint
checkpoint = torch.load("checkpoints/vit/checkpoint_epoch_50.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Custom Learning Rate Schedule
```yaml
training:
  scheduler: step
  scheduler_step_size: 30    # Decay every 30 epochs
  scheduler_gamma: 0.1       # Multiply LR by 0.1
```

---

## References

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [TIMM Documentation](https://github.com/rwightman/pytorch-image-models)

---

For more examples, see [config_examples.py](../../common/config_examples.py)
