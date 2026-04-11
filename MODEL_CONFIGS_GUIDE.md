# Model Configuration Comparison Guide

Quick reference for all model configurations optimized for crop disease detection.

## Models Overview

| Model | Type | Speed | Accuracy | Memory | Best For |
|-------|------|-------|----------|--------|----------|
| **SimpleCNN** | Custom CNN | ⚡⚡⚡ Fast | ⭐⭐ Low | 🟢 Very Low | Baseline, quick testing |
| **DeepCNN** | Custom CNN | ⚡⚡ Medium | ⭐⭐⭐ Medium | 🟢 Low | Faster than pre-trained |
| **AttentionCNN** | Custom CNN + Attention | ⚡⚡ Medium | ⭐⭐⭐ Medium | 🟡 Medium | Feature focus, better accuracy |
| **MobileNetV2** | Pre-trained CNN | ⚡⚡⚡ Fast | ⭐⭐⭐ Medium | 🟢 Very Low | Mobile, edge deployment |
| **EfficientNet-B0** | Pre-trained CNN | ⚡⚡ Medium | ⭐⭐⭐⭐ Good | 🟡 Medium | Best efficiency |
| **ResNet50** | Pre-trained CNN | ⚡⚡ Medium | ⭐⭐⭐⭐ Good | 🟡 Medium | Industry standard |
| **VGG16** | Pre-trained CNN | ⚡ Slow | ⭐⭐⭐⭐ Good | 🔴 High | Classic, powerful |
| **ViT** | Pre-trained Transformer | ⚡ Slow | ⭐⭐⭐⭐⭐ Best | 🔴 Very High | State-of-the-art |

---

## Detailed Configuration Comparison

### Training Parameters

```
Model              | Epochs | Batch | LR     | Optimizer | Scheduler | Early-Stop
-------------------+--------+-------+--------+-----------+-----------+----------
SimpleCNN          | 50     | 32    | 0.001  | Adam      | Step      | 10
DeepCNN            | 80     | 32    | 0.001  | Adam      | Cosine    | 12
AttentionCNN       | 90     | 32    | 0.001  | AdamW     | Cosine    | 15
MobileNetV2        | 80     | 64    | 0.001  | Adam      | Cosine    | 12
EfficientNet-B0    | 100    | 48    | 0.001  | Adam      | Cosine    | 15
ResNet50           | 100    | 32    | 0.001  | Adam      | Cosine    | 15
VGG16              | 100    | 32    | 0.0001 | SGD       | Step      | 15
ViT                | 150    | 16    | 0.0001 | AdamW     | Cosine    | 20
```

### Key Configuration Differences

#### Lightweight Models (Fast Training)
**SimpleCNN, MobileNetV2**
- Fewer epochs (50-80)
- Larger batch sizes (32-64)
- Standard learning rate (0.001)
- Less augmentation

#### Medium Models (Balanced)
**DeepCNN, AttentionCNN, EfficientNet-B0, ResNet50**
- More epochs (80-100)
- Medium batch size (32-48)
- Standard learning rate (0.001)
- Moderate augmentation (0.4-0.5)

#### Heavy Models (Best Quality)
**VGG16, ViT**
- More epochs (100-150)
- Smaller batch size (16-32) for stability
- Lower learning rate (0.0001)
- Stronger augmentation

---

## Quick Decision Guide

### Choose SimpleCNN if...
✅ Testing pipeline quickly  
✅ Limited GPU memory  
✅ Want baseline comparison  
✅ Training time < 5 minutes  

**Typical Accuracy**: 75-80%

### Choose MobileNetV2 if...
✅ Need fast inference (mobile/edge)  
✅ Want pretrained backbone  
✅ Limited memory  
✅ Deployment priority  

**Typical Accuracy**: 85-88%

### Choose EfficientNet-B0 if...
✅ Need good balance of speed/accuracy  
✅ Fast training with decent accuracy  
✅ Good for production  
✅ Single GPU training  

**Typical Accuracy**: 90-92%

### Choose ResNet50 if...
✅ Industry-standard model  
✅ Proven performance  
✅ Lots of documentation  
✅ Don't know what to choose  

**Typical Accuracy**: 90-93%

### Choose VGG16 if...
✅ Want powerful model  
✅ Have GPU memory available  
✅ Accuracy is priority  
✅ Training time is not critical  

**Typical Accuracy**: 91-94%

### Choose ViT if...
✅ Want state-of-the-art results  
✅ Have excellent GPU (RTX 3090+)  
✅ Willing to wait for training  
✅ Maximum accuracy needed  

**Typical Accuracy**: 94-96%

---

## Training Time Estimates

**On single GPU (RTX 3090):**

| Model | Time per Epoch | Total Time (est.) | Total Epochs |
|-------|--------|----------|---------|
| SimpleCNN | ~1 min | ~50 min | 50 |
| DeepCNN | ~2 min | ~160 min | 80 |
| AttentionCNN | ~2.5 min | ~225 min | 90 |
| MobileNetV2 | ~1.5 min | ~120 min | 80 |
| EfficientNet-B0 | ~2 min | ~200 min | 100 |
| ResNet50 | ~2 min | ~200 min | 100 |
| VGG16 | ~3 min | ~300 min | 100 |
| ViT | ~6 min | ~900 min | 150 |

---

## File Locations

```
src/models/
├── simple_cnn/config.yaml      ← Fast baseline training
├── deep_cnn/config.yaml        ← Custom deep model
├── attention_cnn/config.yaml   ← CNN with attention
├── mobilenetv2/config.yaml     ← Lightweight, mobile-friendly
├── efficientnet_b0/config.yaml ← Efficient & powerful
├── resnet50/config.yaml        ← Industry standard
├── vgg16/config.yaml           ← Classic, powerful CNN
└── vit/config.yaml             ← State-of-the-art transformer
```

---

## How to Train Each Model

All models use the same training script pattern. Just navigate to the model directory and run:

```bash
# Example for ResNet50
cd src/models/resnet50
python train.py

# Or for ViT
cd src/models/vit
python train.py

# Or for MobileNetV2
cd src/models/mobilenetv2
python train.py
```

---

## Modifying Configurations

### Common Tweaks

**Out of GPU Memory?**
```yaml
training:
  batch_size: 16  # Reduce from 32
device:
  mixed_precision: true  # Enable if not already
```

**Want Faster Training?**
```yaml
training:
  epochs: 50  # Reduce from 100
data:
  num_workers: 8  # Increase from 4
  augmentation_strength: 0.3  # Reduce from 0.5
```

**Want Better Accuracy?**
```yaml
training:
  epochs: 150  # Increase
  learning_rate: 0.00005  # Decrease (be more careful)
data:
  augmentation_strength: 0.7  # Increase
```

**Want Different Learning Rate Schedule?**
```yaml
# Option 1: Step decay (more aggressive)
training:
  scheduler: step
  scheduler_step_size: 10
  scheduler_gamma: 0.1

# Option 2: Exponential decay (smooth)
training:
  scheduler: exponential
  scheduler_gamma: 0.95

# Option 3: Cosine annealing (smooth to minimum)
training:
  scheduler: cosine
```

---

## Performance Expectations

### Accuracy by Model (on PlantVillage dataset)

```
SimpleCNN           ████████░░ 78-82%
DeepCNN             ███████░░░ 82-86%
AttentionCNN        ███████░░░ 84-88%
MobileNetV2         ████████░░ 85-88%
EfficientNet-B0     █████████░ 90-92%
ResNet50            █████████░ 90-93%
VGG16               █████████░ 91-94%
ViT                 ██████████ 94-96%
```

---

## Recommended Training Pipeline

### 1. Start with SimpleCNN (Quick test)
```bash
cd src/models/simple_cnn
python train.py  # ~50 minutes
```

### 2. Try ResNet50 (Balanced)
```bash
cd src/models/resnet50
python train.py  # ~200 minutes
```

### 3. Compare with EfficientNet
```bash
cd src/models/efficientnet_b0
python train.py  # ~200 minutes
```

### 4. Go for excellence with ViT
```bash
cd src/models/vit
python train.py  # ~900 minutes (15 hours)
```

---

## Model Architecture Notes

### Custom Models (SimpleCNN, DeepCNN, AttentionCNN)
- **Pros**: Fast training, low memory, no pretrained weights required
- **Cons**: Need to train from scratch, may need more data
- **Best for**: Learning, quick experiments, custom architectures

### Pre-trained CNNs (MobileNetV2, EfficientNet, ResNet, VGG)
- **Pros**: Fast convergence, good accuracy, transfer learning
- **Cons**: Larger models, more GPU memory
- **Best for**: Production, when accuracy matters, limited data

### Vision Transformer (ViT)
- **Pros**: State-of-the-art accuracy, handles global context well
- **Cons**: Slow, high memory, needs patience for training
- **Best for**: Final production, when accuracy is critical

---

## Experiment Tracking

Create a results folder to track experiments:

```bash
mkdir experiments
```

After each training, save the history:

```bash
# Copy training history
cp logs/model_name/training_history.json experiments/model_name_results.json
cp checkpoints/model_name/best_model.pt experiments/model_name_best.pt
```

Then compare:
```json
{
  "model": "resnet50",
  "final_val_acc": 0.923,
  "best_epoch": 78,
  "training_time": "195 min",
  "hyperparams": { ... }
}
```

---

## Advanced: Custom Combinations

### Fast + Good Accuracy
```yaml
# Use EfficientNet with:
training:
  epochs: 100
  batch_size: 48
  learning_rate: 0.001
  scheduler: cosine
device:
  mixed_precision: true
```

### Best Accuracy (No Time Limit)
```yaml
# Use ViT with:
training:
  epochs: 200
  batch_size: 16
  learning_rate: 0.00005
  early_stopping_patience: 30
  scheduler: cosine
data:
  augmentation_strength: 0.7
```

### Production Ready
```yaml
# Use ResNet50 with:
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  scheduler: cosine
  early_stopping_patience: 15
device:
  mixed_precision: true
```

---

## Next Steps

1. **Choose a model** based on your requirements
2. **Review the config** in `src/models/model_name/config.yaml`
3. **Run training**: `python train.py`
4. **Monitor progress**: Check `logs/model_name/`
5. **Evaluate results**: Load `checkpoints/model_name/best_model.pt`

---

## FAQ

**Q: Which model should I choose?**
A: Start with ResNet50. It's reliable and well-documented. Then try EfficientNet or ViT if needed.

**Q: Can I run multiple models at once?**
A: Yes, if you have memory. Open multiple terminals and run different models in different GPU IDs.

**Q: How do I use a trained model?**
A: Load it using PyTorch: `torch.load("checkpoints/model_name/best_model.pt")`

**Q: Can I combine models (ensemble)?**
A: Yes! Train multiple models and average their predictions for better accuracy.

---

For detailed training guide, see [VIT_TRAINING_GUIDE.md](../VIT_TRAINING_GUIDE.md) (mostly applies to all models).
