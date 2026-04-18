# Plant Disease Classifier - Inference Bundle

Trained on the PlantVillage dataset (color images) using MobileNetV2
with transfer learning.

## Files

- **best_model.pth** - PyTorch state_dict for the trained MobileNetV2 classifier.
  The final classifier head is:
  `nn.Sequential(nn.Dropout(p=0.1), nn.Linear(1280, 38))`.
  Load with `model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))`.

- **class_labels.json** - Mapping between class indices and human-readable class names.
  - `idx_to_label`: list where `idx_to_label[i]` is the class name for output index `i`.
  - `label_to_idx`: dict mapping class name to index.
  - `num_classes`: 38.

- **README.md** - This file.

## Model architecture

```python
from torchvision.models import mobilenet_v2
import torch.nn as nn

model = mobilenet_v2(weights=None)
in_features = model.classifier[-1].in_features   # 1280
model.classifier = nn.Sequential(
    nn.Dropout(p=0.1),
    nn.Linear(in_features, 38),
)
```

## Inference preprocessing

Input images MUST be preprocessed exactly like validation data during training:

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

- Image size: **224 x 224**
- Color mode: **RGB**
- Normalization: **ImageNet mean/std** (above)

## Minimal inference example

```python
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2

# --- Load labels ---
with open("class_labels.json") as f:
    labels = json.load(f)
idx_to_label = labels["idx_to_label"]
num_classes = labels["num_classes"]

# --- Build model and load weights ---
model = mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.1),
    nn.Linear(model.classifier[-1].in_features, num_classes),
)
state = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

# --- Preprocess ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open("your_leaf.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)  # shape: (1, 3, 224, 224)

# --- Predict ---
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    top5 = torch.topk(probs, k=5)

for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
    print(f"{idx_to_label[idx]}: {score*100:.2f}%")
```

## Training config (for reference)

- Base model: MobileNetV2 (ImageNet-pretrained)
- Learning rate: 0.0001
- Dropout: 0.1
- Optimizer: Adam, weight_decay=1e-4
- LR scheduler: StepLR (step_size=7, gamma=0.1)
- Epochs: 18
- Batch size: 32
- Loss: CrossEntropyLoss
- Input: RGB, 224x224, ImageNet normalization
- Train / Val / Test split: 80 / 10 / 10 (seed=42)
