# DeepLearning_CropDiseaseDetection

Plant disease classification experiments on the PlantVillage dataset.

Dataset:
https://github.com/spMohanty/PlantVillage-Dataset

Reference:

@article{Mohanty_Hughes_Salathé_2016,
    title   = {Using deep learning for image-based plant disease detection},
    volume  = {7},
    DOI     = {10.3389/fpls.2016.01419},
    journal = {Frontiers in Plant Science},
    author  = {Mohanty, Sharada P. and Hughes, David P. and Salathé, Marcel},
    year    = {2016},
    month   = {Sep}
}

## Setup

1. Clone this repository.
2. Place the PlantVillage dataset next to this repository so the default shared loader path works:

```text
adl/
|- DeepLearning_CropDiseaseDetection/
|- PlantVillage-Dataset/
   |- raw/
      |- color/
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install `timm` if you want to run the ViT model:

```bash
pip install timm
```

Run commands from the repository root:

```bash
cd DeepLearning_CropDiseaseDetection
```

## Shared Data Loading

The classic CNN and transfer-learning scripts below use the shared loader in `src/common/`.

- Default dataset root: `../PlantVillage-Dataset/raw/color`
- Data split: random stratified `80/10/10`
- Split is rebuilt each run using the configured seed
- Default checkpoint outputs go under `checkpoints/<model_name>/`

If you want different hyperparameters, edit that model folder's `config.yaml`.

## Runnable Models

These model folders are wired to the current shared `common/` code:

- `src/models/deep_cnn`
- `src/models/efficientnet_b0`
- `src/models/mobilenetv2`
- `src/models/resnet50`
- `src/models/simple_cnn`
- `src/models/attention_cnn`
- `src/models/vit`

## Train And Test

### Deep CNN

```bash
python -m src.models.deep_cnn.train
python -m src.models.deep_cnn.test --checkpoint checkpoints/deep_cnn/best_model.pth
```

### EfficientNet-B0

```bash
python -m src.models.efficientnet_b0.train
python -m src.models.efficientnet_b0.test --checkpoint checkpoints/efficientnet_b0/best_model.pth
```

### MobileNetV2

```bash
python -m src.models.mobilenetv2.train
python -m src.models.mobilenetv2.test --checkpoint checkpoints/mobilenetv2/best_model.pth
```

### ResNet50

```bash
python -m src.models.resnet50.train
python -m src.models.resnet50.test --checkpoint checkpoints/resnet50/best_model.pth
```

### Simple CNN

```bash
python -m src.models.simple_cnn.train
python -m src.models.simple_cnn.test --checkpoint checkpoints/simple_cnn/best_model.pth
```

### Attention CNN

```bash
python src/models/attention_cnn/train.py src/models/attention_cnn/config.yaml
python src/models/attention_cnn/test.py --config src/models/attention_cnn/config.yaml --checkpoint best_model.pt
```

### Vision Transformer

```bash
python src/models/vit/train.py
python src/models/vit/test.py --config src/models/vit/config.yaml --checkpoint best_model.pt
```

## Outputs

Training scripts typically write:

- model checkpoints such as `best_model.pt` or `best_model.pth`
- `history.json`
- `training_curves.png`
- test-time metrics and confusion matrix images for the matching `test.py`

## Notes

- `attention_cnn` and `vit` use the newer config system in `src/common/config.py`.
- `efficientnet_b0`, `mobilenetv2`, and `resnet50` use pretrained torchvision weights by default, so the first run may download model weights.
- `deep_cnn` and `simple_cnn` train from scratch.
