# DeepLearning_CropDiseaseDetection

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
2. Clone the PlantVillage dataset next to this repository so the default paths work:

```text
adl/
|- DeepLearning_CropDiseaseDetection/
|- PlantVillage-Dataset/
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install `timm` as well if you want to run the Vision Transformer model:

```bash
pip install timm
```

## How To Run The Models

Run all commands from the repository root:

```bash
cd DeepLearning_CropDiseaseDetection
```

Before training, update the relevant `config.yaml` file if you want to change:

- dataset path
- batch size
- number of epochs
- optimizer or scheduler
- checkpoint and log directories

The currently implemented model pipelines are:

- `attention_cnn`
- `vit`

The other folders under `src/models/` currently contain config placeholders only and do not yet have runnable `train.py` / `test.py` implementations.

## Train Attention CNN

```bash
python src/models/attention_cnn/train.py src/models/attention_cnn/config.yaml
```

Optional examples:

```bash
python src/models/attention_cnn/train.py src/models/attention_cnn/config.yaml --epochs 10
python src/models/attention_cnn/train.py src/models/attention_cnn/config.yaml --resume-from checkpoints/attention_cnn/best_model.pt
```

## Test Attention CNN

```bash
python src/models/attention_cnn/test.py --config src/models/attention_cnn/config.yaml --checkpoint best_model.pt
```

If `best_model.pt` is not in the current folder, the script will look inside the checkpoint directory defined in the config, which defaults to:

```text
checkpoints/attention_cnn/
```

## Train Vision Transformer

The ViT training script uses parameters from config.yaml, so just adjust the parameters from there.

```bash
python src/models/vit/train.py
```

## Test Vision Transformer

```bash
python src/models/vit/test.py --config src/models/vit/config.yaml --checkpoint best_model.pt
```

If `best_model.pt` is not in the current folder, the script will look inside the checkpoint directory defined in the config, which defaults to:

```text
checkpoints/vit/
```

## Training Outputs

By default, training writes artifacts to:

- `checkpoints/attention_cnn/` or `checkpoints/vit/`
- `logs/attention_cnn/` or `logs/vit/`

The best checkpoint is usually saved as:

```text
best_model.pt
```
