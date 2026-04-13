from __future__ import annotations

import io
import json
from pathlib import Path

import modal
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.models.deep_cnn.model import DeepCNN


APP_NAME = "crop-disease-deep-cnn"
DATASET_MOUNT = Path("/vol/data")
ARTIFACT_MOUNT = Path("/vol/artifacts")
MODEL_DIR = ARTIFACT_MOUNT / "deep_cnn"
LABELS_PATH = MODEL_DIR / "labels.json"
WEIGHTS_PATH = MODEL_DIR / "best_deep_cnn.pt"
CONFIG_PATH = Path("/root/config/deep_cnn_config.yaml")
DATA_VOLUME_NAME = "crop-disease-dataset"
ARTIFACT_VOLUME_NAME = "crop-disease-artifacts"

dataset_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "fastapi[standard]==0.115.0",
        "pillow==10.4.0",
        "numpy==2.1.1",
        "pyyaml==6.0.3",
    )
    .add_local_python_source("src")
    .add_local_file("src/models/deep_cnn/config.yaml", remote_path="/root/config/deep_cnn_config.yaml")
)

app = modal.App(APP_NAME, image=image)


def _preprocess_image(image: Image.Image) -> torch.Tensor:
    resized = image.resize((128, 128))
    array = np.asarray(resized, dtype="float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


@app.cls(
    cpu=2.0,
    memory=4096,
    scaledown_window=300,
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
class DeepCNNPredictor:
    @modal.enter()
    def load(self) -> None:
        artifact_volume.reload()
        if not LABELS_PATH.exists() or not WEIGHTS_PATH.exists():
            raise RuntimeError(
                "Missing deep CNN artifacts in the Modal artifact volume. "
                "Run the remote training command first."
            )

        with LABELS_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.class_names = metadata["class_names"]
        self.display_names = metadata["display_names"]
        self.model = DeepCNN(num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        self.model.eval()

    @modal.method()
    def predict_bytes(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _preprocess_image(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_values, top_indices = torch.topk(probabilities, k=min(5, len(self.class_names)))
        best_index = int(top_indices[0].item())

        return {
            "predicted_class": self.class_names[best_index],
            "display_label": self.display_names[best_index],
            "confidence": float(top_values[0].item()),
            "top_predictions": [
                {
                    "class_name": self.class_names[int(index)],
                    "display_label": self.display_names[int(index)],
                    "confidence": float(value),
                }
                for value, index in zip(top_values.tolist(), top_indices.tolist())
            ],
        }


@app.function(
    image=image,
    gpu="T4",
    cpu=4.0,
    memory=32768,
    timeout=60 * 60 * 24,
    volumes={DATASET_MOUNT: dataset_volume, ARTIFACT_MOUNT: artifact_volume},
)
def train_remote(config_overrides: dict | None = None) -> dict:
    from src.models.deep_cnn.train import run_training
    import yaml

    artifact_volume.reload()
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["train_split"] = str(DATASET_MOUNT / "splits" / "train.txt")
    config["val_split"] = str(DATASET_MOUNT / "splits" / "val.txt")
    config["test_split"] = str(DATASET_MOUNT / "splits" / "test.txt")
    config["dataset_root"] = str(DATASET_MOUNT / "dataset" / "raw" / "color")
    config["output_dir"] = str(MODEL_DIR)

    if config_overrides:
        config.update(config_overrides)

    print("Starting remote training on Modal...", flush=True)
    run_training(config)
    artifact_volume.commit()

    with LABELS_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return {
        "status": "ok",
        "artifact_dir": str(MODEL_DIR),
        "num_classes": metadata["num_classes"],
        "class_names": metadata["class_names"],
    }


@app.function(cpu=1.0, memory=2048, image=image, volumes={ARTIFACT_MOUNT: artifact_volume})
@modal.asgi_app(label="crop-disease-api")
def web_app():
    api = FastAPI(title="Crop Disease Deep CNN")
    predictor = DeepCNNPredictor()

    @api.get("/health")
    async def health() -> dict:
        artifact_volume.reload()
        return {
            "status": "ok",
            "model_loaded": LABELS_PATH.exists() and WEIGHTS_PATH.exists(),
            "num_classes": len(json.loads(LABELS_PATH.read_text(encoding="utf-8"))["class_names"]) if LABELS_PATH.exists() else 0,
        }

    @api.get("/classes")
    async def classes() -> dict:
        artifact_volume.reload()
        if not LABELS_PATH.exists():
            raise HTTPException(status_code=500, detail="labels.json not found in deployed artifacts")
        metadata = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        return {
            "num_classes": metadata["num_classes"],
            "class_names": metadata["class_names"],
            "display_names": metadata["display_names"],
        }

    @api.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty upload")
        try:
            return predictor.predict_bytes.remote(image_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return api
