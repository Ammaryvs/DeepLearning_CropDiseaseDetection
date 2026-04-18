import json
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

# Import the same wrapper class the checkpoint was saved from
sys.path.insert(0, str(Path(__file__).parent))
from src.models.mobilenetv2.model import MobileNetV2Classifier

app = Flask(__name__)

BUNDLE_DIR = Path(__file__).parent / "Inference Bundle"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

with open(BUNDLE_DIR / "class_labels.json") as f:
    _labels = json.load(f)

IDX_TO_LABEL: list[str] = _labels["idx_to_label"]
NUM_CLASSES: int = _labels["num_classes"]

model = MobileNetV2Classifier(num_classes=NUM_CLASSES, pretrained=False, dropout=0.1)
state = torch.load(BUNDLE_DIR / "best_model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_label(raw: str) -> dict:
    """Split 'Plant___Disease' into human-readable parts."""
    if "___" in raw:
        plant, disease = raw.split("___", 1)
    else:
        plant, disease = raw, ""
    plant = plant.replace("_", " ").replace(",", ",")
    disease = disease.replace("_", " ")
    is_healthy = "healthy" in disease.lower()
    return {"plant": plant, "disease": disease, "healthy": is_healthy}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    data = file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Image too large (max 10 MB)"}), 413

    try:
        from io import BytesIO
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot read image"}), 400

    x = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top5 = torch.topk(probs, k=5)

    results = []
    for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        raw = IDX_TO_LABEL[idx]
        parsed = parse_label(raw)
        results.append({
            "label": raw,
            "plant": parsed["plant"],
            "disease": parsed["disease"],
            "healthy": parsed["healthy"],
            "confidence": round(score * 100, 2),
        })

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(debug=True, port=8888)
