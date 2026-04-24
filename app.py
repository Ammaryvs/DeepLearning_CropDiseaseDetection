import json
import numpy as np
from io import BytesIO
from pathlib import Path

import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

BUNDLE_DIR = Path(__file__).parent / "Inference Bundle"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

with open(BUNDLE_DIR / "class_labels.json") as f:
    _labels = json.load(f)

IDX_TO_LABEL: list[str] = _labels["idx_to_label"]

_session = None


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(BUNDLE_DIR / "model.onnx"))
    return _session


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    x = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)   # HWC -> CHW
    return x[np.newaxis]        # add batch dim


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def parse_label(raw: str) -> dict:
    if "___" in raw:
        plant, disease = raw.split("___", 1)
    else:
        plant, disease = raw, ""
    plant   = plant.replace("_", " ")
    disease = disease.replace("_", " ")
    return {"plant": plant, "disease": disease, "healthy": "healthy" in disease.lower()}


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
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot read image"}), 400

    x      = preprocess(img)
    logits = get_session().run(None, {"input": x})[0][0]
    probs  = softmax(logits)
    top5   = probs.argsort()[::-1][:5]

    results = []
    for idx in top5:
        raw    = IDX_TO_LABEL[idx]
        parsed = parse_label(raw)
        results.append({
            "label":      raw,
            "plant":      parsed["plant"],
            "disease":    parsed["disease"],
            "healthy":    parsed["healthy"],
            "confidence": round(float(probs[idx]) * 100, 2),
        })

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(debug=True, port=8888)
