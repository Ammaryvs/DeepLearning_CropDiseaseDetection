from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))


def load_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the trained simple CNN.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/simple_cnn/best_simple_cnn.keras"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/simple_cnn/labels.json"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=128)
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    with args.labels.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    class_names = metadata["class_names"]
    display_names = metadata["display_names"]

    batch = load_image(args.image, (args.image_width, args.image_height))
    predictions = model.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(predictions))

    print(f"Predicted class: {class_names[best_index]}")
    print(f"Display label: {display_names[best_index]}")
    print(f"Confidence: {predictions[best_index]:.4f}")

    top_indices = np.argsort(predictions)[::-1][:5]
    print("Top 5 predictions:")
    for index in top_indices:
        print(f"  {display_names[int(index)]}: {predictions[int(index)]:.4f}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    main()
