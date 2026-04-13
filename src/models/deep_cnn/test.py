from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.deep_cnn.model import DeepCNN


def load_metadata(labels_path: Path) -> dict:
    with labels_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_transform(image_height: int, image_width: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the trained deep CNN.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/deep_cnn/best_deep_cnn.pt"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/deep_cnn/labels.json"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=128)
    args = parser.parse_args()

    metadata = load_metadata(args.labels)
    class_names = metadata["class_names"]
    display_names = metadata["display_names"]

    model = DeepCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = build_transform(args.image_height, args.image_width)(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    best_index = int(probabilities.argmax().item())
    print(f"Predicted class: {class_names[best_index]}")
    print(f"Display label: {display_names[best_index]}")
    print(f"Confidence: {probabilities[best_index].item():.4f}")

    top_values, top_indices = torch.topk(probabilities, k=min(5, len(class_names)))
    print("Top predictions:")
    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        print(f"  {display_names[index]}: {value:.4f}")


if __name__ == "__main__":
    main()
