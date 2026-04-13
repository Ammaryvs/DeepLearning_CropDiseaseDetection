from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.deep_cnn.model import DeepCNN


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    print(f"Loading config from: {config_path}", flush=True)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_label(raw_name: str) -> str:
    _, condition = raw_name.split("___", maxsplit=1)
    return condition


def display_class_name(raw_name: str) -> str:
    return raw_name.replace("_", " ")


def resolve_image_path(split_file: Path, rel_path: str, dataset_root: Path | None = None) -> Path:
    raw_path = Path(rel_path)
    if dataset_root is not None:
        raw_parts = list(raw_path.parts)
        if "color" in raw_parts:
            color_index = raw_parts.index("color")
            candidate = (dataset_root / Path(*raw_parts[color_index + 1 :])).resolve()
            if candidate.exists():
                return candidate

    candidates = [
        (split_file.parent / raw_path).resolve(),
        (split_file.parent.parent / raw_path).resolve(),
        (PROJECT_ROOT / raw_path).resolve(),
        (PROJECT_ROOT.parent / raw_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_split_file(split_file: Path, dataset_root: Path | None = None) -> tuple[list[str], list[str]]:
    print(f"Reading split file: {split_file}", flush=True)
    image_paths: list[str] = []
    labels: list[str] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.split("\t")
            image_paths.append(str(resolve_image_path(split_file, rel_path, dataset_root=dataset_root)))
            labels.append(normalize_label(label))
    return image_paths, labels


def verify_paths_exist(paths: list[str], split_name: str) -> None:
    print(f"Checking {split_name} paths...", flush=True)
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing file referenced by {split_name}:\n{path}\n"
                "Update the split paths or place the PlantVillage dataset where the split files expect it."
            )


class PlantDiseaseDataset(Dataset):
    def __init__(self, image_paths: list[str], labels: list[int], transform: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image), self.labels[index]


def create_transforms(image_height: int, image_width: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_height, image_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(
    config: dict,
    class_names: list[str],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_root = Path(config["dataset_root"]).resolve() if config.get("dataset_root") else None
    train_paths, train_labels = parse_split_file(Path(config["train_split"]), dataset_root=dataset_root)
    val_paths, val_labels = parse_split_file(Path(config["val_split"]), dataset_root=dataset_root)
    test_paths, test_labels = parse_split_file(Path(config["test_split"]), dataset_root=dataset_root)

    verify_paths_exist(train_paths, "train")
    verify_paths_exist(val_paths, "val")
    verify_paths_exist(test_paths, "test")

    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    train_targets = [label_to_index[label] for label in train_labels]
    val_targets = [label_to_index[label] for label in val_labels]
    test_targets = [label_to_index[label] for label in test_labels]

    print(f"Train samples: {len(train_paths)}", flush=True)
    print(f"Validation samples: {len(val_paths)}", flush=True)
    print(f"Test samples: {len(test_paths)}", flush=True)
    print(f"Classes ({len(class_names)}):", flush=True)
    for class_name, count in Counter(train_labels).most_common():
        print(f"  {class_name}: {count}", flush=True)

    train_transform, eval_transform = create_transforms(config["image_height"], config["image_width"])
    train_dataset = PlantDiseaseDataset(train_paths, train_targets, train_transform)
    val_dataset = PlantDiseaseDataset(val_paths, val_targets, eval_transform)
    test_dataset = PlantDiseaseDataset(test_paths, test_targets, eval_transform)

    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += images.size(0)
    return total_loss / total_seen, total_correct / total_seen


def save_metadata(output_dir: Path, class_names: list[str], history: list[dict[str, float]]) -> None:
    metadata = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "display_names": [display_class_name(name) for name in class_names],
        "history": history,
    }
    with (output_dir / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def run_training(config: dict) -> None:
    print("Starting deep CNN training pipeline...", flush=True)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = Path(config["dataset_root"]).resolve() if config.get("dataset_root") else None
    train_paths, train_labels = parse_split_file(Path(config["train_split"]), dataset_root=dataset_root)
    class_names = sorted(set(train_labels))
    train_loader, val_loader, test_loader = create_dataloaders(config, class_names, device)

    print(f"Using device: {device}", flush=True)
    model = DeepCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_deep_cnn.pt"
    final_path = output_dir / "final_deep_cnn.pt"

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    patience = 0

    print("Starting training...", flush=True)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_seen += images.size(0)

        train_loss = running_loss / running_seen
        train_acc = running_correct / running_seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                print("Early stopping triggered.", flush=True)
                break

    print("Evaluating best checkpoint on test set...", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}", flush=True)
    print(f"Test accuracy: {test_acc:.4f}", flush=True)

    torch.save(model.state_dict(), final_path)
    save_metadata(output_dir, class_names, history)
    print(f"Saved artifacts to: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deep CNN on disease-only PlantVillage labels.")
    parser.add_argument("--config", type=Path, default=Path("src/models/deep_cnn/config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_root is not None:
        config["dataset_root"] = str(args.dataset_root)
    run_training(config)


if __name__ == "__main__":
    main()
