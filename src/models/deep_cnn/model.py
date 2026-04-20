from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DeepCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class PlantDiseaseDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: list[int], transform: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image), self.labels[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_label(raw_name: str) -> str:
    _, condition = raw_name.split("___", maxsplit=1)
    return condition


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


def parse_split_file(split_file: Path, dataset_root: Path | None = None) -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    labels: list[str] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.split("\t")
            image_paths.append(resolve_image_path(split_file, rel_path, dataset_root=dataset_root))
            labels.append(normalize_label(label))
    return image_paths, labels


def verify_paths_exist(paths: list[Path], split_name: str) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file referenced by {split_name} split:\n{path}\n"
                "Update the split paths or place the PlantVillage dataset where the split files expect it."
            )


def filter_missing_samples(
    image_paths: list[Path],
    labels: list[str],
    split_name: str,
) -> tuple[list[Path], list[str]]:
    kept_paths: list[Path] = []
    kept_labels: list[str] = []
    missing_paths: list[Path] = []

    for image_path, label in zip(image_paths, labels):
        if image_path.exists():
            kept_paths.append(image_path)
            kept_labels.append(label)
        else:
            missing_paths.append(image_path)

    if missing_paths:
        print(f"Warning: skipped {len(missing_paths)} missing files from {split_name} split.")
        print(f"First missing file: {missing_paths[0]}")

    if not kept_paths:
        raise FileNotFoundError(
            f"All files referenced by the {split_name} split are missing.\n"
            "Check dataset_root or update the split file paths."
        )

    return kept_paths, kept_labels


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


def create_dataloaders(config: dict, device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    dataset_root = Path(config["dataset_root"]).resolve() if config.get("dataset_root") else None
    train_split = Path(config["train_split"])
    val_split = Path(config["val_split"])
    test_split = Path(config["test_split"])

    train_paths, train_labels = parse_split_file(train_split, dataset_root=dataset_root)
    val_paths, val_labels = parse_split_file(val_split, dataset_root=dataset_root)
    test_paths, test_labels = parse_split_file(test_split, dataset_root=dataset_root)

    train_paths, train_labels = filter_missing_samples(train_paths, train_labels, "train")
    val_paths, val_labels = filter_missing_samples(val_paths, val_labels, "val")
    test_paths, test_labels = filter_missing_samples(test_paths, test_labels, "test")

    class_names = sorted(set(train_labels))
    label_to_index = {label: idx for idx, label in enumerate(class_names)}

    train_targets = [label_to_index[label] for label in train_labels]
    val_targets = [label_to_index[label] for label in val_labels]
    test_targets = [label_to_index[label] for label in test_labels]

    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    print(f"Classes ({len(class_names)}):")
    for class_name, count in Counter(train_labels).most_common():
        print(f"  {class_name}: {count}")

    train_transform, eval_transform = create_transforms(config["image_height"], config["image_width"])
    train_dataset = PlantDiseaseDataset(train_paths, train_targets, train_transform)
    val_dataset = PlantDiseaseDataset(val_paths, val_targets, eval_transform)
    test_dataset = PlantDiseaseDataset(test_paths, test_targets, eval_transform)

    num_workers = config.get("num_workers", 0)
    batch_size = config.get("batch_size", 32)
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
    return train_loader, val_loader, test_loader, class_names


def train_one_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    print("\n" + "=" * 60)
    print(f"Running model with {optimizer_name}")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss / train_total:.4f} | "
            f"Train Acc: {train_correct / train_total:.4f} | "
            f"Val Loss: {val_loss / val_total:.4f} | "
            f"Val Acc: {val_correct / val_total:.4f}"
        )

    return model


def evaluate_test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline DeepCNN training loop on split manifest files.")
    parser.add_argument("--config", type=Path, default=Path("src/models/deep_cnn/config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_root is not None:
        config["dataset_root"] = str(args.dataset_root)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = create_dataloaders(config, device)
    epochs = config.get("epochs", 15)
    learning_rate = config.get("learning_rate", 0.001)

    model = DeepCNN(num_classes=len(class_names)).to(device)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_one_model(model, optimizer, "SGD", train_loader, val_loader, device, epochs)
    test_acc = evaluate_test(model, test_loader, device)
    print(f"SGD Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
