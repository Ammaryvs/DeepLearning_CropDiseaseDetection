import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from model import create_mobilenetv2
from src.common.dataloader import create_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def get_dataset_loaders(loaders, dataset_name):
    if dataset_name == "plantvillage":
        return loaders["plantvillage"]["train"], loaders["plantvillage"]["val"]
    if dataset_name == "plantdoc":
        plantdoc_loaders = loaders["plantdoc"]
        if plantdoc_loaders is None:
            raise ValueError("PlantDoc dataset is empty or incomplete.")
        return plantdoc_loaders["train"], plantdoc_loaders["val"]
    if dataset_name == "combined":
        return loaders["combined"]["train"], loaders["combined"]["val"]
    raise ValueError(f"Unknown dataset: {dataset_name}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(loaders["label2idx"])

    train_loader, val_loader = get_dataset_loaders(loaders, args.dataset)

    model = create_mobilenetv2(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        latest_checkpoint_path = save_dir / args.checkpoint
        torch.save(model.state_dict(), latest_checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = save_dir / args.best_checkpoint
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Saved best model to {best_checkpoint_path}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for Plant Disease Detection")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined",
        choices=["plantvillage", "plantdoc", "combined"],
        help="Which dataset to train on",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--step_size", type=int, default=7, help="LR scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR scheduler gamma")
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint", type=str, default="last_model.pth", help="Latest checkpoint filename")
    parser.add_argument("--best_checkpoint", type=str, default="best_model.pth", help="Best checkpoint filename")

    main(parser.parse_args())
