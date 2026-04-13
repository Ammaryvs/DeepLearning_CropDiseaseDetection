import argparse
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from model import create_mobilenetv2
from src.common.dataloader import create_dataloaders


def test_one_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(loaders["label2idx"])

    model = create_mobilenetv2(
        num_classes=num_classes,
        pretrained=False,
        dropout=args.dropout,
    ).to(device)

    checkpoint_path = os.path.join(args.save_dir, args.checkpoint)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()

    if args.dataset == "plantvillage":
        test_loader = loaders["plantvillage"]["test"]
    elif args.dataset == "plantdoc":
        if loaders["plantdoc"] is None:
            raise ValueError("PlantDoc dataset is empty or incomplete.")
        test_loader = loaders["plantdoc"]["test"]
    elif args.dataset == "combined":
        test_loader = loaders["combined"]["test"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    test_loss, test_acc = test_one_epoch(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MobileNetV2 for Plant Disease Detection")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined",
        choices=["plantvillage", "plantdoc", "combined"],
        help="Which dataset to test on",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout used during training")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory of saved models")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Checkpoint filename")

    main(parser.parse_args())
