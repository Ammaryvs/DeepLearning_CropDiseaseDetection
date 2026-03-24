import os
import sys
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from model import VisionTransformer
from src.common.dataloader import create_dataloaders


def test_one_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

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

    model = VisionTransformer(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)

    checkpoint_path = os.path.join(args.save_dir, args.checkpoint)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()

    loaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    test_loader = loaders[args.dataset]["test"]

    # pv_train_loader = loaders["plantvillage"]["train"]
    # pv_val_loader = loaders["plantvillage"]["val"]
    pv_test_loader = loaders["plantvillage"]["test"]

    # pd_train_loader = loaders["plantdoc"]["train"]
    # pd_val_loader = loaders["plantdoc"]["val"]
    pd_test_loader = loaders["plantdoc"]["test"]

    # combined_train_loader = loaders["combined"]["train"]
    # combined_val_loader = loaders["combined"]["val"]
    combined_test_loader = loaders["combined"]["test"]

    # choose which test loader to use
    if args.dataset == "plantvillage":
        test_loader = pv_test_loader
    elif args.dataset == "plantdoc":
        test_loader = pd_test_loader
    elif args.dataset == "combined":
        test_loader = combined_test_loader
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    test_loss, test_acc = test_one_epoch(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Vision Transformer for Plant Disease Detection")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--depth", type=int, default=12, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--dataset", type=str, default="combined",
                        choices=["plantvillage", "plantdoc", "combined"],
                        help="Which dataset to test on")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory of saved models")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Checkpoint filename")

    args = parser.parse_args()
    main(args)