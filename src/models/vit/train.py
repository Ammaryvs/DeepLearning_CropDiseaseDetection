import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from model import VisionTransformer

from src.common.dataloader import create_dataloaders

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loaders = create_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = loaders[args.dataset]["train"]
    val_loader = loaders[args.dataset]["val"]


    # pv_train_loader = loaders["plantvillage"]["train"]
    # pv_val_loader = loaders["plantvillage"]["val"]
    # pv_test_loader = loaders["plantvillage"]["test"]

    # pd_train_loader = loaders["plantdoc"]["train"]
    # pd_val_loader = loaders["plantdoc"]["val"]
    # pd_test_loader = loaders["plantdoc"]["test"]

    # combined_train_loader = loaders["combined"]["train"]
    # combined_val_loader = loaders["combined"]["val"]
    # combined_test_loader = loaders["combined"]["test"]

    # example: train on combined
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print("Best model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer for Plant Disease Detection")
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--patch_size', type=int, default=16, help="Patch size")
    parser.add_argument('--depth', type=int, default=12, help='Transformer depth')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--dataset', type=str, default='combined',
                        choices=['plantvillage', 'plantdoc', 'combined'],
                        help='Which dataset to train on')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)