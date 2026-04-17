import os
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# ── Training / Evaluation loops ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc="train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataloader. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, leave=False, desc="eval"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, filepath: str) -> None:
    """Save a training checkpoint dict to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model, optimizer=None, device='cpu'):
    """
    Load a checkpoint into model (and optionally optimizer).
    Returns (epoch, best_val_acc, label2idx).
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (
        checkpoint.get('epoch', 0),
        checkpoint.get('best_val_acc', 0.0),
        checkpoint.get('label2idx', None),
    )


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path: str) -> None:
    """Plot and save side-by-side loss and accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label='Train')
    ax1.plot(epochs, val_losses, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label='Train')
    ax2.plot(epochs, val_accs, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


# ── JSON helpers ──────────────────────────────────────────────────────────────

def save_json(data: dict, filepath: str) -> None:
    """Serialise a dict to JSON, converting numpy types automatically."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if hasattr(obj, 'item'):    # numpy scalar
            return obj.item()
        if hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        return str(obj)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=_convert)


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
