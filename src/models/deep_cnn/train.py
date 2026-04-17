"""
Training script for DeepCNN on crop disease detection.

Trained from scratch (no pretrained weights). Single-phase training.

Grid search findings (deepcnn.ipynb):
    SGD  lr=0.01  → best result (val acc 0.9845, test acc 0.9834)
    SGD  lr=0.1   → diverged
    Adam lr=0.1   → unstable
    Adam lr=0.01  → good

Usage (run from the project root directory)
-------------------------------------------
    python -m src.models.deep_cnn.train

Outputs saved to checkpoints/deep_cnn/
    best_model.pth       – best checkpoint (by val accuracy)
    training_curves.png  – loss & accuracy curves
    history.json         – per-epoch metrics
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.common.seed import set_seed
from src.common.dataloader import create_dataloaders
from src.common.utils import (
    train_one_epoch, evaluate, save_checkpoint,
    plot_training_curves, save_json, EarlyStopping,
)
from src.models.deep_cnn.model import build_deep_cnn

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_optimizer(opt_name, params, lr, weight_decay):
    if opt_name == 'SGD':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def main():
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg['training']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dataloaders = create_dataloaders(
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
    )
    loaders     = dataloaders['loaders']
    label2idx   = dataloaders['label2idx']
    num_classes = dataloaders['num_classes']
    print(f"Dataset: plantvillage  |  Classes: {num_classes}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_deep_cnn(
        num_classes=num_classes,
        dropout=cfg['model']['dropout'],
    ).to(device)

    criterion  = nn.CrossEntropyLoss()
    opt_name   = cfg['training'].get('optimizer', 'SGD')
    lr         = cfg['training']['lr']
    save_dir   = cfg['training']['save_dir']
    history    = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}

    optimizer  = make_optimizer(opt_name, model.parameters(), lr, cfg['training']['weight_decay'])
    scheduler  = CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
    es         = EarlyStopping(patience=cfg['training']['early_stop_patience'])

    print(f"\n[Training] {opt_name} lr={lr} | epochs={cfg['training']['epochs']}")

    best_val_acc  = 0.0
    best_state    = None

    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, loaders['val'], criterion, device)
        scheduler.step()

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)

        print(f"Epoch {epoch:>2}/{cfg['training']['epochs']} | "
              f"Train Loss {train_loss:.4f}  Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f}  Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best val acc: {best_val_acc:.4f}")

        if es(val_loss):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    # ── Save ──────────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(save_dir, 'best_model.pth')
    save_checkpoint({
        'model_state_dict': best_state,
        'best_val_acc':     best_val_acc,
        'label2idx':        label2idx,
        'num_classes':      num_classes,
    }, ckpt_path)
    print(f"\nBest checkpoint saved to {ckpt_path}  (val acc: {best_val_acc:.4f})")

    plot_training_curves(
        history['train_losses'], history['val_losses'],
        history['train_accs'],   history['val_accs'],
        os.path.join(save_dir, 'training_curves.png'),
    )
    save_json(history, os.path.join(save_dir, 'history.json'))
    print("Training complete.")


if __name__ == '__main__':
    main()
