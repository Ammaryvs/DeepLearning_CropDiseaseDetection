"""
Training script for ResNet50 transfer learning on crop disease detection.

Three-phase strategy
--------------------
Phase 1  Feature extraction  Backbone frozen; only the FC head is updated.
Phase 2  Fine-tuning         layer4 + FC head unfrozen; lower learning rate.
Phase 3  Full fine-tuning    All layers unfrozen; very low learning rate.

Usage (run from the project root directory)
-------------------------------------------
    python -m src.models.resnet50.train

Outputs saved to checkpoints/resnet50/
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
from src.models.resnet50.model import build_resnet50, freeze_backbone, unfreeze_top_layers, unfreeze_all

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_phase(model, loaders, criterion, optimizer, scheduler, device,
              num_epochs, early_stopping, history, phase_name):
    """
    Execute one training phase.

    Updates `history` in-place.
    Returns (best_state_dict, best_val_acc) for this phase.
    """
    best_val_acc = max(history['val_accs']) if history['val_accs'] else 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, loaders['val'], criterion, device)
        scheduler.step()

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)

        print(f"[{phase_name}] Epoch {epoch:>2}/{num_epochs} | "
              f"Train Loss {train_loss:.4f}  Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f}  Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best val acc: {best_val_acc:.4f}")

        if early_stopping(val_loss):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    return best_state, best_val_acc


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
    loaders = dataloaders['loaders']
    label2idx = dataloaders['label2idx']
    num_classes = dataloaders['num_classes']
    print(f"Dataset: plantvillage  |  Classes: {num_classes}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_resnet50(
        num_classes=num_classes,
        pretrained=cfg['model']['pretrained'],
        dropout=cfg['training']['dropout'],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    opt_name  = cfg['training'].get('optimizer', 'AdamW')
    save_dir  = cfg['training']['save_dir']
    history   = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}
    overall_best_state, overall_best_acc = None, 0.0

    # ── Phase 1: Feature Extraction ──────────────────────────────────────────
    p1 = cfg['training']['phase1']
    freeze_backbone(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n[Phase 1] Frozen backbone | {opt_name} lr={p1['lr']} | "
          f"{len(trainable_params)} parameter tensors")

    optimizer = make_optimizer(opt_name, trainable_params, p1['lr'], cfg['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=p1['epochs'])
    es = EarlyStopping(patience=cfg['training']['early_stop_patience'])

    state, acc = run_phase(model, loaders, criterion, optimizer, scheduler,
                           device, p1['epochs'], es, history, 'Phase1')
    if state is not None and acc > overall_best_acc:
        overall_best_acc, overall_best_state = acc, state

    # ── Phase 2: Fine-tuning ──────────────────────────────────────────────────
    p2 = cfg['training']['phase2']
    unfreeze_top_layers(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n[Phase 2] layer4 + head | {opt_name} lr={p2['lr']} | "
          f"{len(trainable_params)} parameter tensors")

    optimizer = make_optimizer(opt_name, trainable_params, p2['lr'], cfg['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=p2['epochs'])
    es = EarlyStopping(patience=cfg['training']['early_stop_patience'])

    state, acc = run_phase(model, loaders, criterion, optimizer, scheduler,
                           device, p2['epochs'], es, history, 'Phase2')
    if state is not None and acc > overall_best_acc:
        overall_best_acc, overall_best_state = acc, state

    # ── Phase 3: Full Fine-tuning ─────────────────────────────────────────────
    p3 = cfg['training']['phase3']
    unfreeze_all(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n[Phase 3] All layers | {opt_name} lr={p3['lr']} | "
          f"{len(trainable_params)} parameter tensors")

    optimizer = make_optimizer(opt_name, trainable_params, p3['lr'], cfg['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=p3['epochs'])
    es = EarlyStopping(patience=cfg['training']['early_stop_patience'])

    state, acc = run_phase(model, loaders, criterion, optimizer, scheduler,
                           device, p3['epochs'], es, history, 'Phase3')
    if state is not None and acc > overall_best_acc:
        overall_best_acc, overall_best_state = acc, state

    # ── Save best checkpoint ──────────────────────────────────────────────────
    ckpt_path = os.path.join(save_dir, 'best_model.pth')
    save_checkpoint({
        'model_state_dict': overall_best_state,
        'best_val_acc': overall_best_acc,
        'label2idx': label2idx,
        'num_classes': num_classes,
    }, ckpt_path)
    print(f"\nBest checkpoint saved to {ckpt_path}  (val acc: {overall_best_acc:.4f})")

    # ── Save curves and history ───────────────────────────────────────────────
    plot_training_curves(
        history['train_losses'], history['val_losses'],
        history['train_accs'], history['val_accs'],
        os.path.join(save_dir, 'training_curves.png'),
    )
    save_json(history, os.path.join(save_dir, 'history.json'))
    print("Training complete.")


if __name__ == '__main__':
    main()
