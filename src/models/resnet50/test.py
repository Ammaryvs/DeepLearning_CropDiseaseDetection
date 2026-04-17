"""
Evaluation script for the trained ResNet50 model.

Loads the best saved checkpoint, runs inference on the test split, and reports:
  - Overall accuracy, precision, recall, F1 (weighted)
  - Full per-class classification report
  - Confusion matrix image (saved to the checkpoint directory)
  - JSON file with summary metrics

Usage (run from the project root directory)
-------------------------------------------
    python -m src.models.resnet50.test
    python -m src.models.resnet50.test --checkpoint path/to/best_model.pth
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.common.dataloader import create_dataloaders
from src.common.utils import evaluate, load_checkpoint, save_json
from src.common.metrics import compute_metrics, print_report, plot_confusion_matrix
from src.models.resnet50.model import build_resnet50

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet50 on the test set.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pth checkpoint. Defaults to config save_dir/best_model.pth')
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dataloaders = create_dataloaders(
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
    )
    loaders = dataloaders['loaders']

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or os.path.join(cfg['training']['save_dir'], 'best_model.pth')
    print(f"Loading checkpoint: {ckpt_path}")

    # Use label mapping stored in the checkpoint to guarantee consistency
    raw_ckpt = torch.load(ckpt_path, map_location=device)
    label2idx = raw_ckpt.get('label2idx') or dataloaders['label2idx']
    num_classes = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(num_classes)]

    model = build_resnet50(num_classes=num_classes, pretrained=False,
                           dropout=cfg['training']['dropout']).to(device)
    load_checkpoint(ckpt_path, model=model, map_location=device)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels = evaluate(
        model, loaders['test'], criterion, device
    )

    print(f"\nTest Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}")

    metrics = compute_metrics(all_labels, all_preds)
    print(f"Precision : {metrics['precision']:.4f}  (weighted)")
    print(f"Recall    : {metrics['recall']:.4f}  (weighted)")
    print(f"F1        : {metrics['f1']:.4f}  (weighted)")

    print("\n--- Per-Class Classification Report ---")
    print_report(all_labels, all_preds, class_names=class_names)

    # ── Save outputs ─────────────────────────────────────────────────────────
    save_dir = cfg['training']['save_dir']
    plot_confusion_matrix(all_labels, all_preds, class_names,
                          save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    save_json({
        'test_loss':           test_loss,
        'test_accuracy':       test_acc,
        'precision_weighted':  metrics['precision'],
        'recall_weighted':     metrics['recall'],
        'f1_weighted':         metrics['f1'],
    }, os.path.join(save_dir, 'test_results.json'))
    print(f"Results saved to {save_dir}/")


if __name__ == '__main__':
    main()
