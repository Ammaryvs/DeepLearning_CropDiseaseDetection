import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred, average: str = 'weighted') -> dict:
    """Return accuracy, precision, recall, and F1 as a dict."""
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall':    recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1':        f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def print_report(y_true, y_pred, class_names=None) -> None:
    """Print a full per-class sklearn classification report."""
    import numpy as np
    present_labels = sorted(set(y_true) | set(y_pred))
    target_names = [class_names[i] for i in present_labels] if class_names else None
    print(classification_report(y_true, y_pred, labels=present_labels,
                                target_names=target_names, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str, max_classes: int = 20) -> None:
    """
    Plot and save a confusion matrix.

    To keep the figure readable, at most `max_classes` classes are shown
    (the ones with the most samples).
    """
    cm = confusion_matrix(y_true, y_pred)
    n_total = len(class_names)

    # select the top-N most frequent classes for display
    class_totals = cm.sum(axis=1)
    top_idx = sorted(class_totals.argsort()[::-1][:max_classes])

    cm_sub = cm[top_idx][:, top_idx]
    names_sub = [class_names[i] for i in top_idx]
    n = len(names_sub)

    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))
    im = ax.imshow(cm_sub, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names_sub, rotation=90, fontsize=7)
    ax.set_yticklabels(names_sub, fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    title = f'Confusion Matrix (top {n} of {n_total} classes)' if n < n_total else 'Confusion Matrix'
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
