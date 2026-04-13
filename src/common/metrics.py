"""Common metric utilities for classification training loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch


@dataclass
class AverageMeter:
    """Track the running average of a scalar metric."""

    name: str
    value: float = 0.0
    avg: float = 0.0
    total: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.value = 0.0
        self.avg = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = float(value)
        self.total += float(value) * n
        self.count += n
        self.avg = self.total / self.count if self.count else 0.0


@dataclass
class MetricTracker:
    """Maintain multiple named AverageMeters."""

    metric_names: Sequence[str]
    meters: MutableMapping[str, AverageMeter] = field(init=False)

    def __post_init__(self) -> None:
        self.meters = {name: AverageMeter(name=name) for name in self.metric_names}

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter(name=name)
        self.meters[name].update(value, n=n)

    def average(self, name: str) -> float:
        return self.meters[name].avg

    def result(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}


def accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> List[float]:
    """Compute top-k accuracy percentages for a batch."""
    if logits.ndim != 2:
        raise ValueError("Expected logits to have shape [batch_size, num_classes].")
    if targets.ndim != 1:
        raise ValueError("Expected targets to have shape [batch_size].")
    if logits.size(0) != targets.size(0):
        raise ValueError("Batch size mismatch between logits and targets.")

    if not topk:
        return []

    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    batch_size = targets.size(0)
    results: List[float] = []
    for k in topk:
        k = min(k, logits.size(1))
        correct_k = correct[:k].reshape(-1).float().sum().item()
        results.append(100.0 * correct_k / batch_size if batch_size else 0.0)
    return results


def _to_long_tensor(values: Iterable[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().long().view(-1)
    return torch.as_tensor(list(values), dtype=torch.long)


def confusion_matrix(
    predictions: Iterable[int] | torch.Tensor,
    targets: Iterable[int] | torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build a confusion matrix with shape [num_classes, num_classes]."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    preds = _to_long_tensor(predictions)
    labels = _to_long_tensor(targets)

    if preds.numel() != labels.numel():
        raise ValueError("Predictions and targets must have the same number of elements.")

    valid = (
        (labels >= 0)
        & (labels < num_classes)
        & (preds >= 0)
        & (preds < num_classes)
    )
    preds = preds[valid]
    labels = labels[valid]

    encoded = labels * num_classes + preds
    matrix = torch.bincount(encoded, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def classification_metrics(
    predictions: Iterable[int] | torch.Tensor,
    targets: Iterable[int] | torch.Tensor,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Compute aggregate and per-class metrics from predictions."""
    cm = confusion_matrix(predictions, targets, num_classes)
    tp = cm.diag().float()
    support = cm.sum(dim=1).float()
    predicted = cm.sum(dim=0).float()

    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(tp),
    )

    total = cm.sum().item()
    correct = tp.sum().item()
    weights = support / support.sum().clamp_min(1.0)

    per_class = {}
    for idx in range(num_classes):
        name = class_names[idx] if class_names is not None else str(idx)
        per_class[name] = {
            "precision": float(precision[idx].item()),
            "recall": float(recall[idx].item()),
            "f1": float(f1[idx].item()),
            "support": int(support[idx].item()),
        }

    return {
        "accuracy": float(correct / total) if total else 0.0,
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "weighted_precision": float((precision * weights).sum().item()),
        "weighted_recall": float((recall * weights).sum().item()),
        "weighted_f1": float((f1 * weights).sum().item()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return top-1 accuracy as a ratio in [0, 1]."""
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())
