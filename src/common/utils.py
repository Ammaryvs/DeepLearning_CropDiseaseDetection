"""General-purpose helpers for training, checkpointing, and experiment logging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import torch


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_string(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return a timestamp string for filenames or run directories."""
    return datetime.now().strftime(fmt)


def save_json(data: Mapping[str, Any], path: str | Path, indent: int = 2) -> Path:
    """Save a dictionary to disk as JSON."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent)
    return output_path


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON content from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def save_checkpoint(
    state: Mapping[str, Any],
    checkpoint_dir: str | Path,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
    best_filename: str = "best_model.pt",
) -> Path:
    """Save a training checkpoint and optionally update the best checkpoint."""
    checkpoint_dir = ensure_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / filename
    torch.save(dict(state), checkpoint_path)

    if is_best:
        best_path = checkpoint_dir / best_filename
        torch.save(dict(state), best_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore model, optimizer, and scheduler state."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


@dataclass
class History:
    """Simple container for tracking metrics across epochs."""

    values: MutableMapping[str, List[float]] = field(default_factory=dict)

    def update(self, **metrics: float) -> None:
        for name, value in metrics.items():
            self.values.setdefault(name, []).append(float(value))

    def latest(self) -> Dict[str, float]:
        return {
            name: metric_values[-1]
            for name, metric_values in self.values.items()
            if metric_values
        }

    def to_dict(self) -> Dict[str, List[float]]:
        return dict(self.values)

    def save(self, path: str | Path, indent: int = 2) -> Path:
        return save_json(self.values, path, indent=indent)


@dataclass
class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    patience: int = 10
    mode: str = "min"
    min_delta: float = 0.0
    best_score: Optional[float] = None
    num_bad_epochs: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        return score > (self.best_score + self.min_delta)

    def step(self, score: float) -> bool:
        """Update early stopping state. Returns True if the score improved."""
        if self._is_improvement(score):
            self.best_score = score
            self.num_bad_epochs = 0
            self.should_stop = False
            return True

        self.num_bad_epochs += 1
        self.should_stop = self.num_bad_epochs >= self.patience
        return False
