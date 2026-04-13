"""Reproducibility helpers for training and evaluation."""

from __future__ import annotations

import os
import random
from typing import Optional

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected but kept optional here
    np = None


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker based on PyTorch's initial worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    if np is not None:
        np.random.seed(worker_seed + worker_id)


def create_generator(seed: Optional[int] = None) -> torch.Generator:
    """Create a PyTorch generator seeded for deterministic sampling."""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator
