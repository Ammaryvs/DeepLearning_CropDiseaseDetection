"""Run a parameter sweep for the Vision Transformer model."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from common.config import FullConfig
from common.utils import ensure_dir, save_json
from train import train


DEFAULT_LRS = [0.001, 0.0001]
DEFAULT_WEIGHT_DECAYS = [0.01, 0.05]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_run_name(learning_rate: float, weight_decay: float) -> str:
    return f"lr_{learning_rate:g}_wd_{weight_decay:g}".replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViT hyperparameter sweep.")
    parser.add_argument("--config", default="src/models/vit/config.yaml")
    parser.add_argument("--output", default="results/vit_sweep_results.json")
    parser.add_argument("--learning-rates", default="0.001, 0.0001")
    parser.add_argument("--weight-decays", default="0.01,0.05")
    args = parser.parse_args()

    base_config = FullConfig.from_yaml(args.config)
    learning_rates = parse_float_list(args.learning_rates)
    weight_decays = parse_float_list(args.weight_decays)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    runs = []
    for learning_rate, weight_decay in itertools.product(learning_rates, weight_decays):
        config = FullConfig.from_dict(base_config.to_dict())
        run_name = build_run_name(learning_rate, weight_decay)

        config.training.learning_rate = learning_rate
        config.training.weight_decay = weight_decay
        config.checkpoint.checkpoint_dir = str(Path(base_config.checkpoint.checkpoint_dir) / run_name)
        config.checkpoint.log_dir = str(Path(base_config.checkpoint.log_dir) / run_name)

        result = train(config_path=args.config, config=config, run_name=run_name)
        runs.append(result)

        save_json(
            {
                "model": "vit",
                "base_config": args.config,
                "parameter_grid": {
                    "learning_rate": learning_rates,
                    "weight_decay": weight_decays,
                },
                "num_runs": len(runs),
                "runs": runs,
            },
            output_path,
        )


if __name__ == "__main__":
    main()
