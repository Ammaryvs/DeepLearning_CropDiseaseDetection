"""Run a parameter sweep for the AttentionCNN model."""

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


DEFAULT_LRS = [0.1, 0.01, 0.001]
DEFAULT_AUG_STRENGTHS = [0.3, 0.5, 0.7]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_run_name(learning_rate: float, augmentation_strength: float) -> str:
    return f"lr_{learning_rate:g}_aug_{augmentation_strength:g}".replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AttentionCNN hyperparameter sweep.")
    parser.add_argument("--config", default="src/models/attention_cnn/config.yaml")
    parser.add_argument("--output", default="results/attention_cnn_sweep_results.json")
    parser.add_argument("--learning-rates", default="0.1,0.01,0.001")
    parser.add_argument("--augmentation-strengths", default="0.3,0.5,0.7")
    args = parser.parse_args()

    base_config = FullConfig.from_yaml(args.config)
    learning_rates = parse_float_list(args.learning_rates)
    augmentation_strengths = parse_float_list(args.augmentation_strengths)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    runs = []
    for learning_rate, augmentation_strength in itertools.product(learning_rates, augmentation_strengths):
        config = FullConfig.from_dict(base_config.to_dict())
        run_name = build_run_name(learning_rate, augmentation_strength)

        config.training.learning_rate = learning_rate
        config.data.augmentation_strength = augmentation_strength
        config.checkpoint.checkpoint_dir = str(Path(base_config.checkpoint.checkpoint_dir) / run_name)
        config.checkpoint.log_dir = str(Path(base_config.checkpoint.log_dir) / run_name)

        result = train(config_path=args.config, config=config, run_name=run_name)
        runs.append(result)

        save_json(
            {
                "model": "attention_cnn",
                "base_config": args.config,
                "parameter_grid": {
                    "learning_rate": learning_rates,
                    "augmentation_strength": augmentation_strengths,
                },
                "num_runs": len(runs),
                "runs": runs,
            },
            output_path,
        )


if __name__ == "__main__":
    main()
