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


DEFAULT_LRS = [0.1, 0.001]
DEFAULT_AUG_STRENGTHS = [0.3, 0.7]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_run_specs(raw: str) -> list[tuple[float, float]]:
    runs: list[tuple[float, float]] = []
    for item in raw.split(","):
        spec = item.strip()
        if not spec:
            continue

        parts = [part.strip() for part in spec.split(":")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid run spec '{spec}'. Expected format 'learning_rate:augmentation_strength'."
            )

        runs.append((float(parts[0]), float(parts[1])))

    return runs


def build_run_name(learning_rate: float, augmentation_strength: float) -> str:
    return f"lr_{learning_rate:g}_aug_{augmentation_strength:g}".replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AttentionCNN hyperparameter sweep.")
    parser.add_argument("--config", default="src/models/attention_cnn/config.yaml")
    parser.add_argument("--output", default="results/attention_cnn_sweep_results.json")
    parser.add_argument("--learning-rates", default="0.1,0.01,0.001")
    parser.add_argument("--augmentation-strengths", default="0.3,0.5,0.7")
    parser.add_argument(
        "--runs",
        default=None,
        help="Exact runs to execute, e.g. '0.01:0.5,0.001:0.7'. Overrides the Cartesian product.",
    )
    parser.add_argument(
        "--resume-runs",
        default=None,
        help="Subset of --runs to resume from checkpoint, e.g. '0.01:0.5'.",
    )
    parser.add_argument(
        "--resume-checkpoint-name",
        default="best_model.pt",
        help="Checkpoint filename inside each run directory to resume from.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the total number of epochs to train each selected run up to.",
    )
    args = parser.parse_args()

    base_config = FullConfig.from_yaml(args.config)
    if args.runs:
        selected_runs = parse_run_specs(args.runs)
        learning_rates = sorted({learning_rate for learning_rate, _ in selected_runs})
        augmentation_strengths = sorted({augmentation_strength for _, augmentation_strength in selected_runs})
    else:
        learning_rates = parse_float_list(args.learning_rates)
        augmentation_strengths = parse_float_list(args.augmentation_strengths)
        selected_runs = list(itertools.product(learning_rates, augmentation_strengths))

    resume_runs = set(parse_run_specs(args.resume_runs)) if args.resume_runs else set()
    missing_resume_runs = resume_runs.difference(selected_runs)
    if missing_resume_runs:
        formatted = ", ".join(f"{lr:g}:{aug:g}" for lr, aug in sorted(missing_resume_runs))
        raise ValueError(f"--resume-runs contains runs not present in this sweep: {formatted}")

    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    runs = []
    for learning_rate, augmentation_strength in selected_runs:
        config = FullConfig.from_dict(base_config.to_dict())
        run_name = build_run_name(learning_rate, augmentation_strength)

        config.training.learning_rate = learning_rate
        config.data.augmentation_strength = augmentation_strength
        config.checkpoint.checkpoint_dir = str(Path(base_config.checkpoint.checkpoint_dir) / run_name)
        config.checkpoint.log_dir = str(Path(base_config.checkpoint.log_dir) / run_name)

        resume_from = None
        if (learning_rate, augmentation_strength) in resume_runs:
            resume_path = Path(config.checkpoint.checkpoint_dir) / args.resume_checkpoint_name
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found for {run_name}: {resume_path}")
            resume_from = str(resume_path)

        result = train(
            config_path=args.config,
            config=config,
            run_name=run_name,
            resume_from=resume_from,
            epochs_override=args.epochs,
        )
        runs.append(result)

        save_json(
            {
                "model": "attention_cnn",
                "base_config": args.config,
                "parameter_grid": {
                    "learning_rate": learning_rates,
                    "augmentation_strength": augmentation_strengths,
                },
                "selected_runs": [
                    {
                        "learning_rate": learning_rate,
                        "augmentation_strength": augmentation_strength,
                        "run_name": build_run_name(learning_rate, augmentation_strength),
                        "resume": (learning_rate, augmentation_strength) in resume_runs,
                    }
                    for learning_rate, augmentation_strength in selected_runs
                ],
                "num_runs": len(runs),
                "runs": runs,
            },
            output_path,
        )


if __name__ == "__main__":
    main()
