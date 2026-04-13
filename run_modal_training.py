from __future__ import annotations

import argparse
from pathlib import Path

import modal

from modal_app import DATA_VOLUME_NAME, app, dataset_volume, train_remote


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = (PROJECT_ROOT.parent / "PlantVillage-Dataset" / "raw" / "color").resolve()
DEFAULT_SPLIT_DIR = (PROJECT_ROOT / "data" / "splits").resolve()


def upload_training_data(dataset_root: Path, split_dir: Path, force: bool) -> None:
    print(f"Uploading split files from: {split_dir}", flush=True)
    print(f"Uploading dataset files from: {dataset_root}", flush=True)
    with dataset_volume.batch_upload(force=force) as batch:
        batch.put_directory(str(split_dir), "/splits")
        batch.put_directory(str(dataset_root), "/dataset/raw/color")
    print(f"Upload complete to Modal Volume '{DATA_VOLUME_NAME}'.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload PlantVillage data and train the deep CNN on Modal.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gpu", type=str, default=None, help="Reserved for future use; GPU is configured in modal_app.py")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--force-upload", action="store_true")
    args = parser.parse_args()

    if not args.skip_upload:
        if not args.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
        if not args.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {args.split_dir}")
        upload_training_data(args.dataset_root, args.split_dir, force=args.force_upload)

    overrides: dict[str, int | float] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    print("Launching remote training job on Modal...", flush=True)
    with modal.enable_output(), app.run():
        result = train_remote.remote(overrides if overrides else None)

    print("Remote training finished.", flush=True)
    print(result, flush=True)


if __name__ == "__main__":
    main()
