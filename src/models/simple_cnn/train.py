from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import tensorflow as tf
import yaml
from tensorflow import keras

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.simple_cnn.model import build_simple_cnn


AUTOTUNE = tf.data.AUTOTUNE
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path: Path) -> dict:
    print(f"Loading config from: {config_path}", flush=True)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_split_file(split_file: Path) -> tuple[list[str], list[str]]:
    print(f"Reading split file: {split_file}", flush=True)
    image_paths: list[str] = []
    labels: list[str] = []

    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.split("\t")
            resolved = resolve_image_path(split_file, rel_path)
            image_paths.append(str(resolved))
            labels.append(label)

    return image_paths, labels


def resolve_image_path(split_file: Path, rel_path: str) -> Path:
    raw_path = Path(rel_path)
    candidates = [
        (split_file.parent / raw_path).resolve(),
        (split_file.parent.parent / raw_path).resolve(),
        (PROJECT_ROOT / raw_path).resolve(),
        (PROJECT_ROOT.parent / raw_path).resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def encode_labels(labels: Iterable[str], class_names: list[str]) -> list[int]:
    label_to_index = {label: index for index, label in enumerate(class_names)}
    return [label_to_index[label] for label in labels]


def normalize_label(raw_name: str) -> str:
    _, condition = raw_name.split("___", maxsplit=1)
    return condition


def display_class_name(raw_name: str) -> str:
    return raw_name.replace("_", " ")


def build_dataset(
    image_paths: list[str],
    labels: list[int],
    image_size: tuple[int, int],
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    split_name = "train" if training else "eval"
    print(
        f"Building {split_name} dataset with {len(image_paths)} images, batch_size={batch_size}, image_size={image_size}",
        flush=True,
    )
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((path_ds, label_ds))

    if training:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 8192), reshuffle_each_iteration=True)

    def _load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        return image, label

    dataset = dataset.map(_load_image, num_parallel_calls=AUTOTUNE)

    if training:
        augmenter = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.08),
                keras.layers.RandomZoom(0.1),
            ],
            name="augmentation",
        )
        dataset = dataset.map(
            lambda image, label: (augmenter(image, training=True), label),
            num_parallel_calls=AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def verify_paths_exist(paths: list[str], split_name: str) -> None:
    print(f"Checking {split_name} paths...", flush=True)
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing file referenced by {split_name}:\n{path}\n"
                "Update the split paths or place the PlantVillage dataset where the split files expect it."
            )


def save_metadata(output_dir: Path, class_names: list[str], history: keras.callbacks.History) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "display_names": [display_class_name(name) for name in class_names],
        "history": history.history,
    }

    with (output_dir / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple TensorFlow CNN on PlantVillage split files.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/models/simple_cnn/config.yaml"),
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print("TensorFlow loaded. Starting training pipeline setup...", flush=True)
    tf.random.set_seed(config["seed"])

    image_height = config["image_height"]
    image_width = config["image_width"]
    image_size = (image_height, image_width)
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

    train_split = Path(config["train_split"])
    val_split = Path(config["val_split"])
    test_split = Path(config["test_split"])
    output_dir = Path(config["output_dir"])

    print("Parsing train/val/test split files...", flush=True)
    train_paths, train_labels = parse_split_file(train_split)
    val_paths, val_labels = parse_split_file(val_split)
    test_paths, test_labels = parse_split_file(test_split)

    print("Loaded split files.", flush=True)
    verify_paths_exist(train_paths, "train")
    verify_paths_exist(val_paths, "val")
    verify_paths_exist(test_paths, "test")

    train_labels = [normalize_label(label) for label in train_labels]
    val_labels = [normalize_label(label) for label in val_labels]
    test_labels = [normalize_label(label) for label in test_labels]

    class_names = sorted(set(train_labels))
    train_targets = encode_labels(train_labels, class_names)
    val_targets = encode_labels(val_labels, class_names)
    test_targets = encode_labels(test_labels, class_names)

    print(f"Train samples: {len(train_paths)}", flush=True)
    print(f"Validation samples: {len(val_paths)}", flush=True)
    print(f"Test samples: {len(test_paths)}", flush=True)
    print(f"Classes ({len(class_names)}):", flush=True)
    for class_name, count in Counter(train_labels).most_common():
        print(f"  {class_name}: {count}", flush=True)

    train_ds = build_dataset(train_paths, train_targets, image_size, batch_size, training=True)
    val_ds = build_dataset(val_paths, val_targets, image_size, batch_size, training=False)
    test_ds = build_dataset(test_paths, test_targets, image_size, batch_size, training=False)

    print("Building CNN model...", flush=True)
    model = build_simple_cnn(
        input_shape=(image_height, image_width, 3),
        num_classes=len(class_names),
    )

    print("Compiling model...", flush=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_simple_cnn.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print("Starting training...", flush=True)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("Training finished. Evaluating on test set...", flush=True)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}", flush=True)
    print(f"Test accuracy: {test_accuracy:.4f}", flush=True)

    print(f"Saving model artifacts to: {output_dir}", flush=True)
    model.save(output_dir / "final_simple_cnn.keras")
    save_metadata(output_dir, class_names, history)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
