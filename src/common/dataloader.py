import random
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import (
    PlantDiseaseDataset,
    PLANTVILLAGE_DIR,
    train_transform,
    val_transform,
)
from .seed import create_generator, seed_worker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def resolve_data_root(root_dir=None):
    if root_dir is None:
        return PLANTVILLAGE_DIR

    path = Path(root_dir)
    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            Path.cwd() / path,
            PROJECT_ROOT / path,
            WORKSPACE_ROOT / path,
            path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve dataset directory: {root_dir}")

def list_files_by_class(root_dir, ext_patterns=("*.jpg", "*.jpeg", "*.png")):
    root_dir = resolve_data_root(root_dir)
    classes = []
    samples = []

    if not root_dir.exists():
        print(f"Directory not found: {root_dir}. Treating as empty dataset.")
        return classes, samples

    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        classes.append(class_name)

        file_list = []
        for ext in ext_patterns:
            file_list.extend(class_dir.glob(ext))

        for p in sorted(file_list):
            samples.append((str(p), class_name))

    print(f"Found {len(classes)} classes and {len(samples)} samples in {root_dir}")
    return sorted(classes), samples


def split_samples(samples, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = random.Random(seed)
    samples_by_class = defaultdict(list)
    for sample in samples:
        samples_by_class[sample[1]].append(sample)

    train = []
    val = []
    test = []

    for class_name in sorted(samples_by_class):
        class_samples = samples_by_class[class_name][:]
        rng.shuffle(class_samples)

        n = len(class_samples)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        train.extend(class_samples[:n_train])
        val.extend(class_samples[n_train:n_train + n_val])
        test.extend(class_samples[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(f"Split {len(samples)} samples into {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test


def write_split_file(split_path, samples):
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as f:
        for image_path, label_name in samples:
            f.write(f"{image_path}\t{label_name}\n")

    print(f"Wrote {len(samples)} samples to {split_path}")


def save_splits(train_samples, val_samples, test_samples, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "splits"

    write_split_file(output_dir / "train.txt", train_samples)
    write_split_file(output_dir / "val.txt", val_samples)
    write_split_file(output_dir / "test.txt", test_samples)


def create_dataloaders(batch_size=32, num_workers=4):
    # scan PlantVillage dataset
    pv_classes, pv_samples = list_files_by_class(PLANTVILLAGE_DIR)

    # label mapping
    label2idx = {label: i for i, label in enumerate(pv_classes)}

    # split into train/val/test (80/10/10)
    pv_train_samples, pv_val_samples, pv_test_samples = split_samples(pv_samples, 0.8, 0.1, 0.1)

    # datasets
    pv_train_ds = PlantDiseaseDataset(pv_train_samples, label2idx, transform=train_transform)
    pv_val_ds = PlantDiseaseDataset(pv_val_samples, label2idx, transform=val_transform)
    pv_test_ds = PlantDiseaseDataset(pv_test_samples, label2idx, transform=val_transform)

    # loaders
    loaders = {
        "train": DataLoader(pv_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(pv_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(pv_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return {
        "loaders": loaders,
        "label2idx": label2idx,
        "num_classes": len(pv_classes),
    }


def create_plantvillage_dataloaders(
    root_dir=None,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    seed=42,
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1,
    train_tfms=None,
    val_tfms=None,
):
    root_dir = resolve_data_root(root_dir)
    classes, samples = list_files_by_class(root_dir)
    label2idx = {label: idx for idx, label in enumerate(classes)}

    train_samples, val_samples, test_samples = split_samples(
        samples,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    train_ds = PlantDiseaseDataset(
        train_samples,
        label2idx,
        transform=train_tfms if train_tfms is not None else train_transform,
    )
    val_ds = PlantDiseaseDataset(
        val_samples,
        label2idx,
        transform=val_tfms if val_tfms is not None else val_transform,
    )
    test_ds = PlantDiseaseDataset(
        test_samples,
        label2idx,
        transform=val_tfms if val_tfms is not None else val_transform,
    )

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=create_generator(seed),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=create_generator(seed),
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=create_generator(seed),
        ),
    }

    return {
        "root_dir": str(root_dir),
        "loaders": loaders,
        "datasets": {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
        },
        "label2idx": label2idx,
        "num_classes": len(classes),
        "classes": classes,
        "split_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
    }


def create_and_save_splits():
    pv_classes, pv_samples = list_files_by_class(PLANTVILLAGE_DIR)

    pv_train_samples, pv_val_samples, pv_test_samples = split_samples(pv_samples, 0.8, 0.1, 0.1)

    save_splits(pv_train_samples, pv_val_samples, pv_test_samples)

    return {
        "train": pv_train_samples,
        "val": pv_val_samples,
        "test": pv_test_samples,
    }


if __name__ == "__main__":
    create_and_save_splits()
