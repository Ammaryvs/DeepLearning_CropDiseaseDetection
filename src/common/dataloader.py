import random
from torch.utils.data import DataLoader, ConcatDataset
from src.common.dataset import (
    PlantDiseaseDataset,
    PLANTDOC_TEST_DIR,
    PLANTDOC_TRAIN_DIR,
    PLANTVILLAGE_DIR,
    train_transform,
    val_transform,
)

def list_files_by_class(root_dir, ext_patterns=("*.jpg", "*.jpeg", "*.png")):
    classes = []
    samples = []

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

    samples = samples[:]  # avoid modifying original list
    random.Random(seed).shuffle(samples)

    n = len(samples)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    print(f"Split {len(samples)} samples into {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test


def create_dataloaders(batch_size=32, num_workers=4):
    # scan datasets once
    pv_classes, pv_samples = list_files_by_class(PLANTVILLAGE_DIR)
    pd_train_classes, pd_train_samples = list_files_by_class(PLANTDOC_TRAIN_DIR)
    pd_test_classes, pd_test_samples = list_files_by_class(PLANTDOC_TEST_DIR)

    # unified label mapping
    all_labels = sorted(set(pv_classes) | set(pd_train_classes) | set(pd_test_classes))
    label2idx = {label: i for i, label in enumerate(all_labels)}

    # split
    pv_train_samples, pv_val_samples, pv_test_samples = split_samples(pv_samples, 0.8, 0.1, 0.1)
    pd_train_samples, pd_val_samples, _ = split_samples(pd_train_samples, 0.8, 0.1, 0.1)

    # keep original PlantDoc test set
    pd_test_samples_full = pd_test_samples

    # datasets
    pv_train_ds = PlantDiseaseDataset(pv_train_samples, label2idx, transform=train_transform)
    pv_val_ds = PlantDiseaseDataset(pv_val_samples, label2idx, transform=val_transform)
    pv_test_ds = PlantDiseaseDataset(pv_test_samples, label2idx, transform=val_transform)

    pd_train_ds = PlantDiseaseDataset(pd_train_samples, label2idx, transform=train_transform)
    pd_val_ds = PlantDiseaseDataset(pd_val_samples, label2idx, transform=val_transform)
    pd_test_ds = PlantDiseaseDataset(pd_test_samples_full, label2idx, transform=val_transform)

    # individual loaders
    pv_loaders = {
        "train": DataLoader(pv_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(pv_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(pv_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    pd_loaders = {
        "train": DataLoader(pd_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(pd_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(pd_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    # combined datasets (skip empty PlantDoc splits to avoid DataLoader crash on Windows)
    def concat_nonempty(*datasets):
        non_empty = [ds for ds in datasets if len(ds) > 0]
        return ConcatDataset(non_empty) if len(non_empty) > 1 else non_empty[0]

    combined_train_ds = concat_nonempty(pv_train_ds, pd_train_ds)
    combined_val_ds = concat_nonempty(pv_val_ds, pd_val_ds)
    combined_test_ds = concat_nonempty(pv_test_ds, pd_test_ds)

    combined_loaders = {
        "train": DataLoader(combined_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(combined_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(combined_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return {
        "plantvillage": pv_loaders,
        "plantdoc": pd_loaders,
        "combined": combined_loaders,
        "label2idx": label2idx,
    }