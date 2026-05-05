from collections import Counter

import timm.data
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import IMG_SIZE, NUM_WORKERS, TaskSpec


def _train_tf(mean, std):
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _eval_tf(mean, std):
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_norm(model):
    cfg = timm.data.resolve_data_config({}, model=model)
    return cfg["mean"], cfg["std"]


def make_loaders(task: TaskSpec, model, batch_size: int):
    mean, std = get_norm(model)
    train_tf, eval_tf = _train_tf(mean, std), _eval_tf(mean, std)

    train_ds = datasets.ImageFolder(task.train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(task.val_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(task.test_dir, transform=eval_tf)

    if not (train_ds.classes == val_ds.classes == test_ds.classes):
        raise ValueError(
            f"Class mismatch: train={train_ds.classes} val={val_ds.classes} test={test_ds.classes}"
        )

    extra_loaders = {}
    for name, path in task.extra_test_dirs.items():
        ds = datasets.ImageFolder(path, transform=eval_tf)
        if ds.classes != train_ds.classes:
            raise ValueError(f"Extra test '{name}' classes differ: {ds.classes}")
        extra_loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    counts = Counter(label for _, label in train_ds.samples)
    n, k = len(train_ds), len(train_ds.classes)
    class_weights = torch.tensor(
        [n / (k * counts[i]) for i in range(k)], dtype=torch.float32
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "extra_tests": extra_loaders,
        "classes": train_ds.classes,
        "class_weights": class_weights,
        "train_counts": {train_ds.classes[i]: counts[i] for i in range(k)},
    }
