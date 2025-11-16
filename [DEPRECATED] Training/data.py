"""Data loading utilities for RecycleVision."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from config import DataConfig


def _build_transforms(config: DataConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return training and validation transforms."""
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    train_transforms = [
        transforms.Resize(int(config.image_size * 1.15), interpolation=InterpolationMode.BILINEAR, antialias=True)
    ]
    if config.augment:
        train_transforms.extend(
            [
                transforms.RandomResizedCrop(config.image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
            ]
        )
    else:
        train_transforms.append(transforms.CenterCrop(config.image_size))
    train_transforms.extend([
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.3), ratio=(0.3, 3.3)),
    ])

    val_transforms = transforms.Compose(
        [
            transforms.Resize(
                config.image_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transforms.Compose(train_transforms), val_transforms


def _resolve_train_val_dirs(config: DataConfig) -> Tuple[Path, Path | None]:
    train_dir = config.train_dir or config.root / "train"
    val_dir = config.val_dir
    if val_dir is None:
        candidate = config.root / "val"
        val_dir = candidate if candidate.exists() else None
    return train_dir, val_dir


def build_datasets(config: DataConfig) -> Tuple[Dataset, Dataset]:
    """Create the training and validation datasets from ImageFolder directories."""
    train_transform, val_transform = _build_transforms(config)
    train_dir, val_dir = _resolve_train_val_dirs(config)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

    if val_dir and val_dir.exists():
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    else:
        val_size = max(1, int(0.2 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        val_dataset.dataset.transform = val_transform  # type: ignore[attr-defined]
    return train_dataset, val_dataset


def build_loaders(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """Return PyTorch data loaders for training and validation."""
    train_dataset, val_dataset = build_datasets(config)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    return train_loader, val_loader


def build_inference_transform(config: DataConfig) -> transforms.Compose:
    """Return the validation transform for inference pipelines."""
    _, val_transform = _build_transforms(config)
    return val_transform


def class_names_from_data(config: DataConfig) -> List[str]:
    """Derive class names from the training directory layout."""
    train_dir, _ = _resolve_train_val_dirs(config)
    dataset = datasets.ImageFolder(train_dir)
    return list(dataset.classes)
