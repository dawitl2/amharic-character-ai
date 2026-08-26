"""Maintainable training components for the active CharacterCNN."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from data_splits import load_or_create_split_manifest, split_indices
from preprocessing import CharacterTransform, PreprocessingSpec
from project_paths import DATA_DIR, SPLIT_MANIFEST_PATH


@dataclass(frozen=True)
class TrainingSettings:
    max_epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.01
    optimizer_name: str = "SGD"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-5
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class TrainingData:
    dataset: ImageFolder
    manifest: dict
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def create_training_data(settings: TrainingSettings, spec: PreprocessingSpec) -> TrainingData:
    dataset = ImageFolder(DATA_DIR, transform=CharacterTransform(spec))
    manifest = load_or_create_split_manifest(dataset, DATA_DIR, SPLIT_MANIFEST_PATH)
    indices = split_indices(dataset, DATA_DIR, manifest)
    generator = torch.Generator().manual_seed(settings.seed)

    def loader(split_name: str, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            Subset(dataset, indices[split_name]),
            batch_size=settings.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
        )

    return TrainingData(
        dataset=dataset,
        manifest=manifest,
        train_loader=loader("train", shuffle=True),
        validation_loader=loader("validation", shuffle=False),
        test_loader=loader("test", shuffle=False),
    )
