"""Maintainable training components for the active CharacterCNN."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from checkpoints import CHECKPOINT_FORMAT_VERSION
from cnn_model import ARCHITECTURE_NAME
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


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    correct: int
    total: int


@dataclass
class TrainingProgress:
    cumulative_epochs: int = 0
    best_validation_accuracy: float = float("-inf")
    best_validation_loss: float = float("inf")
    epoch_of_best_checkpoint: int | None = None
    epochs_without_improvement: int = 0


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


def train_one_epoch(model, loader, loss_function, optimizer, device: torch.device) -> EpochMetrics:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return EpochMetrics(loss_sum / total, 100.0 * correct / total, correct, total)


def evaluate_model(model, loader, loss_function, device: torch.device) -> EpochMetrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss_sum += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
    return EpochMetrics(loss_sum / total, 100.0 * correct / total, correct, total)


def build_training_metadata(
    settings: TrainingSettings,
    data: TrainingData,
    spec: PreprocessingSpec,
    progress: TrainingProgress,
    *,
    test_accuracy: float | None,
) -> dict[str, Any]:
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "class_to_idx": data.dataset.class_to_idx,
        "num_classes": len(data.dataset.classes),
        "preprocessing": spec.to_metadata(),
        "best_validation_accuracy": progress.best_validation_accuracy,
        "test_accuracy": test_accuracy,
        "epoch_of_best_checkpoint": progress.epoch_of_best_checkpoint,
        "cumulative_epochs_trained": progress.cumulative_epochs,
        "optimizer": {"name": settings.optimizer_name, "learning_rate": settings.learning_rate},
        "batch_size": settings.batch_size,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "max",
            "factor": settings.scheduler_factor,
            "patience": settings.scheduler_patience,
            "min_lr": settings.scheduler_min_lr,
        },
        "early_stopping": {
            "patience": settings.early_stopping_patience,
            "minimum_validation_accuracy_delta": settings.early_stopping_min_delta,
        },
        "dataset": {
            "total": len(data.dataset),
            "train": data.manifest["counts"]["train"],
            "validation": data.manifest["counts"]["validation"],
            "test": data.manifest["counts"]["test"],
        },
        "split": {
            "strategy": data.manifest["strategy"],
            "seed": data.manifest["seed"],
            "dataset_signature": data.manifest["dataset_signature"],
            "manifest": "cnn_data_split.json",
        },
    }


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    metadata: dict[str, Any],
    progress: TrainingProgress,
) -> dict[str, Any]:
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metadata": metadata,
        "training_progress": {
            "cumulative_epochs": progress.cumulative_epochs,
            "best_validation_accuracy": progress.best_validation_accuracy,
            "best_validation_loss": progress.best_validation_loss,
            "epoch_of_best_checkpoint": progress.epoch_of_best_checkpoint,
            "epochs_without_improvement": progress.epochs_without_improvement,
        },
    }
