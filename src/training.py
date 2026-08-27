"""Maintainable training components for the active CharacterCNN."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointCompatibilityError,
    save_checkpoint_atomic,
    save_json_atomic,
    validate_cnn_metadata,
)
from cnn_model import ARCHITECTURE_NAME, CharacterCNN
from data_splits import load_or_create_split_manifest, split_indices
from dataset_contract import validate_training_dataset
from preprocessing import CharacterTransform, PreprocessingSpec
from project_paths import (
    BEST_CNN_CHECKPOINT,
    CNN_CONFIG_PATH,
    DATA_DIR,
    LATEST_CNN_CHECKPOINT,
    METRICS_CSV_PATH,
    SPLIT_MANIFEST_PATH,
)
from training_artifacts import archive_active_training_artifacts


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
    fresh_start: bool = False
    progress_interval_batches: int = 250
    num_workers: int = 0


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


def create_training_data(
    settings: TrainingSettings,
    spec: PreprocessingSpec,
    dataset: ImageFolder | None = None,
) -> TrainingData:
    if dataset is None:
        dataset = ImageFolder(DATA_DIR, transform=CharacterTransform(spec))
    validate_training_dataset(dataset)
    manifest = load_or_create_split_manifest(dataset, DATA_DIR, SPLIT_MANIFEST_PATH)
    indices = split_indices(dataset, DATA_DIR, manifest)
    validate_split_coverage(dataset, indices)
    generator = torch.Generator().manual_seed(settings.seed)

    def loader(split_name: str, *, shuffle: bool) -> DataLoader:
        options = {
            "dataset": Subset(dataset, indices[split_name]),
            "batch_size": settings.batch_size,
            "shuffle": shuffle,
            "generator": generator if shuffle else None,
            "num_workers": settings.num_workers,
            "persistent_workers": settings.num_workers > 0,
        }
        if settings.num_workers > 0:
            options["prefetch_factor"] = 2
        return DataLoader(
            **options,
        )

    return TrainingData(
        dataset=dataset,
        manifest=manifest,
        train_loader=loader("train", shuffle=True),
        validation_loader=loader("validation", shuffle=False),
        test_loader=loader("test", shuffle=False),
    )


def validate_split_coverage(dataset, indices: dict[str, list[int]]) -> None:
    """Require every class in every partition so all reported metrics are meaningful."""
    expected = set(range(len(dataset.classes)))
    for split_name in ("train", "validation", "test"):
        split_indices_for_name = indices.get(split_name, [])
        represented = {
            int(dataset.samples[index][1]) for index in split_indices_for_name
        }
        missing = expected - represented
        if missing:
            raise ValueError(
                f"The {split_name} split is missing {len(missing)} dataset classes. "
                "Add independent images or provenance groups for every class."
            )


def _print_batch_progress(
    phase: str,
    batch_number: int,
    batch_count: int,
    correct: int,
    total: int,
) -> None:
    print(
        f"{phase}: batch {batch_number:,}/{batch_count:,} "
        f"({100.0 * batch_number / batch_count:.1f}%), "
        f"running accuracy {100.0 * correct / total:.2f}%",
        flush=True,
    )


def train_one_epoch(
    model,
    loader,
    loss_function,
    optimizer,
    device: torch.device,
    *,
    epoch: int | None = None,
    progress_interval: int = 250,
) -> EpochMetrics:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    batch_count = len(loader)
    for batch_number, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
        if epoch is not None and (
            batch_number % progress_interval == 0 or batch_number == batch_count
        ):
            _print_batch_progress(
                f"Epoch {epoch} training", batch_number, batch_count, correct, total
            )
    return EpochMetrics(loss_sum / total, 100.0 * correct / total, correct, total)


def evaluate_model(
    model,
    loader,
    loss_function,
    device: torch.device,
    *,
    phase: str | None = None,
    progress_interval: int = 250,
) -> EpochMetrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        batch_count = len(loader)
        for batch_number, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss_sum += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
            if phase is not None and (
                batch_number % progress_interval == 0 or batch_number == batch_count
            ):
                _print_batch_progress(
                    phase, batch_number, batch_count, correct, total
                )
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
        "data_loader_workers": settings.num_workers,
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


def resume_training_state(
    checkpoint_path: Path,
    model,
    optimizer,
    scheduler,
    data: TrainingData,
    spec: PreprocessingSpec,
    settings: TrainingSettings,
    device: torch.device,
) -> TrainingProgress:
    if not Path(checkpoint_path).is_file():
        return TrainingProgress()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    metadata = checkpoint.get("metadata", {})
    validate_cnn_metadata(metadata)
    if metadata["class_to_idx"] != data.dataset.class_to_idx:
        raise CheckpointCompatibilityError("Resume checkpoint class mapping differs from the dataset.")
    if PreprocessingSpec.from_metadata(metadata) != spec:
        raise CheckpointCompatibilityError("Resume checkpoint preprocessing differs from training.")
    if metadata.get("split", {}).get("dataset_signature") != data.manifest["dataset_signature"]:
        raise CheckpointCompatibilityError("Resume checkpoint was trained with a different data split.")
    saved_optimizer = metadata.get("optimizer", {})
    if saved_optimizer.get("name") != settings.optimizer_name:
        raise CheckpointCompatibilityError("Resume checkpoint uses a different optimizer.")
    if float(saved_optimizer.get("learning_rate")) != settings.learning_rate:
        raise CheckpointCompatibilityError("Resume checkpoint uses a different base learning rate.")

    required = ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise CheckpointCompatibilityError(f"Resume checkpoint is missing: {', '.join(missing)}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    saved_progress = checkpoint.get("training_progress", {})
    return TrainingProgress(
        cumulative_epochs=int(saved_progress.get("cumulative_epochs", 0)),
        best_validation_accuracy=float(saved_progress.get("best_validation_accuracy", float("-inf"))),
        best_validation_loss=float(saved_progress.get("best_validation_loss", float("inf"))),
        epoch_of_best_checkpoint=saved_progress.get("epoch_of_best_checkpoint"),
        epochs_without_improvement=int(saved_progress.get("epochs_without_improvement", 0)),
    )


def _append_metrics(epoch: int, train: EpochMetrics, validation: EpochMetrics, learning_rate: float) -> None:
    write_header = not METRICS_CSV_PATH.exists()
    with METRICS_CSV_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(("epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "learning_rate"))
        writer.writerow(
            (
                epoch,
                f"{train.loss:.6f}",
                f"{train.accuracy:.4f}",
                f"{validation.loss:.6f}",
                f"{validation.accuracy:.4f}",
                f"{learning_rate:.8f}",
            )
        )


def run_training(settings: TrainingSettings) -> dict[str, Any]:
    set_deterministic_seed(settings.seed)
    spec = PreprocessingSpec()
    dataset = ImageFolder(DATA_DIR, transform=CharacterTransform(spec))
    dataset_report = validate_training_dataset(dataset)
    archive_path = None
    if settings.fresh_start:
        archive_path = archive_active_training_artifacts()
        if archive_path is not None:
            print(f"Archived previous training artifacts: {archive_path}")
    data = create_training_data(settings, spec, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterCNN(num_classes=len(data.dataset.classes)).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=settings.scheduler_factor,
        patience=settings.scheduler_patience,
        min_lr=settings.scheduler_min_lr,
    )
    progress = resume_training_state(
        LATEST_CNN_CHECKPOINT,
        model,
        optimizer,
        scheduler,
        data,
        spec,
        settings,
        device,
    )

    print(f"Architecture: {ARCHITECTURE_NAME}")
    print(
        f"Dataset: {dataset_report.class_count} classes, {dataset_report.image_count} images, "
        f"{dataset_report.minimum_images_per_class}-{dataset_report.maximum_images_per_class} per class"
    )
    print(f"Device: {device}")
    print(f"Maximum cumulative epochs: {settings.max_epochs}")
    print(f"Optimizer: {settings.optimizer_name}")
    print(f"Learning rate: {settings.learning_rate}")
    print(f"Batch size: {settings.batch_size}")
    print(f"Data loader workers: {settings.num_workers}")
    print(
        "Scheduler: ReduceLROnPlateau "
        f"(factor={settings.scheduler_factor}, patience={settings.scheduler_patience}, "
        f"min_lr={settings.scheduler_min_lr})"
    )
    print(
        f"Early stopping: patience={settings.early_stopping_patience}, "
        f"minimum accuracy delta={settings.early_stopping_min_delta:.2f} percentage points"
    )
    print(f"Split: {data.manifest['strategy']} {data.manifest['counts']}")
    if progress.cumulative_epochs:
        print(f"Resumed from cumulative epoch {progress.cumulative_epochs}: {LATEST_CNN_CHECKPOINT}")
    elif settings.fresh_start:
        print("Fresh training requested; starting from random weights.")
    else:
        print("No compatible latest CNN checkpoint found; starting from random weights.")

    for epoch in range(progress.cumulative_epochs + 1, settings.max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            data.train_loader,
            loss_function,
            optimizer,
            device,
            epoch=epoch,
            progress_interval=settings.progress_interval_batches,
        )
        validation_metrics = evaluate_model(
            model,
            data.validation_loader,
            loss_function,
            device,
            phase=f"Epoch {epoch} validation",
            progress_interval=settings.progress_interval_batches,
        )
        previous_best_accuracy = progress.best_validation_accuracy
        previous_best_loss = progress.best_validation_loss
        is_best = (
            validation_metrics.accuracy > previous_best_accuracy
            or (
                validation_metrics.accuracy == previous_best_accuracy
                and validation_metrics.loss < previous_best_loss
            )
        )
        meaningful_improvement = (
            validation_metrics.accuracy >= previous_best_accuracy + settings.early_stopping_min_delta
            or (
                validation_metrics.accuracy >= previous_best_accuracy
                and validation_metrics.loss < previous_best_loss - 1e-4
            )
        )
        progress.cumulative_epochs = epoch
        if is_best:
            progress.best_validation_accuracy = validation_metrics.accuracy
            progress.best_validation_loss = validation_metrics.loss
            progress.epoch_of_best_checkpoint = epoch
        progress.epochs_without_improvement = (
            0 if meaningful_improvement else progress.epochs_without_improvement + 1
        )
        scheduler.step(validation_metrics.accuracy)

        metadata = build_training_metadata(settings, data, spec, progress, test_accuracy=None)
        payload = checkpoint_payload(model, optimizer, scheduler, metadata, progress)
        if is_best:
            save_checkpoint_atomic(BEST_CNN_CHECKPOINT, payload)
            save_json_atomic(CNN_CONFIG_PATH, metadata)
        save_checkpoint_atomic(LATEST_CNN_CHECKPOINT, payload)
        current_lr = float(optimizer.param_groups[0]["lr"])
        _append_metrics(epoch, train_metrics, validation_metrics, current_lr)
        marker = " best" if is_best else ""
        print(
            f"Epoch {epoch:3d} | train {train_metrics.accuracy:6.2f}% "
            f"loss {train_metrics.loss:.4f} | validation {validation_metrics.accuracy:6.2f}% "
            f"loss {validation_metrics.loss:.4f} | lr {current_lr:.6f}{marker}"
        )
        if progress.epochs_without_improvement >= settings.early_stopping_patience:
            print(f"Early stopping after {progress.epochs_without_improvement} unimproved epochs.")
            break

    if not BEST_CNN_CHECKPOINT.is_file():
        raise RuntimeError("Training finished without producing a best CNN checkpoint.")
    best_checkpoint = torch.load(BEST_CNN_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    test_metrics = evaluate_model(
        model,
        data.test_loader,
        loss_function,
        device,
        phase="Final test",
        progress_interval=settings.progress_interval_batches,
    )
    final_metadata = build_training_metadata(
        settings, data, spec, progress, test_accuracy=test_metrics.accuracy
    )
    best_checkpoint["metadata"] = final_metadata
    save_checkpoint_atomic(BEST_CNN_CHECKPOINT, best_checkpoint)
    save_json_atomic(CNN_CONFIG_PATH, final_metadata)
    if LATEST_CNN_CHECKPOINT.is_file():
        latest_checkpoint = torch.load(LATEST_CNN_CHECKPOINT, map_location="cpu", weights_only=True)
        latest_checkpoint["metadata"] = final_metadata
        save_checkpoint_atomic(LATEST_CNN_CHECKPOINT, latest_checkpoint)
    print(f"Best validation accuracy: {progress.best_validation_accuracy:.2f}%")
    print(f"Independent test accuracy: {test_metrics.accuracy:.2f}%")
    return final_metadata
