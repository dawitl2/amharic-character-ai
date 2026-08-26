"""Migrate a verified historical CNN state dict to the strict active format."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointCompatibilityError,
    save_checkpoint_atomic,
    save_json_atomic,
)
from cnn_model import ARCHITECTURE_NAME, CharacterCNN
from preprocessing import PreprocessingSpec
from project_paths import (
    BEST_CNN_CHECKPOINT,
    CNN_CONFIG_PATH,
    HISTORICAL_BEST_WEIGHTS,
    HISTORICAL_CONFIG_PATH,
)


def build_metadata(config: dict, source_weights: Path) -> dict:
    class_to_idx = config.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise CheckpointCompatibilityError("Historical configuration has no class mapping.")

    spec = PreprocessingSpec.from_metadata(config)
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_to_idx),
        "preprocessing": spec.to_metadata(),
        "best_validation_accuracy": config.get(
            "best_validation_accuracy", config.get("best_val_accuracy")
        ),
        "test_accuracy": config.get("test_accuracy"),
        "epoch_of_best_checkpoint": config.get("epoch_of_best_checkpoint", config.get("best_epoch")),
        "cumulative_epochs_trained": config.get(
            "cumulative_epochs_trained", config.get("epochs_trained", 0)
        ),
        "optimizer": {
            "name": config.get("optimizer", "SGD"),
            "learning_rate": config.get("learning_rate", 0.01),
        },
        "batch_size": config.get("batch_size", 64),
        "scheduler": config.get("scheduler"),
        "early_stopping": config.get("early_stopping"),
        "dataset": {
            "total": config.get("dataset_total"),
            "train": config.get("train_count"),
            "validation": config.get("val_count"),
            "test": config.get("test_count"),
        },
        "split": {
            "strategy": config.get("split_strategy", "historical_random_stratified"),
            "seed": config.get("split_seed", 42),
            "note": "Migrated metrics describe the historical per-image random split.",
        },
        "migration": {"source_weights": str(source_weights.resolve())},
    }


def migrate_historical_best(
    config_path: Path = HISTORICAL_CONFIG_PATH,
    weights_path: Path = HISTORICAL_BEST_WEIGHTS,
    output_config_path: Path = CNN_CONFIG_PATH,
    output_checkpoint_path: Path = BEST_CNN_CHECKPOINT,
) -> tuple[Path, Path]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or "conv1.weight" not in state_dict:
        raise CheckpointCompatibilityError(
            f"{weights_path} is not a CharacterCNN state dict; migration stopped."
        )

    metadata = build_metadata(config, Path(weights_path))
    model = CharacterCNN(num_classes=len(metadata["class_to_idx"]))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(
            f"Historical weights do not match the declared class mapping: {error}"
        ) from error

    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": state_dict,
        "metadata": metadata,
    }
    save_checkpoint_atomic(output_checkpoint_path, payload)
    save_json_atomic(output_config_path, metadata)
    return Path(output_config_path), Path(output_checkpoint_path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    config_path, checkpoint_path = migrate_historical_best()
    print(f"Architecture: {ARCHITECTURE_NAME}")
    print(f"Configuration: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
