"""Strict, self-describing checkpoint handling for the active CNN."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cnn_model import ARCHITECTURE_NAME, INPUT_CHANNELS, CharacterCNN
from preprocessing import PreprocessingSpec


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointCompatibilityError(RuntimeError):
    """Raised when model artifacts do not describe the active CNN exactly."""


@dataclass(frozen=True)
class ModelBundle:
    model: CharacterCNN
    metadata: dict[str, Any]
    preprocessing: PreprocessingSpec
    checkpoint_path: Path

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {index: label for label, index in self.metadata["class_to_idx"].items()}


def validate_cnn_metadata(metadata: dict[str, Any]) -> None:
    architecture = metadata.get("architecture")
    if architecture != ARCHITECTURE_NAME:
        raise CheckpointCompatibilityError(
            f"Expected architecture '{ARCHITECTURE_NAME}', found {architecture!r}. "
            "Legacy linear and ambiguous checkpoints are not valid CNN artifacts."
        )

    class_to_idx = metadata.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise CheckpointCompatibilityError("Checkpoint metadata has no class_to_idx mapping.")
    indices = sorted(class_to_idx.values())
    if indices != list(range(len(indices))):
        raise CheckpointCompatibilityError("Class indices must be unique and contiguous from zero.")
    if int(metadata.get("num_classes", len(indices))) != len(indices):
        raise CheckpointCompatibilityError("num_classes does not match class_to_idx.")

    spec = PreprocessingSpec.from_metadata(metadata)
    if spec.channels != INPUT_CHANNELS:
        raise CheckpointCompatibilityError(
            f"CharacterCNN expects {INPUT_CHANNELS} input channel, metadata declares {spec.channels}."
        )
    if (spec.width, spec.height) != (64, 64):
        raise CheckpointCompatibilityError(
            f"CharacterCNN expects 64 x 64 input, metadata declares {spec.width} x {spec.height}."
        )


def _validate_state_dict(state_dict: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    if "classifier.weight" not in state_dict or "conv1.weight" not in state_dict:
        raise CheckpointCompatibilityError(
            "Checkpoint tensor names do not match CharacterCNN; this may be a legacy linear model."
        )
    output_classes = int(state_dict["classifier.weight"].shape[0])
    expected_classes = len(metadata["class_to_idx"])
    if output_classes != expected_classes:
        raise CheckpointCompatibilityError(
            f"Checkpoint outputs {output_classes} classes but metadata declares {expected_classes}."
        )
    input_channels = int(state_dict["conv1.weight"].shape[1])
    if input_channels != INPUT_CHANNELS:
        raise CheckpointCompatibilityError(
            f"Checkpoint expects {input_channels} channels; CharacterCNN expects {INPUT_CHANNELS}."
        )


def load_cnn_bundle(config_path: Path, checkpoint_path: Path) -> ModelBundle:
    """Load the active CNN or fail before any inference can occur."""
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"CNN configuration not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_cnn_metadata(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise CheckpointCompatibilityError(
            "Active CNN checkpoints must be self-describing dictionaries, not bare weight files."
        )
    checkpoint_metadata = checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, dict):
        raise CheckpointCompatibilityError("Checkpoint is missing its metadata block.")
    validate_cnn_metadata(checkpoint_metadata)
    if checkpoint_metadata["class_to_idx"] != config["class_to_idx"]:
        raise CheckpointCompatibilityError(
            "Checkpoint and configuration class mappings differ; refusing unsafe label decoding."
        )
    if PreprocessingSpec.from_metadata(checkpoint_metadata) != PreprocessingSpec.from_metadata(config):
        raise CheckpointCompatibilityError(
            "Checkpoint and configuration preprocessing specifications differ."
        )

    state_dict = checkpoint["model_state_dict"]
    _validate_state_dict(state_dict, checkpoint_metadata)
    model = CharacterCNN(num_classes=len(config["class_to_idx"]))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"CharacterCNN tensor shapes are incompatible: {error}") from error
    model.eval()
    return ModelBundle(
        model=model,
        metadata=config,
        preprocessing=PreprocessingSpec.from_metadata(config),
        checkpoint_path=checkpoint_path,
    )


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
