"""Single inference entry point shared by the CLI, diagnostics, and GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from checkpoints import ModelBundle, load_cnn_bundle
from preprocessing import preprocess_image
from project_paths import BEST_CNN_CHECKPOINT, CNN_CONFIG_PATH, DATA_DIR


@dataclass(frozen=True)
class RankedPrediction:
    character: str
    probability: float
    class_index: int


@dataclass(frozen=True)
class Prediction:
    predicted_character: str
    confidence: float
    probabilities: tuple[float, ...]
    logits: tuple[float, ...]
    top_predictions: tuple[RankedPrediction, ...]
    probability_sum: float
    tensor_shape: tuple[int, ...]


class InferenceEngine:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    @classmethod
    def from_artifacts(
        cls,
        config_path: Path = CNN_CONFIG_PATH,
        checkpoint_path: Path = BEST_CNN_CHECKPOINT,
    ) -> "InferenceEngine":
        return cls(load_cnn_bundle(config_path, checkpoint_path))

    def startup_summary(self) -> str:
        metadata = self.bundle.metadata
        classes = ", ".join(self.bundle.idx_to_class.values())
        accuracy = metadata.get("best_validation_accuracy")
        accuracy_text = "N/A" if accuracy is None else f"{accuracy:.2f}%"
        return "\n".join(
            (
                f"Architecture: {metadata['architecture']}",
                f"Checkpoint: {self.bundle.checkpoint_path}",
                f"Classes: {classes}",
                f"Best validation accuracy: {accuracy_text}",
            )
        )

    def predict_image(
        self,
        image: Image.Image,
        *,
        prepare_external: bool,
        top_k: int = 3,
    ) -> Prediction:
        tensor = preprocess_image(
            image,
            self.bundle.preprocessing,
            prepare_external=prepare_external,
        ).unsqueeze(0)
        if tensor.shape != (
            1,
            1,
            self.bundle.preprocessing.height,
            self.bundle.preprocessing.width,
        ):
            raise RuntimeError(f"Unexpected CNN tensor shape: {tuple(tensor.shape)}")

        self.bundle.model.eval()
        with torch.inference_mode():
            logits = self.bundle.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
        probability_sum = float(probabilities[0].sum().item())
        if abs(probability_sum - 1.0) > 1e-5:
            raise RuntimeError(f"Softmax probabilities sum to {probability_sum}, expected 1.0.")

        count = min(max(1, top_k), probabilities.shape[1])
        top_probabilities, top_indices = probabilities[0].topk(count)
        ranked = tuple(
            RankedPrediction(
                character=self.bundle.idx_to_class[int(index)],
                probability=float(probability),
                class_index=int(index),
            )
            for probability, index in zip(top_probabilities.tolist(), top_indices.tolist())
        )
        return Prediction(
            predicted_character=ranked[0].character,
            confidence=ranked[0].probability,
            probabilities=tuple(float(value) for value in probabilities[0].tolist()),
            logits=tuple(float(value) for value in logits[0].tolist()),
            top_predictions=ranked,
            probability_sum=probability_sum,
            tensor_shape=tuple(tensor.shape),
        )

    def predict_path(self, image_path: Path, *, top_k: int = 3) -> Prediction:
        path = Path(image_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        prepare_external = not path.is_relative_to(DATA_DIR.resolve())
        with Image.open(path) as image:
            return self.predict_image(image, prepare_external=prepare_external, top_k=top_k)


def dataset_label_for_path(image_path: Path) -> str | None:
    path = Path(image_path).resolve()
    try:
        relative_path = path.relative_to(DATA_DIR.resolve())
    except ValueError:
        return None
    if len(relative_path.parts) < 2:
        return None
    return relative_path.parts[0]
