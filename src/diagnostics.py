"""Separate CNN sanity checks for training, validation, test, and external data."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from data_splits import load_or_create_split_manifest, split_indices
from inference import InferenceEngine
from preprocessing import CharacterTransform
from project_paths import DATA_DIR, SPLIT_MANIFEST_PATH


@dataclass(frozen=True)
class EvaluationFailure:
    image_path: Path
    correct_character: str
    predicted_character: str
    confidence: float


@dataclass(frozen=True)
class EvaluationResult:
    split_name: str
    correct: int
    total: int
    mean_confidence: float
    failures: tuple[EvaluationFailure, ...]

    @property
    def accuracy(self) -> float | None:
        return None if self.total == 0 else 100.0 * self.correct / self.total


def load_diagnostic_dataset(engine: InferenceEngine):
    dataset = ImageFolder(
        DATA_DIR,
        transform=CharacterTransform(engine.bundle.preprocessing),
    )
    if dataset.class_to_idx != engine.bundle.metadata["class_to_idx"]:
        raise RuntimeError("Dataset class mapping differs from the loaded CNN checkpoint.")
    manifest = load_or_create_split_manifest(dataset, DATA_DIR, SPLIT_MANIFEST_PATH)
    return dataset, manifest, split_indices(dataset, DATA_DIR, manifest)


def evaluate_indices(
    engine: InferenceEngine,
    dataset: ImageFolder,
    indices: list[int],
    split_name: str,
    *,
    limit: int | None = None,
    seed: int = 42,
    failure_limit: int = 50,
) -> EvaluationResult:
    selected_indices = list(indices)
    if limit is not None and limit < len(selected_indices):
        selected_indices = random.Random(seed).sample(selected_indices, limit)
    loader = DataLoader(Subset(dataset, selected_indices), batch_size=256, shuffle=False)
    index_cursor = 0
    correct = 0
    confidence_sum = 0.0
    failures = []
    engine.bundle.model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = engine.bundle.model(images)
            probabilities = torch.softmax(logits, dim=1)
            if not torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(probabilities.shape[0]),
                atol=1e-5,
            ):
                raise RuntimeError("A diagnostic Softmax row does not sum to one.")
            batch_confidences, predictions = probabilities.max(dim=1)
            correct += int((predictions == labels).sum().item())
            confidence_sum += float(batch_confidences.sum().item())
            for offset, (label, prediction, confidence) in enumerate(
                zip(labels.tolist(), predictions.tolist(), batch_confidences.tolist())
            ):
                if label == prediction or len(failures) >= failure_limit:
                    continue
                dataset_index = selected_indices[index_cursor + offset]
                failures.append(
                    EvaluationFailure(
                        image_path=Path(dataset.samples[dataset_index][0]),
                        correct_character=dataset.classes[label],
                        predicted_character=engine.bundle.idx_to_class[prediction],
                        confidence=confidence,
                    )
                )
            index_cursor += len(labels)
    total = len(selected_indices)
    return EvaluationResult(
        split_name=split_name,
        correct=correct,
        total=total,
        mean_confidence=0.0 if total == 0 else confidence_sum / total,
        failures=tuple(failures),
    )


def evaluate_external_directory(
    engine: InferenceEngine,
    external_dir: Path,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> EvaluationResult:
    supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [
        path
        for path in Path(external_dir).rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_suffixes
        and path.parent.name in engine.bundle.metadata["class_to_idx"]
    ]
    paths.sort()
    if limit is not None and limit < len(paths):
        paths = random.Random(seed).sample(paths, limit)
    correct = 0
    confidence_sum = 0.0
    failures = []
    for path in paths:
        prediction = engine.predict_path(path)
        label = path.parent.name
        confidence_sum += prediction.confidence
        if prediction.predicted_character == label:
            correct += 1
        elif len(failures) < 50:
            failures.append(
                EvaluationFailure(
                    image_path=path,
                    correct_character=label,
                    predicted_character=prediction.predicted_character,
                    confidence=prediction.confidence,
                )
            )
    return EvaluationResult(
        split_name="external",
        correct=correct,
        total=len(paths),
        mean_confidence=0.0 if not paths else confidence_sum / len(paths),
        failures=tuple(failures),
    )


def _print_result(result: EvaluationResult) -> None:
    if result.accuracy is None:
        print(f"{result.split_name.title():<20} Accuracy: N/A (no labeled samples supplied)")
        return
    print(
        f"{result.split_name.title():<20} Accuracy: {result.accuracy:6.2f}% "
        f"({result.correct}/{result.total}), mean confidence {result.mean_confidence:.2%}"
    )
    for failure in result.failures[:10]:
        print(
            f"  FAIL {failure.image_path.name}: correct {failure.correct_character}, "
            f"predicted {failure.predicted_character}, confidence {failure.confidence:.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation", "test"),
        default="all",
    )
    parser.add_argument("--limit", type=int, help="Random samples per selected split")
    parser.add_argument(
        "--external-dir",
        type=Path,
        help="Optional labeled external tree: external_dir/<character>/<image>",
    )
    arguments = parser.parse_args()

    engine = InferenceEngine.from_artifacts()
    print(engine.startup_summary())
    dataset, _, indices = load_diagnostic_dataset(engine)
    requested = ("train", "validation", "test") if arguments.split == "all" else (arguments.split,)
    for split_name in requested:
        _print_result(
            evaluate_indices(
                engine,
                dataset,
                indices[split_name],
                split_name,
                limit=arguments.limit,
            )
        )
    external = (
        evaluate_external_directory(engine, arguments.external_dir, limit=arguments.limit)
        if arguments.external_dir
        else EvaluationResult("external", 0, 0, 0.0, ())
    )
    _print_result(external)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
