"""Read-only dataset and model checks to run before a long training job."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from cnn_model import CharacterCNN
from data_splits import canonical_content_digest
from dataset_contract import validate_training_dataset
from project_paths import DATA_DIR


def select_paths(dataset: ImageFolder, images_per_class: int | None) -> list[tuple[Path, str]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for raw_path, class_index in dataset.samples:
        by_class[dataset.classes[int(class_index)]].append(Path(raw_path))
    selected = []
    for class_name in dataset.classes:
        paths = sorted(by_class[class_name])
        if images_per_class is not None and images_per_class < len(paths):
            paths = random.Random(42 + dataset.class_to_idx[class_name]).sample(
                paths, images_per_class
            )
        selected.extend((path, class_name) for path in paths)
    return selected


def inspect_image(item: tuple[Path, str]) -> tuple[Path, str, str]:
    path, class_name = item
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        extrema = image.convert("L").getextrema()
    if extrema[0] == extrema[1]:
        raise ValueError(f"Image has no visible intensity variation: {path}")
    return path, class_name, canonical_content_digest(path)


def verify_images(
    dataset: ImageFolder,
    *,
    images_per_class: int | None,
    workers: int = 8,
) -> int:
    selected = select_paths(dataset, images_per_class)
    digest_classes: dict[str, tuple[str, Path]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for path, class_name, digest in executor.map(inspect_image, selected):
            previous = digest_classes.get(digest)
            if previous is not None and previous[0] != class_name:
                raise ValueError(
                    "Identical normalized image content has conflicting labels: "
                    f"{previous[1]} ({previous[0]}) and {path} ({class_name})"
                )
            digest_classes[digest] = (class_name, path)
    return len(selected)


def verify_model_output(class_count: int) -> int:
    model = CharacterCNN(num_classes=class_count)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 1, 64, 64))
    if tuple(output.shape) != (2, class_count):
        raise RuntimeError(f"CNN produced {tuple(output.shape)}, expected (2, {class_count}).")
    return sum(parameter.numel() for parameter in model.parameters())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Decode and fingerprint every image instead of a deterministic sample",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=20,
        help="Images checked per class unless --all-images is supplied (default: 20)",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    dataset = ImageFolder(DATA_DIR)
    report = validate_training_dataset(dataset)
    checked = verify_images(
        dataset,
        images_per_class=None if arguments.all_images else arguments.images_per_class,
        workers=arguments.workers,
    )
    parameter_count = verify_model_output(report.class_count)
    print(
        f"PASS dataset contract: {report.class_count} classes, {report.image_count} images, "
        f"{report.minimum_images_per_class}-{report.maximum_images_per_class} per class"
    )
    print(f"PASS image decoding/content check: {checked} images")
    print(f"PASS CNN output shape: {report.class_count} logits ({parameter_count:,} parameters)")
    print("No training was performed.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
