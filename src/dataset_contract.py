"""Validation for the canonical Ethiopic training dataset."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from project_paths import CHARACTERS_MANIFEST_PATH


@dataclass(frozen=True)
class DatasetReport:
    class_count: int
    image_count: int
    minimum_images_per_class: int
    maximum_images_per_class: int


def load_expected_characters(path: Path = CHARACTERS_MANIFEST_PATH) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        characters = json.load(handle)
    if not isinstance(characters, list) or not characters:
        raise ValueError(f"Character manifest must be a non-empty JSON list: {path}")
    if any(not isinstance(character, str) or len(character) != 1 for character in characters):
        raise ValueError("Every character manifest entry must be one Unicode character.")
    if len(set(characters)) != len(characters):
        raise ValueError("Character manifest contains duplicate characters.")
    unassigned = [character for character in characters if unicodedata.category(character) == "Cn"]
    if unassigned:
        codepoints = ", ".join(f"U+{ord(character):04X}" for character in unassigned)
        raise ValueError(f"Character manifest contains unassigned Unicode code points: {codepoints}")
    return characters


def validate_training_dataset(
    dataset,
    *,
    character_manifest_path: Path = CHARACTERS_MANIFEST_PATH,
    minimum_images_per_class: int = 3,
) -> DatasetReport:
    expected = load_expected_characters(character_manifest_path)
    actual = list(dataset.classes)
    if actual != expected:
        missing = [character for character in expected if character not in actual]
        unexpected = [character for character in actual if character not in expected]
        raise ValueError(
            "Dataset classes do not exactly match src/characters.json "
            f"(missing={len(missing)}, unexpected={len(unexpected)}, "
            f"order_matches={set(actual) == set(expected)})."
        )

    counts = Counter(int(class_index) for _, class_index in dataset.samples)
    too_small = [
        actual[index]
        for index in range(len(actual))
        if counts[index] < minimum_images_per_class
    ]
    if too_small:
        raise ValueError(
            f"Every class needs at least {minimum_images_per_class} images; "
            f"{len(too_small)} classes are below the minimum."
        )
    per_class = [counts[index] for index in range(len(actual))]
    return DatasetReport(
        class_count=len(actual),
        image_count=len(dataset),
        minimum_images_per_class=min(per_class),
        maximum_images_per_class=max(per_class),
    )
