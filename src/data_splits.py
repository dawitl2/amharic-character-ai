"""Deterministic, group-aware train/validation/test split manifests."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from checkpoints import save_json_atomic


SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_STRATEGY = "stratified_provenance_and_duplicate_groups_v2"
_VARIANT_SUFFIX = re.compile(
    r"(?i)(?:[_-](?:aug(?:mentation)?|variant|copy|rotated|shifted|blurred)[_-]?\d+)$"
)


def canonical_content_digest(image_path: Path) -> str:
    """Hash normalized pixels so identical 64/128 renderings stay together."""
    with Image.open(image_path) as image:
        normalized = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
        pixels = np.asarray(normalized, dtype=np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def sample_group_id(image_path: Path, class_name: str) -> str:
    """Use explicit augmentation provenance when present, otherwise exact content."""
    base_stem = _VARIANT_SUFFIX.sub("", image_path.stem)
    if base_stem != image_path.stem:
        return f"{class_name}:source:{base_stem}"
    return f"{class_name}:pixels:{canonical_content_digest(image_path)}"


def _dataset_records(dataset, data_root: Path) -> list[dict[str, Any]]:
    records = []
    for index, (raw_path, class_index) in enumerate(dataset.samples):
        path = Path(raw_path).resolve()
        relative_path = path.relative_to(data_root.resolve()).as_posix()
        records.append(
            {
                "index": index,
                "path": relative_path,
                "class_index": int(class_index),
                "class_name": dataset.classes[int(class_index)],
                "group": sample_group_id(path, dataset.classes[int(class_index)]),
            }
        )
    return records


def _dataset_signature(records: list[dict[str, Any]], class_to_idx: dict[str, int]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(class_to_idx, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(
            f"\n{record['path']}:{record['class_index']}:{record['group']}".encode("utf-8")
        )
    return digest.hexdigest()


def _inventory_signature(dataset, data_root: Path) -> str:
    """Quickly detect replaced files before reusing an expensive split manifest."""
    root = data_root.resolve()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(dataset.class_to_idx, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    for raw_path, class_index in sorted(dataset.samples):
        path = Path(raw_path).resolve()
        stat = path.stat()
        relative_path = path.relative_to(root).as_posix()
        digest.update(
            f"\n{relative_path}:{int(class_index)}:{stat.st_size}:{stat.st_mtime_ns}".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def create_split_manifest(dataset, data_root: Path, *, seed: int = SPLIT_SEED) -> dict[str, Any]:
    records = _dataset_records(dataset, data_root)
    grouped_by_class: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped_by_class[record["class_index"]][record["group"]].append(record)

    split_paths = {name: [] for name in SPLIT_RATIOS}
    for class_index, groups in sorted(grouped_by_class.items()):
        group_items = list(groups.items())
        class_seed = seed + int(hashlib.sha256(str(class_index).encode()).hexdigest()[:8], 16)
        random.Random(class_seed).shuffle(group_items)
        class_total = sum(len(group) for _, group in group_items)
        targets = {name: class_total * ratio for name, ratio in SPLIT_RATIOS.items()}
        counts = {name: 0 for name in SPLIT_RATIOS}

        for _, group_records in group_items:
            destination = min(
                SPLIT_RATIOS,
                key=lambda name: (counts[name] / max(targets[name], 1), list(SPLIT_RATIOS).index(name)),
            )
            split_paths[destination].extend(record["path"] for record in group_records)
            counts[destination] += len(group_records)

    for paths in split_paths.values():
        paths.sort()
    duplicate_group_count = sum(
        1 for groups in grouped_by_class.values() for records_in_group in groups.values()
        if len(records_in_group) > 1
    )
    return {
        "format_version": 1,
        "strategy": SPLIT_STRATEGY,
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "class_to_idx": dataset.class_to_idx,
        "dataset_signature": _dataset_signature(records, dataset.class_to_idx),
        "inventory_signature": _inventory_signature(dataset, data_root),
        "dataset_total": len(records),
        "counts": {name: len(paths) for name, paths in split_paths.items()},
        "duplicate_or_variant_group_count": duplicate_group_count,
        "splits": split_paths,
    }


def load_or_create_split_manifest(dataset, data_root: Path, manifest_path: Path) -> dict[str, Any]:
    if Path(manifest_path).is_file():
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        current_paths = {
            Path(path).resolve().relative_to(Path(data_root).resolve()).as_posix()
            for path, _ in dataset.samples
        }
        saved_paths = {
            path for split_paths in manifest.get("splits", {}).values() for path in split_paths
        }
        if (
            manifest.get("strategy") == SPLIT_STRATEGY
            and manifest.get("class_to_idx") == dataset.class_to_idx
            and current_paths == saved_paths
            and manifest.get("inventory_signature")
            == _inventory_signature(dataset, data_root)
        ):
            return manifest

    manifest = create_split_manifest(dataset, data_root)
    save_json_atomic(manifest_path, manifest)
    return manifest


def split_indices(dataset, data_root: Path, manifest: dict[str, Any]) -> dict[str, list[int]]:
    index_by_path = {
        Path(path).resolve().relative_to(Path(data_root).resolve()).as_posix(): index
        for index, (path, _) in enumerate(dataset.samples)
    }
    result = {}
    for split_name, paths in manifest["splits"].items():
        try:
            result[split_name] = [index_by_path[path] for path in paths]
        except KeyError as error:
            raise ValueError(f"Split manifest references a missing dataset image: {error.args[0]}") from error
    all_indices = [index for indices in result.values() for index in indices]
    if len(all_indices) != len(dataset) or len(set(all_indices)) != len(dataset):
        raise ValueError("Split manifest must contain every dataset image exactly once.")
    return result
