"""Deterministic, group-aware train/validation/test split manifests."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_STRATEGY = "stratified_provenance_and_duplicate_groups_v1"
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
