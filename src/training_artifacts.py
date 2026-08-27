"""Lifecycle helpers for recoverable fresh training runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from project_paths import (
    BEST_CNN_CHECKPOINT,
    CNN_CONFIG_PATH,
    LATEST_CNN_CHECKPOINT,
    METRICS_CSV_PATH,
    MODEL_DIR,
    SPLIT_MANIFEST_PATH,
)


ACTIVE_TRAINING_ARTIFACTS = (
    BEST_CNN_CHECKPOINT,
    LATEST_CNN_CHECKPOINT,
    CNN_CONFIG_PATH,
    SPLIT_MANIFEST_PATH,
    METRICS_CSV_PATH,
)


def archive_active_training_artifacts(
    paths: Iterable[Path] = ACTIVE_TRAINING_ARTIFACTS,
    *,
    archive_root: Path = MODEL_DIR / "archive",
) -> Path | None:
    """Move existing active artifacts into one timestamped, recoverable directory."""
    existing = [Path(path) for path in paths if Path(path).is_file()]
    if not existing:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = Path(archive_root) / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    for source in existing:
        source.replace(destination / source.name)
    return destination
