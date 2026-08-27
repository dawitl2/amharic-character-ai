"""Isolated artifact paths used only by the preserved linear lessons."""

from project_paths import MODEL_DIR


LEGACY_LINEAR_DIR = MODEL_DIR / "legacy_linear"
LEGACY_LINEAR_WEIGHTS = LEGACY_LINEAR_DIR / "linear_model_weights.pth"
LEGACY_LINEAR_CONFIG = LEGACY_LINEAR_DIR / "linear_model_config.json"
