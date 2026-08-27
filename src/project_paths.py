"""Project paths resolved independently of the current working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHARACTERS_MANIFEST_PATH = PROJECT_ROOT / "src" / "characters.json"
MODEL_DIR = PROJECT_ROOT / "models"
CNN_CONFIG_PATH = MODEL_DIR / "cnn_model_config.json"
BEST_CNN_CHECKPOINT = MODEL_DIR / "best_cnn_model.pth"
LATEST_CNN_CHECKPOINT = MODEL_DIR / "latest_cnn_checkpoint.pth"
SPLIT_MANIFEST_PATH = MODEL_DIR / "cnn_data_split.json"
METRICS_CSV_PATH = MODEL_DIR / "cnn_training_metrics.csv"

# Read-only migration inputs. Active code never writes to these ambiguous names.
HISTORICAL_CONFIG_PATH = MODEL_DIR / "model_config.json"
HISTORICAL_BEST_WEIGHTS = MODEL_DIR / "best_model_weights.pth"
