"""Train or resume the active CharacterCNN model."""

import argparse
import sys

from training import TrainingSettings, run_training


def parse_args() -> argparse.Namespace:
    defaults = TrainingSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=defaults.max_epochs,
        help="Maximum cumulative epoch, including resumed epochs (default: 200)",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--scheduler-patience", type=int, default=defaults.scheduler_patience
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=defaults.early_stopping_patience
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    arguments = parse_args()
    settings = TrainingSettings(
        max_epochs=arguments.max_epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        scheduler_patience=arguments.scheduler_patience,
        early_stopping_patience=arguments.early_stopping_patience,
    )
    run_training(settings)
