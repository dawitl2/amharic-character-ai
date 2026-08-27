"""Train or resume the active CharacterCNN model."""

import argparse
import sys

from training import TrainingSettings, run_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = TrainingSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=defaults.max_epochs,
        help="Maximum cumulative epoch, including resumed epochs (default: 200)",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.num_workers,
        help="Background image-loading workers (default: 0; try 2 for CPU training)",
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--scheduler-patience", type=int, default=defaults.scheduler_patience
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=defaults.early_stopping_patience
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Archive active checkpoints, metrics, and split data before training",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    arguments = parse_args()
    settings = TrainingSettings(
        max_epochs=arguments.max_epochs,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        learning_rate=arguments.learning_rate,
        scheduler_patience=arguments.scheduler_patience,
        early_stopping_patience=arguments.early_stopping_patience,
        fresh_start=arguments.fresh,
    )
    run_training(settings)
