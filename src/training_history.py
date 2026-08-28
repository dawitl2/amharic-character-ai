"""Load and visualize real per-epoch CNN training history."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

from project_paths import METRICS_CSV_PATH


@dataclass(frozen=True)
class TrainingEpoch:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    learning_rate: float


def load_training_history(path: Path = METRICS_CSV_PATH) -> tuple[TrainingEpoch, ...]:
    path = Path(path)
    if not path.is_file():
        return ()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(
            TrainingEpoch(
                epoch=int(row["epoch"]),
                train_loss=float(row["train_loss"]),
                train_accuracy=float(row["train_accuracy"]),
                validation_loss=float(row["validation_loss"]),
                validation_accuracy=float(row["validation_accuracy"]),
                learning_rate=float(row["learning_rate"]),
            )
            for row in csv.DictReader(handle)
        )
    return rows


def build_training_figure(
    history: tuple[TrainingEpoch, ...], *, best_epoch: int | None
) -> Figure:
    figure = Figure(figsize=(10.5, 6.6), dpi=100, facecolor="#F7F9FC")
    accuracy_axis, loss_axis, learning_rate_axis = figure.subplots(3, 1)
    if not history:
        accuracy_axis.text(
            0.5,
            0.5,
            "No saved training history is available.",
            ha="center",
            va="center",
        )
        loss_axis.set_visible(False)
        learning_rate_axis.set_visible(False)
        return figure

    epochs = [row.epoch for row in history]
    accuracy_axis.plot(
        epochs,
        [row.train_accuracy for row in history],
        label="Training",
        color="#246BFD",
        linewidth=2,
    )
    accuracy_axis.plot(
        epochs,
        [row.validation_accuracy for row in history],
        label="Validation",
        color="#16A34A",
        linewidth=2,
    )
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.legend(loc="lower right", frameon=False)

    loss_axis.plot(
        epochs,
        [row.train_loss for row in history],
        label="Training",
        color="#246BFD",
        linewidth=2,
    )
    loss_axis.plot(
        epochs,
        [row.validation_loss for row in history],
        label="Validation",
        color="#F59E0B",
        linewidth=2,
    )
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.legend(loc="upper right", frameon=False)

    learning_rate_axis.plot(
        epochs,
        [row.learning_rate for row in history],
        color="#7C3AED",
        linewidth=2,
    )
    learning_rate_axis.set_ylabel("Learning rate")
    learning_rate_axis.set_xlabel("Epoch")

    for axis in (accuracy_axis, loss_axis, learning_rate_axis):
        axis.grid(alpha=0.18)
        axis.set_facecolor("#FFFFFF")
        axis.spines[["top", "right"]].set_visible(False)
        if best_epoch is not None:
            axis.axvline(
                best_epoch,
                color="#DC2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
            )
    accuracy_axis.set_title(
        "Saved CNN learning history"
        + (f" — best checkpoint epoch {best_epoch}" if best_epoch else ""),
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(pad=1.7)
    return figure
