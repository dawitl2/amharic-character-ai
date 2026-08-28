"""Model, pipeline, and training-history pages for the desktop app."""

from __future__ import annotations

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui_theme import (
    BACKGROUND,
    BORDER,
    MUTED,
    PRIMARY,
    SOFT_BLUE,
    SURFACE,
    SURFACE_MUTED,
    TEXT,
    card,
    muted_label,
    section_title,
)
from training_history import build_training_figure, load_training_history


def _accuracy(value) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%"


class ModelInfoPage(ctk.CTkFrame):
    def __init__(self, parent, bundle):
        super().__init__(parent, fg_color=BACKGROUND)
        metadata = bundle.metadata
        parameter_count = sum(
            parameter.numel() for parameter in bundle.model.parameters()
        )
        dataset = metadata.get("dataset", {})
        preprocessing = metadata.get("preprocessing", {})

        ctk.CTkLabel(
            self,
            text="Model Information",
            font=("Segoe UI", 24, "bold"),
            text_color=TEXT,
            anchor="w",
        ).pack(fill="x", padx=22, pady=(20, 4))
        muted_label(
            self,
            "Values below come from the active checkpoint and configuration.",
        ).pack(fill="x", padx=22, pady=(0, 14))

        metrics = ctk.CTkFrame(self, fg_color="transparent")
        metrics.pack(fill="x", padx=16)
        items = (
            ("Classes", len(metadata.get("class_to_idx", {}))),
            ("Parameters", f"{parameter_count:,}"),
            ("Best validation", _accuracy(metadata.get("best_validation_accuracy"))),
            ("Independent test", _accuracy(metadata.get("test_accuracy"))),
            ("Best epoch", metadata.get("epoch_of_best_checkpoint", "N/A")),
        )
        for label, value in items:
            metric = card(metrics, width=190, height=90)
            metric.pack(side="left", fill="x", expand=True, padx=6)
            metric.pack_propagate(False)
            muted_label(metric, label).pack(anchor="w", padx=14, pady=(14, 3))
            ctk.CTkLabel(
                metric,
                text=str(value),
                font=("Segoe UI", 19, "bold"),
                text_color=TEXT,
            ).pack(anchor="w", padx=14)

        details = card(self)
        details.pack(fill="x", padx=22, pady=18)
        section_title(details, "Architecture and training contract").pack(
            fill="x", padx=18, pady=(15, 8)
        )
        rows = (
            ("Architecture", metadata.get("architecture", "Unknown")),
            ("Checkpoint", bundle.checkpoint_path.resolve()),
            (
                "Input",
                f"{preprocessing.get('width', 64)} × {preprocessing.get('height', 64)} grayscale",
            ),
            ("Tensor shape", "[batch, 1, 64, 64]"),
            ("Normalization", "identity: mean 0.0, standard deviation 1.0"),
            ("Optimizer", metadata.get("optimizer", {}).get("name", "Unknown")),
            (
                "Base learning rate",
                metadata.get("optimizer", {}).get("learning_rate", "Unknown"),
            ),
            ("Batch size", metadata.get("batch_size", "Unknown")),
            (
                "Scheduler",
                (metadata.get("scheduler") or {}).get("name", "None"),
            ),
            (
                "Epochs trained",
                metadata.get("cumulative_epochs_trained", "Unknown"),
            ),
            ("Dataset images", f"{dataset.get('total', 'Unknown'):,}" if isinstance(dataset.get('total'), int) else "Unknown"),
            ("Training images", f"{dataset.get('train', 'Unknown'):,}" if isinstance(dataset.get('train'), int) else "Unknown"),
            ("Validation images", f"{dataset.get('validation', 'Unknown'):,}" if isinstance(dataset.get('validation'), int) else "Unknown"),
            ("Test images", f"{dataset.get('test', 'Unknown'):,}" if isinstance(dataset.get('test'), int) else "Unknown"),
        )
        for label, value in rows:
            row = ctk.CTkFrame(details, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=4)
            ctk.CTkLabel(
                row,
                text=label,
                width=220,
                anchor="w",
                font=("Segoe UI", 12, "bold"),
                text_color=TEXT,
            ).pack(side="left")
            ctk.CTkLabel(
                row,
                text=str(value),
                anchor="w",
                font=("Segoe UI", 12),
                text_color=MUTED,
            ).pack(side="left", fill="x", expand=True)
        muted_label(
            details,
            "The translation provider is external. It does not recognize images and is not part of the CNN.",
            wraplength=900,
            justify="left",
        ).pack(fill="x", padx=18, pady=(12, 16))


class PipelinePage(ctk.CTkFrame):
    FLOWS = {
        "Character": (
            ("IMAGE", "One printed glyph"),
            ("PREPROCESS", "Crop • center • 64×64 grayscale"),
            ("CNN", "Convolutional features"),
            ("LOGITS", "290 raw class scores"),
            ("SOFTMAX", "Probabilities sum to 1"),
            ("CHARACTER", "Decoded with class mapping"),
        ),
        "Word": (
            ("WORD IMAGE", "Printed Ethiopic word"),
            ("OPENCV", "Threshold and component grouping"),
            ("CROPS", "Left-to-right character regions"),
            ("CNN", "Our trained character model"),
            ("SEQUENCE", "Characters reconstructed"),
            ("TRANSLATE", "Optional external text step"),
        ),
        "Sentence": (
            ("LINE IMAGE", "Printed line or sentence"),
            ("OPENCV", "Lines, gaps, words, characters"),
            ("READING ORDER", "Top-to-bottom, left-to-right"),
            ("CNN", "Each crop classified locally"),
            ("AMHARIC TEXT", "Words and spaces restored"),
            ("ENGLISH", "Optional translation request"),
        ),
    }

    def __init__(self, parent):
        super().__init__(parent, fg_color=BACKGROUND)
        ctk.CTkLabel(
            self,
            text="How the OCR System Works",
            font=("Segoe UI", 24, "bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=22, pady=(20, 4))
        muted_label(
            self,
            "OpenCV locates structure. The CNN supplies recognition intelligence.",
        ).pack(anchor="w", padx=22, pady=(0, 14))
        tabs = ctk.CTkTabview(
            self,
            fg_color=SURFACE,
            border_color=BORDER,
            border_width=1,
            segmented_button_selected_color=PRIMARY,
            segmented_button_unselected_color=SOFT_BLUE,
            text_color=TEXT,
        )
        tabs.pack(fill="both", expand=True, padx=22, pady=(0, 22))
        for flow_name, steps in self.FLOWS.items():
            tab = tabs.add(flow_name)
            self._build_flow(tab, steps)

    @staticmethod
    def _build_flow(parent, steps) -> None:
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=28, pady=28)
        container.grid_columnconfigure((0, 1, 2), weight=1, uniform="flow")
        container.grid_rowconfigure((0, 1), weight=1, uniform="flow")
        for index, (title, description) in enumerate(steps):
            step = ctk.CTkFrame(
                container,
                fg_color=SURFACE_MUTED,
                border_color=BORDER,
                border_width=1,
                corner_radius=9,
            )
            step.grid(
                row=index // 3,
                column=index % 3,
                sticky="nsew",
                padx=10,
                pady=10,
            )
            ctk.CTkLabel(
                step,
                text=str(index + 1),
                width=42,
                height=42,
                corner_radius=21,
                fg_color=SOFT_BLUE,
                text_color=PRIMARY,
                font=("Segoe UI", 15, "bold"),
            ).pack(anchor="w", padx=18, pady=(18, 8))
            copy = ctk.CTkFrame(step, fg_color="transparent")
            copy.pack(fill="both", expand=True, padx=18, pady=(0, 18))
            ctk.CTkLabel(
                copy,
                text=title,
                font=("Segoe UI", 13, "bold"),
                text_color=TEXT,
            ).pack(anchor="w")
            muted_label(copy, description, wraplength=340, justify="left").pack(
                anchor="w", pady=(5, 0)
            )


class TrainingGraphsPage(ctk.CTkFrame):
    def __init__(self, parent, metadata):
        super().__init__(parent, fg_color=BACKGROUND)
        ctk.CTkLabel(
            self,
            text="Training History",
            font=("Segoe UI", 24, "bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=22, pady=(20, 4))
        history = load_training_history()
        muted_label(
            self,
            f"Showing {len(history)} saved epoch(s). No values are inferred or fabricated.",
        ).pack(anchor="w", padx=22, pady=(0, 12))
        graph_card = card(self)
        graph_card.pack(fill="both", expand=True, padx=22, pady=(0, 22))
        figure = build_training_figure(
            history, best_epoch=metadata.get("epoch_of_best_checkpoint")
        )
        self.canvas = FigureCanvasTkAgg(figure, master=graph_card)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
