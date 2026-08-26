"""Lightweight desktop interface for the active CharacterCNN model."""

from __future__ import annotations

import random
import queue
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from diagnostics import evaluate_indices, load_diagnostic_dataset
from inference import InferenceEngine, Prediction, dataset_label_for_path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

BACKGROUND = "#F5F7FA"
SURFACE = "#FFFFFF"
BORDER = "#D8DEE8"
TEXT = "#18212F"
MUTED = "#667085"
PRIMARY = "#2457A7"
PRIMARY_HOVER = "#19437F"
SUCCESS = "#157347"
DANGER = "#B42318"
SOFT_BLUE = "#EDF3FC"
SOFT_GREEN = "#ECF8F1"
SOFT_RED = "#FEF0EE"


class AmharicAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Ethiopic Character Recognizer")
        self.geometry("1240x780")
        self.minsize(1080, 700)
        self.configure(fg_color=BACKGROUND)

        try:
            self.engine = InferenceEngine.from_artifacts()
            self.dataset, self.split_manifest, self.split_indices = load_diagnostic_dataset(
                self.engine
            )
        except Exception as error:
            messagebox.showerror("CNN model could not be loaded", str(error))
            self.destroy()
            raise
        print(self.engine.startup_summary())

        self.preview_image = None
        self.sample_images = []
        self.current_path: Path | None = None
        self.current_prediction: Prediction | None = None
        self.inference_generation = 0
        self.worker_queue: queue.Queue = queue.Queue()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_workspace()
        self._build_status_bar()
        self._refresh_samples()
        self.after(50, self._poll_worker_queue)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=76)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        ctk.CTkLabel(
            header,
            text="Ethiopic Character Recognizer",
            font=("Segoe UI", 22, "bold"),
            text_color=TEXT,
        ).pack(side="left", padx=24, pady=18)
        ctk.CTkLabel(
            header,
            text="CharacterCNN  •  64 × 64 grayscale",
            font=("Segoe UI", 12),
            text_color=MUTED,
        ).pack(side="right", padx=24)

    def _build_workspace(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=18, pady=18)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0, minsize=270)
        body.grid_columnconfigure(1, weight=1, minsize=430)
        body.grid_columnconfigure(2, weight=0, minsize=300)

        self.left_panel = self._panel(body)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.center_panel = self._panel(body)
        self.center_panel.grid(row=0, column=1, sticky="nsew", padx=0)
        self.right_panel = self._panel(body)
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        self._build_source_panel()
        self._build_prediction_panel()
        self._build_model_panel()

    @staticmethod
    def _panel(parent) -> ctk.CTkFrame:
        return ctk.CTkFrame(
            parent,
            fg_color=SURFACE,
            corner_radius=5,
            border_width=1,
            border_color=BORDER,
        )

    @staticmethod
    def _section_title(parent, title: str, subtitle: str = "") -> None:
        ctk.CTkLabel(
            parent,
            text=title,
            font=("Segoe UI", 15, "bold"),
            text_color=TEXT,
            anchor="w",
        ).pack(fill="x")
        if subtitle:
            ctk.CTkLabel(
                parent,
                text=subtitle,
                font=("Segoe UI", 11),
                text_color=MUTED,
                anchor="w",
                justify="left",
                wraplength=235,
            ).pack(fill="x", pady=(2, 10))

    def _build_source_panel(self) -> None:
        content = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=16, pady=16)
        self._section_title(content, "Image source", "Choose a held-out sample or an external image.")

        self.split_var = ctk.StringVar(value="test")
        self.split_menu = ctk.CTkOptionMenu(
            content,
            values=["train", "validation", "test"],
            variable=self.split_var,
            command=lambda _: self._refresh_samples(),
            fg_color=SOFT_BLUE,
            button_color=PRIMARY,
            button_hover_color=PRIMARY_HOVER,
            text_color=TEXT,
            height=34,
        )
        self.split_menu.pack(fill="x", pady=(0, 8))
        ctk.CTkButton(
            content,
            text="Refresh labeled samples",
            command=self._refresh_samples,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            height=34,
            corner_radius=4,
        ).pack(fill="x")

        self.sample_list = ctk.CTkScrollableFrame(
            content, fg_color=BACKGROUND, corner_radius=4, height=135
        )
        self.sample_list.pack(fill="x", pady=12)

        ctk.CTkButton(
            content,
            text="Upload external image…",
            command=self._upload_image,
            fg_color=SURFACE,
            hover_color=SOFT_BLUE,
            text_color=PRIMARY,
            border_width=1,
            border_color=PRIMARY,
            height=36,
            corner_radius=4,
        ).pack(fill="x")
        ctk.CTkLabel(
            content,
            text="Known label for external image (optional)",
            font=("Segoe UI", 11),
            text_color=MUTED,
            anchor="w",
        ).pack(fill="x", pady=(12, 4))
        self.manual_label_var = ctk.StringVar(value="Unknown")
        self.manual_label_menu = ctk.CTkOptionMenu(
            content,
            values=["Unknown", *self.engine.bundle.idx_to_class.values()],
            variable=self.manual_label_var,
            command=lambda _: self._render_correctness(),
            fg_color=BACKGROUND,
            button_color=BORDER,
            button_hover_color="#C5CEDB",
            text_color=TEXT,
            height=32,
        )
        self.manual_label_menu.pack(fill="x")

        ctk.CTkFrame(content, fg_color=BORDER, height=1).pack(fill="x", pady=16)
        self._section_title(content, "Automatic evaluation", "Random labeled samples from one explicit split.")
        options = ctk.CTkFrame(content, fg_color="transparent")
        options.pack(fill="x")
        self.test_count_var = ctk.StringVar(value="25")
        ctk.CTkOptionMenu(
            options,
            values=["10", "25", "50", "100"],
            variable=self.test_count_var,
            width=82,
            fg_color=BACKGROUND,
            button_color=BORDER,
            button_hover_color="#C5CEDB",
            text_color=TEXT,
        ).pack(side="left")
        self.run_test_button = ctk.CTkButton(
            options,
            text="Run test",
            width=120,
            command=self._run_automatic_test,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            corner_radius=4,
        )
        self.run_test_button.pack(side="right")
        self.test_output = ctk.CTkTextbox(
            content,
            height=85,
            fg_color=BACKGROUND,
            border_width=1,
            border_color=BORDER,
            corner_radius=4,
            font=("Consolas", 10),
            text_color=TEXT,
        )
        self.test_output.pack(fill="both", expand=True, pady=(10, 0))
        self._set_text(self.test_output, "No automatic test has been run.")

    def _build_prediction_panel(self) -> None:
        content = ctk.CTkFrame(self.center_panel, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=24, pady=18)
        self._section_title(content, "Current prediction", "Model confidence applies only to this image.")

        self.preview = ctk.CTkLabel(
            content,
            text="Select an image",
            width=300,
            height=275,
            fg_color=BACKGROUND,
            corner_radius=4,
            text_color=MUTED,
        )
        self.preview.pack(fill="x", pady=(4, 14))

        result_row = ctk.CTkFrame(content, fg_color="transparent")
        result_row.pack(fill="x")
        self.predicted_character_label = ctk.CTkLabel(
            result_row, text="—", font=("Segoe UI", 64, "bold"), text_color=TEXT
        )
        self.predicted_character_label.pack(side="left", padx=(10, 24))
        summary = ctk.CTkFrame(result_row, fg_color="transparent")
        summary.pack(side="left", fill="x", expand=True)
        self.confidence_label = ctk.CTkLabel(
            summary,
            text="Confidence —",
            font=("Segoe UI", 18, "bold"),
            text_color=TEXT,
            anchor="w",
        )
        self.confidence_label.pack(fill="x")
        self.correct_answer_label = ctk.CTkLabel(
            summary, text="Correct answer: Unknown", font=("Segoe UI", 12), text_color=MUTED, anchor="w"
        )
        self.correct_answer_label.pack(fill="x", pady=(5, 0))
        self.result_label = ctk.CTkLabel(
            summary, text="Result: Not scored", font=("Segoe UI", 12, "bold"), text_color=MUTED, anchor="w"
        )
        self.result_label.pack(fill="x", pady=(3, 0))

        ctk.CTkLabel(
            content,
            text="Top predictions",
            font=("Segoe UI", 13, "bold"),
            text_color=TEXT,
            anchor="w",
        ).pack(fill="x", pady=(18, 6))
        self.top_rows = []
        for _ in range(3):
            row = ctk.CTkFrame(content, fg_color="transparent")
            row.pack(fill="x", pady=3)
            label = ctk.CTkLabel(row, text="—", width=75, anchor="w", text_color=TEXT)
            label.pack(side="left")
            bar = ctk.CTkProgressBar(row, height=8, fg_color=BORDER, progress_color=PRIMARY)
            bar.set(0)
            bar.pack(side="left", fill="x", expand=True, padx=8)
            value = ctk.CTkLabel(row, text="—", width=52, anchor="e", text_color=MUTED)
            value.pack(side="right")
            self.top_rows.append((label, bar, value))

        self.input_details_label = ctk.CTkLabel(
            content,
            text="Input: —",
            font=("Consolas", 10),
            text_color=MUTED,
            anchor="w",
            justify="left",
        )
        self.input_details_label.pack(fill="x", pady=(16, 0))

    def _build_model_panel(self) -> None:
        content = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=16, pady=16)
        self._section_title(content, "Model information", "Saved checkpoint metadata, not current-image confidence.")
        metadata = self.engine.bundle.metadata
        optimizer = metadata.get("optimizer") or {}
        if isinstance(optimizer, str):
            optimizer = {"name": optimizer, "learning_rate": metadata.get("learning_rate")}
        scheduler = metadata.get("scheduler") or {}
        dataset = metadata.get("dataset") or {}
        split = metadata.get("split") or {}
        rows = (
            ("Model", metadata.get("architecture")),
            ("Checkpoint", self.engine.bundle.checkpoint_path.name),
            ("Format version", metadata.get("checkpoint_format_version", "Not recorded")),
            ("Input", "64 × 64 grayscale"),
            ("Classes", str(metadata.get("num_classes"))),
            ("Best validation", self._format_accuracy(metadata.get("best_validation_accuracy"))),
            ("Test accuracy", self._format_accuracy(metadata.get("test_accuracy"))),
            ("Best epoch", metadata.get("epoch_of_best_checkpoint", "Not recorded")),
            ("Total epochs", metadata.get("cumulative_epochs_trained", 0)),
            ("Optimizer", optimizer.get("name", "Not recorded")),
            ("Learning rate", optimizer.get("learning_rate", "Not recorded")),
            ("Batch size", metadata.get("batch_size", "Not recorded")),
            ("Scheduler", scheduler.get("name", "Not recorded")),
            ("Dataset", dataset.get("total", "Not recorded")),
            ("Split", split.get("strategy", "Not recorded")),
        )
        for name, value in rows:
            row = ctk.CTkFrame(content, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(
                row, text=name, font=("Segoe UI", 10), text_color=MUTED, anchor="w", height=20
            ).pack(side="left")
            ctk.CTkLabel(
                row,
                text=str(value),
                font=("Segoe UI", 10, "bold"),
                text_color=TEXT,
                anchor="e",
                wraplength=155,
                justify="right",
            ).pack(side="right", fill="x", expand=True)

        ctk.CTkFrame(content, fg_color=BORDER, height=1).pack(fill="x", pady=14)
        ctk.CTkLabel(
            content,
            text="External preprocessing",
            font=("Segoe UI", 12, "bold"),
            text_color=TEXT,
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            content,
            text="Grayscale → foreground crop → aspect-preserving fit → centered 64 × 64 canvas → tensor",
            font=("Segoe UI", 11),
            text_color=MUTED,
            justify="left",
            anchor="w",
            wraplength=255,
        ).pack(fill="x", pady=(4, 0))

    def _build_status_bar(self) -> None:
        self.status_label = ctk.CTkLabel(
            self,
            text=f"Ready  •  {self.engine.bundle.checkpoint_path}",
            font=("Segoe UI", 10),
            text_color=MUTED,
            anchor="w",
            height=28,
        )
        self.status_label.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 5))

    @staticmethod
    def _format_accuracy(value) -> str:
        return "N/A" if value is None else f"{float(value):.2f}%"

    @staticmethod
    def _set_text(textbox: ctk.CTkTextbox, text: str) -> None:
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("1.0", text)
        textbox.configure(state="disabled")

    def _refresh_samples(self) -> None:
        if not hasattr(self, "sample_list"):
            return
        for child in self.sample_list.winfo_children():
            child.destroy()
        split_name = self.split_var.get()
        indices = self.split_indices[split_name]
        chosen = random.sample(indices, min(5, len(indices)))
        self.sample_images = []
        for dataset_index in chosen:
            path = Path(self.dataset.samples[dataset_index][0])
            label = self.dataset.classes[self.dataset.samples[dataset_index][1]]
            with Image.open(path) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((34, 34), Image.Resampling.LANCZOS)
                ctk_image = ctk.CTkImage(light_image=thumbnail.copy(), size=thumbnail.size)
            self.sample_images.append(ctk_image)
            ctk.CTkButton(
                self.sample_list,
                text=f"{label}   {path.name[:21]}{'…' if len(path.name) > 21 else ''}",
                image=ctk_image,
                compound="left",
                anchor="w",
                command=lambda selected=path: self._select_image(selected),
                fg_color=SURFACE,
                hover_color=SOFT_BLUE,
                text_color=TEXT,
                border_width=1,
                border_color=BORDER,
                corner_radius=3,
                height=42,
            ).pack(fill="x", pady=2)

    def _upload_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select character image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if selected:
            self.manual_label_var.set("Unknown")
            self._select_image(Path(selected))

    def _select_image(self, image_path: Path) -> None:
        self.current_path = Path(image_path).resolve()
        self.current_prediction = None
        self.inference_generation += 1
        generation = self.inference_generation
        with Image.open(self.current_path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((370, 260), Image.Resampling.LANCZOS)
            self.preview_image = ctk.CTkImage(light_image=preview.copy(), size=preview.size)
        self.preview.configure(image=self.preview_image, text="")
        self.predicted_character_label.configure(text="…")
        self.confidence_label.configure(text="Running CharacterCNN…")
        self.status_label.configure(text=f"Predicting  •  {self.current_path.name}")
        threading.Thread(
            target=self._predict_worker,
            args=(self.current_path, generation),
            daemon=True,
        ).start()

    def _predict_worker(self, path: Path, generation: int) -> None:
        try:
            prediction = self.engine.predict_path(path, top_k=3)
        except Exception as error:
            self.worker_queue.put(("prediction_error", str(error)))
            return
        self.worker_queue.put(("prediction", path, generation, prediction))

    def _show_prediction(self, path: Path, generation: int, prediction: Prediction) -> None:
        if generation != self.inference_generation:
            return
        self.current_prediction = prediction
        self.predicted_character_label.configure(text=prediction.predicted_character)
        self.confidence_label.configure(text=f"Confidence {prediction.confidence:.1%}")
        for row, candidate in zip(self.top_rows, prediction.top_predictions):
            label, bar, value = row
            label.configure(text=candidate.character)
            bar.set(candidate.probability)
            value.configure(text=f"{candidate.probability:.1%}")
        self.input_details_label.configure(
            text=(
                f"Tensor {prediction.tensor_shape}  •  probability sum {prediction.probability_sum:.6f}\n"
                f"Source {path.name}"
            )
        )
        self.status_label.configure(text=f"Ready  •  {self.engine.bundle.checkpoint_path}")
        self._render_correctness()

    def _render_correctness(self) -> None:
        if self.current_path is None:
            return
        correct_answer = dataset_label_for_path(self.current_path)
        if correct_answer is None and self.manual_label_var.get() != "Unknown":
            correct_answer = self.manual_label_var.get()
        self.correct_answer_label.configure(
            text=f"Correct answer: {correct_answer or 'Unknown'}"
        )
        if self.current_prediction is None or correct_answer is None:
            self.result_label.configure(text="Result: Not scored", text_color=MUTED)
            return
        is_correct = self.current_prediction.predicted_character == correct_answer
        self.result_label.configure(
            text="Result: CORRECT" if is_correct else "Result: WRONG",
            text_color=SUCCESS if is_correct else DANGER,
        )

    def _run_automatic_test(self) -> None:
        split_name = self.split_var.get()
        count = int(self.test_count_var.get())
        self.run_test_button.configure(state="disabled", text="Running…")
        self._set_text(self.test_output, f"Evaluating {count} random {split_name} samples…")
        threading.Thread(
            target=self._automatic_test_worker,
            args=(split_name, count),
            daemon=True,
        ).start()

    def _automatic_test_worker(self, split_name: str, count: int) -> None:
        try:
            result = evaluate_indices(
                self.engine,
                self.dataset,
                self.split_indices[split_name],
                split_name,
                limit=count,
                seed=random.randrange(1_000_000),
            )
            lines = [
                f"Split: {split_name}",
                f"Correct: {result.correct}",
                f"Wrong: {result.total - result.correct}",
                f"Accuracy: {result.accuracy:.2f}%",
                "",
                "Failures:",
            ]
            if result.failures:
                lines.extend(
                    f"{failure.image_path.name}: {failure.correct_character} → "
                    f"{failure.predicted_character} ({failure.confidence:.1%})"
                    for failure in result.failures
                )
            else:
                lines.append("None")
            output = "\n".join(lines)
        except Exception as error:
            output = f"Automatic test failed:\n{error}"
        self.worker_queue.put(("automatic_test", output))

    def _finish_automatic_test(self, output: str) -> None:
        self._set_text(self.test_output, output)
        self.run_test_button.configure(state="normal", text="Run test")

    def _poll_worker_queue(self) -> None:
        while True:
            try:
                message = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            if message[0] == "prediction":
                _, path, generation, prediction = message
                self._show_prediction(path, generation, prediction)
            elif message[0] == "prediction_error":
                messagebox.showerror("Prediction failed", message[1])
            elif message[0] == "automatic_test":
                self._finish_automatic_test(message[1])
        self.after(50, self._poll_worker_queue)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    app = AmharicAIApp()
    app.mainloop()
