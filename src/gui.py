"""Lightweight desktop interface for the active CharacterCNN model."""

from __future__ import annotations

import json
import random
import queue
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from diagnostics import evaluate_labeled_paths
from dataset_contract import load_expected_characters
from inference import InferenceEngine, Prediction, dataset_label_for_path
from project_paths import DATA_DIR, SPLIT_MANIFEST_PATH

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


def model_information(metadata: dict) -> dict[str, object]:
    dataset = metadata.get("dataset", {})
    scheduler = metadata.get("scheduler") or {}
    return {
        "cumulative_epochs": metadata.get("cumulative_epochs_trained", "Unknown"),
        "train_samples": dataset.get("train", "Unknown"),
        "validation_samples": dataset.get("validation", "Unknown"),
        "test_samples": dataset.get("test", "Unknown"),
        "scheduler": scheduler.get("name", "None"),
    }


def format_top_predictions(prediction: Prediction) -> str:
    return "   ".join(
        f"{rank}. {candidate.character} — {candidate.probability:.1%}"
        for rank, candidate in enumerate(prediction.top_predictions, start=1)
    )

class AmharicAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Ethiopic Character Recognizer")
        self.geometry("1160x760")
        self.minsize(1050, 680)
        self.configure(fg_color=BACKGROUND)

        try:
            self.engine = InferenceEngine.from_artifacts()
            expected_mapping = {
                character: index
                for index, character in enumerate(load_expected_characters())
            }
            if self.engine.bundle.metadata["class_to_idx"] != expected_mapping:
                raise RuntimeError(
                    "The active CNN checkpoint does not match src/characters.json. "
                    "Train the full dataset with --fresh before opening the GUI."
                )
            with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
                self.split_manifest = json.load(handle)
            if self.split_manifest.get("class_to_idx") != self.engine.bundle.metadata["class_to_idx"]:
                raise RuntimeError(
                    "Data split class mapping differs from the active CNN checkpoint."
                )
            self.split_paths = {
                name: [DATA_DIR / relative_path for relative_path in relative_paths]
                for name, relative_paths in self.split_manifest.get("splits", {}).items()
            }
            checkpoint_split = self.engine.bundle.metadata.get("split", {})
            self.manifest_is_checkpoint_split = (
                checkpoint_split.get("strategy") == self.split_manifest.get("strategy")
                and checkpoint_split.get("dataset_signature") == self.split_manifest.get("dataset_signature")
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

        self._build_header()
        self._build_tabs()
        self._build_status_bar()
        self._refresh_samples()
        self.after(50, self._poll_worker_queue)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=76)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header,
            text="Ethiopic Character Recognizer",
            font=("Segoe UI", 22, "bold"),
            text_color=TEXT,
        ).pack(side="left", padx=24, pady=18)
        
        num_classes = len(self.engine.bundle.metadata.get("class_to_idx", {}))
        ctk.CTkLabel(
            header,
            text=f"Active Model: CNN | Checkpoint: {self.engine.bundle.checkpoint_path.name} | Classes: {num_classes}",
            font=("Segoe UI", 12),
            text_color=MUTED,
        ).pack(side="right", padx=24)

    def _build_tabs(self) -> None:
        self.tabview = ctk.CTkTabview(self, fg_color=SURFACE, bg_color=BACKGROUND, text_color=TEXT)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.tab_predict = self.tabview.add("Predict")
        self.tab_test = self.tabview.add("Test Model")
        self.tab_info = self.tabview.add("Model Information")
        
        self._build_predict_tab()
        self._build_test_tab()
        self._build_info_tab()

    def _build_predict_tab(self) -> None:
        self.tab_predict.grid_columnconfigure(0, weight=1)
        self.tab_predict.grid_columnconfigure(1, weight=2)
        
        # Left side: Image selection
        left_panel = ctk.CTkFrame(self.tab_predict, fg_color="transparent")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(left_panel, text="Image Source", font=("Segoe UI", 15, "bold"), text_color=TEXT, anchor="w").pack(fill="x")
        
        self.split_var = ctk.StringVar(value="test")
        self.split_menu = ctk.CTkOptionMenu(
            left_panel, values=["train", "validation", "test"], variable=self.split_var,
            command=lambda _: self._refresh_samples(), fg_color=SOFT_BLUE, button_color=PRIMARY, text_color=TEXT
        )
        self.split_menu.pack(fill="x", pady=10)
        
        ctk.CTkButton(left_panel, text="Refresh Samples", command=self._refresh_samples, fg_color=PRIMARY, hover_color=PRIMARY_HOVER).pack(fill="x")
        
        self.sample_list = ctk.CTkScrollableFrame(left_panel, fg_color=BACKGROUND, corner_radius=4, height=200)
        self.sample_list.pack(fill="x", pady=10)
        
        ctk.CTkButton(left_panel, text="Upload External Image", command=self._upload_image, fg_color=SURFACE, hover_color=SOFT_BLUE, text_color=PRIMARY, border_color=PRIMARY, border_width=1).pack(fill="x")

        # Right side: Prediction Result
        right_panel = ctk.CTkFrame(self.tab_predict, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.preview = ctk.CTkLabel(right_panel, text="Select an image to preview", fg_color=BACKGROUND, corner_radius=6, height=260)
        self.preview.pack(fill="x", pady=(0, 20))
        
        res_frame = ctk.CTkFrame(right_panel, fg_color=SURFACE, border_color=BORDER, border_width=1)
        res_frame.pack(fill="x", pady=10)
        
        self.predicted_character_label = ctk.CTkLabel(res_frame, text="?", font=("Segoe UI", 72, "bold"), text_color=TEXT)
        self.predicted_character_label.pack(pady=10)
        
        self.confidence_label = ctk.CTkLabel(res_frame, text="Confidence: --", font=("Segoe UI", 16), text_color=MUTED)
        self.confidence_label.pack(pady=5)

        self.top_predictions_label = ctk.CTkLabel(
            res_frame, text="Top predictions: --", font=("Segoe UI", 13), text_color=MUTED
        )
        self.top_predictions_label.pack(pady=5)
        
        self.correct_answer_label = ctk.CTkLabel(res_frame, text="Correct Answer: --", font=("Segoe UI", 14), text_color=TEXT)
        self.correct_answer_label.pack(pady=5)
        
        self.result_label = ctk.CTkLabel(res_frame, text="Result: Not scored", font=("Segoe UI", 16, "bold"), text_color=MUTED)
        self.result_label.pack(pady=5)
        
        self.manual_label_var = ctk.StringVar(value="Unknown")

    def _build_test_tab(self) -> None:
        content = ctk.CTkFrame(self.tab_test, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        controls = ctk.CTkFrame(content, fg_color="transparent")
        controls.pack(fill="x", pady=10)
        
        ctk.CTkLabel(controls, text="Samples to test:", text_color=TEXT).pack(side="left", padx=5)
        self.test_count_var = ctk.StringVar(value="25")
        ctk.CTkOptionMenu(controls, variable=self.test_count_var, values=["10", "25", "50", "100", "250"], width=80, fg_color=SOFT_BLUE, text_color=TEXT).pack(side="left", padx=5)
        
        self.run_test_button = ctk.CTkButton(controls, text="Run Automatic Test", command=self._run_automatic_test, fg_color=PRIMARY)
        self.run_test_button.pack(side="left", padx=20)
        
        self.test_output = ctk.CTkTextbox(content, fg_color=BACKGROUND, text_color=TEXT, font=("Consolas", 12))
        self.test_output.pack(fill="both", expand=True, pady=10)
        self.test_output.configure(state="disabled")

    def _build_info_tab(self) -> None:
        content = ctk.CTkScrollableFrame(self.tab_info, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        meta = self.engine.bundle.metadata
        info = model_information(meta)
        
        def add_info(label, value):
            f = ctk.CTkFrame(content, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=label, width=200, anchor="w", font=("Segoe UI", 12, "bold"), text_color=TEXT).pack(side="left")
            ctk.CTkLabel(f, text=str(value), anchor="w", text_color=MUTED).pack(side="left", fill="x", expand=True)

        add_info("Architecture", meta.get("architecture", "Unknown"))
        add_info("Number of Classes", len(meta.get("class_to_idx", {})))
        add_info("Best Validation Accuracy", self._format_accuracy(meta.get("best_validation_accuracy")))
        add_info("Test Accuracy", self._format_accuracy(meta.get("test_accuracy")))
        add_info("Best Epoch", meta.get("epoch_of_best_checkpoint", "Unknown"))
        add_info("Cumulative Epochs", info["cumulative_epochs"])
        add_info("Batch Size", meta.get("batch_size", "Unknown"))
        add_info("Optimizer", meta.get("optimizer", {}).get("name", "Unknown"))
        add_info("Learning Rate", meta.get("optimizer", {}).get("learning_rate", "Unknown"))
        add_info("Scheduler", info["scheduler"])
        add_info("Checkpoint Path", self.engine.bundle.checkpoint_path.resolve())
        
        add_info("Train Samples", info["train_samples"])
        add_info("Validation Samples", info["validation_samples"])
        add_info("Test Samples", info["test_samples"])

    def _build_status_bar(self) -> None:
        self.status_bar = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=32)
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.pack_propagate(False)
        self.status_label = ctk.CTkLabel(self.status_bar, text="Ready", font=("Segoe UI", 11), text_color=MUTED)
        self.status_label.pack(side="left", padx=16, pady=4)

    def _format_accuracy(self, value) -> str:
        return "N/A" if value is None else f"{float(value):.2f}%"

    def _set_text(self, textbox: ctk.CTkTextbox, text: str) -> None:
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
        paths = [p for p in self.split_paths.get(split_name, []) if p.exists()]
        if not paths:
            return
        chosen = random.sample(paths, min(5, len(paths)))
        self.sample_images = []
        for path in chosen:
            label = dataset_label_for_path(path)
            with Image.open(path) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((34, 34), Image.Resampling.LANCZOS)
                ctk_image = ctk.CTkImage(light_image=thumbnail.copy(), size=thumbnail.size)
            self.sample_images.append(ctk_image)
            ctk.CTkButton(
                self.sample_list, text=f"{label}   {path.name[:21]}", image=ctk_image, compound="left", anchor="w",
                command=lambda selected=path: self._select_image(selected), fg_color=SURFACE, hover_color=SOFT_BLUE,
                text_color=TEXT, border_width=1, border_color=BORDER, height=42
            ).pack(fill="x", pady=2)

    def _upload_image(self) -> None:
        selected = filedialog.askopenfilename(title="Select character image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
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
        self.predicted_character_label.configure(text="...")
        self.confidence_label.configure(text="Predicting...")
        self.top_predictions_label.configure(text="Top predictions: --")
        self.status_label.configure(text=f"Predicting {self.current_path.name}")
        threading.Thread(target=self._predict_worker, args=(self.current_path, generation), daemon=True).start()

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
        self.confidence_label.configure(text=f"Confidence: {prediction.confidence:.1%}")
        self.top_predictions_label.configure(
            text=f"Top predictions: {format_top_predictions(prediction)}"
        )
        
        self.status_label.configure(text="Ready")
        self._render_correctness()

    def _render_correctness(self) -> None:
        if self.current_path is None:
            return
        correct_answer = dataset_label_for_path(self.current_path) or "Unknown"

        self.correct_answer_label.configure(text=f"Correct Answer: {correct_answer}")
        
        if self.current_prediction is None or correct_answer == "Unknown":
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
        self.run_test_button.configure(state="disabled", text="Running...")
        self._set_text(self.test_output, f"Evaluating {count} random {split_name} samples...")
        threading.Thread(target=self._automatic_test_worker, args=(split_name, count), daemon=True).start()

    def _automatic_test_worker(self, split_name: str, count: int) -> None:
        try:
            if not self.manifest_is_checkpoint_split:
                raise RuntimeError(
                    "The active split was not used by this checkpoint. "
                    "Fresh training is required before held-out testing."
                )
            paths = [p for p in self.split_paths.get(split_name, []) if p.exists()]
            if not paths:
                raise RuntimeError(f"No current images exist in the {split_name} split.")
            result = evaluate_labeled_paths(self.engine, paths, split_name, limit=count, seed=random.randrange(1_000_000))
            lines = [
                f"Partition: {split_name}",
                "Held out from checkpoint: YES",
                f"Correct: {result.correct}",
                f"Wrong: {result.total - result.correct}",
                f"Accuracy: {result.accuracy:.2f}%",
                "",
                "Failures:",
            ]
            if result.failures:
                lines.extend(f"{f.image_path.name}: {f.correct_character} -> {f.predicted_character} ({f.confidence:.1%})" for f in result.failures)
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
                self._show_prediction(message[1], message[2], message[3])
            elif message[0] == "prediction_error":
                messagebox.showerror("Prediction failed", message[1])
            elif message[0] == "automatic_test":
                self._finish_automatic_test(message[1])
        self.after(50, self._poll_worker_queue)

if __name__ == "__main__":
    app = AmharicAIApp()
    app.mainloop()
