"""Professional desktop interface for character, word, and sentence OCR."""

from __future__ import annotations

import json
import queue
import random
import sys
import threading
from pathlib import Path
from tkinter import TclError, filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from dataset_contract import load_expected_characters
from diagnostics import evaluate_labeled_paths
from gui_insights import ModelInfoPage, PipelinePage, TrainingGraphsPage
from gui_ocr import OCRModePage
from gui_theme import (
    BACKGROUND,
    BORDER,
    DANGER,
    MUTED,
    PRIMARY,
    PRIMARY_HOVER,
    SOFT_BLUE,
    SOFT_GREEN,
    SUCCESS,
    SURFACE,
    SURFACE_MUTED,
    TEXT,
    card,
    muted_label,
    section_title,
)
from inference import InferenceEngine, Prediction, dataset_label_for_path
from project_paths import DATA_DIR, SPLIT_MANIFEST_PATH
from translation import MyMemoryTranslationProvider


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


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
    PAGE_NAMES = (
        "Character",
        "Word OCR",
        "Sentence OCR",
        "Evaluate",
        "Model Info",
        "Pipeline",
        "Training Graphs",
    )

    def __init__(self):
        super().__init__()
        self.title("Ethiopic OCR — Character, Word, and Sentence Recognition")
        self.geometry("1440x900")
        self.minsize(1180, 760)
        self.configure(fg_color=BACKGROUND)
        try:
            self.state("zoomed")
        except TclError:
            pass

        self.engine = self._load_active_model()
        self.worker_queue: queue.Queue = queue.Queue()
        self.translator = MyMemoryTranslationProvider()
        self.preview_image = None
        self.sample_images = []
        self.current_path: Path | None = None
        self.current_prediction: Prediction | None = None
        self.inference_generation = 0
        self.pages: dict[str, ctk.CTkFrame] = {}

        self._build_header()
        self._build_navigation()
        self._build_workspace()
        self._build_status_bar()
        self._show_page("Character")
        self._refresh_samples()
        self.after(50, self._poll_worker_queue)
        print(self.engine.startup_summary())

    def _load_active_model(self) -> InferenceEngine:
        try:
            engine = InferenceEngine.from_artifacts()
            expected_mapping = {
                character: index
                for index, character in enumerate(load_expected_characters())
            }
            if engine.bundle.metadata["class_to_idx"] != expected_mapping:
                raise RuntimeError(
                    "The active CNN checkpoint does not match src/characters.json."
                )
            with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
                self.split_manifest = json.load(handle)
            if self.split_manifest.get("class_to_idx") != expected_mapping:
                raise RuntimeError(
                    "Data split class mapping differs from the active CNN checkpoint."
                )
            self.split_paths = {
                name: [DATA_DIR / relative_path for relative_path in relative_paths]
                for name, relative_paths in self.split_manifest.get("splits", {}).items()
            }
            checkpoint_split = engine.bundle.metadata.get("split", {})
            self.manifest_is_checkpoint_split = (
                checkpoint_split.get("strategy")
                == self.split_manifest.get("strategy")
                and checkpoint_split.get("dataset_signature")
                == self.split_manifest.get("dataset_signature")
            )
            return engine
        except Exception as error:
            messagebox.showerror("CNN model could not be loaded", str(error))
            self.destroy()
            raise

    def _build_header(self) -> None:
        header = ctk.CTkFrame(
            self,
            fg_color=SURFACE,
            corner_radius=0,
            height=92,
            border_color=BORDER,
            border_width=1,
        )
        header.pack(fill="x")
        header.pack_propagate(False)
        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.pack(side="left", padx=26, pady=16)
        ctk.CTkLabel(
            brand,
            text="ፊ",
            width=48,
            height=48,
            corner_radius=11,
            fg_color=PRIMARY,
            text_color="#FFFFFF",
            font=("Nyala", 30, "bold"),
        ).pack(side="left", padx=(0, 13))
        names = ctk.CTkFrame(brand, fg_color="transparent")
        names.pack(side="left")
        ctk.CTkLabel(
            names,
            text="Ethiopic OCR",
            font=("Segoe UI", 22, "bold"),
            text_color=TEXT,
        ).pack(anchor="w")
        muted_label(
            names,
            "Character intelligence trained locally • word and sentence structure via OpenCV",
        ).pack(anchor="w")

        metadata = self.engine.bundle.metadata
        badges = (
            ("Model", metadata.get("architecture", "CNN")),
            ("Checkpoint", self.engine.bundle.checkpoint_path.name),
            ("Classes", len(metadata.get("class_to_idx", {}))),
        )
        badge_row = ctk.CTkFrame(header, fg_color="transparent")
        badge_row.pack(side="right", padx=26)
        for label, value in badges:
            badge = ctk.CTkFrame(
                badge_row,
                fg_color=SURFACE_MUTED,
                border_color=BORDER,
                border_width=1,
                corner_radius=8,
            )
            badge.pack(side="left", padx=5)
            ctk.CTkLabel(
                badge,
                text=f"{label}: {value}",
                text_color=TEXT,
                font=("Segoe UI", 12),
            ).pack(padx=13, pady=9)

    def _build_navigation(self) -> None:
        navigation = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=56)
        navigation.pack(fill="x")
        navigation.pack_propagate(False)
        self.navigation = ctk.CTkSegmentedButton(
            navigation,
            values=list(self.PAGE_NAMES),
            command=self._show_page,
            selected_color=PRIMARY,
            selected_hover_color=PRIMARY_HOVER,
            unselected_color=SURFACE,
            unselected_hover_color=SOFT_BLUE,
            text_color=TEXT,
            font=("Segoe UI", 12, "bold"),
            height=36,
        )
        self.navigation.pack(padx=24, pady=9)
        self.navigation.set("Character")

    def _build_workspace(self) -> None:
        self.workspace = ctk.CTkFrame(self, fg_color=BACKGROUND, corner_radius=0)
        self.workspace.pack(fill="both", expand=True)
        self.workspace.grid_rowconfigure(0, weight=1)
        self.workspace.grid_columnconfigure(0, weight=1)

        self.pages["Character"] = self._build_character_page()
        self.pages["Evaluate"] = self._build_evaluation_page()
        self.pages["Model Info"] = ModelInfoPage(
            self.workspace, self.engine.bundle
        )
        self.pages["Pipeline"] = PipelinePage(self.workspace)
        self.pages["Training Graphs"] = TrainingGraphsPage(
            self.workspace, self.engine.bundle.metadata
        )
        for mode, page_name in (("word", "Word OCR"), ("sentence", "Sentence OCR")):
            self.pages[page_name] = OCRModePage(
                self.workspace,
                mode=mode,
                character_engine=self.engine,
                worker_queue=self.worker_queue,
                status_callback=self._set_status,
                translator=self.translator,
            )
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

    def _show_page(self, page_name: str) -> None:
        self.pages[page_name].tkraise()
        if hasattr(self, "navigation"):
            self.navigation.set(page_name)
        self._set_status(f"{page_name} ready")

    def _build_character_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.workspace, fg_color=BACKGROUND)
        page.grid_columnconfigure(0, weight=0, minsize=350)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        browser = card(page, corner_radius=0, border_width=0)
        browser.grid(row=0, column=0, sticky="nsew")
        section_title(browser, "Dataset browser").pack(
            fill="x", padx=22, pady=(20, 10)
        )
        self.split_var = ctk.StringVar(value="test")
        self.split_menu = ctk.CTkOptionMenu(
            browser,
            values=["train", "validation", "test"],
            variable=self.split_var,
            command=lambda _: self._refresh_samples(),
            fg_color=SURFACE_MUTED,
            button_color=PRIMARY,
            text_color=TEXT,
            height=38,
        )
        self.split_menu.pack(fill="x", padx=22, pady=5)
        refresh_row = ctk.CTkFrame(browser, fg_color="transparent")
        refresh_row.pack(fill="x", padx=22, pady=7)
        self.sample_count_label = muted_label(refresh_row, "")
        self.sample_count_label.pack(side="left")
        ctk.CTkButton(
            refresh_row,
            text="Refresh",
            command=self._refresh_samples,
            width=86,
            height=30,
            fg_color=SURFACE,
            hover_color=SOFT_BLUE,
            text_color=PRIMARY,
            border_color=BORDER,
            border_width=1,
        ).pack(side="right")
        self.sample_list = ctk.CTkScrollableFrame(
            browser,
            fg_color=SURFACE_MUTED,
            corner_radius=7,
        )
        self.sample_list.pack(fill="both", expand=True, padx=22, pady=7)
        ctk.CTkButton(
            browser,
            text="Upload External Character",
            command=self._upload_image,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            height=42,
        ).pack(fill="x", padx=22, pady=(8, 20))

        workspace = ctk.CTkFrame(page, fg_color=BACKGROUND)
        workspace.grid(row=0, column=1, sticky="nsew", padx=20, pady=18)
        workspace.grid_columnconfigure((0, 1), weight=1)
        workspace.grid_rowconfigure(0, weight=3)
        workspace.grid_rowconfigure(1, weight=1)

        preview_card = card(workspace)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        section_title(preview_card, "Input preview").pack(
            fill="x", padx=18, pady=(15, 8)
        )
        self.preview = ctk.CTkLabel(
            preview_card,
            text="Select a dataset sample or upload an image",
            fg_color=SURFACE_MUTED,
            text_color=MUTED,
            corner_radius=7,
        )
        self.preview.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        result_card = card(workspace)
        result_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
        section_title(result_card, "Prediction result").pack(
            fill="x", padx=18, pady=(15, 6)
        )
        self.predicted_character_label = ctk.CTkLabel(
            result_card,
            text="?",
            font=("Nyala", 84, "bold"),
            text_color=TEXT,
        )
        self.predicted_character_label.pack(pady=(12, 0))
        self.confidence_label = ctk.CTkLabel(
            result_card,
            text="Confidence: —",
            font=("Segoe UI", 20, "bold"),
            text_color=PRIMARY,
        )
        self.confidence_label.pack()
        self.top_predictions_label = ctk.CTkLabel(
            result_card,
            text="Top predictions: —",
            font=("Segoe UI", 13),
            text_color=MUTED,
            wraplength=530,
        )
        self.top_predictions_label.pack(padx=20, pady=14)
        answer_row = ctk.CTkFrame(result_card, fg_color=SURFACE_MUTED, corner_radius=7)
        answer_row.pack(fill="x", padx=18, pady=8)
        self.correct_answer_label = ctk.CTkLabel(
            answer_row,
            text="Correct answer: Unknown",
            font=("Segoe UI", 14),
            text_color=TEXT,
        )
        self.correct_answer_label.pack(side="left", padx=14, pady=12)
        self.result_label = ctk.CTkLabel(
            answer_row,
            text="NOT SCORED",
            font=("Segoe UI", 13, "bold"),
            text_color=MUTED,
        )
        self.result_label.pack(side="right", padx=14)

        preprocessing = card(workspace)
        preprocessing.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        section_title(preprocessing, "Shared preprocessing contract").pack(
            fill="x", padx=16, pady=(13, 7)
        )
        for text in (
            "✓ Grayscale input",
            "✓ External whitespace crop and centered fit",
            "✓ 64 × 64 • tensor [1, 1, 64, 64]",
            "✓ Identity normalization • values [0, 1]",
        ):
            ctk.CTkLabel(
                preprocessing,
                text=text,
                text_color=SUCCESS,
                font=("Segoe UI", 12),
                anchor="w",
            ).pack(fill="x", padx=16, pady=2)

        model_card = card(workspace)
        model_card.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        section_title(model_card, "Inference summary").pack(
            fill="x", padx=16, pady=(13, 7)
        )
        parameter_count = sum(
            parameter.numel() for parameter in self.engine.bundle.model.parameters()
        )
        for label, value in (
            ("Model", self.engine.bundle.metadata["architecture"]),
            ("Parameters", f"{parameter_count:,}"),
            ("Classes", len(self.engine.bundle.metadata["class_to_idx"])),
            ("Device", "CPU"),
        ):
            row = ctk.CTkFrame(model_card, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=2)
            muted_label(row, label).pack(side="left")
            ctk.CTkLabel(row, text=str(value), text_color=TEXT).pack(side="right")
        return page

    def _build_evaluation_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.workspace, fg_color=BACKGROUND)
        heading = ctk.CTkFrame(page, fg_color="transparent")
        heading.pack(fill="x", padx=24, pady=(22, 10))
        ctk.CTkLabel(
            heading,
            text="Held-out Evaluation",
            font=("Segoe UI", 24, "bold"),
            text_color=TEXT,
        ).pack(side="left")
        muted_label(
            heading,
            "Random labeled samples remain separate from uploaded external images.",
        ).pack(side="left", padx=18)
        controls = card(page)
        controls.pack(fill="x", padx=24, pady=8)
        muted_label(controls, "Samples").pack(side="left", padx=(16, 5), pady=14)
        self.test_count_var = ctk.StringVar(value="25")
        ctk.CTkOptionMenu(
            controls,
            variable=self.test_count_var,
            values=["10", "25", "50", "100", "250"],
            width=86,
            fg_color=SOFT_BLUE,
            button_color=PRIMARY,
            text_color=TEXT,
        ).pack(side="left", padx=5)
        self.run_test_button = ctk.CTkButton(
            controls,
            text="Run Automatic Test",
            command=self._run_automatic_test,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
        )
        self.run_test_button.pack(side="left", padx=14)
        muted_label(
            controls,
            "Partition follows the Dataset Browser selection.",
        ).pack(side="right", padx=16)
        output_card = card(page)
        output_card.pack(fill="both", expand=True, padx=24, pady=(8, 22))
        self.test_output = ctk.CTkTextbox(
            output_card,
            fg_color=SURFACE_MUTED,
            text_color=TEXT,
            font=("Consolas", 13),
            border_width=0,
        )
        self.test_output.pack(fill="both", expand=True, padx=14, pady=14)
        self.test_output.configure(state="disabled")
        return page

    def _build_status_bar(self) -> None:
        self.status_bar = ctk.CTkFrame(
            self, fg_color=SURFACE, corner_radius=0, height=30
        )
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.pack_propagate(False)
        self.status_indicator = ctk.CTkLabel(
            self.status_bar, text="●", text_color=SUCCESS, font=("Segoe UI", 12)
        )
        self.status_indicator.pack(side="left", padx=(16, 5))
        self.status_label = muted_label(self.status_bar, "Ready")
        self.status_label.pack(side="left")
        muted_label(self.status_bar, "OCR intelligence: local CharacterCNN").pack(
            side="right", padx=16
        )

    def _set_status(self, text: str) -> None:
        if hasattr(self, "status_label"):
            self.status_label.configure(text=text)

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
        paths = [path for path in self.split_paths.get(split_name, []) if path.exists()]
        self.sample_count_label.configure(text=f"{len(paths):,} available samples")
        if not paths:
            muted_label(self.sample_list, "No current paths in this partition.").pack(
                padx=12, pady=20
            )
            return
        chosen = random.sample(paths, min(7, len(paths)))
        self.sample_images = []
        for path in chosen:
            label = dataset_label_for_path(path)
            with Image.open(path) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((42, 42), Image.Resampling.LANCZOS)
            reference = ctk.CTkImage(light_image=thumbnail.copy(), size=thumbnail.size)
            self.sample_images.append(reference)
            ctk.CTkButton(
                self.sample_list,
                text=f"{label}   {path.name[:24]}",
                image=reference,
                compound="left",
                anchor="w",
                command=lambda selected=path: self._select_image(selected),
                fg_color=SURFACE,
                hover_color=SOFT_BLUE,
                text_color=TEXT,
                border_width=1,
                border_color=BORDER,
                height=52,
            ).pack(fill="x", pady=3)

    def _upload_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select character image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if selected:
            self._select_image(Path(selected))

    def _select_image(self, image_path: Path) -> None:
        self.current_path = Path(image_path).resolve()
        self.current_prediction = None
        self.inference_generation += 1
        generation = self.inference_generation
        with Image.open(self.current_path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((570, 430), Image.Resampling.LANCZOS)
        self.preview_image = ctk.CTkImage(
            light_image=preview.copy(), dark_image=preview.copy(), size=preview.size
        )
        self.preview.configure(image=self.preview_image, text="")
        self.predicted_character_label.configure(text="…")
        self.confidence_label.configure(text="Predicting…")
        self.top_predictions_label.configure(text="Top predictions: —")
        self._set_status(f"Predicting {self.current_path.name}")
        threading.Thread(
            target=self._predict_worker,
            args=(self.current_path, generation),
            daemon=True,
        ).start()

    def _predict_worker(self, path: Path, generation: int) -> None:
        try:
            prediction = self.engine.predict_path(path, top_k=3)
            self.worker_queue.put(("prediction", path, generation, prediction))
        except Exception as error:
            self.worker_queue.put(("prediction_error", str(error)))

    def _show_prediction(
        self, path: Path, generation: int, prediction: Prediction
    ) -> None:
        if generation != self.inference_generation:
            return
        self.current_prediction = prediction
        self.predicted_character_label.configure(text=prediction.predicted_character)
        self.confidence_label.configure(
            text=f"Confidence: {prediction.confidence:.1%}"
        )
        self.top_predictions_label.configure(
            text=f"Top predictions: {format_top_predictions(prediction)}"
        )
        self._set_status("Character prediction complete")
        self._render_correctness()

    def _render_correctness(self) -> None:
        if self.current_path is None:
            return
        correct_answer = dataset_label_for_path(self.current_path) or "Unknown"
        self.correct_answer_label.configure(text=f"Correct answer: {correct_answer}")
        if self.current_prediction is None or correct_answer == "Unknown":
            self.result_label.configure(text="NOT SCORED", text_color=MUTED)
            return
        is_correct = self.current_prediction.predicted_character == correct_answer
        self.result_label.configure(
            text="✓ CORRECT" if is_correct else "✕ WRONG",
            text_color=SUCCESS if is_correct else DANGER,
            fg_color=SOFT_GREEN if is_correct else SURFACE_MUTED,
            corner_radius=6,
        )

    def _run_automatic_test(self) -> None:
        split_name = self.split_var.get()
        count = int(self.test_count_var.get())
        self.run_test_button.configure(state="disabled", text="Running…")
        self._set_text(
            self.test_output, f"Evaluating {count} random {split_name} samples…"
        )
        threading.Thread(
            target=self._automatic_test_worker,
            args=(split_name, count),
            daemon=True,
        ).start()

    def _automatic_test_worker(self, split_name: str, count: int) -> None:
        try:
            if not self.manifest_is_checkpoint_split:
                raise RuntimeError(
                    "The active split was not used by this checkpoint; held-out claims are unsafe."
                )
            paths = [path for path in self.split_paths.get(split_name, []) if path.exists()]
            if not paths:
                raise RuntimeError(f"No current images exist in the {split_name} split.")
            result = evaluate_labeled_paths(
                self.engine,
                paths,
                split_name,
                limit=count,
                seed=random.randrange(1_000_000),
            )
            lines = [
                f"Partition: {split_name}",
                "Held out from checkpoint: YES",
                f"Correct: {result.correct}",
                f"Wrong: {result.total - result.correct}",
                f"Accuracy: {result.accuracy:.2f}%",
                f"Mean confidence: {result.mean_confidence:.2%}",
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
        self.run_test_button.configure(state="normal", text="Run Automatic Test")
        self._set_status("Evaluation complete")

    def _poll_worker_queue(self) -> None:
        while True:
            try:
                message = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            kind = message[0]
            if kind == "prediction":
                self._show_prediction(message[1], message[2], message[3])
            elif kind == "prediction_error":
                messagebox.showerror("Prediction failed", message[1])
            elif kind == "automatic_test":
                self._finish_automatic_test(message[1])
            elif kind == "ocr_result":
                message[1].apply_result(message[2])
            elif kind == "ocr_error":
                message[1].apply_error(message[2])
            elif kind == "translation_result":
                message[1].apply_translation(message[2])
            elif kind == "translation_error":
                message[1].apply_translation_error(message[2])
        self.after(50, self._poll_worker_queue)


if __name__ == "__main__":
    app = AmharicAIApp()
    app.mainloop()
