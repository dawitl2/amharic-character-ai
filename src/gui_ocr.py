"""Reusable Word OCR and Sentence OCR desktop pages."""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image

from gui_theme import (
    BACKGROUND,
    BORDER,
    MUTED,
    PRIMARY,
    PRIMARY_HOVER,
    SOFT_BLUE,
    SUCCESS,
    SURFACE,
    SURFACE_MUTED,
    TEXT,
    WARNING,
    card,
    muted_label,
    section_title,
)
from ocr_engine import OCREngine, OCRResult
from translation import MyMemoryTranslationProvider


IMAGE_TYPES = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]


class OCRModePage(ctk.CTkFrame):
    def __init__(
        self,
        parent,
        *,
        mode: str,
        character_engine,
        worker_queue,
        status_callback,
        translator: MyMemoryTranslationProvider,
    ):
        super().__init__(parent, fg_color=BACKGROUND)
        self.mode = mode
        self.character_engine = character_engine
        self.worker_queue = worker_queue
        self.status_callback = status_callback
        self.translator = translator
        self.source_path: Path | None = None
        self.source_image: Image.Image | None = None
        self.result: OCRResult | None = None
        self.preview_mode = "segmentation"
        self.image_references = []
        self.threshold_var = ctk.StringVar(value="50%")
        self._build()

    @property
    def title(self) -> str:
        return "Word Recognition" if self.mode == "word" else "Sentence Recognition"

    def _build(self) -> None:
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=3)
        self.grid_rowconfigure(2, weight=2)

        heading = ctk.CTkFrame(self, fg_color="transparent")
        heading.grid(row=0, column=0, columnspan=2, sticky="ew", padx=22, pady=(18, 10))
        ctk.CTkLabel(
            heading,
            text=self.title,
            font=("Segoe UI", 23, "bold"),
            text_color=TEXT,
        ).pack(side="left")
        muted_label(
            heading,
            text=(
                "OpenCV locates characters; the active CNN identifies them."
                if self.mode == "word"
                else "OpenCV finds words and characters; the CNN reconstructs reading order."
            ),
        ).pack(side="left", padx=18)
        self.upload_button = ctk.CTkButton(
            heading,
            text=f"Upload {self.mode.title()} Image",
            command=self._upload,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            height=38,
        )
        self.upload_button.pack(side="right")

        preview_card = card(self)
        preview_card.grid(row=1, column=0, sticky="nsew", padx=(22, 9), pady=8)
        section_title(preview_card, "Input and segmentation preview").pack(
            fill="x", padx=18, pady=(14, 8)
        )
        self.preview = ctk.CTkLabel(
            preview_card,
            text=f"Upload a printed {self.mode} image",
            fg_color=SURFACE_MUTED,
            text_color=MUTED,
            corner_radius=7,
        )
        self.preview.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        preview_controls = ctk.CTkFrame(preview_card, fg_color="transparent")
        preview_controls.pack(fill="x", padx=16, pady=(0, 14))
        self.preview_toggle = ctk.CTkSegmentedButton(
            preview_controls,
            values=["Original", "Segmentation", "Binary"],
            command=self._change_preview,
            selected_color=PRIMARY,
            selected_hover_color=PRIMARY_HOVER,
            unselected_color=SOFT_BLUE,
            unselected_hover_color="#DCE9FF",
            text_color=TEXT,
        )
        self.preview_toggle.set("Segmentation")
        self.preview_toggle.pack(side="left")
        muted_label(preview_controls, "Blue = word, green = character").pack(
            side="right"
        )

        result_card = card(self)
        result_card.grid(row=1, column=1, sticky="nsew", padx=(9, 22), pady=8)
        section_title(result_card, "Recognition result").pack(
            fill="x", padx=18, pady=(14, 8)
        )
        threshold_row = ctk.CTkFrame(result_card, fg_color="transparent")
        threshold_row.pack(fill="x", padx=18)
        muted_label(threshold_row, "Uncertain below").pack(side="left")
        ctk.CTkOptionMenu(
            threshold_row,
            values=["30%", "50%", "70%", "90%"],
            variable=self.threshold_var,
            width=82,
            fg_color=SOFT_BLUE,
            button_color=PRIMARY,
            text_color=TEXT,
        ).pack(side="left", padx=8)
        self.summary_label = muted_label(threshold_row, "No image analyzed")
        self.summary_label.pack(side="right")

        ctk.CTkLabel(
            result_card,
            text="Recognized Amharic",
            font=("Segoe UI", 12, "bold"),
            text_color=MUTED,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(14, 4))
        self.amharic_text = ctk.CTkTextbox(
            result_card,
            height=92,
            fg_color=SURFACE_MUTED,
            border_color=BORDER,
            border_width=1,
            text_color=TEXT,
            font=("Nyala", 24),
            wrap="word",
        )
        self.amharic_text.pack(fill="x", padx=18)
        self._set_text(self.amharic_text, "—")

        ctk.CTkLabel(
            result_card,
            text="English translation",
            font=("Segoe UI", 12, "bold"),
            text_color=MUTED,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(14, 4))
        self.english_text = ctk.CTkTextbox(
            result_card,
            height=72,
            fg_color=SURFACE_MUTED,
            border_color=BORDER,
            border_width=1,
            text_color=TEXT,
            font=("Segoe UI", 15),
            wrap="word",
        )
        self.english_text.pack(fill="x", padx=18)
        self._set_text(self.english_text, "Translation is optional and requires internet.")

        action_row = ctk.CTkFrame(result_card, fg_color="transparent")
        action_row.pack(fill="x", padx=18, pady=14)
        self.translate_button = ctk.CTkButton(
            action_row,
            text="Translate",
            command=self._translate,
            state="disabled",
            width=110,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
        )
        self.translate_button.pack(side="left")
        ctk.CTkButton(
            action_row,
            text="Copy Amharic",
            command=lambda: self._copy(self.amharic_text.get("1.0", "end")),
            width=112,
            fg_color=SURFACE,
            hover_color=SOFT_BLUE,
            text_color=PRIMARY,
            border_color=BORDER,
            border_width=1,
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            action_row,
            text="Copy English",
            command=lambda: self._copy(self.english_text.get("1.0", "end")),
            width=112,
            fg_color=SURFACE,
            hover_color=SOFT_BLUE,
            text_color=PRIMARY,
            border_color=BORDER,
            border_width=1,
        ).pack(side="left")
        muted_label(action_row, "Provider: MyMemory (text only)").pack(side="right")

        diagnostics = card(self)
        diagnostics.grid(
            row=2, column=0, columnspan=2, sticky="nsew", padx=22, pady=(8, 18)
        )
        section_title(diagnostics, "Detected character diagnostics").pack(
            fill="x", padx=18, pady=(12, 6)
        )
        self.character_strip = ctk.CTkScrollableFrame(
            diagnostics,
            orientation="horizontal",
            fg_color=SURFACE_MUTED,
            height=142,
            corner_radius=6,
        )
        self.character_strip.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        muted_label(
            self.character_strip,
            "Each crop will show its raw CNN prediction, confidence, and uncertainty state.",
        ).pack(padx=16, pady=36)

    @staticmethod
    def _set_text(widget: ctk.CTkTextbox, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _copy(self, value: str) -> None:
        cleaned = value.strip()
        if not cleaned or cleaned == "—":
            return
        self.clipboard_clear()
        self.clipboard_append(cleaned)
        self.status_callback("Copied to clipboard")

    def _upload(self) -> None:
        selected = filedialog.askopenfilename(
            title=f"Select printed {self.mode} image", filetypes=IMAGE_TYPES
        )
        if not selected:
            return
        self.source_path = Path(selected)
        with Image.open(self.source_path) as image:
            self.source_image = image.convert("RGB")
        self._display_image(self.source_image)
        self.upload_button.configure(state="disabled", text="Analyzing…")
        self.status_callback(f"Segmenting {self.source_path.name}")
        threshold = int(self.threshold_var.get().rstrip("%")) / 100.0
        threading.Thread(
            target=self._ocr_worker,
            args=(self.source_image.copy(), threshold),
            daemon=True,
        ).start()

    def _ocr_worker(self, image: Image.Image, threshold: float) -> None:
        try:
            engine = OCREngine(
                self.character_engine, confidence_threshold=threshold
            )
            result = engine.recognize(image, mode=self.mode)
            self.worker_queue.put(("ocr_result", self, result))
        except Exception as error:
            self.worker_queue.put(("ocr_error", self, str(error)))

    def apply_result(self, result: OCRResult) -> None:
        self.result = result
        self.upload_button.configure(
            state="normal", text=f"Upload {self.mode.title()} Image"
        )
        self._set_text(
            self.amharic_text,
            result.text or "No character regions were detected.",
        )
        self._set_text(
            self.english_text,
            "Select Translate to send the recognized text to MyMemory.",
        )
        self.translate_button.configure(state="normal" if result.text else "disabled")
        self.summary_label.configure(
            text=(
                f"{len(result.words)} word(s) • {len(result.characters)} character(s) • "
                f"{result.uncertain_count} uncertain"
            )
        )
        self.preview_mode = "segmentation"
        self.preview_toggle.set("Segmentation")
        self._display_image(result.overlay)
        self._render_diagnostics(result)
        self.status_callback("OCR complete")

    def apply_error(self, message: str) -> None:
        self.upload_button.configure(
            state="normal", text=f"Upload {self.mode.title()} Image"
        )
        self.status_callback("OCR failed")
        self._set_text(self.amharic_text, f"OCR failed: {message}")

    def _change_preview(self, selection: str) -> None:
        self.preview_mode = selection.lower()
        if self.source_image is None:
            return
        if self.preview_mode == "original" or self.result is None:
            image = self.source_image
        elif self.preview_mode == "binary":
            image = self.result.segmentation.binary_preview.convert("RGB")
        else:
            image = self.result.overlay
        self._display_image(image)

    def _display_image(self, image: Image.Image) -> None:
        display = image.copy()
        display.thumbnail((620, 340), Image.Resampling.LANCZOS)
        reference = ctk.CTkImage(
            light_image=display, dark_image=display, size=display.size
        )
        self.image_references = [reference, *self.image_references[:80]]
        self.preview.configure(image=reference, text="")

    def _render_diagnostics(self, result: OCRResult) -> None:
        for child in self.character_strip.winfo_children():
            child.destroy()
        self.image_references = self.image_references[:1]
        for index, character in enumerate(result.characters, start=1):
            frame = ctk.CTkFrame(
                self.character_strip,
                fg_color=SURFACE,
                border_color=WARNING if character.uncertain else BORDER,
                border_width=1,
                corner_radius=7,
                width=150,
                height=118,
            )
            frame.pack(side="left", padx=5, pady=4)
            frame.pack_propagate(False)
            crop = character.crop.convert("RGB")
            crop.thumbnail((52, 52), Image.Resampling.LANCZOS)
            reference = ctk.CTkImage(light_image=crop, size=crop.size)
            self.image_references.append(reference)
            ctk.CTkLabel(frame, image=reference, text="").pack(side="left", padx=8)
            details = ctk.CTkFrame(frame, fg_color="transparent")
            details.pack(side="left", fill="both", expand=True, pady=10)
            ctk.CTkLabel(
                details,
                text=f"{index}. {character.prediction.predicted_character}",
                font=("Nyala", 22, "bold"),
                text_color=TEXT,
            ).pack(anchor="w")
            ctk.CTkLabel(
                details,
                text=f"{character.prediction.confidence:.1%}",
                text_color=WARNING if character.uncertain else SUCCESS,
                font=("Segoe UI", 12, "bold"),
            ).pack(anchor="w")
            ctk.CTkLabel(
                details,
                text="uncertain" if character.uncertain else "accepted",
                text_color=MUTED,
                font=("Segoe UI", 10),
            ).pack(anchor="w")

    def _translate(self) -> None:
        if self.result is None or not self.result.text:
            return
        self.translate_button.configure(state="disabled", text="Translating…")
        self.status_callback("Requesting optional English translation")
        threading.Thread(
            target=self._translation_worker,
            args=(self.result.raw_text,),
            daemon=True,
        ).start()

    def _translation_worker(self, text: str) -> None:
        try:
            result = self.translator.translate(text)
            self.worker_queue.put(("translation_result", self, result))
        except Exception as error:
            self.worker_queue.put(("translation_error", self, str(error)))

    def apply_translation(self, result) -> None:
        self.translate_button.configure(state="normal", text="Translate")
        self._set_text(self.english_text, result.translated_text)
        self.status_callback(f"Translation complete via {result.provider}")

    def apply_translation_error(self, message: str) -> None:
        self.translate_button.configure(state="normal", text="Translate")
        self._set_text(
            self.english_text,
            f"Translation unavailable. The Amharic OCR result is preserved.\n{message}",
        )
        self.status_callback("Translation unavailable")
