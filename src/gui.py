import os
import sys
import json
import random
import threading
import time
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import customtkinter as ctk
import tkinter.filedialog as fd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cnn_model import CNNModel

ctk.set_appearance_mode("Dark")

# ─── Color Palette ───────────────────────────────────────────────────────────
BG            = "#212121"
SURFACE       = "#2F2F2F"
SURFACE_SOFT  = "#3A3A3A"
BORDER        = "#424242"
TEXT_PRIMARY  = "#ECECEC"
TEXT_SECONDARY= "#B4B4B4"
TEXT_MUTED    = "#828282"
ACCENT        = "#10A37F"
ACCENT_HOVER  = "#1A7F64"
ACCENT_LIGHT  = "#2A4B42"
ACCENT_TINT   = "#1D362F"
SIDEBAR       = "#171717"
SIDEBAR_SOFT  = "#212121"
SIDEBAR_BORDER= "#2F2F2F"
SIDEBAR_TEXT  = "#ECECEC"
SIDEBAR_MUTED = "#9B9B9B"
DANGER        = "#EF4444"
DANGER_LIGHT  = "#4A2222"
SUCCESS       = "#10A37F"
SUCCESS_LIGHT = "#2A4B42"

# ─── Fonts ───────────────────────────────────────────────────────────────────
FONT_TITLE    = ("Segoe UI", 24, "bold")
FONT_SUBTITLE = ("Segoe UI", 14)
FONT_HEADING  = ("Segoe UI", 17, "bold")
FONT_BODY     = ("Segoe UI", 13)
FONT_SMALL    = ("Segoe UI", 11)
FONT_TINY     = ("Segoe UI", 10, "bold")
FONT_CHAR     = ("Segoe UI", 72, "bold")
FONT_CONF     = ("Segoe UI", 32, "bold")
FONT_MONO     = ("Consolas", 10)


def get_random_images(num_images=5):
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    all_images = list(data_dir.rglob("*.png"))
    if not all_images:
        return []
    return random.sample(all_images, min(num_images, len(all_images)))


class AmharicAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Amharic Character AI")
        self.geometry("1180x760")
        self.minsize(1020, 700)
        self.configure(fg_color=BG)

        # Load model once
        self._load_model()

        # Two‑column grid: left selector | right results
        self.grid_columnconfigure(0, weight=0, minsize=310)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

        # Initial state
        self._shuffle_images()

    # ── Model ────────────────────────────────────────────────────────────────
    def _load_model(self):
        config_path = "models/model_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.idx_to_class = {v: k for k, v in self.config["class_to_idx"].items()}

        self.model = CNNModel(num_classes=len(self.idx_to_class))
        
        best_weights_path = "models/best_model_weights.pth"
        
        if os.path.exists(best_weights_path):
            self.model.load_state_dict(torch.load(best_weights_path))
        else:
            print("Warning: best_model_weights.pth not found. Model will output random predictions.")
            
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.config["image_height"], self.config["image_width"])),
            transforms.ToTensor()
        ])

    # ── Left Panel ───────────────────────────────────────────────────────────
    def _build_left_panel(self):
        self.left = ctk.CTkFrame(self, fg_color=SIDEBAR, corner_radius=0)
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_propagate(False)
        self.left.configure(width=310)

        # Product identity
        title_area = ctk.CTkFrame(self.left, fg_color="transparent")
        title_area.pack(fill="x", padx=26, pady=(24, 0))

        mark = ctk.CTkFrame(title_area, width=42, height=42,
                            fg_color=ACCENT, corner_radius=12)
        mark.pack(side="left")
        mark.pack_propagate(False)
        ctk.CTkLabel(mark, text="ሀ", font=("Segoe UI", 22, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor="center")

        brand_copy = ctk.CTkFrame(title_area, fg_color="transparent")
        brand_copy.pack(side="left", padx=(12, 0))
        ctk.CTkLabel(brand_copy, text="Fidel Vision",
                     font=("Segoe UI", 17, "bold"), text_color=SIDEBAR_TEXT,
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(brand_copy, text="AMHARIC CHARACTER AI",
                     font=("Segoe UI", 9, "bold"), text_color=ACCENT,
                     anchor="w").pack(anchor="w", pady=(1, 0))

        ctk.CTkFrame(self.left, fg_color=SIDEBAR_BORDER, height=1).pack(
            fill="x", padx=26, pady=(20, 18))

        intro = ctk.CTkFrame(self.left, fg_color="transparent")
        intro.pack(fill="x", padx=26)
        ctk.CTkLabel(intro, text="Choose a sample",
                     font=FONT_HEADING, text_color=SIDEBAR_TEXT,
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(intro,
                     text="Five fresh images are ready. Select one to begin.",
                     font=FONT_SMALL, text_color=SIDEBAR_MUTED,
                     anchor="w", justify="left").pack(anchor="w", pady=(3, 0))

        # Image grid – 5 images arranged vertically as cards
        self.card_container = ctk.CTkFrame(self.left, fg_color="transparent")
        self.card_container.pack(fill="both", expand=True, padx=24, pady=(10, 7))

        # Compact action tray stays visible at supported window sizes.
        btn_area = ctk.CTkFrame(self.left, fg_color=SIDEBAR_SOFT,
                                corner_radius=16, border_width=1,
                                border_color=SIDEBAR_BORDER)
        btn_area.pack(fill="x", padx=20, pady=(0, 14))

        self.upload_btn = ctk.CTkButton(
            btn_area, text="Upload an image", font=("Segoe UI", 12, "bold"),
            fg_color=ACCENT, text_color="white",
            hover_color=ACCENT_HOVER, border_width=0,
            corner_radius=10, height=36,
            command=self._upload_image)
        self.upload_btn.pack(fill="x", padx=12, pady=(10, 7))

        self.shuffle_btn = ctk.CTkButton(
            btn_area, text="Shuffle samples", font=("Segoe UI", 12, "bold"),
            fg_color="transparent", text_color=SIDEBAR_TEXT,
            hover_color=SIDEBAR_BORDER, border_width=1,
            border_color=SIDEBAR_BORDER, corner_radius=10, height=34,
            command=self._shuffle_images)
        self.shuffle_btn.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkFrame(btn_area, fg_color=SIDEBAR_BORDER, height=1).pack(
            fill="x", padx=12, pady=(0, 8))
        
        self.test_session_var = ctk.BooleanVar(value=False)
        self.test_session_switch = ctk.CTkSwitch(
            btn_area, text="Evaluation session", font=("Segoe UI", 11, "bold"),
            text_color=SIDEBAR_TEXT, progress_color=ACCENT,
            button_color=SURFACE, button_hover_color=ACCENT_TINT,
            variable=self.test_session_var, command=self._toggle_test_session)
        self.test_session_switch.pack(anchor="w", padx=12, pady=(0, 7))

        test_options = ctk.CTkFrame(btn_area, fg_color="transparent")
        self.test_options = test_options

        self.test_count_entry = ctk.CTkEntry(
            test_options, placeholder_text="Count", width=72, height=28,
            font=FONT_SMALL, fg_color=SIDEBAR, text_color=SIDEBAR_TEXT,
            border_color=SIDEBAR_BORDER, corner_radius=8)
        self.test_count_entry.insert(0, "10")
        self.test_count_entry.pack(side="left")
        self.test_count_entry.configure(state="disabled")

        self.auto_test_var = ctk.BooleanVar(value=False)
        self.auto_test_switch = ctk.CTkSwitch(
            test_options, text="Auto-run", font=FONT_SMALL,
            text_color=SIDEBAR_MUTED, progress_color=ACCENT,
            button_color=SURFACE, button_hover_color=ACCENT_TINT,
            variable=self.auto_test_var, width=42)
        self.auto_test_switch.pack(side="right")
        self.auto_test_switch.configure(state="disabled")

        self.run_test_btn = ctk.CTkButton(
            btn_area, text="Start evaluation", font=("Segoe UI", 11, "bold"),
            fg_color=ACCENT, text_color="white", hover_color=ACCENT_HOVER,
            corner_radius=9, height=32, command=self._start_test_session)
        self.run_test_btn.configure(state="disabled")

    # ── Right Panel ──────────────────────────────────────────────────────────
    def _build_right_panel(self):
        self.right = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.right.grid(row=0, column=1, sticky="nsew")
        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.right, fg_color=SURFACE, corner_radius=0,
                              height=76, border_width=1, border_color=BORDER)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        header_copy = ctk.CTkFrame(header, fg_color="transparent")
        header_copy.pack(side="left", padx=28, pady=15)
        ctk.CTkLabel(header_copy, text="Recognition workspace",
                     font=("Segoe UI", 16, "bold"), text_color=TEXT_PRIMARY,
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(header_copy, text="Inspect a sample and review the model output",
                     font=FONT_SMALL, text_color=TEXT_MUTED,
                     anchor="w").pack(anchor="w", pady=(2, 0))

        header_btns = ctk.CTkFrame(header, fg_color="transparent")
        header_btns.pack(side="right", padx=22)

        # Session Tracker Banner (hidden by default)
        self.session_banner = ctk.CTkFrame(self.right, fg_color=ACCENT_LIGHT, corner_radius=8,
                                           border_width=1, border_color=ACCENT)
        self.session_banner.place(relx=0.54, y=17, anchor="n")
        self.session_banner.place_forget()
        
        self.session_label = ctk.CTkLabel(self.session_banner, text="", font=("Segoe UI", 13, "bold"), text_color=ACCENT)
        self.session_label.pack(padx=24, pady=8)

        # Session State Variables
        self.session_active = False
        self.session_target = 0
        self.session_current = 0
        self.session_correct = 0

        ctk.CTkButton(header_btns, text="Statistics", width=92, height=36,
                      font=("Segoe UI", 11, "bold"),
                      fg_color=SURFACE_SOFT, text_color=TEXT_SECONDARY,
                      hover_color=BORDER, corner_radius=9,
                      border_width=1, border_color=BORDER,
                      command=self._show_stats).pack(side="left", padx=(0, 8))

        ctk.CTkButton(header_btns, text="About", width=72, height=36,
                      font=("Segoe UI", 11, "bold"),
                      fg_color=TEXT_PRIMARY, text_color="white",
                      hover_color="#1E293B", corner_radius=9,
                      command=self._show_settings).pack(side="left")

        # The workspace scrolls so developer details remain accessible.
        self.center = ctk.CTkScrollableFrame(
            self.right, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=TEXT_MUTED)
        self.center.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

        self._show_empty_state()

    def _show_stats(self):
        """Renders training statistics inline in the right panel."""
        self._clear_center()

        top_row = ctk.CTkFrame(self.center, fg_color="transparent")
        top_row.pack(fill="x", pady=(0, 4))
        ctk.CTkButton(top_row, text="✕  Back", width=80, height=32,
                      font=FONT_SMALL, fg_color="transparent",
                      text_color=TEXT_MUTED, hover_color=BORDER,
                      corner_radius=6,
                      command=self._show_empty_state).pack(anchor="w")

        card = ctk.CTkFrame(self.center, fg_color=SURFACE,
                            corner_radius=16, border_width=1,
                            border_color=BORDER)
        card.pack(padx=20, pady=(0, 10))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=32, pady=28)

        ctk.CTkLabel(inner, text="Training Statistics",
                     font=FONT_HEADING, text_color=TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkFrame(inner, fg_color=BORDER, height=1).pack(fill="x", pady=(8, 14))

        epochs = self.config.get("epochs_trained", 100)
        test_acc = self.config.get("test_accuracy", 66.63)
        best_val = self.config.get("best_val_accuracy", "N/A")
        dataset_size = 6000
        total_forward_passes = epochs * dataset_size
        
        # LinearModel: Linear(64*64, 3) = 4096 * 3 + 3 = 12291
        model_params = 12291

        import datetime
        today = datetime.datetime.now().strftime("%B %d, %Y")

        lines = [
            ("Date", today),
            ("Total Parameters", f"{model_params:,} saved weights"),
            ("Total Epochs", f"{epochs} complete cycles"),
            ("Images Processed", f"~{total_forward_passes:,} forward passes"),
            ("Best Val Accuracy", f"{best_val}%"),
            ("Final Eval Accuracy", f"{test_acc}%"),

        ]

        for label, value in lines:
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=label, font=("Segoe UI", 11, "bold"),
                         text_color=TEXT_SECONDARY, width=130, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, font=FONT_SMALL,
                         text_color=TEXT_PRIMARY, anchor="w").pack(side="left", padx=(8, 0))

    def _show_settings(self):
        """Renders project status information inline in the right panel."""
        self._clear_center()

        # Back / close button row
        top_row = ctk.CTkFrame(self.center, fg_color="transparent")
        top_row.pack(fill="x", pady=(0, 4))

        ctk.CTkButton(top_row, text="✕  Back", width=80, height=32,
                      font=FONT_SMALL, fg_color="transparent",
                      text_color=TEXT_MUTED, hover_color=BORDER,
                      corner_radius=6,
                      command=self._show_empty_state).pack(anchor="w")

        # Card
        card = ctk.CTkFrame(self.center, fg_color=SURFACE,
                            corner_radius=16, border_width=1,
                            border_color=BORDER)
        card.pack(padx=20, pady=(0, 10))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=32, pady=28)

        # Header
        ctk.CTkLabel(inner, text="Project Status",
                     font=FONT_HEADING, text_color=TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkFrame(inner, fg_color=BORDER, height=1).pack(fill="x", pady=(8, 14))

        # Info rows
        num_classes = len(self.config.get("class_to_idx", {}))
        class_names = ", ".join(self.config.get("class_to_idx", {}).keys())
        epochs = self.config.get("epochs_trained", 15)
        test_acc = self.config.get("test_accuracy", "N/A")

        lines = [
            ("Stage", "Phase 21 — Real Inference"),
            ("Architecture", self.config.get("architecture", "LinearModel")),
            ("Classes", f"{num_classes}  ({class_names})"),
            ("Image Size", f"{self.config.get('image_width', 64)} × {self.config.get('image_height', 64)} px"),
            ("Dataset", "~6,000 augmented synthetic images"),
            ("Training", f"{epochs} epochs, SGD (lr=0.01), batch size 64"),
            ("Test Accuracy", f"{test_acc}%"),
            ("Capabilities", "Recognizes ሀ, ለ, መ from synthetic images"),
        ]

        limitations = [
            "Only 3 Amharic characters are supported.",
            "Model is a simple linear network (no CNN yet).",
            "Trained on synthetic data only, not real handwriting.",
            "Accuracy is limited due to heavy augmentation vs. simple architecture.",
        ]

        for label, value in lines:
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=label, font=("Segoe UI", 11, "bold"),
                         text_color=TEXT_SECONDARY, width=110, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, font=FONT_SMALL,
                         text_color=TEXT_PRIMARY, anchor="w",
                         wraplength=300).pack(side="left", padx=(8, 0))

        ctk.CTkFrame(inner, fg_color=BORDER, height=1).pack(fill="x", pady=(12, 8))
        ctk.CTkLabel(inner, text="Known Limitations",
                     font=("Segoe UI", 11, "bold"), text_color=DANGER,
                     anchor="w").pack(anchor="w")

        for lim in limitations:
            ctk.CTkLabel(inner, text=f"•  {lim}", font=FONT_SMALL,
                         text_color=TEXT_SECONDARY, anchor="w",
                         wraplength=380).pack(anchor="w", pady=1)

    def _clear_center(self):
        for w in self.center.winfo_children():
            w.destroy()

    def _show_empty_state(self):
        self._clear_center()

        hero = ctk.CTkFrame(self.center, fg_color=SURFACE,
                            corner_radius=24, border_width=1,
                            border_color=BORDER)
        hero.pack(padx=40, pady=(74, 30))

        inner = ctk.CTkFrame(hero, fg_color="transparent")
        inner.pack(padx=70, pady=54)

        badge = ctk.CTkLabel(inner, text="  MODEL READY  ",
                             font=("Segoe UI", 9, "bold"),
                             text_color=ACCENT, fg_color=ACCENT_LIGHT,
                             corner_radius=10, height=24)
        badge.pack(pady=(0, 18))

        glyph = ctk.CTkFrame(inner, width=96, height=96,
                             fg_color=ACCENT_LIGHT, corner_radius=28,
                             border_width=1, border_color=ACCENT_TINT)
        glyph.pack()
        glyph.pack_propagate(False)
        ctk.CTkLabel(glyph, text="ሀ", font=("Segoe UI", 48, "bold"),
                     text_color=ACCENT).place(relx=0.5, rely=0.48, anchor="center")

        ctk.CTkLabel(inner, text="Recognize Amharic characters",
                     font=("Segoe UI", 26, "bold"),
                     text_color=TEXT_PRIMARY).pack(pady=(22, 7))
        ctk.CTkLabel(
            inner,
            text="Choose a dataset sample or upload your own image.\nThe prediction and confidence will appear here.",
            font=FONT_SUBTITLE, text_color=TEXT_SECONDARY,
            justify="center").pack()

        steps = ctk.CTkFrame(inner, fg_color="transparent")
        steps.pack(pady=(28, 0))
        for number, label in (("01", "Select image"), ("02", "Run model"), ("03", "Review result")):
            step = ctk.CTkFrame(steps, fg_color=SURFACE_SOFT,
                                corner_radius=12, border_width=1,
                                border_color=BORDER)
            step.pack(side="left", padx=5)
            ctk.CTkLabel(step, text=number, font=("Segoe UI", 10, "bold"),
                         text_color=ACCENT).pack(side="left", padx=(12, 7), pady=9)
            ctk.CTkLabel(step, text=label, font=FONT_SMALL,
                         text_color=TEXT_SECONDARY).pack(side="left", padx=(0, 12), pady=9)

    def _show_loading(self):
        self._clear_center()

        loading_card = ctk.CTkFrame(self.center, fg_color=SURFACE,
                                    corner_radius=22, border_width=1,
                                    border_color=BORDER)
        loading_card.pack(padx=40, pady=(150, 30))
        loading_inner = ctk.CTkFrame(loading_card, fg_color="transparent")
        loading_inner.pack(padx=76, pady=48)

        ctk.CTkLabel(loading_inner, text="AI",
                     font=("Segoe UI", 13, "bold"), text_color="white",
                     fg_color=ACCENT, width=48, height=48,
                     corner_radius=15).pack(pady=(0, 16))
        self.loading_label = ctk.CTkLabel(loading_inner, text="Analyzing image",
                                          font=("Segoe UI", 18, "bold"),
                                          text_color=TEXT_PRIMARY)
        self.loading_label.pack(pady=(0, 6))
        ctk.CTkLabel(loading_inner, text="Preparing the image for recognition",
                     font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=(0, 18))

        self.progress = ctk.CTkProgressBar(loading_inner, width=280, height=8,
                                           progress_color=ACCENT,
                                           fg_color=BORDER, corner_radius=4)
        self.progress.set(0)
        self.progress.pack()

    def _show_result(self, char, conf, logits_list, probs_list, source_file, true_label, image_path):
        self._clear_center()

        is_low = conf < 80.0
        is_known = true_label in self.config.get("class_to_idx", {})
        is_correct = (char == true_label) if is_known else False
        
        # Track this result in the active test session (if any)
        self._track_session_result(char, true_label)

        # ── Result card ──────────────────────────────────────────────────
        card = ctk.CTkFrame(self.center, fg_color=SURFACE,
                            corner_radius=22, border_width=1,
                            border_color=BORDER)
        card.pack(padx=36, pady=(22, 30))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=48, pady=38)

        # Status pill
        pill_bg = DANGER_LIGHT if is_low else ACCENT_LIGHT
        pill_fg = DANGER if is_low else ACCENT
        pill_text = "Low confidence" if is_low else "High confidence"

        pill = ctk.CTkLabel(inner, text=f"  {pill_text}  ",
                            font=FONT_TINY, text_color=pill_fg,
                            fg_color=pill_bg, corner_radius=10, height=22)
        pill.pack(pady=(0, 16))

        # ── Side-by-side: selected image | predicted character ───────
        pair_frame = ctk.CTkFrame(inner, fg_color="transparent")
        pair_frame.pack(pady=(0, 8))

        # Left column: Selected Image
        left_col = ctk.CTkFrame(pair_frame, fg_color="transparent")
        left_col.pack(side="left", padx=(0, 24))

        ctk.CTkLabel(left_col, text="Selected Image",
                     font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_SECONDARY).pack(pady=(0, 8))

        sel_img = Image.open(image_path).convert("RGB").resize((96, 96), Image.Resampling.LANCZOS)
        self._result_img_ref = ctk.CTkImage(light_image=sel_img, size=(96, 96))

        img_card = ctk.CTkFrame(left_col, fg_color=BG, corner_radius=12,
                                border_width=1, border_color=BORDER)
        img_card.pack()
        ctk.CTkLabel(img_card, image=self._result_img_ref, text="").pack(padx=12, pady=12)

        # Arrow
        ctk.CTkLabel(pair_frame, text="→", font=("Segoe UI", 28),
                     text_color=TEXT_MUTED).pack(side="left", padx=(0, 24), pady=(20, 0))

        # Right column: Prediction
        right_col = ctk.CTkFrame(pair_frame, fg_color="transparent")
        right_col.pack(side="left")

        ctk.CTkLabel(right_col, text="Prediction",
                     font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_SECONDARY).pack(pady=(0, 8))

        pred_card = ctk.CTkFrame(right_col, fg_color=ACCENT_LIGHT if not is_low else DANGER_LIGHT,
                                  corner_radius=12, border_width=1,
                                  border_color=ACCENT if not is_low else DANGER)
        pred_card.pack()
        ctk.CTkLabel(pred_card, text=char, font=FONT_CHAR,
                     text_color=TEXT_PRIMARY).pack(padx=24, pady=16)

        # Thin line
        ctk.CTkFrame(inner, fg_color=BORDER, height=1).pack(fill="x", pady=(12, 16))

        # Confidence section
        conf_color = DANGER if is_low else ACCENT

        ctk.CTkLabel(inner, text=f"{conf:.1f}%", font=FONT_CONF,
                     text_color=conf_color).pack()

        bar = ctk.CTkProgressBar(inner, width=200, height=6,
                                  progress_color=conf_color,
                                  fg_color=BORDER, corner_radius=3)
        bar.set(conf / 100.0)
        bar.pack(pady=(6, 4))

        ctk.CTkLabel(inner, text="Model Confidence",
                     font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=(0, 16))

        # Thin line
        ctk.CTkFrame(inner, fg_color=BORDER, height=1).pack(fill="x", pady=(0, 16))

        # Outcome summary
        is_known = true_label in self.config.get("class_to_idx", {})
        if is_known:
            is_correct = (char == true_label)
            result_text = "CORRECT" if is_correct else "WRONG"
            result_color = SUCCESS if is_correct else DANGER
            result_bg = SUCCESS_LIGHT if is_correct else DANGER_LIGHT
            ans_text = true_label
        else:
            result_text = "USER UPLOAD"
            result_color = TEXT_MUTED
            result_bg = SURFACE_SOFT
            ans_text = "N/A"

        outcome_header = ctk.CTkFrame(inner, fg_color="transparent")
        outcome_header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(outcome_header, text="Outcome summary",
                     font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkLabel(outcome_header, text=f"  {result_text}  ",
                     font=("Segoe UI", 9, "bold"),
                     text_color=result_color, fg_color=result_bg,
                     corner_radius=9, height=22).pack(side="right")

        summary = ctk.CTkFrame(inner, fg_color=SURFACE_SOFT,
                               corner_radius=12, border_width=1,
                               border_color=BORDER)
        summary.pack(fill="x", pady=(0, 14))

        metrics = (
            ("PREDICTION", char),
            ("EXPECTED", ans_text),
            ("CONFIDENCE", f"{conf:.1f}%"),
        )
        for metric_index, (label, value) in enumerate(metrics):
            metric = ctk.CTkFrame(summary, fg_color="transparent")
            metric.pack(side="left", expand=True, fill="both", padx=14, pady=13)
            ctk.CTkLabel(metric, text=label, font=("Segoe UI", 8, "bold"),
                         text_color=TEXT_MUTED).pack()
            ctk.CTkLabel(metric, text=value, font=("Segoe UI", 15, "bold"),
                         text_color=TEXT_PRIMARY).pack(pady=(3, 0))
            if metric_index < len(metrics) - 1:
                ctk.CTkFrame(summary, fg_color=BORDER, width=1).pack(
                    side="left", fill="y", pady=10)

        # Developer details
        ctk.CTkLabel(inner, text="Developer details", font=("Segoe UI", 11, "bold"),
                     text_color=TEXT_SECONDARY, anchor="w").pack(anchor="w", pady=(2, 7))

        details = (
            f"Source:   {source_file}\n"
            f"Logits:   {logits_list}\n"
            f"Probs:    {probs_list}"
        )
        detail_panel = ctk.CTkFrame(inner, fg_color="#F1F5F9",
                                    corner_radius=10)
        detail_panel.pack(fill="x")
        ctk.CTkLabel(detail_panel, text=details, font=FONT_MONO,
                     text_color=TEXT_SECONDARY, justify="left",
                     anchor="w").pack(fill="x", padx=14, pady=11)

    # ── Image Cards ──────────────────────────────────────────────────────────
    def _shuffle_images(self):
        for w in self.card_container.winfo_children():
            w.destroy()

        self.thumb_refs = []
        images = get_random_images(5)

        for index, path in enumerate(images, start=1):
            img = Image.open(path).convert("RGB").resize((38, 38), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, size=(38, 38))
            self.thumb_refs.append(ctk_img)

            card = ctk.CTkButton(
                self.card_container,
                text=f"   Sample {index}   ·   {path.parent.name}",
                image=ctk_img,
                compound="left",
                anchor="w",
                font=("Segoe UI", 11, "bold"),
                text_color=SIDEBAR_TEXT,
                fg_color=SIDEBAR_SOFT,
                hover_color=SIDEBAR_BORDER,
                border_width=1,
                border_color=SIDEBAR_BORDER,
                corner_radius=12,
                height=48,
                command=lambda p=path: self._on_select(p))
            card.pack(fill="x", pady=2)

        self._show_empty_state()

    def _upload_image(self):
        file_path = fd.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self._on_select(Path(file_path))

    def _on_select(self, path):
        # Cancel any previous thread's UI updates by incrementing ID
        if not hasattr(self, '_inference_id'):
            self._inference_id = 0
        self._inference_id += 1
        
        self._show_loading()
        threading.Thread(target=self._run_inference, args=(path, self._inference_id), daemon=True).start()

    def _run_inference(self, image_path, inf_id):
        # Smooth progress animation
        for i in range(1, 6):
            time.sleep(0.15)
            if self._inference_id == inf_id:
                self.after(0, lambda v=i/10: self.progress.set(v))

        if self._inference_id == inf_id:
            self.after(0, lambda: self.loading_label.configure(text="Running model"))

        # ALWAYS convert to RGB before transform to perfectly match training ImageFolder
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)

        probs = F.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

        char = self.idx_to_class[top_idx.item()]
        conf = top_prob.item() * 100

        for i in range(6, 11):
            time.sleep(0.1)
            if self._inference_id == inf_id:
                self.after(0, lambda v=i/10: self.progress.set(v))

        logits_list = [round(float(x), 2) for x in logits.numpy()[0]]
        probs_list = [round(float(x), 3) for x in probs.numpy()[0]]
        src = os.path.basename(image_path)
        lbl = image_path.parent.name

        time.sleep(0.2)
        if self._inference_id == inf_id:
            self.after(0, self._show_result, char, conf, logits_list, probs_list, src, lbl, image_path)

    # ── Test Mode ─────────────────────────────────────────────────────────────
    def _toggle_test_session(self):
        on = self.test_session_var.get()
        if on:
            self.test_options.pack(fill="x", padx=12, pady=(0, 7))
            self.run_test_btn.pack(fill="x", padx=12, pady=(0, 10))
            self.test_count_entry.configure(state="normal")
            self.run_test_btn.configure(state="normal")
            self.auto_test_switch.configure(state="normal")
            self.upload_btn.configure(state="disabled")
        else:
            self.test_options.pack_forget()
            self.run_test_btn.pack_forget()
            self.test_count_entry.configure(state="disabled")
            self.run_test_btn.configure(state="disabled")
            self.auto_test_switch.configure(state="disabled")
            self.auto_test_var.set(False)
            self.upload_btn.configure(state="normal")
            if self.session_active:
                self._end_session(show_results=False)

    def _start_test_session(self):
        try:
            count = int(self.test_count_entry.get())
            if count <= 0: return
        except ValueError:
            return
            
        self.session_active = True
        self.session_target = count
        self.session_current = 0
        self.session_correct = 0
        self.session_wrong = 0
        
        self.run_test_btn.configure(state="disabled", text="Running...")
        self.test_count_entry.configure(state="disabled")
        self.test_session_switch.configure(state="disabled")
        self.auto_test_switch.configure(state="disabled")
        self.upload_btn.configure(state="disabled")
        
        self._update_session_banner()
        self.session_banner.place(relx=0.5, y=16, anchor="n")
        
        # Auto-shuffle to give fresh images
        self._shuffle_images()
        
        # If auto mode is on, start the automatic visual loop
        if self.auto_test_var.get():
            if not hasattr(self, '_inference_id'):
                self._inference_id = 0
            self._inference_id += 1
            threading.Thread(
                target=self._run_auto_test_loop,
                args=(count, self._inference_id),
                daemon=True
            ).start()

    def _run_auto_test_loop(self, count, inf_id):
        """Auto mode: visually clicks through images for the user."""
        images = get_random_images(count)
        for img_path in images:
            if self._inference_id != inf_id or not getattr(self, 'session_active', False):
                return
            
            # Show loading
            self.after(0, self._show_loading)
            time.sleep(0.4)
            if self._inference_id != inf_id: return
            
            self.after(0, lambda: self.loading_label.configure(text="Evaluating..."))
            for i in range(1, 6):
                time.sleep(0.06)
                if self._inference_id == inf_id:
                    self.after(0, lambda v=i/10: self.progress.set(v))

            # Run inference
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(tensor)

            probs = F.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
            char = self.idx_to_class[top_idx.item()]
            conf = top_prob.item() * 100

            for i in range(6, 11):
                time.sleep(0.06)
                if self._inference_id == inf_id:
                    self.after(0, lambda v=i/10: self.progress.set(v))

            logits_list = [round(float(x), 2) for x in logits.numpy()[0]]
            probs_list = [round(float(x), 3) for x in probs.numpy()[0]]
            src = os.path.basename(img_path)
            lbl = img_path.parent.name
            
            time.sleep(0.2)
            if self._inference_id == inf_id:
                self.after(0, self._show_result, char, conf, logits_list, probs_list, src, lbl, img_path)
            
            # Pause so user can see each result
            time.sleep(1.5)

    def _track_session_result(self, predicted, true_label):
        """Called after every prediction while session is active."""
        if not self.session_active:
            return
            
        self.session_current += 1
        is_known = true_label in self.config.get("class_to_idx", {})
        if is_known and predicted == true_label:
            self.session_correct += 1
        else:
            self.session_wrong += 1
        
        self._update_session_banner()
        
        if self.session_current >= self.session_target:
            self.session_active = False  # prevent double-fire
            self.after(1200, self._end_session)

    def _update_session_banner(self):
        done = self.session_current
        target = self.session_target
        remaining = target - done
        pct = (self.session_correct / done * 100) if done > 0 else 0
        
        self.session_label.configure(
            text=f"  {done}/{target} done   ·   ✓ {self.session_correct}   ✗ {self.session_wrong}   ·   {remaining} left   ·   {pct:.0f}%  "
        )
        
    def _end_session(self, show_results=True):
        was_active = self.session_active or True  # may already be False from track
        done = self.session_current
        correct = self.session_correct
        
        self.session_active = False
        self.session_banner.place_forget()
        self.run_test_btn.configure(state="normal", text="▶ Start Session")
        self.test_session_switch.configure(state="normal")
        self.auto_test_switch.configure(state="normal" if self.test_session_var.get() else "disabled")
        
        if self.test_session_var.get():
            self.test_count_entry.configure(state="normal")
            self.upload_btn.configure(state="disabled")
        else:
            self.upload_btn.configure(state="normal")
            
        if show_results and done > 0:
            accuracy = (correct / done) * 100
            self._show_session_result(correct, done, accuracy)

    def _show_session_result(self, correct, total, accuracy):
        self._clear_center()

        card = ctk.CTkFrame(self.center, fg_color=SURFACE,
                            corner_radius=20, border_width=1,
                            border_color=BORDER)
        card.pack(padx=20, pady=40)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=48, pady=40)
        
        # Title
        ctk.CTkLabel(inner, text="Test Session Complete",
                     font=("Segoe UI", 20, "bold"), text_color=TEXT_PRIMARY).pack(pady=(0, 8))
        ctk.CTkLabel(inner, text=f"Evaluated {total} images",
                     font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=(0, 24))
        
        # Score boxes side by side
        score_frame = ctk.CTkFrame(inner, fg_color="transparent")
        score_frame.pack(fill="x", pady=(0, 24))
        
        wrong = total - correct
        
        # Correct box
        corr_box = ctk.CTkFrame(score_frame, fg_color="#ECFDF5", corner_radius=12,
                                border_width=1, border_color="#10B981")
        corr_box.pack(side="left", padx=8, expand=True, fill="both")
        ctk.CTkLabel(corr_box, text="CORRECT", font=FONT_TINY,
                     text_color="#10B981").pack(pady=(14, 2))
        ctk.CTkLabel(corr_box, text=str(correct), font=("Segoe UI", 36, "bold"),
                     text_color="#10B981").pack(pady=(0, 14))
        
        # Wrong box
        wrong_box = ctk.CTkFrame(score_frame, fg_color=DANGER_LIGHT, corner_radius=12,
                                 border_width=1, border_color=DANGER)
        wrong_box.pack(side="left", padx=8, expand=True, fill="both")
        ctk.CTkLabel(wrong_box, text="WRONG", font=FONT_TINY,
                     text_color=DANGER).pack(pady=(14, 2))
        ctk.CTkLabel(wrong_box, text=str(wrong), font=("Segoe UI", 36, "bold"),
                     text_color=DANGER).pack(pady=(0, 14))

        # Accuracy percentage
        color = "#10B981" if accuracy >= 75 else DANGER
        
        ctk.CTkLabel(inner, text=f"{accuracy:.1f}%",
                     font=("Segoe UI", 42, "bold"), text_color=color).pack(pady=(0, 4))
        ctk.CTkLabel(inner, text="Final Accuracy",
                     font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=(0, 8))
                     
        bar = ctk.CTkProgressBar(inner, width=280, height=8,
                                 progress_color=color, fg_color=BORDER, corner_radius=4)
        bar.set(accuracy / 100.0)
        bar.pack(pady=(0, 28))
                     
        ctk.CTkButton(inner, text="Done", width=140, height=40, font=FONT_BODY,
                      fg_color=ACCENT, text_color="white",
                      hover_color="#059669", corner_radius=8,
                      command=self._show_empty_state).pack()


if __name__ == "__main__":
    app = AmharicAIApp()
    app.mainloop()
