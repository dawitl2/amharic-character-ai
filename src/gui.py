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

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_model import SimpleModel

ctk.set_appearance_mode("Light")

# ─── Color Palette ───────────────────────────────────────────────────────────
BG           = "#FAFAFA"
SURFACE      = "#FFFFFF"
BORDER       = "#E8E8E8"
TEXT_PRIMARY  = "#1A1A1A"
TEXT_SECONDARY= "#6B6B6B"
TEXT_MUTED    = "#9CA3AF"
ACCENT        = "#0EA47A"
ACCENT_HOVER  = "#0C8A66"
ACCENT_LIGHT  = "#ECFDF5"
DANGER        = "#EF4444"
DANGER_LIGHT  = "#FEF2F2"

# ─── Fonts ───────────────────────────────────────────────────────────────────
FONT_TITLE    = ("Segoe UI", 22, "bold")
FONT_SUBTITLE = ("Segoe UI", 13)
FONT_HEADING  = ("Segoe UI", 16, "bold")
FONT_BODY     = ("Segoe UI", 13)
FONT_SMALL    = ("Segoe UI", 11)
FONT_TINY     = ("Segoe UI", 10)
FONT_CHAR     = ("Segoe UI", 64, "bold")
FONT_CONF     = ("Segoe UI", 28, "bold")
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
        self.geometry("960x640")
        self.minsize(860, 580)
        self.configure(fg_color=BG)

        # Load model once
        self._load_model()

        # Two‑column grid: left selector | right results
        self.grid_columnconfigure(0, weight=0, minsize=200)
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

        self.model = SimpleModel()
        
        best_weights_path = "models/best_model_weights.pth"
        legacy_weights_path = "models/simple_model_weights.pth"
        
        if os.path.exists(best_weights_path):
            self.model.load_state_dict(torch.load(best_weights_path))
        else:
            self.model.load_state_dict(torch.load(legacy_weights_path))
            
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.config["image_height"], self.config["image_width"])),
            transforms.ToTensor()
        ])

    # ── Left Panel ───────────────────────────────────────────────────────────
    def _build_left_panel(self):
        self.left = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0,
                                  border_width=1, border_color=BORDER)
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_propagate(False)
        self.left.configure(width=200)

        # Title area
        title_area = ctk.CTkFrame(self.left, fg_color="transparent")
        title_area.pack(fill="x", padx=24, pady=(28, 0))

        ctk.CTkLabel(title_area, text="Select an Image",
                     font=FONT_HEADING, text_color=TEXT_PRIMARY,
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(title_area,
                     text="5 random samples from the dataset.\nClick one to run inference.",
                     font=FONT_SMALL, text_color=TEXT_MUTED,
                     anchor="w", justify="left").pack(anchor="w", pady=(4, 0))

        # Thin separator
        sep = ctk.CTkFrame(self.left, fg_color=BORDER, height=1)
        sep.pack(fill="x", padx=24, pady=(20, 0))

        # Image grid – 5 images arranged vertically as cards
        self.card_container = ctk.CTkFrame(self.left, fg_color="transparent")
        self.card_container.pack(fill="both", expand=True, padx=24, pady=(16, 8))

        # Shuffle button pinned to bottom
        btn_area = ctk.CTkFrame(self.left, fg_color="transparent")
        btn_area.pack(fill="x", padx=24, pady=(0, 24))

        self.shuffle_btn = ctk.CTkButton(
            btn_area, text="↻  Shuffle", font=FONT_BODY,
            fg_color=BG, text_color=TEXT_PRIMARY,
            hover_color=BORDER, border_width=1, border_color=BORDER,
            corner_radius=8, height=40,
            command=self._shuffle_images)
        self.shuffle_btn.pack(fill="x")

    # ── Right Panel ──────────────────────────────────────────────────────────
    def _build_right_panel(self):
        self.right = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.right.grid(row=0, column=1, sticky="nsew")

        # Top-right header buttons
        header_btns = ctk.CTkFrame(self.right, fg_color="transparent")
        header_btns.place(relx=1.0, x=-16, y=16, anchor="ne")

        ctk.CTkButton(header_btns, text="📊 Stats", width=70, height=32,
                      font=("Segoe UI", 12, "bold"),
                      fg_color="transparent", text_color=TEXT_MUTED,
                      hover_color=BORDER, corner_radius=6,
                      command=self._show_stats).pack(side="left", padx=(0, 8))

        ctk.CTkButton(header_btns, text="ℹ Info", width=60, height=32,
                      font=("Segoe UI", 12, "bold"),
                      fg_color="transparent", text_color=TEXT_MUTED,
                      hover_color=BORDER, corner_radius=6,
                      command=self._show_settings).pack(side="left")

        # Center wrapper
        self.center = ctk.CTkFrame(self.right, fg_color="transparent")
        self.center.place(relx=0.5, rely=0.5, anchor="center")

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
        
        # SimpleModel: Linear(64*64, 3) = 4096 * 3 + 3 = 12291
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
            ("Architecture", self.config.get("architecture", "SimpleModel")),
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
        ctk.CTkLabel(self.center, text="No image selected",
                     font=FONT_SUBTITLE, text_color=TEXT_MUTED).pack(pady=(0, 6))
        ctk.CTkLabel(self.center,
                     text="Pick a sample from the left panel to begin.",
                     font=FONT_SMALL, text_color=TEXT_MUTED).pack()

    def _show_loading(self):
        self._clear_center()
        self.loading_label = ctk.CTkLabel(self.center, text="Analyzing",
                                          font=FONT_SUBTITLE,
                                          text_color=TEXT_SECONDARY)
        self.loading_label.pack(pady=(0, 12))

        self.progress = ctk.CTkProgressBar(self.center, width=220,
                                            progress_color=ACCENT,
                                            fg_color=BORDER)
        self.progress.set(0)
        self.progress.pack()

    def _show_result(self, char, conf, logits_list, probs_list, source_file, true_label, image_path):
        self._clear_center()

        is_low = conf < 80.0

        # ── Result card ──────────────────────────────────────────────────
        card = ctk.CTkFrame(self.center, fg_color=SURFACE,
                            corner_radius=16, border_width=1,
                            border_color=BORDER)
        card.pack(padx=20, pady=10)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=36, pady=32)

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
        ctk.CTkLabel(inner, text="Outcome", font=("Segoe UI", 11, "bold"),
                     text_color=TEXT_SECONDARY, anchor="w").pack(anchor="w")
                     
        is_correct = (char == true_label)
        result_text = "CORRECT" if is_correct else "WRONG"
        result_color = "#10B981" if is_correct else DANGER

        outcome = (
            f"Prediction:     {char}\n"
            f"Correct Answer: {true_label}\n"
            f"Confidence:     {conf:.1f}%\n"
            f"Result:         {result_text}"
        )
        ctk.CTkLabel(inner, text=outcome, font=FONT_MONO,
                     text_color=result_color, justify="left",
                     anchor="w").pack(anchor="w", pady=(6, 12))

        # Developer details
        ctk.CTkLabel(inner, text="Developer Details", font=("Segoe UI", 11, "bold"),
                     text_color=TEXT_SECONDARY, anchor="w").pack(anchor="w")

        details = (
            f"Source:   {source_file}\n"
            f"Logits:   {logits_list}\n"
            f"Probs:    {probs_list}"
        )
        ctk.CTkLabel(inner, text=details, font=FONT_MONO,
                     text_color=TEXT_MUTED, justify="left",
                     anchor="w").pack(anchor="w", pady=(6, 0))

    # ── Image Cards ──────────────────────────────────────────────────────────
    def _shuffle_images(self):
        for w in self.card_container.winfo_children():
            w.destroy()

        self.thumb_refs = []
        images = get_random_images(5)

        for path in images:
            img = Image.open(path).convert("RGB").resize((48, 48), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, size=(48, 48))
            self.thumb_refs.append(ctk_img)

            card = ctk.CTkButton(
                self.card_container,
                text="",
                image=ctk_img,
                fg_color="transparent",
                hover_color=ACCENT_LIGHT,
                corner_radius=10,
                width=64, height=64,
                command=lambda p=path: self._on_select(p))
            card.pack(pady=4)

        self._show_empty_state()

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


if __name__ == "__main__":
    app = AmharicAIApp()
    app.mainloop()
