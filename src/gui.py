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
ctk.set_default_color_theme("green")

def get_random_images(num_images=5):
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    all_images = list(data_dir.rglob("*.png"))
    if not all_images:
        return []
    return random.sample(all_images, min(num_images, len(all_images)))

class AmharicAIChat(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Amharic Character AI")
        self.geometry("900x700")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header = ctk.CTkFrame(self, corner_radius=0, fg_color="#FFFFFF", height=60)
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.pack_propagate(False)
        ctk.CTkLabel(self.header, text="Amharic OCR Assistant", font=ctk.CTkFont(size=20, weight="bold"), text_color="#202123").pack(pady=15)
        
        # Chat area
        self.chat_frame = ctk.CTkScrollableFrame(self, fg_color="#F7F7F8")
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        
        # Load Model
        self.load_model()
        
        # Start initial prompt
        self.prompt_user_for_image()
        
    def load_model(self):
        config_path = "models/model_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.idx_to_class = {v: k for k, v in self.config["class_to_idx"].items()}
        self.model = SimpleModel()
        self.model.load_state_dict(torch.load("models/simple_model_weights.pth"))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.config["image_height"], self.config["image_width"])),
            transforms.ToTensor()
        ])
        
    def add_assistant_message(self, text, is_warning=False):
        msg_frame = ctk.CTkFrame(self.chat_frame, fg_color="#F7F7F8", corner_radius=0)
        msg_frame.pack(fill="x", padx=40, pady=10)
        
        icon = "⚠️" if is_warning else "🤖"
        color = "#D9534F" if is_warning else "#10A37F"
        
        icon_lbl = ctk.CTkLabel(msg_frame, text=icon, font=ctk.CTkFont(size=24), text_color=color, width=40)
        icon_lbl.pack(side="left", anchor="n", padx=(0, 10))
        
        text_lbl = ctk.CTkLabel(msg_frame, text=text, font=ctk.CTkFont(size=14), text_color="#343541", justify="left", wraplength=700)
        text_lbl.pack(side="left", anchor="n", pady=5)
        
        # Scroll to bottom
        self.after(50, self.scroll_to_bottom)
        return msg_frame, text_lbl
        
    def add_user_message(self, img_path):
        msg_frame = ctk.CTkFrame(self.chat_frame, fg_color="#FFFFFF", corner_radius=0)
        msg_frame.pack(fill="x", padx=40, pady=10)
        
        icon_lbl = ctk.CTkLabel(msg_frame, text="👤", font=ctk.CTkFont(size=24), text_color="#555555", width=40)
        icon_lbl.pack(side="left", anchor="n", padx=(0, 10))
        
        img = Image.open(img_path).resize((64, 64), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=img, size=(64, 64))
        
        img_lbl = ctk.CTkLabel(msg_frame, image=ctk_img, text="")
        img_lbl.image = ctk_img
        img_lbl.pack(side="left", anchor="n", pady=5)
        
        self.after(50, self.scroll_to_bottom)
        
    def prompt_user_for_image(self):
        msg_frame, _ = self.add_assistant_message("I am ready. Here are 5 random images from the dataset. Please click one to analyze:")
        
        # Add images below the message
        images = get_random_images(5)
        if not images:
            self.add_assistant_message("Error: No images found in data/ directory.", is_warning=True)
            return
            
        img_container = ctk.CTkFrame(msg_frame, fg_color="transparent")
        img_container.pack(anchor="w", padx=50, pady=10)
        
        self.selection_buttons = []
        for path in images:
            img = Image.open(path).resize((64, 64), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, size=(64, 64))
            
            btn = ctk.CTkButton(img_container, image=ctk_img, text="", width=64, height=64, 
                                fg_color="#E5E5E5", hover_color="#D1D5DB",
                                command=lambda p=path: self.on_image_selected(p))
            btn.pack(side="left", padx=10)
            self.selection_buttons.append(btn)
            
    def on_image_selected(self, path):
        # Disable buttons so user can't click multiple
        for btn in self.selection_buttons:
            btn.configure(state="disabled")
            
        self.add_user_message(path)
        
        # Create a placeholder "Analyzing..." message
        analysis_frame, text_lbl = self.add_assistant_message("Analyzing features...")
        
        # Run inference progressively in a thread
        threading.Thread(target=self.run_inference_progressive, args=(path, text_lbl, analysis_frame)).start()
        
    def run_inference_progressive(self, image_path, text_lbl, analysis_frame):
        # Fake delay for "progressive" feeling
        time.sleep(0.8)
        self.after(0, lambda: text_lbl.configure(text="Applying Softmax probabilities..."))
        time.sleep(0.8)
        
        # Real inference
        img = Image.open(image_path)
        tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(tensor)
            
        probs = F.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)
        
        char = self.idx_to_class[top_idx.item()]
        conf = top_prob.item() * 100
        
        is_warning = conf < 80.0
        final_text = f"**Prediction:** {char}\n**Confidence:** {conf:.1f}%\n\n"
        
        dev_info = (
            f"Developer Logs:\n"
            f"• Source File: {os.path.basename(image_path)}\n"
            f"• True Label: {image_path.parent.name}\n"
            f"• Raw Logits: {[round(float(x), 2) for x in logits.numpy()[0]]}\n"
            f"• Probabilities: {[round(float(x), 3) for x in probs.numpy()[0]]}"
        )
        
        final_text += dev_info
        
        # Update the UI
        self.after(0, self.finalize_analysis, text_lbl, final_text, is_warning)
        
    def finalize_analysis(self, text_lbl, final_text, is_warning):
        text_lbl.configure(text=final_text)
        if is_warning:
            text_lbl.configure(text_color="#D9534F")
            
        # Provide a button to analyze another
        btn = ctk.CTkButton(self.chat_frame, text="Analyze Another Image", fg_color="#10A37F", hover_color="#0B7A5E",
                            command=self.prompt_user_for_image)
        btn.pack(pady=20)
        self.scroll_to_bottom()
        
    def scroll_to_bottom(self):
        self.chat_frame._parent_canvas.yview_moveto(1.0)

if __name__ == "__main__":
    app = AmharicAIChat()
    app.mainloop()
