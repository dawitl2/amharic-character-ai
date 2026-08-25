import os
import sys
import json
import random
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

# Ensure simple_model can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_model import SimpleModel

def get_random_images(num_images=5):
    """Selects random images from the dataset to display in the UI."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
        
    all_images = list(data_dir.rglob("*.png"))
    if not all_images:
        return []
        
    return random.sample(all_images, min(num_images, len(all_images)))

class AmharicOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Amharic Character OCR (Light Mode)")
        self.root.geometry("800x600")
        self.root.configure(bg="#F7F7F8") # ChatGPT light mode background
        
        # Load Model
        self.load_model()
        
        # UI Setup
        self.setup_ui()
        
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
        
    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="Amharic Character AI", font=("Arial", 20, "bold"), bg="#F7F7F8", fg="#2D2D2D")
        header.pack(pady=20)
        
        instructions = tk.Label(self.root, text="Select one of the random images below to run real-time inference:", font=("Arial", 12), bg="#F7F7F8", fg="#555555")
        instructions.pack(pady=5)
        
        # Image Frame
        self.image_frame = tk.Frame(self.root, bg="#F7F7F8")
        self.image_frame.pack(pady=20)
        
        self.load_random_images_ui()
        
        # Refresh Button
        refresh_btn = tk.Button(self.root, text="Load New Images", command=self.load_random_images_ui, font=("Arial", 10), bg="#10A37F", fg="white", relief="flat", padx=10, pady=5)
        refresh_btn.pack(pady=10)
        
        # Results Frame
        self.results_frame = tk.Frame(self.root, bg="white", highlightbackground="#E5E5E5", highlightthickness=1, bd=0)
        self.results_frame.pack(pady=20, padx=40, fill="x")
        
        self.result_label = tk.Label(self.results_frame, text="Awaiting selection...", font=("Arial", 16), bg="white", fg="#2D2D2D", pady=10)
        self.result_label.pack()
        
        self.developer_details = tk.Label(self.results_frame, text="", font=("Courier", 10), bg="white", fg="#888888", justify="left")
        self.developer_details.pack(pady=5)
        
    def load_random_images_ui(self):
        # Clear existing images
        for widget in self.image_frame.winfo_children():
            widget.destroy()
            
        images = get_random_images(5)
        self.photo_images = [] # keep reference to avoid garbage collection
        
        for img_path in images:
            img = Image.open(img_path)
            img = img.resize((80, 80), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.photo_images.append(photo)
            
            btn = tk.Button(self.image_frame, image=photo, command=lambda p=img_path: self.run_inference(p), bg="white", relief="solid", bd=1)
            btn.pack(side="left", padx=10)
            
    def run_inference(self, image_path):
        img = Image.open(image_path)
        tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(tensor)
            
        probs = F.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)
        
        char = self.idx_to_class[top_idx.item()]
        conf = top_prob.item() * 100
        
        self.result_label.config(text=f"Prediction: {char}  |  Confidence: {conf:.1f}%")
        
        dev_info = (
            f"Developer Details:\n"
            f"File: {os.path.basename(image_path)}\n"
            f"Ground Truth Dir: {image_path.parent.name}\n"
            f"Raw Logits: {logits.numpy()[0]}\n"
            f"Probabilities: {probs.numpy()[0]}"
        )
        self.developer_details.config(text=dev_info)

if __name__ == "__main__":
    root = tk.Tk()
    app = AmharicOCRApp(root)
    root.mainloop()
