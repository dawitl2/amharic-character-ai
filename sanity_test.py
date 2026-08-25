import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from simple_model import SimpleModel

with open("models/model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

model = SimpleModel()
model.load_state_dict(torch.load("models/best_model_weights.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config["image_height"], config["image_width"])),
    transforms.ToTensor()
])

idx_to_class = {v: k for k, v in config["class_to_idx"].items()}

correct = 0
total = 0
results = {k: {"correct": 0, "total": 0} for k in idx_to_class.values()}

for cls_name in idx_to_class.values():
    img_dir = Path("data") / cls_name
    for i, img_path in enumerate(list(img_dir.glob("*.png"))[:100]): # test 100 per class
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
        top_idx = torch.argmax(logits, dim=1).item()
        pred_cls = idx_to_class[top_idx]
        
        total += 1
        results[cls_name]["total"] += 1
        if pred_cls == cls_name:
            correct += 1
            results[cls_name]["correct"] += 1

print(f"Overall Accuracy on 300 train images: {correct/total*100:.2f}%")
for k, v in results.items():
    print(f"Class {k}: {v['correct']}/{v['total']} ({v['correct']/v['total']*100:.2f}%)")
