import os
import sys
import json
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# Ensure we're running from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cnn_model import CNNModel

def main():
    config_path = "models/model_config.json"
    weights_path = "models/best_model_weights.pth"
    
    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        print("Please train the CNN first to generate weights and config.")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    if "CNNModel" not in config.get("architecture", ""):
        print("The saved model is not a CNN. Please run train.py to train the CNN first.")
        return

    # Load Model
    idx_to_class = {v: k for k, v in config["class_to_idx"].items()}
    model = CNNModel(num_classes=len(idx_to_class))
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # Pick a random image from data folder
    from pathlib import Path
    data_dir = Path("data")
    all_images = list(data_dir.rglob("*.png"))
    if not all_images:
        print("No images found in data folder.")
        return
        
    image_path = random.choice(all_images)
    print(f"Visualizing features for: {image_path}")
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    img = Image.open(image_path).convert("L")
    x = transform(img).unsqueeze(0)  # Add batch dimension [1, 1, 64, 64]
    
    # We want to intercept the output after conv1 and conv2
    # We can do this by running the layers manually
    with torch.no_grad():
        import torch.nn.functional as F
        # Layer 1
        conv1_out = model.conv1(x)
        pool1_out = model.pool1(F.relu(conv1_out))
        
        # Layer 2
        conv2_out = model.conv2(pool1_out)
        pool2_out = model.pool2(F.relu(conv2_out))
        
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Original Image
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title(f"Original Image (Class: {image_path.parent.name})")
    axs[0].axis('off')
    
    # Conv1 Features (16 channels)
    # We will plot the first 16 channels in a 4x4 grid inside the second subplot
    # Actually, matplotlib subplots inside subplots is tricky. Let's just make a huge figure
    plt.close(fig)
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"CNN Feature Maps for {image_path.parent.name}", fontsize=16)
    
    # Original
    ax_orig = fig.add_subplot(2, 5, 1)
    ax_orig.imshow(img, cmap='gray')
    ax_orig.set_title("Original")
    ax_orig.axis('off')
    
    # Conv1 (Show 4 channels)
    for i in range(4):
        ax = fig.add_subplot(2, 5, i + 2)
        ax.imshow(pool1_out[0, i].numpy(), cmap='viridis')
        ax.set_title(f"Conv1 Filter {i}")
        ax.axis('off')
        
    # Conv2 (Show 5 channels)
    for i in range(5):
        ax = fig.add_subplot(2, 5, i + 6)
        ax.imshow(pool2_out[0, i].numpy(), cmap='plasma')
        ax.set_title(f"Conv2 Filter {i}")
        ax.axis('off')
        
    plt.tight_layout()
    
    # Save the visualization
    out_path = "models/cnn_visualization.png"
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
