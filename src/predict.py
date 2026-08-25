import argparse
import torch
import json
import os
import sys
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

from simple_model import SimpleModel

def predict_character(image_path):
    # 1. Load Configuration
    config_path = "models/model_config.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Please train and save the model first.")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    idx_to_class = {v: k for k, v in config["class_to_idx"].items()}

    # 2. Recreate Model Architecture and Load Weights
    model = SimpleModel()
    weights_path = "models/simple_model_weights.pth"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 3. Preprocess the Image
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)
        
    # Standardize image to match training transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config["image_height"], config["image_width"])),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) # Add batch dimension -> [1, 1, 64, 64]
    
    # 4. Run Model
    with torch.no_grad():
        logits = model(image_tensor)
        
    # 5. Convert Logits to Probabilities using Softmax
    probabilities = F.softmax(logits, dim=1)
    
    # 6. Select highest probability
    top_prob, top_class_idx = torch.max(probabilities, dim=1)
    
    # 7. Convert class number to character
    predicted_character = idx_to_class[top_class_idx.item()]
    confidence_percentage = top_prob.item() * 100
    
    # 8. Display Prediction
    print(f"Prediction: {predicted_character}")
    print(f"Confidence: {confidence_percentage:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict an Amharic character from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    
    predict_character(args.image_path)
