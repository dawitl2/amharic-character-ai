import torch
import json
import os
from torchvision import transforms
from PIL import Image
import sys
sys.stdout.reconfigure(encoding='utf-8')

from simple_model import SimpleModel

print("--- Phase 20: Loading the Model ---")

# 1. Load Configuration
config_path = "models/model_config.json"
if not os.path.exists(config_path):
    raise FileNotFoundError("Configuration file not found. Did you run save_model.py?")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"Loaded config for architecture: {config['architecture']}")

# We must reverse the mapping to turn predicted index back to character
idx_to_class = {v: k for k, v in config["class_to_idx"].items()}

# 2. Recreate Model Architecture
# The code must EXACTLY match the architecture that was used to save the weights!
model = SimpleModel()

# 3. Load Saved Weights
weights_path = "models/simple_model_weights.pth"
model.load_state_dict(torch.load(weights_path))
print(f"Successfully loaded learned weights from '{weights_path}'")

# 4. Set to Evaluation Mode (CRITICAL for inference)
model.eval()

# 5. Test Inference
# Let's grab an image directly from the dataset folder to verify it works
test_image_path = "data/ሀ/synthetic_001.png"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

image = Image.open(test_image_path)
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0) # Add batch dimension: [1, 1, 64, 64]

with torch.no_grad():
    logits = model(image_tensor)
    prediction_idx = torch.argmax(logits, dim=1).item()
    
predicted_character = idx_to_class[prediction_idx]

print("\n--- Test Inference ---")
print(f"Loaded image: {test_image_path}")
print(f"Model prediction: {predicted_character}")
if predicted_character == "ሀ":
    print("Prediction is correct! The loaded model works perfectly.")
else:
    print("Prediction was incorrect. The model may not be trained well enough.")
