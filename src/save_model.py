import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import json

from legacy_linear_paths import LEGACY_LINEAR_CONFIG, LEGACY_LINEAR_DIR, LEGACY_LINEAR_WEIGHTS
from linear_model import LinearModel

# 1. Setup DataLoader to train the model quickly
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
dataset = ImageFolder(root="data", transform=transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 2. Train the Model
model = LinearModel()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Training model to prepare for saving...")
model.train()
for epoch in range(1, 15):
    for images, labels_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_function(model(images), labels_batch)
        loss.backward()
        optimizer.step()

# 3. Save the Model
LEGACY_LINEAR_DIR.mkdir(parents=True, exist_ok=True)
model_path = LEGACY_LINEAR_WEIGHTS

# Save only the learned weights (state_dict), which is best practice in PyTorch
torch.save(model.state_dict(), model_path)
print(f"\nModel weights saved to '{model_path}'!")

# 4. Save Model Configuration and Class Mapping
config = {
    "image_width": 64,
    "image_height": 64,
    "channels": 1,
    "architecture": "LinearModel (Flatten -> Linear)",
    "classes": dataset.classes,
    "class_to_idx": dataset.class_to_idx
}

config_path = LEGACY_LINEAR_CONFIG
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

print(f"Model configuration saved to '{config_path}'!")
