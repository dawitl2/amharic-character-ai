import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from simple_model import SimpleModel

# 1. Setup DataLoader
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
dataset = ImageFolder(root="data", transform=transform)

indices = list(range(len(dataset)))
labels = dataset.targets

train_indices, _ = train_test_split(
    indices, test_size=0.30, random_state=42, stratify=labels
)
train_dataset = Subset(dataset, train_indices)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 2. Setup Model, Loss, and Optimizer
model = SimpleModel()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Training Loop
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = loss_function(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Update model weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * images.size(0)
        
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
    # Calculate epoch metrics
    epoch_loss = total_loss / total_samples
    epoch_accuracy = (correct_predictions / total_samples) * 100
    
    print(f"Epoch {epoch}")
    print(f"Loss: {epoch_loss:.2f}")
    print(f"Accuracy: {epoch_accuracy:.0f}%")
    print()
