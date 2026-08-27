import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from linear_model import LinearModel

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
model = LinearModel()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Observe Learning (Record Metrics)
num_epochs = 20
epoch_losses = []
epoch_accuracies = []

for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = total_loss / total_samples
    epoch_accuracy = (correct_predictions / total_samples) * 100
    
    epoch_losses.append(epoch_loss)
    epoch_accuracies.append(epoch_accuracy)
    print(f"Epoch {epoch} - Loss: {epoch_loss:.2f}, Accuracy: {epoch_accuracy:.0f}%")

# 4. Plot Learning Curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', color='red')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), epoch_accuracies, marker='o', color='blue')
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)

plt.tight_layout()
plt.savefig("learning_curve.png")
print("\nLearning curve plot saved as 'learning_curve.png'.")
