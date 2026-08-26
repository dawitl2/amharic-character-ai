import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from linear_model import LinearModel

# 1. Setup DataLoaders to INTENTIONALLY overfit
# We will use a tiny training set (e.g., 6 images) and validate on the rest.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
dataset = ImageFolder(root="data", transform=transform)

indices = list(range(len(dataset)))
labels = dataset.targets

# Split 20% train (tiny), 80% validation (huge) to force overfitting
train_idx, val_idx = train_test_split(
    indices, test_size=0.80, random_state=42, stratify=labels
)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=2, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=4, shuffle=False)

model = LinearModel()
loss_function = nn.CrossEntropyLoss()
# Use a slightly higher learning rate to make it overfit faster
optimizer = optim.SGD(model.parameters(), lr=0.05) 

num_epochs = 40
train_accuracies = []
val_accuracies = []

for epoch in range(1, num_epochs + 1):
    
    # --- TRAINING PHASE ---
    model.train()
    train_correct = 0
    train_total = 0
    for images, labels_batch in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels_batch)
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(logits, dim=1)
        train_correct += (predictions == labels_batch).sum().item()
        train_total += labels_batch.size(0)
        
    train_acc = (train_correct / train_total) * 100
    train_accuracies.append(train_acc)
    
    # --- VALIDATION PHASE ---
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_logits = model(val_images)
            val_predictions = torch.argmax(val_logits, dim=1)
            val_correct += (val_predictions == val_labels).sum().item()
            val_total += val_labels.size(0)
            
    val_acc = (val_correct / val_total) * 100
    val_accuracies.append(val_acc)
    
    print(f"Epoch {epoch:2d} | Train: {train_acc:3.0f}% | Val: {val_acc:3.0f}%")

# Plotting the overfitting phenomenon
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', color='blue', label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, marker='x', color='red', label='Validation Accuracy')
plt.title("Overfitting Demonstration")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("overfitting_curve.png")
print("\nSaved overfitting plot as 'overfitting_curve.png'.")
