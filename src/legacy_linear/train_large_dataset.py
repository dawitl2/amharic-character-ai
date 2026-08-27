import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt

from linear_model import LinearModel

print("--- Training on the MASSIVE Dataset ---")

# 1. Setup DataLoaders
# We increase batch size to 64 to train faster on thousands of images!
batch_size = 64 

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
dataset = ImageFolder(root="data", transform=transform)

print(f"Total images found: {len(dataset)}")

indices = list(range(len(dataset)))
labels = dataset.targets

# Train (70%), Val (15%), Test (15%)
train_idx, temp_idx, _, temp_labels = train_test_split(
    indices, labels, test_size=0.30, random_state=42, stratify=labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=42, stratify=temp_labels
)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

# 2. Setup Model, Loss, and Optimizer
model = LinearModel(num_classes=len(dataset.classes))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 15
train_accuracies = []
val_accuracies = []

print("\n--- Training Loop ---")
for epoch in range(1, num_epochs + 1):
    # Train
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
    
    # Validate
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
    
    print(f"Epoch {epoch:2d} | Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}%")

# 3. Final Test Evaluation
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_logits = model(test_images)
        test_predictions = torch.argmax(test_logits, dim=1)
        test_correct += (test_predictions == test_labels).sum().item()
        test_total += test_labels.size(0)

test_acc = (test_correct / test_total) * 100

print("\n--- Final Results ---")
print(f"TEST ACCURACY ON MASSIVE DATASET: {test_acc:.2f}%")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, marker='x', label='Validation Accuracy')
plt.title("Learning Curves on Large Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.savefig("large_dataset_curve.png")
print("Saved learning curve to 'large_dataset_curve.png'")
