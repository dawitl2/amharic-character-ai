import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from simple_model import SimpleModel

# 1. Setup DataLoaders (Train, Validation, and Test)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
dataset = ImageFolder(root="data", transform=transform)

indices = list(range(len(dataset)))
labels = dataset.targets

# Split 70% train, 30% temporary remaining
train_idx, temp_idx, _, temp_labels = train_test_split(
    indices, labels, test_size=0.30, random_state=42, stratify=labels
)
# Split the remaining 30% into 15% validation and 15% test
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=42, stratify=temp_labels
)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=4, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=4, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=4, shuffle=False)

# 2. Setup Model, Loss, and Optimizer
model = SimpleModel()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Train the model (using Train & Validation to guide us)
num_epochs = 20

print("--- TRAINING PHASE ---")
for epoch in range(1, num_epochs + 1):
    
    # Train
    model.train()
    for images, labels_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_function(model(images), labels_batch)
        loss.backward()
        optimizer.step()
        
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
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d} | Val Acc: {val_acc:3.0f}%")

print("\nTraining Complete. Decisions Frozen.")
print("Proceeding to Phase 17: Final Test Evaluation...\n")

# 4. Final Test Evaluation (Only done ONCE at the very end!)
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

print("==================================")
print(f"FINAL TEST ACCURACY: {test_acc:.1f}%")
print("==================================")
