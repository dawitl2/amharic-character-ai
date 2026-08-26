import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from linear_model import LinearModel

# 1. Setup DataLoaders (Train and Validation)
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

# 2. Setup Model, Loss, and Optimizer
model = LinearModel()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Training Loop with Validation
num_epochs = 15

for epoch in range(1, num_epochs + 1):
    
    # --- TRAINING PHASE ---
    model.train() # Set model to training mode
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
    
    # --- VALIDATION PHASE ---
    model.eval() # Set model to evaluation mode
    val_correct = 0
    val_total = 0
    
    # Disable gradient calculations to save memory and compute
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_logits = model(val_images)
            val_predictions = torch.argmax(val_logits, dim=1)
            val_correct += (val_predictions == val_labels).sum().item()
            val_total += val_labels.size(0)
            
    val_acc = (val_correct / val_total) * 100
    
    print(f"Epoch {epoch:2d} | Train Acc: {train_acc:3.0f}% | Val Acc: {val_acc:3.0f}%")
