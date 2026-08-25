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

# 3. Get one real batch
images, labels = next(iter(train_loader))

# 4. Training Step
optimizer.zero_grad()           # Clear previous gradients
logits = model(images)          # Generate logits
loss = loss_function(logits, labels) # Calculate loss
loss.backward()                 # Backpropagate loss
optimizer.step()                # Update model weights

# 5. Calculate Predictions and Accuracy
predictions = torch.argmax(logits, dim=1)
correct = (predictions == labels).sum().item()
accuracy = (correct / labels.size(0)) * 100

print("First Complete Training Step Finished!\n")
print(f"Batch Loss:     {loss.item():.4f}")
print(f"Predicted IDs:  {predictions.tolist()}")
print(f"Actual IDs:     {labels.tolist()}")
print(f"Batch Accuracy: {accuracy:.1f}%")
