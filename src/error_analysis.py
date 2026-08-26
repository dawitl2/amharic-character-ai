import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

from linear_model import LinearModel

# 1. Setup DataLoaders
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
dataset = ImageFolder(root="data", transform=transform)

indices = list(range(len(dataset)))
labels = dataset.targets

train_idx, temp_idx, _, temp_labels = train_test_split(
    indices, labels, test_size=0.30, random_state=42, stratify=labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=42, stratify=temp_labels
)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=4, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=4, shuffle=False)

# 2. Train Model quickly
model = LinearModel()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Training model for error analysis...")
model.train()
for epoch in range(1, 21):
    for images, labels_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_function(model(images), labels_batch)
        loss.backward()
        optimizer.step()

# 3. Error Analysis on Test Set
model.eval()
all_predictions = []
all_targets = []

incorrect_examples = []

with torch.no_grad():
    for test_images, test_labels in test_loader:
        logits = model(test_images)
        predictions = torch.argmax(logits, dim=1)
        
        all_predictions.extend(predictions.tolist())
        all_targets.extend(test_labels.tolist())
        
        # Track incorrect predictions
        for i in range(len(test_labels)):
            if predictions[i] != test_labels[i]:
                incorrect_examples.append({
                    'image': test_images[i],
                    'predicted': predictions[i].item(),
                    'actual': test_labels[i].item()
                })

print("\n--- Error Analysis ---")
if len(incorrect_examples) == 0:
    print("Perfect accuracy! No errors to display. (Expected for this tiny dataset)")
else:
    print(f"Found {len(incorrect_examples)} errors.")

class_names = dataset.classes

# Confusion Matrix
cm = confusion_matrix(all_targets, all_predictions)
print("\nConfusion Matrix:")
print(cm)

# Classification Report (Per-class accuracy)
print("\nClassification Report:")
print(classification_report(all_targets, all_predictions, target_names=class_names))

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nSaved confusion matrix plot as 'confusion_matrix.png'.")
