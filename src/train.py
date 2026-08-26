"""
Continuous Training Pipeline
============================
Run this from the project root:

    .venv\Scripts\python.exe src/train.py

This script trains the SimpleModel. It automatically detects and resumes
from previous checkpoints, saves the best model based on validation accuracy,
and updates the configuration for the GUI.
"""
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_model import SimpleModel

NUM_EPOCHS_TO_RUN = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.01

MODEL_DIR = "models"
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
LATEST_CHECKPOINT = os.path.join(MODEL_DIR, "latest_checkpoint.pth")
BEST_MODEL_WEIGHTS = os.path.join(MODEL_DIR, "best_model_weights.pth")
LEGACY_WEIGHTS = os.path.join(MODEL_DIR, "simple_model_weights.pth")

print(f"--- Training Pipeline ---")
print(f"Targeting {NUM_EPOCHS_TO_RUN} additional epochs.")
print(f"Batch size: {BATCH_SIZE} | Learning rate: {LEARNING_RATE}")
print()

# 1. Load Dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
dataset = ImageFolder(root="data", transform=transform)
print(f"Total images: {len(dataset)}")
print(f"Classes: {dataset.class_to_idx}")

# 2. Split: 70% train, 15% val, 15% test
indices = list(range(len(dataset)))
labels = dataset.targets

train_idx, temp_idx, _, temp_labels = train_test_split(
    indices, labels, test_size=0.30, random_state=42, stratify=labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=42, stratify=temp_labels
)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
print()

# 3. Model, Loss, Optimizer
model = SimpleModel(num_classes=len(dataset.classes))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# --- Safety Check & Resume Logic ---
start_epoch = 1
best_val_acc = 0.0
cumulative_epochs = 0

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        old_config = json.load(f)
    
    # Safety Check: Architecture and Classes must match
    if old_config.get("architecture") != "SimpleModel (Flatten -> Linear)":
        print("⚠️  WARNING: Architecture mismatch! Expected SimpleModel. Aborting.")
        sys.exit(1)
    
    if old_config.get("class_to_idx") != dataset.class_to_idx:
        print("⚠️  WARNING: Class mapping mismatch! The dataset classes have expanded from 3 to 10.")
        print("⚠️  Starting training from scratch to accommodate new classes.")
        ignore_checkpoint = True
    else:
        ignore_checkpoint = False
        cumulative_epochs = old_config.get("epochs_trained", 0)
else:
    ignore_checkpoint = False

if os.path.exists(LATEST_CHECKPOINT) and not ignore_checkpoint:
    print(f"Resuming from {LATEST_CHECKPOINT}...")
    checkpoint = torch.load(LATEST_CHECKPOINT)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    print(f"Loaded previous state. Best Validation Accuracy so far: {best_val_acc:.2f}%")
else:
    print("Starting training from scratch with random weights.")

print()
print(f"{'Epoch':>6} | {'Train Acc':>10} | {'Val Acc':>10} | {'Train Loss':>11}")
print("-" * 50)

for epoch in range(start_epoch, NUM_EPOCHS_TO_RUN + 1):
    # Train
    model.train()
    train_correct = 0
    train_total = 0
    epoch_loss = 0.0

    for images, labels_batch in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == labels_batch).sum().item()
        train_total += labels_batch.size(0)

    train_acc = (train_correct / train_total) * 100
    avg_loss = epoch_loss / len(train_loader)

    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_imgs, val_lbls in val_loader:
            val_logits = model(val_imgs)
            val_preds = torch.argmax(val_logits, dim=1)
            val_correct += (val_preds == val_lbls).sum().item()
            val_total += val_lbls.size(0)

    val_acc = (val_correct / val_total) * 100

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_WEIGHTS)
        best_msg = "⭐ NEW BEST MODEL SAVED!"
    else:
        best_msg = ""

    print(f"{epoch:>6} | {train_acc:>9.1f}% | {val_acc:>9.1f}% | {avg_loss:>11.4f}  {best_msg}")
    
    cumulative_epochs += 1
    
    # Save the latest state
    torch.save({
        "epoch": cumulative_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }, LATEST_CHECKPOINT)

# 4. Final Test
print("\nLoading best model for final independent test evaluation...")
model.load_state_dict(torch.load(BEST_MODEL_WEIGHTS))
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for test_imgs, test_lbls in test_loader:
        test_logits = model(test_imgs)
        test_preds = torch.argmax(test_logits, dim=1)
        test_correct += (test_preds == test_lbls).sum().item()
        test_total += test_lbls.size(0)

test_acc = (test_correct / test_total) * 100

print()
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Final Test Accuracy:      {test_acc:.2f}%")

# 5. Save Configuration for GUI
os.makedirs("models", exist_ok=True)

# Also update the legacy weights path for older scripts that might expect it
torch.save(model.state_dict(), LEGACY_WEIGHTS)
print(f"Backed up final model to {LEGACY_WEIGHTS}")

config = {
    "architecture": "SimpleModel (Flatten -> Linear)",
    "image_width": 64,
    "image_height": 64,
    "class_to_idx": dataset.class_to_idx,
    "epochs_trained": cumulative_epochs,
    "best_val_accuracy": round(best_val_acc, 2),
    "test_accuracy": round(test_acc, 2)
}
with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    
print(f"Saved config to {CONFIG_PATH}")
print()
print("Done! The GUI and pipeline are now fully updated.")