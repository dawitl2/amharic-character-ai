import torch
import torch.nn as nn
from linear_model import LinearModel

model = LinearModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

fake_images = torch.randn(4, 1, 64, 64)
correct_labels = torch.tensor([0, 1, 2, 0])
loss_function = nn.CrossEntropyLoss()

print("1. BEFORE UPDATE")
print("First bias value:", model.classifier.bias.data[0].item())

optimizer.zero_grad()

outputs = model(fake_images)
initial_loss = loss_function(outputs, correct_labels)
print("Initial Loss:", initial_loss.item())

initial_loss.backward()
print("Gradient for first bias:", model.classifier.bias.grad[0].item())

optimizer.step()

print("\n2. AFTER UPDATE")
print("First bias value changed to:", model.classifier.bias.data[0].item())

outputs_again = model(fake_images)
new_loss = loss_function(outputs_again, correct_labels)
print("New Loss:", new_loss.item())

if new_loss < initial_loss:
    print("The update successfully reduced the loss!")
else:
    print("The update increased the loss (can happen randomly in single steps).")
