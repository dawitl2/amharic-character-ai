import torch
import torch.nn as nn

outputs = torch.tensor([
    [2.5, 0.4, -0.2],
    [0.3, 1.8, 0.1],
    [0.2, 0.5, 2.1],
    [1.7, 0.6, 0.2]
], requires_grad=True)

correct_labels = torch.tensor([
    0,
    1,
    2,
    0
])

loss_function = nn.CrossEntropyLoss()

loss = loss_function(
    outputs,
    correct_labels
)

print("Loss:")
print(loss)

loss.backward()

print()

print("Gradients:")
print(outputs.grad)