import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
            64 * 64,
            3
        )

    def forward(self, x):

        x = self.flatten(x)

        x = self.classifier(x)

        return x


model = SimpleModel()

print(model)