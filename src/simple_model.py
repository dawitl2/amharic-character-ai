import torch
import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
            64 * 64,
            num_classes
        )

    def forward(self, x):

        x = self.flatten(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = SimpleModel()

    fake_images = torch.randn(4, 1, 64, 64)

    outputs = model(fake_images)

    predictions = torch.argmax(outputs, dim=1)

    print("Raw outputs:")
    print(outputs)

    print()

    print("Predicted class numbers:")
    print(predictions)