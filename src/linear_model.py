import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """
    LINEAR MODEL (Phase 7 - Phase 24)
    This is the original, simple neural network.
    It uses a single linear (fully-connected) layer directly on the flattened pixels.
    While useful for learning the basics, it struggles with complex variations.
    """

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
    model = LinearModel()

    fake_images = torch.randn(4, 1, 64, 64)

    outputs = model(fake_images)

    predictions = torch.argmax(outputs, dim=1)

    print("Raw outputs:")
    print(outputs)

    print()

    print("Predicted class numbers:")
    print(predictions)