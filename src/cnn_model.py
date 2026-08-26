import torch.nn as nn
import torch.nn.functional as F

ARCHITECTURE_NAME = "CharacterCNN"
INPUT_CHANNELS = 1
INPUT_HEIGHT = 64
INPUT_WIDTH = 64


class CharacterCNN(nn.Module):
    """
    CONVOLUTIONAL NEURAL NETWORK (Phase 25+)
    This architecture learns hierarchical features (edges, strokes) directly from the image pixels.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        # Input shape: [batch_size, 1, 64, 64]
        # 1st Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape after pool1: [batch_size, 16, 32, 32]
        
        # 2nd Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape after pool2: [batch_size, 32, 16, 16]
        
        # Fully Connected Layer
        self.flatten = nn.Flatten()
        
        # 32 channels * 16 * 16 spatial dimensions = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply conv1 -> ReLU -> max pooling
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Apply conv2 -> ReLU -> max pooling
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten and pass to dense layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    import torch

    model = CharacterCNN()
    fake_images = torch.randn(4, 1, 64, 64)
    outputs = model(fake_images)
    print("CNN output shape:", outputs.shape)
