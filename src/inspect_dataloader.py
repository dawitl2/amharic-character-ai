from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

dataset = ImageFolder(
    root="data",
    transform=transform
)

indices = list(range(len(dataset)))
labels = dataset.targets

train_indices, remaining_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

train_dataset = Subset(dataset, train_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("Images batch shape:", images.shape)
print("Labels batch shape:", labels.shape)
print("Labels:", labels)