from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

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

remaining_labels = [labels[i] for i in remaining_indices]

validation_indices, test_indices = train_test_split(
    remaining_indices,
    test_size=0.50,
    random_state=42,
    stratify=remaining_labels
)

train_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, validation_indices)
test_dataset = Subset(dataset, test_indices)

print("Total images:", len(dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Test images:", len(test_dataset))