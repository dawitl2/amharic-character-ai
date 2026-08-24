from PIL import Image
from torchvision import transforms

image = Image.open("data/ሀ/sample_001.png")

to_tensor = transforms.ToTensor()

image_tensor = to_tensor(image)

print("Tensor shape:", image_tensor.shape)

print("Minimum pixel value:", image_tensor.min())
print("Maximum pixel value:", image_tensor.max())