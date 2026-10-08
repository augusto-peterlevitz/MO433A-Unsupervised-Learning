import torch
from PIL import Image
from torchvision import transforms
from data.eurosat import EuroSAT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = EuroSAT(
    root="datasets/EuroSAT_RGB",
    transform=None,
)

print("Number of images:", len(dataset))
print("Classes:", dataset.classes)

image, label = dataset[0]

print("Image:", image.size)
print("Label:", label)
