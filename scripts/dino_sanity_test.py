import torch
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

teacher = torch.hub.load(
    "facebookresearch/dino:main",
    "dino_vits16",
    pretrained=False,
)

weights = torch.load(
    "dino_deitsmall16_pretrain.pth",
    map_location="cpu",
    weights_only=True,
)

teacher.load_state_dict(weights, strict=True)

teacher.eval()

for param in teacher.parameters():
    param.requires_grad = False

teacher = teacher.to(device)

print(f"Device: {device}")
print(f"Parameters: {sum(p.numel() for p in teacher.parameters()):,}")

image = Image.new("RGB", (256, 256), color="gray")

x = teacher_transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    features = teacher(x)

print("Input:", x.shape)
print("Features:", features.shape)
print("Feature dtype:", features.dtype)
print("Feature min:", features.min().item())
print("Feature max:", features.max().item())
print("Feature mean:", features.mean().item())
print("Feature std:", features.std().item())