from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


eval_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


dino_eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
