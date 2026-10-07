from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class EuroSAT(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.samples = []
        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        for class_name in self.classes:
            class_dir = self.root / class_name
            label = self.class_to_idx[class_name]

            for image_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label