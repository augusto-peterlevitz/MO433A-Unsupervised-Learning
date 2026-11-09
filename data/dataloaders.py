
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.eurosat import EuroSAT
from lightly.transforms.vicreg_transform import VICRegTransform
import numpy as np

# --------------------------------------------------
# 1. Normalization
# --------------------------------------------------

# ImageNet normalization is a reasonable starting point
# for the RGB student model.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# --------------------------------------------------
# 2. Training transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        size=64,
        scale=(0.6, 1.0),
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        )
    ], p=0.8),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# VICReg needs two independently augmented views
# of the same image.
# class TwoViewsTransform:
#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, image):
#         view1 = self.transform(image)
#         view2 = self.transform(image)
#         return view1, view2


# vicreg_transform = TwoViewsTransform(train_transform)
vicreg_transform = VICRegTransform(
    input_size=64,
)

# --------------------------------------------------
# 3. Evaluation transform
# --------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# --------------------------------------------------
# 4. Dataset wrapper
# --------------------------------------------------

class EuroSATSubset(Dataset):
    def __init__(
        self,
        root,
        indices,
        transform=None,
        return_labels=True,
    ):
        self.dataset = EuroSAT(
            root=root,
            transform=transform,
        )

        self.indices = list(indices)
        self.return_labels = return_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]

        image, label = self.dataset[original_index]

        if self.return_labels:
            return image, label

        return image


# --------------------------------------------------
# 5. Load saved indices
# --------------------------------------------------

def load_indices(train_pkl, test_pkl):
    with open(train_pkl, "rb") as f:
        train_subsets = pickle.load(f)

    with open(test_pkl, "rb") as f:
        test_indices = pickle.load(f)

    # All training images, without using their labels.
    train_indices = train_subsets[(1.0, 0)]

    return train_subsets, train_indices, test_indices


# --------------------------------------------------
# 6. DataLoader factory
# --------------------------------------------------

def create_loader(
    root,
    indices,
    transform,
    batch_size,
    shuffle,
    return_labels,
    num_workers=4,
    drop_last=False,
    pin_memory=True,
):
    dataset = EuroSATSubset(
        root=root,
        indices=indices,
        transform=transform,
        return_labels=return_labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )

    return loader

def get_data(requested_indices, feature_dir):

    features = np.load(feature_dir / "features.npy")
    labels = np.load(feature_dir / "labels.npy")
    indices = np.load(feature_dir / "indices.npy")

    row_by_index = {int(index): row for row, index in enumerate(indices)}
    missing = [int(index) for index in requested_indices if int(index) not in row_by_index]
    
    if missing:
        raise ValueError(f"Feature cache is missing dataset indices, e.g. {missing[:5]}.")
    
    rows = np.asarray([row_by_index[int(index)] for index in requested_indices])

    requested_features = features[rows]
    requested_labels = labels[rows]

    return requested_features, requested_labels