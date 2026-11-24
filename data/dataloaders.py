
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from data.eurosat import EuroSAT
import numpy as np


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


class UnlabeledDataset(Dataset):
    """Wrap a labeled dataset so training loops receive images only."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image


def load_indices(train_pkl, test_pkl):
    with open(train_pkl, "rb") as f:
        train_subsets = pickle.load(f)

    with open(test_pkl, "rb") as f:
        test_indices = pickle.load(f)

    # All training images, without using their labels.
    train_indices = train_subsets[(1.0, 0)]

    return train_subsets, train_indices, test_indices

def load_test_indices(test_path):
    with test_path.open("rb") as file:
        indices = np.asarray(pickle.load(file), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"Test indices in {test_path} must be a non-empty 1-D array.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Test indices in {test_path} contain duplicates.")
    return indices


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