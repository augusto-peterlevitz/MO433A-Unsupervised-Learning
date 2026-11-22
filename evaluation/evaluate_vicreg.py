import argparse
import json
import random
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from data.eurosat import EuroSAT
from models.student_cnn import StudentCNN


@dataclass
class EvaluationConfig:
    dataset_root: Path = Path("datasets/EuroSAT_RGB")
    subsets_path: Path = Path("datasets/eurosat_train_subsets.pkl")
    test_indices_path: Path = Path("datasets/eurosat_test_subset.pkl")
    checkpoint: Path = Path("checkpoints/vicreg_width_1.0/student_backbone.pth")
    output_dir: Path = Path("evaluation/results/vicreg_width_1.0")
    width_mult: float = 1.0
    batch_size: int = 128
    num_workers: int = 8
    seed: int = 42

    def validate(self):
        if self.width_mult <= 0 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError(
                "width_mult and batch_size must be positive; "
                "num_workers must be nonnegative"
            )
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 4294967295")


def load_config(path):
    with open(path, encoding="utf-8") as config_file:
        values = json.load(config_file)
    if not isinstance(values, dict):
        raise ValueError("Evaluation configuration must be a JSON object")

    config_fields = {field.name: field for field in fields(EvaluationConfig)}
    unknown_fields = set(values) - set(config_fields)
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown evaluation configuration fields: {names}")

    path_fields = {
        "dataset_root",
        "subsets_path",
        "test_indices_path",
        "checkpoint",
        "output_dir",
    }
    integer_fields = {"batch_size", "num_workers", "seed"}
    for name, value in values.items():
        if name in path_fields:
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string path")
            values[name] = Path(value)
        elif name in integer_fields and type(value) is not int:
            raise ValueError(f"{name} must be an integer")
        elif name == "width_mult":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("width_mult must be a number")
            values[name] = float(value)

    config = EvaluationConfig(**values)
    config.validate()
    return config


# Use the same deterministic preprocessing for every encoder.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VICReg backbone using a JSON configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluate_vicreg.json"),
        help="Path to the evaluation configuration (default: configs/evaluate_vicreg.json).",
    )
    return parser.parse_args()


def load_indices(subsets_path, test_indices_path):
    """Load the full training indices and held-out test indices."""
    import pickle

    with open(subsets_path, "rb") as file:
        train_subsets = pickle.load(file)

    with open(test_indices_path, "rb") as file:
        test_indices = pickle.load(file)

    # The full training split is the reference set for k-NN.
    train_indices = train_subsets[(1.0, 0)]

    return train_indices, test_indices


def load_backbone(checkpoint_path, width_mult, device):
    """Create the student backbone and load its trained weights."""
    backbone = StudentCNN(width_mult=width_mult)

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        state_dict = torch.load(checkpoint_path, map_location=device)

    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()

    return backbone


@torch.inference_mode()
def extract_features(model, loader, device):
    """
    Extract one feature vector per image.

    The model is kept in evaluation mode and no gradients are calculated.
    """
    all_features = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)

        features = model(images)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def evaluate_knn(train_features, train_labels, test_features, test_labels):
    """Evaluate cosine k-NN using the training split as reference."""
    # The project specifies k = min(20, examples in the smallest class).
    _, class_counts = np.unique(train_labels, return_counts=True)
    k = min(20, int(class_counts.min()))

    # L2 normalization makes cosine distance operate on normalized vectors.
    train_normalized = normalize(train_features, norm="l2")
    test_normalized = normalize(test_features, norm="l2")

    classifier = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
    )
    classifier.fit(train_normalized, train_labels)

    predictions = classifier.predict(test_normalized)
    accuracy = accuracy_score(test_labels, predictions)

    return {
        "k": k,
        "accuracy": float(accuracy),
    }


def evaluate_kmeans(test_features, test_labels, num_classes, seed):
    """Evaluate unsupervised clusters against test labels."""
    normalized_features = normalize(test_features, norm="l2")

    clustering = KMeans(
        n_clusters=num_classes,
        random_state=seed,
        n_init=10,
    )
    cluster_ids = clustering.fit_predict(normalized_features)

    nmi = normalized_mutual_info_score(test_labels, cluster_ids)
    ari = adjusted_rand_score(test_labels, cluster_ids)

    return {
        "nmi": float(nmi),
        "ari": float(ari),
    }


def calculate_embedding_std(features):
    """
    Calculate the mean per-dimension standard deviation
    after L2-normalizing each feature vector.
    """
    normalized_features = normalize(features, norm="l2")
    per_dimension_std = normalized_features.std(axis=0)

    return {
        "mean_dimension_std": float(per_dimension_std.mean()),
        "min_dimension_std": float(per_dimension_std.min()),
        "max_dimension_std": float(per_dimension_std.max()),
    }


def main():
    config = load_config(parse_args().config)

    if not config.checkpoint.is_file():
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {config.checkpoint}"
        )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_indices, test_indices = load_indices(
        config.subsets_path,
        config.test_indices_path,
    )

    dataset = EuroSAT(
        root=config.dataset_root,
        transform=EVAL_TRANSFORM,
    )

    train_dataset = Subset(dataset, list(map(int, train_indices)))
    test_dataset = Subset(dataset, list(map(int, test_indices)))

    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
        "generator": torch.Generator().manual_seed(config.seed),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options,
    )

    backbone = load_backbone(
        config.checkpoint,
        config.width_mult,
        device,
    )

    print(f"Feature dimension: {backbone.feature_dim}")
    print(
        "Backbone parameters: "
        f"{sum(p.numel() for p in backbone.parameters()):,}"
    )

    # Extract features only once. All diagnostics reuse these arrays.
    train_features, train_labels = extract_features(
        backbone,
        train_loader,
        device,
    )
    test_features, test_labels = extract_features(
        backbone,
        test_loader,
        device,
    )

    print("\nEvaluating k-NN...")
    knn_results = evaluate_knn(
        train_features,
        train_labels,
        test_features,
        test_labels,
    )

    print("Evaluating k-means...")
    kmeans_results = evaluate_kmeans(
        test_features,
        test_labels,
        num_classes=len(dataset.classes),
        seed=config.seed,
    )

    print("Checking embedding standard deviation...")
    std_results = calculate_embedding_std(test_features)

    results = {
        "checkpoint": str(config.checkpoint),
        "width_mult": config.width_mult,
        "seed": config.seed,
        "feature_dimension": int(train_features.shape[1]),
        "num_train_images": int(len(train_labels)),
        "num_test_images": int(len(test_labels)),
        "knn": knn_results,
        "kmeans": kmeans_results,
        "embedding_std": std_results,
    }

    # Save raw features so later probes can reuse them.
    features_path = config.output_dir / "features.npz"
    np.savez_compressed(
        features_path,
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
    )

    results_path = config.output_dir / "metrics.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")

    print("\n========== VICReg Evaluation ==========")
    print(f"k-NN accuracy:       {knn_results['accuracy']:.4f}")
    print(f"k-NN neighbors:      {knn_results['k']}")
    print(f"k-means NMI:         {kmeans_results['nmi']:.4f}")
    print(f"k-means ARI:         {kmeans_results['ari']:.4f}")
    print(
        "Mean embedding std: "
        f"{std_results['mean_dimension_std']:.6f}"
    )
    print("=======================================")
    print(f"Metrics saved to:  {results_path}")
    print(f"Features saved to: {features_path}")


if __name__ == "__main__":
    main()