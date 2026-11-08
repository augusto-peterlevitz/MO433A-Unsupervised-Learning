import argparse
from pathlib import Path

import numpy as np
import torch

from data.dataloaders import create_loader, eval_transform, load_indices
from models.student_cnn import StudentCNN
from training.train_vicreg import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract deterministic raw VICReg backbone features for EuroSAT."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vicreg.json"),
        help="VICReg training configuration (default: configs/vicreg.json).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Backbone checkpoint (defaults to student_backbone.pth in config output_dir).",
    )
    parser.add_argument(
        "--eurosat-root",
        type=Path,
        help="EuroSAT root (defaults to dataset_root in the config).",
    )
    parser.add_argument(
        "--train-pkl",
        type=Path,
        help="Training subset indices (defaults to subsets_path in the config).",
    )
    parser.add_argument(
        "--test-pkl",
        type=Path,
        default=Path("datasets/eurosat_test_subset.pkl"),
        help="Held-out test indices pickle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Feature output directory (defaults to datasets/features/vicreg_width_<width>).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Extraction batch size (defaults to batch_size in the config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="DataLoader worker count (defaults to num_workers in the config).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_root = args.eurosat_root or config.dataset_root
    train_path = args.train_pkl or config.subsets_path
    checkpoint_path = args.checkpoint_path or (
        config.output_dir / "student_backbone.pth"
    )
    output_dir = args.output_dir or (
        Path("datasets/features")
        / f"vicreg_width_{config.width_mult:g}"
    )
    batch_size = (
        config.batch_size if args.batch_size is None else args.batch_size
    )
    num_workers = (
        config.num_workers if args.num_workers is None else args.num_workers
    )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    _, train_indices, test_indices = load_indices(
        train_path,
        args.test_pkl)
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Training and test indices must not overlap.")
    sample_indices = np.concatenate((train_indices, test_indices))
    if np.unique(sample_indices).size != sample_indices.size:
        raise ValueError("Feature extraction indices must be unique.")

    dataloader = create_loader(
        root=dataset_root,
        indices=sample_indices,
        transform=eval_transform,
        batch_size=batch_size,
        shuffle=False,
        return_labels=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")

    student = StudentCNN(width_mult=config.width_mult)
    student.load_weights(checkpoint_path)
    student.to(device)
    
    features, labels = student.extract_features(
        dataloader,
        device,
    )

    if features.shape[0] != len(dataloader.dataset) or labels.shape[0] != len(dataloader.dataset):
        raise RuntimeError(
            "Extracted feature and label counts do not match dataset."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "indices.npy", sample_indices)

    print("\n--- Feature cache verification ---")
    print("Raw features shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("Indices shape:", sample_indices.shape)
    print("Saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
