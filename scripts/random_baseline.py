import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from data.dataloaders import create_loader, eval_transform, load_indices
from models.student_cnn import build_student


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract row-A features from randomly initialized student CNNs."
    )
    parser.add_argument("--eurosat-root", type=Path, default=Path("datasets/EuroSAT_RGB"))
    parser.add_argument("--train-pkl", type=Path, default=Path("datasets/eurosat_train_subsets.pkl"))
    parser.add_argument("--test-pkl", type=Path, default=Path("datasets/eurosat_test_subset.pkl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/features/random_baseline"),
        help="Root directory for row-A feature caches.",
    )
    parser.add_argument("--student-cnn", choices=("small", "large"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")
    minimum_seeds = 3 if args.student_cnn == "small" else 2
    if len(args.seeds) < minimum_seeds:
        raise ValueError(
            f"The {args.student_cnn} student requires at least {minimum_seeds} seeds."
        )
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be unique nonnegative integers.")

    _, train_indices, test_indices = load_indices(args.train_pkl, args.test_pkl)
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Training and test indices must not overlap.")

    sample_indices = np.concatenate((train_indices, test_indices))
    if np.unique(sample_indices).size != sample_indices.size:
        raise ValueError("Feature extraction indices must be unique.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloader = create_loader(
        root=args.eurosat_root,
        indices=sample_indices,
        transform=eval_transform,
        batch_size=args.batch_size,
        shuffle=False,
        return_labels=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    size_name = "small_cnn" if args.student_cnn == "small" else "larger_cnn"
    for seed in args.seeds:
        set_seed(seed)

        student = build_student(args.student_cnn)
        num_parameters = sum(parameter.numel() for parameter in student.parameters())
        student.to(device)

        # Row A deliberately uses no pretraining. eval() keeps BatchNorm
        # on its default running statistics instead of batch statistics.
        student.eval()
        features, labels = student.extract_features(dataloader, device)

        output_dir = args.output_dir / size_name / f"seed{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "features.npy", features)
        np.save(output_dir / "labels.npy", labels)
        np.save(output_dir / "indices.npy", sample_indices)

        metadata = {
            "model": f"Random {args.student_cnn} StudentCNN",
            "encoder": f"random_baseline_{size_name}_seed{seed}",
            "approach": "random_baseline",
            "seed": seed,
            "student_cnn": args.student_cnn,
            "pretrained": False,
            "batch_norm_mode": "eval",
            "num_parameters": num_parameters,
            "feature_dim": int(features.shape[1]),
            "num_samples": int(features.shape[0]),
            "dataset": "EuroSAT RGB",
            "batch_size": args.batch_size,
            "device": str(device),
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print(
            f"Saved row A cache: student_cnn={args.student_cnn}, seed={seed}, "
            f"parameters={num_parameters}, feature_dim={features.shape[1]} "
            f"to {output_dir}"
        )


if __name__ == "__main__":
    main()
