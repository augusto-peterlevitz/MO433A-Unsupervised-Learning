import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from data.dataloaders import create_loader, eval_transform, load_indices
from models.student_cnn import StudentCNN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract row-A features from randomly initialized student CNNs."
    )
    parser.add_argument("--eurosat-root", type=Path, default=Path("datasets/EuroSAT_RGB"))
    parser.add_argument("--train-pkl", type=Path, default=Path("datasets/eurosat_train_subsets.pkl"))
    parser.add_argument("--test-pkl", type=Path, default=Path("datasets/eurosat_test_subset.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/features"))
    parser.add_argument("--width-mults", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
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
    if not args.width_mults or any(width <= 0 for width in args.width_mults):
        raise ValueError("width-mults must contain positive values.")
    if not args.seeds:
        raise ValueError("At least one random seed is required.")

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

    for width_mult in args.width_mults:
        for seed in args.seeds:
            set_seed(seed)

            student = StudentCNN(width_mult=width_mult)
            num_parameters = sum(parameter.numel() for parameter in student.parameters())
            student.to(device)

            # Row A deliberately uses no pretraining. eval() keeps BatchNorm
            # on its default running statistics instead of batch statistics.
            student.eval()
            features, labels = student.extract_features(dataloader, device)

            output_dir = (
                args.output_dir
                / f"random_width_{width_mult:g}_seed_{seed}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "features.npy", features)
            np.save(output_dir / "labels.npy", labels)
            np.save(output_dir / "indices.npy", sample_indices)

            metadata = {
                "model": f"Random StudentCNN width {width_mult:g}",
                "row": "A",
                "seed": seed,
                "width_mult": width_mult,
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
                f"Saved row A cache: width={width_mult:g}, seed={seed}, "
                f"parameters={num_parameters}, feature_dim={features.shape[1]} "
                f"to {output_dir}"
            )


if __name__ == "__main__":
    main()
