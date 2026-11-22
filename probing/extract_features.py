import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.dataloaders import create_loader, eval_transform, load_indices
from models.student_cnn import build_student
from training.train_vicreg import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract deterministic VICReg or distilled student features for EuroSAT."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vicreg.json"),
        help="Base VICReg configuration or a specific seed's saved configuration.",
    )
    parser.add_argument(
        "--student-cnn",
        choices=("small", "large"),
        help="Architecture to extract when using --seeds; defaults to the config.",
    )
    parser.add_argument(
        "--approach",
        choices=("vicreg", "distillation"),
        help="Student training approach; defaults to the config.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Extract these trained seeds from the selected approach's checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Root containing <size>/seed<seed>/ run folders.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Single-run backbone checkpoint; unavailable with --seeds.",
    )
    parser.add_argument("--eurosat-root", type=Path)
    parser.add_argument("--train-pkl", type=Path)
    parser.add_argument(
        "--test-pkl",
        type=Path,
        default=Path("datasets/eurosat_test_subset.pkl"),
        help="Held-out test indices pickle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Single-run output directory, or root directory when using --seeds "
            "(default is selected by approach, architecture, and seed)."
        ),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def extract(config, args, output_dir, checkpoint_path):
    dataset_root = args.eurosat_root or config.dataset_root
    train_path = args.train_pkl or config.subsets_path
    batch_size = config.batch_size if args.batch_size is None else args.batch_size
    num_workers = config.num_workers if args.num_workers is None else args.num_workers
    if batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if num_workers < 0:
        raise ValueError("num-workers must be nonnegative.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Student backbone checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, train_indices, test_indices = load_indices(train_path, args.test_pkl)
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
        pin_memory=(device.type == "cuda"),
    )

    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")

    student = build_student(config.student_cnn)
    student.load_weights(checkpoint_path)
    student.to(device)
    features, labels = student.extract_features(dataloader, device)

    if (
        features.shape[0] != len(dataloader.dataset)
        or labels.shape[0] != len(dataloader.dataset)
    ):
        raise RuntimeError(
            "Extracted feature and label counts do not match dataset."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "indices.npy", sample_indices)

    metadata = {
        "model": f"{config.approach} {config.student_cnn} StudentCNN",
        "encoder": f"{config.experiment_name}_seed{config.seed}",
        "approach": config.approach,
        "seed": config.seed,
        "student_cnn": config.student_cnn,
        "checkpoint": str(checkpoint_path),
        "num_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
        "dataset": "EuroSAT RGB",
        "batch_size": batch_size,
        "device": str(device),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved features to: {output_dir.resolve()}")


def main():
    args = parse_args()
    base_config = load_config(args.config)
    approach = args.approach or base_config.approach
    if approach not in {"vicreg", "distillation"}:
        raise ValueError(
            "probing.extract_features supports only 'vicreg' and 'distillation'; "
            "use the dedicated extractor for random_baseline or teacher_encoding."
        )

    if not args.seeds:
        if args.approach and args.approach != base_config.approach:
            raise ValueError(
                "--approach must match the approach recorded in the config."
            )
        if args.student_cnn and args.student_cnn != base_config.student_cnn:
            raise ValueError(
                "--student-cnn can only override the architecture when --seeds is used."
            )
        checkpoint_path = args.checkpoint_path or (
            base_config.output_dir / "student_backbone.pth"
        )
        output_dir = args.output_dir or (
            Path("datasets/features")
            / approach
            / ("small_cnn" if base_config.student_cnn == "small" else "larger_cnn")
            / f"seed{base_config.seed}"
        )
        extract(base_config, args, output_dir, checkpoint_path)
        return

    if args.checkpoint_path:
        raise ValueError(
            "--checkpoint-path cannot be combined with --seeds; each seed has "
            "its own student_backbone.pth."
        )
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain unique nonnegative integers.")

    student_cnn = args.student_cnn or base_config.student_cnn
    size_name = "small_cnn" if student_cnn == "small" else "larger_cnn"
    checkpoint_root = args.checkpoint_root or Path("checkpoints") / approach
    feature_root = args.output_dir or Path("datasets/features") / approach

    for seed in args.seeds:
        run_dir = checkpoint_root / size_name / f"seed{seed}"
        run_config_path = run_dir / "vicreg.json"
        if run_config_path.is_file():
            config = load_config(run_config_path)
            if (
                config.student_cnn != student_cnn
                or config.seed != seed
                or config.approach != approach
            ):
                raise ValueError(
                    f"Run config {run_config_path} does not match requested "
                    f"approach={approach}, student={student_cnn}, seed={seed}."
                )
        else:
            config = base_config
            config.student_cnn = student_cnn
            config.approach = approach
            config.seed = seed
            config.output_dir = run_dir

        print(
            f"\nExtracting approach={approach}: "
            f"student_cnn={student_cnn}, seed={seed}"
        )
        extract(
            config,
            args,
            output_dir=feature_root / size_name / f"seed{seed}",
            checkpoint_path=run_dir / "student_backbone.pth",
        )


if __name__ == "__main__":
    main()
