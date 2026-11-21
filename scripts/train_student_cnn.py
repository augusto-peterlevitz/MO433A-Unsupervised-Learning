import argparse
from dataclasses import replace
from pathlib import Path

from training.train_vicreg import load_config, train


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run row-B VICReg training for a named StudentCNN architecture "
            "across the requested random seeds."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vicreg.json"),
        help="Base VICReg training settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/row_b_vicreg"),
        help="Root directory for size- and seed-specific checkpoints.",
    )
    parser.add_argument(
        "--student-cnn",
        choices=("small", "large"),
        required=True,
        help="StudentCNN architecture variant to train.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Random seeds for this architecture; project minimum is 3 small, 2 large.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    minimum_seeds = 3 if args.student_cnn == "small" else 2
    if len(args.seeds) < minimum_seeds:
        raise ValueError(
            f"The {args.student_cnn} student requires at least {minimum_seeds} seeds."
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds cannot contain duplicates.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be nonnegative.")

    base_config = load_config(args.config)
    size_name = "small_cnn" if args.student_cnn == "small" else "larger_cnn"
    for seed in args.seeds:
        output_dir = args.output_dir / size_name / f"seed{seed}"
        config = replace(
            base_config,
            experiment_name=f"vicreg_{size_name}",
            student_cnn=args.student_cnn,
            seed=seed,
            output_dir=output_dir,
        )
        config.validate()
        print(
            f"\nStarting row B: student_cnn={args.student_cnn}, seed={seed}"
        )
        train(config)


if __name__ == "__main__":
    main()
