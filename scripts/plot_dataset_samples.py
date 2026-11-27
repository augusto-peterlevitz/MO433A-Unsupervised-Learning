import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from data.eurosat import EuroSAT


SAMPLES_PER_CLASS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a labeled grid of EuroSAT sample images."
    )
    parser.add_argument(
        "--eurosat-root",
        type=Path,
        default=Path("datasets/EuroSAT_RGB"),
        help="EuroSAT dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/eurosat_samples.png"),
        help="Path for the generated figure.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = EuroSAT(root=args.eurosat_root)
    samples_by_class = [[] for _ in dataset.classes]
    for image_path, label in dataset.samples:
        samples_by_class[label].append(image_path)

    rng = random.Random(args.random_state)
    selected_samples = []
    for class_name, class_samples in zip(dataset.classes, samples_by_class):
        if len(class_samples) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Class {class_name!r} has only {len(class_samples)} images; "
                f"{SAMPLES_PER_CLASS} are required."
            )
        selected_samples.append(rng.sample(class_samples, SAMPLES_PER_CLASS))

    figure, axes = plt.subplots(
        len(dataset.classes),
        SAMPLES_PER_CLASS,
        figsize=(12, 2.0 * len(dataset.classes)),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.20,
        right=0.995,
        bottom=0.005,
        top=0.995,
        wspace=0.01,
        hspace=0.01,
    )
    for row, (class_name, class_samples) in enumerate(
        zip(dataset.classes, selected_samples)
    ):
        for column, image_path in enumerate(class_samples):
            with Image.open(image_path) as image:
                axes[row, column].imshow(image.convert("RGB"))
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(
            class_name,
            rotation=0,
            ha="right",
            va="center",
            labelpad=10,
            fontweight="bold",
            fontsize=14,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved dataset sample figure to {args.output}")


if __name__ == "__main__":
    main()
