import argparse
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from data.eurosat import EuroSAT


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed dataset root and output pickle path.
    """
    parser = argparse.ArgumentParser(
        description="Generate labeled EuroSAT subsets and save their indices."
    )
    parser.add_argument(
        "--eurosat-root",
        type=Path,
        default=Path("datasets/EuroSAT_RGB"),
        help="Path to the EuroSAT dataset directory (default: datasets/EuroSAT_RGB)",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=Path("datasets/eurosat_subsets.pkl"),
        help="Path where the subset indices pickle will be saved (default: label_subsets.pkl)",
    )
    return parser.parse_args()


def generate_subsets(labels):
    """Generate stratified EuroSAT index subsets for several fractions.

    Parameters
    ----------
    labels : numpy.ndarray
        Class label for each dataset sample, in dataset index order.

    Returns
    -------
    dict
        Mapping from ``(fraction, draw)`` pairs to arrays of sample indices.
    """
    subsets = {}

    for frac in [0.01, 0.10, 0.50, 1.00]:
        for draw in range(3):
            if frac == 1.0:
                subsets[(frac, draw)] = np.arange(len(labels))
            else:
                indices, _ = train_test_split(
                    np.arange(len(labels)),
                    train_size=frac,
                    stratify=labels,
                    random_state=draw,
                )
                subsets[(frac, draw)] = indices

    return subsets


def check_and_count_subsets(labels, subsets):
    """Print dataset totals and class counts for each generated subset.

    Parameters
    ----------
    labels : numpy.ndarray
        Class label for each dataset sample, in dataset index order.
    subsets : dict
        Mapping from ``(fraction, draw)`` pairs to arrays of sample indices.

    Returns
    -------
    None
    """
    print(f"Number of samples: {len(labels)}")
    print(f"Number of classes: {len(np.unique(labels))}")

    for (frac, draw), indices in subsets.items():
        selected_labels = labels[indices]
        counts = Counter(selected_labels)

        print(
            f"frac={frac:.2f}, "
            f"draw={draw}, "
            f"n={len(indices)}, "
            f"class_counts={dict(sorted(counts.items()))}"
        )


def main():
    """Load EuroSAT, generate and report subsets, then save them to a pickle."""
    args = parse_args()

    dataset = EuroSAT(
        root=args.eurosat_root,
        transform=None,
    )

    # Get labels in the exact same order as dataset indices.
    labels = np.array([label for _, label in dataset])

    subsets = generate_subsets(labels)
    check_and_count_subsets(labels, subsets)

    with args.output_pkl.open("wb") as f:
        pickle.dump(subsets, f)

    print(f"Saved label subsets to {args.output_pkl}")


if __name__ == "__main__":
    main()