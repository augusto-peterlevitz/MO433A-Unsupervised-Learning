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
        description="Generate labeled EuroSAT training and test subsets."
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
        default=Path("datasets/eurosat_train_subsets.pkl"),
        help="Path where the training subset indices pickle will be saved.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Fraction of the dataset held out for final testing (default: 0.20).",
    )
    parser.add_argument(
        "--test-output-pkl",
        type=Path,
        default=Path("datasets/eurosat_test_subset.pkl"),
        help="Path where the held-out test indices pickle will be saved.",
    )
    return parser.parse_args()


def generate_subsets(labels, candidate_indices=None):
    """Generate stratified EuroSAT training subsets from candidate indices.

    Parameters
    ----------
    labels : numpy.ndarray
        Class label for each dataset sample, in dataset index order.
    candidate_indices : array-like, optional
        Indices available for training. Defaults to all dataset indices.

    Returns
    -------
    dict
        Mapping from ``(fraction, draw)`` pairs to arrays of sample indices.
    """
    if candidate_indices is None:
        candidate_indices = np.arange(len(labels))
    else:
        candidate_indices = np.asarray(candidate_indices)

    subsets = {}

    for frac in [0.01, 0.10, 0.50, 1.00]:
        for draw in range(3):
            if frac == 1.0:
                subsets[(frac, draw)] = candidate_indices.copy()
            else:
                indices, _ = train_test_split(
                    candidate_indices,
                    train_size=frac,
                    stratify=labels[candidate_indices],
                    random_state=draw,
                )
                subsets[(frac, draw)] = indices

    return subsets


def generate_test_subset(labels, test_fraction):
    """Return stratified train and held-out test indices."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=test_fraction,
        stratify=labels,
        random_state=42,
    )
    return train_indices, test_indices


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
    """Load EuroSAT, generate subsets, and save training and test indices."""
    args = parse_args()
    if args.output_pkl.resolve() == args.test_output_pkl.resolve():
        raise ValueError("Training and test output pickle paths must be different")

    dataset = EuroSAT(
        root=args.eurosat_root,
        transform=None,
    )

    # Get labels in the exact same order as dataset indices.
    labels = np.array([label for _, label in dataset])

    train_indices, test_indices = generate_test_subset(
        labels, args.test_fraction
    )
    subsets = generate_subsets(labels, train_indices)
    check_and_count_subsets(labels, subsets)
    print(
        f"Final test subset: fraction={args.test_fraction:.2f}, "
        f"n={len(test_indices)}, "
        f"class_counts={dict(sorted(Counter(labels[test_indices]).items()))}"
    )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(subsets, f)
    args.test_output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.test_output_pkl.open("wb") as f:
        pickle.dump(test_indices, f)

    print(f"Saved training subsets to {args.output_pkl}")
    print(f"Saved final test indices to {args.test_output_pkl}")


if __name__ == "__main__":
    main()