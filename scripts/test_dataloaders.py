import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build EuroSAT data loaders and report their dataset sizes."
    )
    parser.add_argument(
        "--eurosat-root",
        type=Path,
        default=Path("datasets/EuroSAT_RGB"),
        help="Path to the EuroSAT dataset directory.",
    )
    parser.add_argument(
        "--train-pkl",
        type=Path,
        default=Path("datasets/eurosat_train_subsets.pkl"),
        help="Path to the pickle containing training subset indices.",
    )
    parser.add_argument(
        "--test-pkl",
        type=Path,
        default=Path("datasets/eurosat_test_subset.pkl"),
        help="Path to the pickle containing held-out test indices.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from data.dataloaders import (
        load_indices,
        create_loader,
        train_transform,
        vicreg_transform,
        eval_transform,
    )

    train_subsets, train_indices, test_indices = load_indices(
        args.train_pkl,
        args.test_pkl,
    )

    vicreg_loader = create_loader(
        root=args.eurosat_root,
        indices=train_indices,
        transform=vicreg_transform,
        batch_size=64,
        shuffle=True,
        return_labels=False,
        drop_last=True,
    )

    distillation_loader = create_loader(
        root=args.eurosat_root,
        indices=train_indices,
        transform=train_transform,
        batch_size=64,
        shuffle=True,
        return_labels=False,
        drop_last=True,
    )

    # Example: 1% labeled subset, draw 0.
    labeled_indices = train_subsets[(0.01, 0)]
    probe_train_loader = create_loader(
        root=args.eurosat_root,
        indices=labeled_indices,
        transform=eval_transform,
        batch_size=128,
        shuffle=True,
        return_labels=True,
    )

    test_loader = create_loader(
        root=args.eurosat_root,
        indices=test_indices,
        transform=eval_transform,
        batch_size=128,
        shuffle=False,
        return_labels=True,
    )

    print("VICReg images:", len(vicreg_loader.dataset))
    print("Distillation images:", len(distillation_loader.dataset))
    print("Labeled training images:", len(probe_train_loader.dataset))
    print("Test images:", len(test_loader.dataset))


if __name__ == "__main__":
    main()
