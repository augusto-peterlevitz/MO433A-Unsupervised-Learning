import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from data.dataloaders import get_data, load_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check cached representations with k-NN, k-means, embedding "
            "variation, and optional t-SNE."
        )
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        action="append",
        required=True,
        help="Feature cache directory; repeat once for each encoder.",
    )
    parser.add_argument(
        "--test-pkl",
        type=Path,
        default=Path("datasets/eurosat_test_subset.pkl"),
        help="Pickle containing held-out test indices.",
    )
    parser.add_argument(
        "--train-pkl",
        type=Path,
        default=Path("datasets/eurosat_train_subsets.pkl"),
        help="Pickle containing training indices and labeled probe subsets.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/probes"))
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--collapse-only",
        action="store_true",
        help="Save collapse indicators without computing t-SNE projections.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    _, train_indices, test_indices = load_indices(args.train_pkl, args.test_pkl)
    encoder_data = []
    reference_train_labels = None
    reference_test_labels = None

    for feature_dir in args.features_dir:
        train_features, train_labels = get_data(train_indices, feature_dir)
        test_features, test_labels = get_data(test_indices, feature_dir)
        if (
            train_features.ndim != 2
            or train_features.shape[0] != train_indices.size
        ):
            raise ValueError(
                f"Expected a 2-D training feature matrix for {feature_dir} with "
                f"{train_indices.size} rows; got {train_features.shape}."
            )
        if (
            test_features.ndim != 2
            or test_features.shape[0] != test_indices.size
        ):
            raise ValueError(
                f"Expected a 2-D feature matrix for {feature_dir} with "
                f"{test_indices.size} rows; got {test_features.shape}."
            )
        if train_features.shape[1] != test_features.shape[1]:
            raise ValueError(
                f"Training and test feature dimensions differ in {feature_dir}."
            )
        if (
            not np.isfinite(train_features).all()
            or not np.isfinite(test_features).all()
        ):
            raise ValueError(f"Features in {feature_dir} contain non-finite values.")

        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)
        if reference_train_labels is None:
            reference_train_labels = train_labels
        elif not np.array_equal(reference_train_labels, train_labels):
            raise ValueError(
                f"Training labels in {feature_dir} do not match the other feature caches."
            )
        if reference_test_labels is None:
            reference_test_labels = test_labels
        elif not np.array_equal(reference_test_labels, test_labels):
            raise ValueError(
                f"Test labels in {feature_dir} do not match the other feature caches."
            )

        # Cosine k-NN and k-means operate on unit-length features. Keep the
        # original train/test labels only for the supervised k-NN check.
        train_features = normalize(train_features, norm="l2")
        test_features = normalize(test_features, norm="l2")
        encoder_data.append(
            (
                feature_dir.as_posix(),
                train_features,
                train_labels,
                test_features,
                test_labels,
            )
        )

    labels = reference_test_labels
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("At least two test classes are required for diagnostics.")
    if not args.collapse_only and (
        args.perplexity <= 0 or args.perplexity >= test_indices.size
    ):
        raise ValueError(
            f"t-SNE perplexity must be greater than 0 and less than the "
            f"number of test samples ({test_indices.size})."
        )

    metrics = []
    projections = []
    for (
        encoder,
        train_features,
        train_labels,
        test_features,
        test_labels,
    ) in encoder_data:
        _, class_counts = np.unique(train_labels, return_counts=True)
        if class_counts.size == 0:
            raise ValueError(f"Training labels in {encoder} contain no classes.")
        n_neighbors = min(20, int(class_counts.min()))
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(train_features, train_labels)
        knn_accuracy = knn.score(test_features, test_labels)

        dimension_stds = test_features.std(axis=0)
        clusters = KMeans(
            n_clusters=classes.size,
            n_init=10,
            random_state=args.random_state,
        ).fit_predict(test_features)
        metrics.append({
            "encoder": encoder,
            "n_train_samples": int(train_labels.size),
            "n_test_samples": int(test_labels.size),
            "n_classes": int(classes.size),
            "knn_k": n_neighbors,
            "knn_accuracy": float(knn_accuracy),
            "chance_accuracy": float(1.0 / classes.size),
            "embedding_std_by_dimension": json.dumps(
                [float(value) for value in dimension_stds]
            ),
            "embedding_std_mean": float(dimension_stds.mean()),
            "embedding_std_min": float(dimension_stds.min()),
            "embedding_std_max": float(dimension_stds.max()),
            "nmi": float(normalized_mutual_info_score(test_labels, clusters)),
            "ari": float(adjusted_rand_score(test_labels, clusters)),
        })

        if not args.collapse_only:
            coordinates = TSNE(
                n_components=2,
                perplexity=args.perplexity,
                init="pca",
                learning_rate="auto",
                n_iter=1000,
                random_state=args.random_state,
            ).fit_transform(test_features)
            projections.append((encoder, coordinates))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_filename = (
        "collapse_metrics.csv" if args.collapse_only else "representation_metrics.csv"
    )
    metrics_path = args.output_dir / metrics_filename
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    if args.collapse_only:
        print(f"Saved collapse indicators to {metrics_path}")
        return

    projection_rows = []
    for encoder, coordinates in projections:
        projection_rows.extend({
            "encoder": encoder,
            "dataset_index": int(dataset_index),
            "label": label,
            "x": float(coordinate[0]),
            "y": float(coordinate[1]),
        } for dataset_index, label, coordinate in zip(
            test_indices, labels, coordinates
        ))
    projection_path = args.output_dir / "tsne_coordinates.csv"
    with projection_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=projection_rows[0].keys())
        writer.writeheader()
        writer.writerows(projection_rows)

    n_encoders = len(encoder_data)
    ncols = min(2, n_encoders)
    nrows = (n_encoders + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    color_map = plt.get_cmap("tab20", classes.size)
    for axis, (encoder, coordinates) in zip(axes.flat, projections):
        for class_index, label in enumerate(classes):
            selected = labels == label
            axis.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                s=10,
                alpha=0.7,
                color=color_map(class_index),
                label=str(label),
                rasterized=True,
            )
        axis.set_title(encoder)
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")

    for axis in axes.flat[n_encoders:]:
        axis.set_visible(False)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, title="True label", loc="outside right center")
    plot_path = args.output_dir / "tsne_projections.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    print(f"Saved representation metrics to {metrics_path}")
    print(f"Saved t-SNE coordinates to {projection_path}")
    print(f"Saved projection panel to {plot_path}")


if __name__ == "__main__":
    main()
