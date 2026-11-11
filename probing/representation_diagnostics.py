import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Characterize cached test features with k-means and 2-D t-SNE."
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_test_indices(test_path):
    with test_path.open("rb") as file:
        indices = np.asarray(pickle.load(file), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"Test indices in {test_path} must be a non-empty 1-D array.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Test indices in {test_path} contain duplicates.")
    return indices


def load_cached_features(requested_indices, feature_dir):
    features = np.load(feature_dir / "features.npy")
    labels = np.load(feature_dir / "labels.npy")
    indices = np.load(feature_dir / "indices.npy")
    if features.ndim != 2 or labels.ndim != 1 or indices.ndim != 1:
        raise ValueError(f"Invalid feature cache array shapes in {feature_dir}.")
    if not (features.shape[0] == labels.size == indices.size):
        raise ValueError(f"Feature, label, and index counts differ in {feature_dir}.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Feature cache indices in {feature_dir} contain duplicates.")

    row_by_index = {int(index): row for row, index in enumerate(indices)}
    missing = [
        int(index) for index in requested_indices if int(index) not in row_by_index
    ]
    if missing:
        raise ValueError(
            f"Feature cache {feature_dir} is missing dataset indices, "
            f"e.g. {missing[:5]}."
        )
    rows = np.asarray([row_by_index[int(index)] for index in requested_indices])
    return features[rows], labels[rows]


def main():
    args = parse_args()
    if len(set(path.name for path in args.features_dir)) != len(args.features_dir):
        raise ValueError("Feature cache directory names must be unique.")

    test_indices = load_test_indices(args.test_pkl)
    encoder_data = []
    reference_labels = None

    for feature_dir in args.features_dir:
        features, labels = load_cached_features(test_indices, feature_dir)
        if features.ndim != 2 or features.shape[0] != test_indices.size:
            raise ValueError(
                f"Expected a 2-D feature matrix for {feature_dir} with "
                f"{test_indices.size} rows; got {features.shape}."
            )
        if not np.isfinite(features).all():
            raise ValueError(f"Features in {feature_dir} contain non-finite values.")

        labels = np.asarray(labels)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise ValueError(
                f"Test labels in {feature_dir} do not match the other feature caches."
            )

        # Unit-length features put embeddings from differently scaled encoders
        # on a comparable scale for these distance-based diagnostics.
        features = normalize(features, norm="l2")
        encoder_data.append((feature_dir.name, features, labels))

    labels = reference_labels
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("At least two test classes are required for diagnostics.")
    if args.perplexity <= 0 or args.perplexity >= test_indices.size:
        raise ValueError(
            f"t-SNE perplexity must be greater than 0 and less than the "
            f"number of test samples ({test_indices.size})."
        )

    metrics = []
    projections = []
    for encoder, features, encoder_labels in encoder_data:
        clusters = KMeans(
            n_clusters=classes.size,
            n_init=10,
            random_state=args.random_state,
        ).fit_predict(features)
        metrics.append({
            "encoder": encoder,
            "n_test_samples": int(encoder_labels.size),
            "n_classes": int(classes.size),
            "nmi": float(normalized_mutual_info_score(encoder_labels, clusters)),
            "ari": float(adjusted_rand_score(encoder_labels, clusters)),
        })

        coordinates = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            init="pca",
            learning_rate="auto",
            n_iter=1000,
            random_state=args.random_state,
        ).fit_transform(features)
        projections.append((encoder, coordinates))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "representation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

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

    print(f"Saved k-means NMI/ARI results to {metrics_path}")
    print(f"Saved t-SNE coordinates to {projection_path}")
    print(f"Saved projection panel to {plot_path}")


if __name__ == "__main__":
    main()
