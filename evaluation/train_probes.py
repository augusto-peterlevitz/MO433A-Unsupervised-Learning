"""Fit and evaluate k-NN, linear SVM, and RBF SVM probes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from evaluation.svm_utils import (
    align_subset_rows,
    load_feature_splits,
    load_indices,
    make_linear_pipeline,
    make_rbf_pipeline,
    stratified_cap,
    subset_combinations,
)


def parse_feature_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Feature sources must use NAME=PATH, for example dino=datasets/features/dino_vits16."
        )
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Feature source name and path must be non-empty.")
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the three project probes with shared SVM parameters."
    )
    parser.add_argument(
        "--feature",
        action="append",
        type=parse_feature_spec,
        required=True,
        metavar="NAME=PATH",
        help=(
            "Feature source. Repeat once per encoder. Supports a directory with "
            "features.npy/labels.npy or features_full.npz."
        ),
    )
    parser.add_argument(
        "--svm-params",
        type=Path,
        default=Path(
            "evaluation/results/vicreg_width_1.0/svm_hyperparameters.json"
        ),
        help="Parameters produced by search_svm_hyperparameters.",
    )
    parser.add_argument(
        "--train-subsets",
        type=Path,
        default=Path("datasets/eurosat_train_subsets.pkl"),
    )
    parser.add_argument(
        "--test-indices",
        type=Path,
        default=Path("datasets/eurosat_test_subset.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results/all_probes"),
    )
    return parser.parse_args()


def load_svm_parameters(path):
    with path.open(encoding="utf-8") as file:
        parameters = json.load(file)
    required = ("linear_svm_C", "rbf_svm_C", "rbf_svm_gamma", "rbf_max_samples")
    missing = [key for key in required if key not in parameters]
    if missing:
        raise ValueError(f"SVM parameter file is missing: {', '.join(missing)}.")
    return parameters


def evaluate_knn(train_features, train_labels, test_features, test_labels):
    minimum_class_count = int(np.unique(train_labels, return_counts=True)[1].min())
    neighbors = min(20, minimum_class_count)
    probe = Pipeline(
        [
            ("l2_normalize", Normalizer(norm="l2")),
            ("standardize", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=neighbors, metric="cosine", algorithm="brute"
                ),
            ),
        ]
    )
    probe.fit(train_features, train_labels)
    accuracy = accuracy_score(test_labels, probe.predict(test_features))
    return float(accuracy), neighbors, len(train_labels)


def evaluate_svm(
    train_features,
    train_labels,
    test_features,
    test_labels,
    parameters,
    probe_name,
):
    if probe_name == "linear_svm":
        probe = make_linear_pipeline(parameters["linear_svm_C"])
    elif probe_name == "rbf_svm":
        train_features, train_labels = stratified_cap(
            train_features, train_labels, int(parameters["rbf_max_samples"])
        )
        probe = make_rbf_pipeline(
            parameters["rbf_svm_C"], parameters["rbf_svm_gamma"]
        )
    else:
        raise ValueError(f"Unknown SVM probe: {probe_name}")

    probe.fit(train_features, train_labels)
    accuracy = accuracy_score(test_labels, probe.predict(test_features))
    return float(accuracy), len(train_labels)


def run_evaluation(
    feature_sources,
    subsets,
    train_indices,
    test_indices,
    svm_parameters,
):
    all_results = []
    reference_train_labels = None
    reference_test_labels = None

    for encoder, feature_path in feature_sources:
        print(f"\nLoading {encoder} features from {feature_path}")
        train_features, train_labels, test_features, test_labels = load_feature_splits(
            feature_path, train_indices, test_indices
        )
        if reference_train_labels is None:
            reference_train_labels = train_labels
        elif not np.array_equal(reference_train_labels, train_labels):
            raise ValueError(
                f"{encoder} training labels do not align with the other encoders."
            )
        if reference_test_labels is None:
            reference_test_labels = test_labels
        elif not np.array_equal(reference_test_labels, test_labels):
            raise ValueError(
                f"{encoder} feature labels do not align with the other encoders."
            )

        for fraction, draw, x_train, y_train in align_subset_rows(
            subsets, train_indices, train_features, train_labels
        ):
            print(
                f"  fraction={fraction:.0%}, draw={draw}, "
                f"training examples={len(y_train)}"
            )
            knn_accuracy, neighbors, knn_count = evaluate_knn(
                x_train, y_train, test_features, test_labels
            )
            all_results.append(
                {
                    "encoder": encoder,
                    "label_fraction": fraction,
                    "draw": draw,
                    "probe": "knn_cosine",
                    "accuracy": knn_accuracy,
                    "n_train_probe": knn_count,
                    "knn_k": neighbors,
                    "svm_C": "",
                    "svm_gamma": "",
                }
            )

            for probe_name in ("linear_svm", "rbf_svm"):
                accuracy, train_count = evaluate_svm(
                    x_train,
                    y_train,
                    test_features,
                    test_labels,
                    svm_parameters,
                    probe_name,
                )
                all_results.append(
                    {
                        "encoder": encoder,
                        "label_fraction": fraction,
                        "draw": draw,
                        "probe": probe_name,
                        "accuracy": accuracy,
                        "n_train_probe": train_count,
                        "knn_k": "",
                        "svm_C": svm_parameters[
                            "linear_svm_C"
                            if probe_name == "linear_svm"
                            else "rbf_svm_C"
                        ],
                        "svm_gamma": (
                            svm_parameters["rbf_svm_gamma"]
                            if probe_name == "rbf_svm"
                            else ""
                        ),
                    }
                )
                print(f"    {probe_name}: {accuracy:.4f}")
            print(f"    knn_cosine (k={neighbors}): {knn_accuracy:.4f}")

    return all_results


def aggregate_results(results):
    grouped = {}
    for row in results:
        key = (row["encoder"], row["label_fraction"], row["probe"])
        grouped.setdefault(key, []).append(row["accuracy"])

    summary = []
    for encoder, fraction, probe in sorted(
        grouped, key=lambda item: (item[0], item[1], item[2])
    ):
        scores = grouped[(encoder, fraction, probe)]
        summary.append(
            {
                "encoder": encoder,
                "label_fraction": fraction,
                "probe": probe,
                "accuracy_mean": float(np.mean(scores)),
                "accuracy_std": (
                    float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                ),
                "draws": len(scores),
            }
        )
    return summary


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    names = [name for name, _ in args.feature]
    if len(set(names)) != len(names):
        raise ValueError("Each --feature encoder name must be unique.")

    subsets, train_indices, test_indices = load_indices(
        args.train_subsets, args.test_indices
    )
    combinations = subset_combinations(subsets)
    svm_parameters = load_svm_parameters(args.svm_params)
    results = run_evaluation(
        args.feature,
        subsets,
        train_indices,
        test_indices,
        svm_parameters,
    )
    summary = aggregate_results(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_draw_path = args.output_dir / "probe_results_by_draw.csv"
    summary_path = args.output_dir / "probe_summary.csv"
    write_csv(
        per_draw_path,
        results,
        (
            "encoder",
            "label_fraction",
            "draw",
            "probe",
            "accuracy",
            "n_train_probe",
            "knn_k",
            "svm_C",
            "svm_gamma",
        ),
    )
    write_csv(
        summary_path,
        summary,
        (
            "encoder",
            "label_fraction",
            "probe",
            "accuracy_mean",
            "accuracy_std",
            "draws",
        ),
    )

    protocol = {
        "svm_hyperparameters": svm_parameters,
        "label_fractions": sorted({fraction for fraction, _ in combinations}),
        "draws": sorted({draw for _, draw in combinations}),
        "subset_combinations": [list(key) for key in combinations],
        "feature_preprocessing": (
            "L2 Normalizer followed by StandardScaler fitted on each probe's "
            "training split only"
        ),
        "rbf_subsampling": (
            "Deterministic stratified sample, capped at the tuning sample limit"
        ),
        "test_split_used_for": ["final probe accuracy only"],
        "results_by_draw": str(per_draw_path),
        "summary": str(summary_path),
    }
    protocol_path = args.output_dir / "probe_protocol.json"
    with protocol_path.open("w", encoding="utf-8") as file:
        json.dump(protocol, file, indent=2)
        file.write("\n")

    print("\nProbe summary (mean ± standard deviation across draws)")
    for row in summary:
        print(
            f"{row['encoder']:16s} {row['label_fraction']:>5.0%} "
            f"{row['probe']:12s} {row['accuracy_mean']:.4f} "
            f"± {row['accuracy_std']:.4f}"
        )
    print(f"\nPer-draw results: {per_draw_path}")
    print(f"Summary:          {summary_path}")
    print(f"Protocol:         {protocol_path}")


if __name__ == "__main__":
    main()
