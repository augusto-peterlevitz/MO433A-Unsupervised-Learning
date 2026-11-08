import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune shared SVM settings on row B and evaluate feature caches."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    tune = commands.add_parser(
        "tune",
        help="Tune C (and gamma) on row B using training-only stratified CV.",
    )
    tune.add_argument("--features-dir", type=Path, required=True)
    tune.add_argument("--train-pkl", type=Path, default=Path("datasets/eurosat_train_subsets.pkl"))
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument("--kernel", choices=("linear", "rbf"), required=True)
    tune.add_argument("--max-tuning-samples", type=int, default=10_000)
    tune.add_argument("--n-jobs", type=int, default=1)

    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate fixed SVM settings on shared subsets and the held-out test set.",
    )
    evaluate.add_argument(
        "--features",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Encoder name and feature-cache directory; repeat for each encoder.",
    )
    evaluate.add_argument("--train-pkl", type=Path, default=Path("datasets/eurosat_train_subsets.pkl"))
    evaluate.add_argument("--test-pkl", type=Path, default=Path("datasets/eurosat_test_subset.pkl"))
    evaluate.add_argument("--params", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_cache(feature_dir):
    features = np.load(feature_dir / "features.npy")
    labels = np.load(feature_dir / "labels.npy")
    indices_path = feature_dir / "indices.npy"
    if not indices_path.exists():
        raise FileNotFoundError(
            f"{indices_path} is missing; rerun feature extraction to save sample indices."
        )
    indices = np.load(indices_path)

    if features.ndim != 2 or labels.ndim != 1 or indices.ndim != 1:
        raise ValueError(f"Invalid feature cache dimensions in {feature_dir}.")
    if features.shape[0] != labels.size or labels.size != indices.size:
        raise ValueError(f"Feature, label, and index counts differ in {feature_dir}.")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features found in {feature_dir}.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Duplicate dataset indices found in {indices_path}.")
    return features, labels, indices


def load_splits(train_path, test_path=None):
    with train_path.open("rb") as file:
        subsets = pickle.load(file)
    if test_path is None:
        test_indices = np.array([], dtype=np.int64)
    else:
        with test_path.open("rb") as file:
            test_indices = np.asarray(pickle.load(file), dtype=np.int64)

    full_train = np.asarray(subsets[(1.0, 0)], dtype=np.int64)
    if np.unique(full_train).size != full_train.size:
        raise ValueError("The full training subset contains duplicate indices.")
    if np.unique(test_indices).size != test_indices.size:
        raise ValueError("The test split contains duplicate indices.")
    if np.intersect1d(full_train, test_indices).size:
        raise ValueError("Training and test indices overlap.")
    return subsets, full_train, test_indices


def rows_for_indices(requested_indices, cached_indices):
    row_by_index = {int(index): row for row, index in enumerate(cached_indices)}
    missing = [int(index) for index in requested_indices if int(index) not in row_by_index]
    if missing:
        raise ValueError(f"Feature cache is missing dataset indices, e.g. {missing[:5]}.")
    return np.asarray([row_by_index[int(index)] for index in requested_indices])


def make_pipeline(kernel, c_value, gamma="scale"):
    classifier = (
        LinearSVC(C=c_value, dual="auto", max_iter=10_000)
        if kernel == "linear"
        else SVC(C=c_value, gamma=gamma, kernel="rbf")
    )
    return Pipeline([
        ("normalize", Normalizer(norm="l2")),
        ("scale", StandardScaler()),
        ("classifier", classifier),
    ])


def tune(args):
    if args.max_tuning_samples < 1:
        raise ValueError("--max-tuning-samples must be positive.")
    features, labels, cached_indices = load_cache(args.features_dir)
    _, full_train, _ = load_splits(args.train_pkl)
    rows = rows_for_indices(full_train, cached_indices)
    x_train = features[rows]
    y_train = labels[rows]

    if args.max_tuning_samples < len(y_train):
        x_train, _, y_train, _ = train_test_split(
            x_train,
            y_train,
            train_size=args.max_tuning_samples,
            stratify=y_train,
            random_state=42,
        )

    class_counts = np.unique(y_train, return_counts=True)[1]
    if class_counts.min() < 5:
        raise ValueError(
            "Each class needs at least five tuning samples for 5-fold stratified CV."
        )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    parameter_grid = {
        "classifier__C": [0.1, 1, 10, 100],
    }
    if args.kernel == "rbf":
        parameter_grid["classifier__gamma"] = ["scale", 0.0001, 0.001, 0.01, 0.1]

    search = GridSearchCV(
        make_pipeline(args.kernel, c_value=1),
        parameter_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=args.n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)

    result = {
        "kernel": args.kernel,
        "C": float(search.best_params_["classifier__C"]),
        "gamma": search.best_params_.get("classifier__gamma"),
        "cv_accuracy": float(search.best_score_),
        "cv_folds": 5,
        "tuning_samples": int(len(y_train)),
        "max_tuning_samples": int(args.max_tuning_samples),
        "features_dir": str(args.features_dir),
        "selection_protocol": "row B, 100% training split only",
        "preprocessing": ["L2 normalization", "StandardScaler fitted per training fold"],
        "parameter_grid": parameter_grid,
        "random_state": 42,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


def parse_named_dirs(values):
    result = []
    seen_names = set()
    for value in values:
        name, separator, directory = value.partition("=")
        if not separator or not name.strip() or not directory.strip():
            raise ValueError(f"Invalid --features value {value!r}; expected NAME=DIR.")
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"Duplicate encoder name: {name}")
        seen_names.add(name)
        result.append((name, Path(directory)))
    return result


def evaluate(args):
    with args.params.open(encoding="utf-8") as file:
        params = json.load(file)
    kernel = params["kernel"]
    if kernel not in ("linear", "rbf"):
        raise ValueError(f"Unsupported kernel in parameter file: {kernel}")
    subsets, _, test_indices = load_splits(args.train_pkl, args.test_pkl)
    encoder_dirs = parse_named_dirs(args.features)
    rows = []

    for encoder, feature_dir in encoder_dirs:
        features, labels, cached_indices = load_cache(feature_dir)
        test_rows = rows_for_indices(test_indices, cached_indices)
        x_test = features[test_rows]
        y_test = labels[test_rows]

        for (fraction, draw), sample_indices in sorted(subsets.items()):
            sample_indices = np.asarray(sample_indices, dtype=np.int64)
            sample_rows = rows_for_indices(sample_indices, cached_indices)
            pipeline = make_pipeline(kernel, params["C"], params.get("gamma") or "scale")
            pipeline.fit(features[sample_rows], labels[sample_rows])
            accuracy = pipeline.score(x_test, y_test)
            rows.append({
                "encoder": encoder,
                "kernel": kernel,
                "fraction": float(fraction),
                "draw": int(draw),
                "n_train": int(sample_rows.size),
                "n_test": int(test_rows.size),
                "accuracy": float(accuracy),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["encoder", "kernel", "fraction", "draw", "n_train", "n_test", "accuracy"]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} results to {args.output}")


def main():
    args = parse_args()
    if args.command == "tune":
        tune(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
