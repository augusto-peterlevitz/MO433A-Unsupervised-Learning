"""Shared data loading and preprocessing for the MO433 SVM probes."""

import pickle
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC


SEED = 42
RBF_MAX_SAMPLES = 10_000


def subset_combinations(subsets):
    """Return the fraction/draw combinations present in a subsets pickle."""
    if not isinstance(subsets, dict):
        raise ValueError("Training subsets file must contain a dictionary.")

    combinations = []
    for key in subsets:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Training subset keys must be (fraction, draw) pairs.")
        fraction, draw = key
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, Real)
            or isinstance(draw, bool)
            or not isinstance(draw, Integral)
        ):
            raise ValueError(
                "Training subset keys must contain numeric fractions and integer draws."
            )
        fraction, draw = float(fraction), int(draw)
        if not 0 < fraction <= 1 or draw < 0:
            raise ValueError(f"Invalid training subset key {(fraction, draw)}.")
        combinations.append((fraction, draw))

    if len(set(combinations)) != len(combinations):
        raise ValueError("Training subsets contain duplicate fraction/draw keys.")
    if (1.0, 0) not in combinations:
        raise KeyError("Training subsets must contain the canonical key (1.0, 0).")
    return tuple(sorted(combinations))


def load_indices(train_subsets_path, test_indices_path):
    with open(train_subsets_path, "rb") as file:
        subsets = pickle.load(file)
    with open(test_indices_path, "rb") as file:
        test_indices = np.asarray(pickle.load(file), dtype=np.int64)

    combinations = subset_combinations(subsets)

    train_indices = np.asarray(subsets[(1.0, 0)], dtype=np.int64)
    if train_indices.ndim != 1 or len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("Canonical training indices must be a unique 1-D array.")
    if test_indices.ndim != 1 or len(np.unique(test_indices)) != len(test_indices):
        raise ValueError("Test indices must be a unique 1-D array.")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Training and test indices overlap.")

    train_index_set = set(train_indices.tolist())
    for key in combinations:
        fraction, _ = key
        indices = np.asarray(subsets[key], dtype=np.int64)
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ValueError(f"Subset {key} must contain unique 1-D indices.")
        if not set(indices.tolist()).issubset(train_index_set):
            raise ValueError(f"Subset {key} contains indices outside the train split.")
        if fraction == 1.0 and not np.array_equal(indices, train_indices):
            raise ValueError(f"Full-label subset {key} differs from the canonical split.")

    return subsets, train_indices, test_indices


def _validate_features(features, labels, description):
    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError(f"{description} features must be a 2-D array.")
    if labels.ndim != 1 or len(features) != len(labels):
        raise ValueError(f"{description} features and labels have incompatible shapes.")
    if len(labels) == 0:
        raise ValueError(f"{description} feature split is empty.")
    if not np.isfinite(features).all():
        raise ValueError(f"{description} features contain non-finite values.")
    return features, labels


def _reorder_split(features, labels, cached_indices, requested_indices, description):
    features, labels = _validate_features(features, labels, description)
    cached_indices = np.asarray(cached_indices, dtype=np.int64)
    if cached_indices.ndim != 1 or len(cached_indices) != len(features):
        raise ValueError(f"{description} cached indices do not match the feature rows.")
    if len(np.unique(cached_indices)) != len(cached_indices):
        raise ValueError(f"{description} cached indices contain duplicates.")
    if set(cached_indices.tolist()) != set(requested_indices.tolist()):
        raise ValueError(f"{description} cached indices do not match the requested split.")

    row_by_index = {
        int(original_index): row
        for row, original_index in enumerate(cached_indices)
    }
    rows = np.asarray([row_by_index[int(index)] for index in requested_indices])
    return features[rows], labels[rows]


def _feature_source_paths(feature_path):
    feature_path = Path(feature_path)
    if feature_path.is_dir():
        return (
            feature_path / "features_full.npz",
            feature_path / "features.npy",
            feature_path / "labels.npy",
        )
    return (
        feature_path if feature_path.suffix == ".npz" else None,
        feature_path.parent / "features.npy",
        feature_path.parent / "labels.npy",
    )


def load_training_features(feature_path, train_indices):
    """Load only the canonical training features, without reading test data."""
    indexed_cache, features_path, labels_path = _feature_source_paths(feature_path)
    if indexed_cache is not None and indexed_cache.exists():
        with np.load(indexed_cache, allow_pickle=False) as cache:
            required = {"train_features", "train_labels", "train_indices"}
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(
                    f"Indexed feature cache {indexed_cache} is missing: "
                    f"{', '.join(sorted(missing))}."
                )
            train_features, train_labels = _reorder_split(
                cache["train_features"],
                cache["train_labels"],
                cache["train_indices"],
                train_indices,
                "Training",
            )
    else:
        if not features_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Expected either {indexed_cache} or both "
                f"{features_path} and {labels_path}."
            )
        features, labels = _validate_features(
            np.load(features_path, mmap_mode="r"),
            np.load(labels_path, mmap_mode="r"),
            "Full-dataset",
        )
        if len(train_indices) and (
            train_indices.min() < 0 or train_indices.max() >= len(features)
        ):
            raise IndexError("A training index is outside the cached feature rows.")
        train_features, train_labels = features[train_indices], labels[train_indices]

    return _validate_features(train_features, train_labels, "Training")


def load_feature_splits(feature_path, train_indices, test_indices):
    """Load either indexed train/test NPZ features or full-dataset NPY features."""
    indexed_cache, features_path, labels_path = _feature_source_paths(feature_path)

    if indexed_cache is not None and indexed_cache.exists():
        with np.load(indexed_cache, allow_pickle=False) as cache:
            required = {
                "train_features",
                "train_labels",
                "train_indices",
                "test_features",
                "test_labels",
                "test_indices",
            }
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(
                    f"Indexed feature cache {indexed_cache} is missing: "
                    f"{', '.join(sorted(missing))}."
                )
            train_features, train_labels = _reorder_split(
                cache["train_features"],
                cache["train_labels"],
                cache["train_indices"],
                train_indices,
                "Training",
            )
            test_features, test_labels = _reorder_split(
                cache["test_features"],
                cache["test_labels"],
                cache["test_indices"],
                test_indices,
                "Test",
            )
    else:
        if not features_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Expected either {indexed_cache} or both "
                f"{features_path} and {labels_path}."
            )
        features, labels = _validate_features(
            np.load(features_path, mmap_mode="r"),
            np.load(labels_path, mmap_mode="r"),
            "Full-dataset",
        )
        all_indices = np.concatenate((train_indices, test_indices))
        if len(all_indices) and (all_indices.min() < 0 or all_indices.max() >= len(features)):
            raise IndexError("A requested split index is outside the cached feature rows.")
        train_features, train_labels = features[train_indices], labels[train_indices]
        test_features, test_labels = features[test_indices], labels[test_indices]

    train_features, train_labels = _validate_features(
        train_features, train_labels, "Training"
    )
    test_features, test_labels = _validate_features(test_features, test_labels, "Test")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Training and test feature dimensions do not match.")
    return train_features, train_labels, test_features, test_labels


def align_subset_rows(subsets, train_indices, train_features, train_labels):
    """Yield labeled split arrays in the cached canonical training-row order."""
    row_by_index = {
        int(original_index): row
        for row, original_index in enumerate(train_indices)
    }
    for fraction, draw in subset_combinations(subsets):
        original_indices = np.asarray(subsets[(fraction, draw)], dtype=np.int64)
        rows = np.asarray(
            [row_by_index[int(index)] for index in original_indices],
            dtype=np.int64,
        )
        yield fraction, draw, train_features[rows], train_labels[rows]


def stratified_cap(features, labels, max_samples=RBF_MAX_SAMPLES):
    """Deterministically cap an RBF training set while preserving class ratios."""
    if len(labels) <= max_samples:
        return features, labels

    from sklearn.model_selection import train_test_split

    selected, _ = train_test_split(
        np.arange(len(labels)),
        train_size=max_samples,
        stratify=labels,
        random_state=SEED,
    )
    return features[selected], labels[selected]


def make_linear_pipeline(c_value):
    return Pipeline(
        [
            ("l2_normalize", Normalizer(norm="l2")),
            ("standardize", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=float(c_value),
                    dual=False,
                    random_state=SEED,
                    max_iter=20_000,
                ),
            ),
        ]
    )


def make_rbf_pipeline(c_value, gamma_value):
    return Pipeline(
        [
            ("l2_normalize", Normalizer(norm="l2")),
            ("standardize", StandardScaler()),
            (
                "svm",
                SVC(kernel="rbf", C=float(c_value), gamma=gamma_value),
            ),
        ]
    )
