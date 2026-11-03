"""Select shared linear and RBF SVM hyperparameters on VICReg training data."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from evaluation.svm_utils import (
    RBF_MAX_SAMPLES,
    SEED,
    load_training_features,
    load_indices,
    make_linear_pipeline,
    make_rbf_pipeline,
    stratified_cap,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune SVM hyperparameters once on the 100% VICReg training split; "
            "the held-out test split is not used."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("evaluation/results/vicreg_width_1.0"),
        help="VICReg feature cache directory or features_full.npz file.",
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
        "--output",
        type=Path,
        default=Path(
            "evaluation/results/vicreg_width_1.0/svm_hyperparameters.json"
        ),
        help="JSON file where the selected parameters will be saved.",
    )
    parser.add_argument(
        "--grid-results",
        type=Path,
        default=None,
        help="Optional CSV path for all cross-validation candidates.",
    )
    return parser.parse_args()


def search(train_features, train_labels):
    class_counts = np.unique(train_labels, return_counts=True)[1]
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("At least two training examples per class are needed for CV.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    linear_search = GridSearchCV(
        estimator=make_linear_pipeline(1.0),
        param_grid={"svm__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        verbose=1,
    )
    linear_search.fit(train_features, train_labels)

    rbf_features, rbf_labels = stratified_cap(
        train_features, train_labels, RBF_MAX_SAMPLES
    )
    rbf_class_counts = np.unique(rbf_labels, return_counts=True)[1]
    rbf_splits = min(3, int(rbf_class_counts.min()))
    if rbf_splits < 2:
        raise ValueError("At least two RBF tuning examples per class are needed for CV.")
    rbf_cv = StratifiedKFold(
        n_splits=rbf_splits, shuffle=True, random_state=SEED
    )
    rbf_search = GridSearchCV(
        estimator=make_rbf_pipeline(1.0, "scale"),
        param_grid={
            "svm__C": [0.1, 1.0, 10.0],
            "svm__gamma": ["scale", 0.01, 0.1],
        },
        scoring="accuracy",
        cv=rbf_cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        verbose=1,
    )
    rbf_search.fit(rbf_features, rbf_labels)

    return {
        "linear_svm_C": float(linear_search.best_params_["svm__C"]),
        "rbf_svm_C": float(rbf_search.best_params_["svm__C"]),
        "rbf_svm_gamma": rbf_search.best_params_["svm__gamma"],
        "linear_cv_best_accuracy": float(linear_search.best_score_),
        "rbf_cv_best_accuracy": float(rbf_search.best_score_),
        "linear_cv_folds": n_splits,
        "rbf_cv_folds": rbf_splits,
        "rbf_tuning_sample_count": int(len(rbf_labels)),
        "rbf_max_samples": RBF_MAX_SAMPLES,
        "tuning_encoder": "VICReg",
        "tuning_label_fraction": 1.0,
        "random_state": SEED,
    }, linear_search, rbf_search


def write_grid_results(output_path, searches):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "probe",
                "params",
                "mean_cv_accuracy",
                "std_cv_accuracy",
                "rank",
            ),
        )
        writer.writeheader()
        for probe_name, grid_search in searches:
            results = grid_search.cv_results_
            for index, params in enumerate(results["params"]):
                writer.writerow(
                    {
                        "probe": probe_name,
                        "params": json.dumps(params, sort_keys=True),
                        "mean_cv_accuracy": float(results["mean_test_score"][index]),
                        "std_cv_accuracy": float(results["std_test_score"][index]),
                        "rank": int(results["rank_test_score"][index]),
                    }
                )


def main():
    args = parse_args()
    subsets, train_indices, _ = load_indices(
        args.train_subsets, args.test_indices
    )
    del subsets
    train_features, train_labels = load_training_features(
        args.feature_dir, train_indices
    )

    print(
        "Tuning on VICReg's 100% labeled training split only: "
        f"{len(train_labels)} examples, {train_features.shape[1]} features."
    )
    parameters, linear_search, rbf_search = search(train_features, train_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=2)
        file.write("\n")

    grid_results_path = args.grid_results
    if grid_results_path is None:
        grid_results_path = args.output.with_name("svm_grid_search.csv")
    write_grid_results(
        grid_results_path,
        (("linear_svm", linear_search), ("rbf_svm", rbf_search)),
    )

    print(f"\nSelected parameters: {json.dumps(parameters, indent=2)}")
    print(f"Saved parameters to: {args.output}")
    print(f"Saved grid-search results to: {grid_results_path}")


if __name__ == "__main__":
    main()
