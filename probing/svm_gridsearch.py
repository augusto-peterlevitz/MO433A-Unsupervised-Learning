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

from data.dataloaders import load_indices, get_data
import numpy as np

RANDOM_SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune shared SVM settings on row B and evaluate feature caches."
    )

    parser.add_argument("--features-dir", dest="feature_dir", type=Path, required=True)
    parser.add_argument("--train-pkl", type=Path, default=Path("datasets/eurosat_train_subsets.pkl"))
    parser.add_argument("--test-pkl", type=Path, default=Path("datasets/eurosat_test_subset.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel", choices=("linear", "rbf"), required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-splits", type=int, default=5, help="Number of splits for cross-validation (default: 5).")

    return parser.parse_args()

def perform_gridsearch(x_train, y_train, n_splits, random_state, pipeline, parameter_grid, n_jobs):

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)

    result = {
        "C": float(search.best_params_["classifier__C"]),
        "gamma": search.best_params_.get("classifier__gamma"),
        "cv_accuracy": float(search.best_score_),
    }

    print(f"Best parameters: {result}")

    return result

def save_results(output_path, results):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Saved results to {output_path}")


def main():
    args = parse_args()

    train_subsets, train_indices, test_indices = load_indices(
        args.train_pkl,
        args.test_pkl,
    )

    x_train, y_train = get_data(train_indices, args.feature_dir)

    parameter_grid = {
            "classifier__C": [0.1, 1, 10, 100],
        }
    if args.kernel == "linear":
        classifier = LinearSVC(C=1, dual="auto", max_iter=10_000)

    elif args.kernel == "rbf":
        classifier = SVC(C=1, gamma="scale", kernel="rbf")
        parameter_grid["classifier__gamma"] = ["scale", 0.0001, 0.001, 0.01, 0.1]
    
    else:
        raise ValueError(f"Unsupported kernel: {args.kernel}")
    
    gridsearch_pipeline = Pipeline([
        ("normalize", Normalizer(norm="l2")),
        ("scale", StandardScaler()),
        ("classifier", classifier),
    ])

    print(f"Performing grid search for {args.kernel} kernel...")
    results = perform_gridsearch(
        x_train=x_train,
        y_train=y_train,
        n_splits=args.n_splits,
        random_state=RANDOM_SEED,
        pipeline=gridsearch_pipeline,
        parameter_grid=parameter_grid,
        n_jobs=args.n_jobs,
    )

    output_path = args.output_dir / f"svm_gridsearch_{args.kernel}.json"
    results["kernel"] = args.kernel
    save_results(output_path, results)

if __name__ == "__main__":
    main()
