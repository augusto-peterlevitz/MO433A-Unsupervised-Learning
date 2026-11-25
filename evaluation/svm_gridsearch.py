import argparse
import json
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC

from data.dataloaders import get_data, load_indices
from evaluation.utils import (
    get_aggregate_encoder_name,
    get_approach,
    get_cache_seed,
    get_encoder_name,
    get_student_size,
    load_feature_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune shared SVM settings on one designated VICReg feature cache. "
            "Use the same resulting parameters for every seed and encoder."
        )
    )

    parser.add_argument("--features-dir", dest="feature_dir", type=Path, required=True)
    parser.add_argument(
        "--train-pkl",
        type=Path,
        default=Path("datasets/eurosat_train_subsets.pkl"),
    )
    parser.add_argument(
        "--test-pkl",
        type=Path,
        default=Path("datasets/eurosat_test_subset.pkl"),
        help="Required by the shared split loader; test data is not used for tuning.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/probes"))
    parser.add_argument("--kernel", choices=("linear", "rbf"), required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for cross-validation splits and LinearSVC (default: 42).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of splits for cross-validation (default: 5).",
    )

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
        "cv_random_seed": random_state,
        "cv_n_splits": n_splits,
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
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive.")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if not args.feature_dir.is_dir():
        raise FileNotFoundError(f"Feature cache not found: {args.feature_dir}")

    _, train_indices, _ = load_indices(
        args.train_pkl,
        args.test_pkl,
    )

    metadata = load_feature_metadata(args.feature_dir)
    approach = get_approach(metadata)
    if approach and approach != "vicreg":
        raise ValueError(
            "SVM hyperparameters must be tuned on a 'vicreg' feature cache."
        )
    x_train, y_train = get_data(train_indices, args.feature_dir)

    parameter_grid = {
            "classifier__C": [0.1, 1, 10, 100],
        }
    if args.kernel == "linear":
        classifier = LinearSVC(
            C=1,
            dual="auto",
            max_iter=100_000,
            random_state=args.seed,
        )

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
        random_state=args.seed,
        pipeline=gridsearch_pipeline,
        parameter_grid=parameter_grid,
        n_jobs=args.n_jobs,
    )

    cache_name = get_aggregate_encoder_name(
        get_encoder_name(args.feature_dir, metadata)
    )
    seed = get_cache_seed(args.feature_dir, metadata)
    student_size = get_student_size(metadata, args.feature_dir)
    if approach and student_size:
        result_dir = args.output_dir / "svm_gridsearch" / approach / student_size
    else:
        result_dir = args.output_dir / "svm_gridsearch" / cache_name
    if seed is not None:
        result_dir /= f"seed{seed}"
    output_path = result_dir / f"svm_gridsearch_{args.kernel}.json"
    results["kernel"] = args.kernel
    results["tuning_encoder"] = (
        f"{approach}/{student_size}" if approach and student_size else cache_name
    )
    results["tuning_seed"] = seed
    save_results(output_path, results)

if __name__ == "__main__":
    main()
