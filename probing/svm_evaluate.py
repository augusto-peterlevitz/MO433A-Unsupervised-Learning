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
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune shared SVM settings on row B and evaluate feature caches."
    )

    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--train-pkl", type=Path, default=Path("datasets/eurosat_train_subsets.pkl"))
    parser.add_argument("--output_dir", type=Path, required=True, default=Path("outputs)"))
    parser.add_argument("--kernel", choices=("linear", "rbf"), required=True)
    parser.add_argument("--max-tuning-samples", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=4)

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

def load_parameters(params_path):
    with params_path.open(encoding="utf-8") as file:
        params = json.load(file)
    return params

def evaluate(output_path, train_subsets, test_indices, args):
    
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

def evaluate(training_pipeline, train_subsets, x_test, y_test):
    results = []
    for (fraction, draw), sample_indices in sorted(train_subsets.items()):
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        x_train, y_train = get_data(sample_indices, args.feature_dir)

        training_pipeline.fit(x_train, y_train)
        accuracy = training_pipeline.score(x_test, y_test)
        results.append({
            "fraction": float(fraction),
            "draw": int(draw),
            "accuracy": float(accuracy),
        })
    
    return pd.DataFrame(results)


def main():
    args = parse_args()

    svm_params = load_parameters(args.params)

    if args.kernel == "linear":
        classifier = LinearSVC(C=svm_params["C"], dual="auto", max_iter=10_000)

    elif args.kernel == "rbf":
        classifier = SVC(C=svm_params["C"], gamma=svm_params.get("gamma") or "scale", kernel="rbf")    
    else:
        raise ValueError(f"Unsupported kernel: {args.kernel}")
    
    training_pipeline = Pipeline([
        ("normalize", Normalizer(norm="l2")),
        ("scale", StandardScaler()),
        ("classifier", classifier),
    ])

    train_subsets, train_indices, test_indices = load_indices(
        args.train_pkl,
        args.test_pkl,
    )
    x_test, y_test = get_data(test_indices, args.feature_dir)

    results = evaluate(training_pipeline, train_subsets, x_test, y_test)

    output_path = args.output_dir / f"svm_evaluation_{args.kernel}.csv"
    results.to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
