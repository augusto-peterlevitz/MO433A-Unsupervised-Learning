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
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune shared SVM settings on row B and evaluate feature caches."
    )

    parser.add_argument("--train-pkl", type=Path, default=Path("datasets/eurosat_train_subsets.pkl"))
    parser.add_argument("--test-pkl", type=Path, default=Path("datasets/eurosat_test_subset.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tuning-samples", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--features-dir", dest="feature_dir", type=Path, required=True)

    return parser.parse_args()

def load_parameters(params_path):
    with params_path.open(encoding="utf-8") as file:
        params = json.load(file)
    return params


def evaluate(training_pipeline, train_subsets, x_test, y_test, feature_dir):
    results = []
    for (fraction, draw), sample_indices in tqdm(sorted(train_subsets.items()), desc="Evaluating subsets"):
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        x_train, y_train = get_data(sample_indices, feature_dir)

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
    print(f"Loaded SVM parameters: {svm_params}")

    if svm_params["kernel"] == "linear":
        classifier = LinearSVC(C=svm_params["C"], dual="auto", max_iter=10_000)

    elif svm_params["kernel"] == "rbf":
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

    results = evaluate(training_pipeline, train_subsets, x_test, y_test, args.feature_dir)

    output_path = args.output_dir / f"svm_evaluation_{svm_params['kernel']}.csv"
    results.to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
