import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from data.dataloaders import load_indices, get_data
import pandas as pd
from tqdm import tqdm

LABEL_FRACTIONS = (0.01, 0.10, 0.50, 1.00)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune shared SVM settings on row B and evaluate feature caches."
    )

    parser.add_argument("--train-pkl", type=Path, default=Path("datasets/eurosat_train_subsets.pkl"))
    parser.add_argument("--test-pkl", type=Path, default=Path("datasets/eurosat_test_subset.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-classes", type=int, required=True, help="Number of classes in the dataset.")
    parser.add_argument("--min-neighbors", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--features-dir", dest="feature_dir", type=Path, required=True)

    return parser.parse_args()


def evaluate(training_pipeline, train_subsets, x_test, y_test, feature_dir):
    results = []

    for (fraction, draw), sample_indices in tqdm(
        sorted(train_subsets.items()), desc="Evaluating label fractions"
    ):
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

    n_neighbors = min(args.min_neighbors, args.n_classes - 1)
    print(f"Using {n_neighbors} neighbors for KNN evaluation (min_n_neighbors={args.min_neighbors}, n_classes={args.n_classes})")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=args.n_jobs)

    training_pipeline = Pipeline([
        ("normalize", Normalizer(norm="l2")),
        ("scale", StandardScaler()),
        ("classifier", knn),
    ])

    train_subsets, train_indices, test_indices = load_indices(
        args.train_pkl,
        args.test_pkl,
    )
    x_test, y_test = get_data(test_indices, args.feature_dir)

    results = evaluate(training_pipeline, train_subsets, x_test, y_test, args.feature_dir)
    encoder_name = args.feature_dir.name
    results.insert(0, "encoder", encoder_name)
    summary = (
        results.groupby("fraction")["accuracy"]
        .agg(mean_accuracy="mean", std_accuracy="std", draws="count")
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir
        / f"knn_evaluation_{encoder_name}.csv"
    )
    results.to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")
    summary_path = (
        args.output_dir
        / f"knn_evaluation_{encoder_name}_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    print(f"Label-efficiency summary saved to {summary_path}")


if __name__ == "__main__":
    main()
