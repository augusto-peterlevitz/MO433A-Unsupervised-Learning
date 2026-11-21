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
from models.utils import load_encoder_cost


# def load_encoder_cost(feature_dir, args):
#     metadata = {}
#     metadata_path = feature_dir / "metadata.json"
#     if metadata_path.exists():
#         with metadata_path.open(encoding="utf-8") as file:
#             metadata = json.load(file)

#     num_parameters = args.num_parameters
#     if num_parameters is None:
#         num_parameters = metadata.get("num_parameters")

#     latency_ms = args.latency_ms
#     if latency_ms is None:
#         latency_ms = metadata.get("latency_ms")

#     return {
#         "num_parameters": num_parameters,
#         "inference_latency_ms": latency_ms,
#         "device": args.device or metadata.get("device"),
#         "batch_size": args.batch_size or metadata.get("batch_size") or 1,
#     }


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
    cost = load_encoder_cost(args.feature_dir)
    results.insert(0, "probe", "knn")
    results.insert(0, "encoder", encoder_name)
    for column, value in cost.items():
        results[column] = value

    summary = (
        results.groupby(["encoder", "probe", "fraction"], dropna=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            draws=("accuracy", "count"),
            num_parameters=("num_parameters", "first"),
            inference_latency_ms=("inference_latency_ms", "first"),
            device=("device", "first"),
            batch_size=("batch_size", "first"),
        )
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
