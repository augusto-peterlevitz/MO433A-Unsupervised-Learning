import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm

from data.dataloaders import get_data, load_indices
from models.utils import load_encoder_cost


def load_feature_metadata(feature_dir):
    metadata_path = feature_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as file:
            return json.load(file)
    return {}


def get_encoder_name(feature_dir, metadata):
    if metadata.get("encoder"):
        return str(metadata["encoder"])
    if feature_dir.name.startswith("seed") and feature_dir.name[4:].isdigit():
        return feature_dir.parent.name
    return feature_dir.name


def get_aggregate_encoder_name(encoder_name):
    return re.sub(r"_seed\d+$", "", encoder_name)


def get_cache_seed(feature_dir, metadata):
    if metadata.get("seed") is not None:
        return metadata["seed"]
    for part in reversed(feature_dir.parts):
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    return None


def get_result_directory(output_dir, feature_dir, metadata, aggregate_encoder):
    parts = feature_dir.parts
    if "random_baseline" in parts:
        baseline_index = parts.index("random_baseline")
        if baseline_index + 1 >= len(parts):
            raise ValueError(
                f"Expected a student-size directory after random_baseline in {feature_dir}."
            )
        return (
            output_dir
            / "svm_evaluation"
            / "random_baseline"
            / parts[baseline_index + 1]
        )

    row = metadata.get("row")
    if row:
        return (
            output_dir
            / "svm_evaluation"
            / str(row).lower()
            / aggregate_encoder
        )
    return output_dir / "svm_evaluation" / aggregate_encoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more feature caches using fixed SVM parameters."
    )
    parser.add_argument(
        "--train-pkl",
        type=Path,
        default=Path("datasets/eurosat_train_subsets.pkl"),
    )
    parser.add_argument(
        "--test-pkl",
        type=Path,
        default=Path("datasets/eurosat_test_subset.pkl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument(
        "--features-dir",
        dest="feature_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more seed-specific or single feature caches.",
    )
    return parser.parse_args()


def load_parameters(params_path):
    with params_path.open(encoding="utf-8") as file:
        return json.load(file)


def create_pipeline(svm_params):
    if svm_params["kernel"] == "linear":
        classifier = LinearSVC(
            C=svm_params["C"],
            dual="auto",
            max_iter=10_000,
        )
    elif svm_params["kernel"] == "rbf":
        classifier = SVC(
            C=svm_params["C"],
            gamma=svm_params.get("gamma") or "scale",
            kernel="rbf",
        )
    else:
        raise ValueError(f"Unsupported kernel: {svm_params['kernel']}")

    return Pipeline([
        ("normalize", Normalizer(norm="l2")),
        ("scale", StandardScaler()),
        ("classifier", classifier),
    ])


def evaluate(train_subsets, x_test, y_test, feature_dir, svm_params):
    results = []
    for (fraction, draw), sample_indices in tqdm(
        sorted(train_subsets.items()),
        desc=f"Evaluating {feature_dir.name}",
    ):
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        x_train, y_train = get_data(sample_indices, feature_dir)
        training_pipeline = create_pipeline(svm_params)
        training_pipeline.fit(x_train, y_train)
        results.append({
            "fraction": float(fraction),
            "draw": int(draw),
            "accuracy": float(training_pipeline.score(x_test, y_test)),
        })
    return pd.DataFrame(results)


def main():
    args = parse_args()

    svm_params = load_parameters(args.params)
    print(f"Loaded SVM parameters: {svm_params}")
    if svm_params.get("kernel") not in {"linear", "rbf"}:
        raise ValueError(f"Unsupported SVM kernel: {svm_params.get('kernel')}")

    train_subsets, train_indices, test_indices = load_indices(
        args.train_pkl,
        args.test_pkl,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_frames = []
    aggregate_output_dirs = {}
    seen_seeds = {}

    for feature_dir in args.feature_dirs:
        if not feature_dir.is_dir():
            raise FileNotFoundError(f"Feature cache not found: {feature_dir}")

        metadata = load_feature_metadata(feature_dir)
        encoder_name = get_encoder_name(feature_dir, metadata)
        aggregate_encoder = get_aggregate_encoder_name(encoder_name)
        random_seed = get_cache_seed(feature_dir, metadata)
        if random_seed is not None:
            seed_key = (aggregate_encoder, str(random_seed))
            if seed_key in seen_seeds:
                raise ValueError(
                    f"Duplicate cache seed {random_seed} for {aggregate_encoder}: "
                    f"{seen_seeds[seed_key]} and {feature_dir}"
                )
            seen_seeds[seed_key] = feature_dir

        x_test, y_test = get_data(test_indices, feature_dir)
        results = evaluate(
            train_subsets,
            x_test,
            y_test,
            feature_dir,
            svm_params,
        )
        cost = load_encoder_cost(feature_dir)
        results.insert(0, "probe", f"{svm_params['kernel']}_svm")
        results.insert(0, "aggregate_encoder", aggregate_encoder)
        results.insert(0, "encoder", encoder_name)
        results["random_seed"] = random_seed
        for column, value in cost.items():
            results[column] = value

        result_root = get_result_directory(
            args.output_dir,
            feature_dir,
            metadata,
            aggregate_encoder,
        )
        seed_dir = result_root / f"seed{random_seed}" if random_seed is not None else result_root
        result_dir = seed_dir / svm_params["kernel"]
        result_dir.mkdir(parents=True, exist_ok=True)
        aggregate_output_dirs[aggregate_encoder] = result_root

        individual_summary = (
            results[
                (results["fraction"] != 1.0) | (results["draw"] == 0)
            ].groupby(["encoder", "probe", "fraction"], dropna=False)
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
        results.to_csv(result_dir / "results.csv", index=False)
        individual_summary.to_csv(result_dir / "summary.csv", index=False)
        print(f"Saved per-seed SVM results to {result_dir}")
        combined_frames.append(results)

    combined = pd.concat(combined_frames, ignore_index=True)
    for aggregate_encoder in combined["aggregate_encoder"].unique():
        encoder_results = combined[
            combined["aggregate_encoder"] == aggregate_encoder
        ].copy()

        # The 100% label subset is stored three times by subset generation.
        # Keep one identical draw per seed so it does not artificially weight
        # the across-seed spread.
        full_fraction = encoder_results["fraction"] == 1.0
        encoder_results = pd.concat([
            encoder_results[~full_fraction],
            encoder_results[full_fraction & (encoder_results["draw"] == 0)],
        ], ignore_index=True)

        combined_summary = (
            encoder_results.groupby(
                ["aggregate_encoder", "probe", "fraction"],
                dropna=False,
            )
            .agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                observations=("accuracy", "count"),
                random_seeds=("random_seed", "nunique"),
                num_parameters=("num_parameters", "first"),
                inference_latency_ms=("inference_latency_ms", "first"),
                device=("device", "first"),
                batch_size=("batch_size", "first"),
            )
            .reset_index()
            .rename(columns={"aggregate_encoder": "encoder"})
        )
        aggregate_dir = (
            aggregate_output_dirs[aggregate_encoder]
            / "aggregate"
            / svm_params["kernel"]
        )
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        encoder_results.to_csv(aggregate_dir / "results.csv", index=False)
        combined_summary.to_csv(aggregate_dir / "summary.csv", index=False)
        print(f"Saved combined seed/draw SVM results to {aggregate_dir}")


if __name__ == "__main__":
    main()
