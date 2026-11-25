import argparse
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from data.dataloaders import load_indices, get_data
import pandas as pd
from tqdm import tqdm
from models.utils import load_encoder_cost
from evaluation.utils import (
    get_aggregate_encoder_name,
    get_cache_seed,
    get_encoder_name,
    get_result_directory,
    load_feature_metadata,
    resolve_feature_dirs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate feature caches with cosine k-NN probes."
    )

    parser.add_argument("--train-pkl", type=Path, default=Path("datasets/eurosat_train_subsets.pkl"))
    parser.add_argument("--test-pkl", type=Path, default=Path("datasets/eurosat_test_subset.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/probes"))
    parser.add_argument("--n-classes", type=int, help="Deprecated; retained for backward-compatible commands.")
    parser.add_argument("--max-neighbors", type=int, default=20, help="Upper bound for k (default: 20).")
    parser.add_argument("--min-neighbors", type=int, help="Deprecated alias for --max-neighbors.")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Feature-cache model directory; append seed<seed> for seeded approaches.",
    )
    parser.add_argument(
        "--approach",
        choices=("random_baseline", "vicreg", "distillation", "teacher_encoding"),
        required=True,
        help="One model approach to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Seed-specific feature caches under --feature-dir (not used for teacher_encoding).",
    )
    parser.add_argument(
        "--student-cnn",
        choices=("small", "large"),
        help="Student size (required for random_baseline, vicreg, and distillation).",
    )
    parser.add_argument(
        "--projector-type",
        choices=("mlp", "linear"),
        help="Distillation projector type (required only for distillation).",
    )
    return parser.parse_args()


def evaluate(train_subsets, x_test, y_test, feature_dir, max_neighbors, n_jobs):
    results = []

    for (fraction, draw), sample_indices in tqdm(
        sorted(train_subsets.items()), desc="Evaluating label fractions"
    ):
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        x_train, y_train = get_data(sample_indices, feature_dir)

        _, class_counts = np.unique(y_train, return_counts=True)
        positive_counts = class_counts[class_counts > 0]
        if positive_counts.size == 0:
            raise ValueError("The labeled subset contains no samples.")

        n_neighbors = min(max_neighbors, int(positive_counts.min()))
        n_neighbors = max(1, n_neighbors)
        n_neighbors = min(n_neighbors, len(y_train))

        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="cosine",
            n_jobs=n_jobs,
        )
        training_pipeline = Pipeline([
            ("normalize", Normalizer(norm="l2")),
            ("scale", StandardScaler()),
            ("classifier", knn),
        ])

        training_pipeline.fit(x_train, y_train)
        accuracy = training_pipeline.score(x_test, y_test)
        results.append({
            "fraction": float(fraction),
            "draw": int(draw),
            "n_neighbors": int(n_neighbors),
            "accuracy": float(accuracy),
        })

    return pd.DataFrame(results)


def main():
    args = parse_args()

    if args.max_neighbors < 1:
        raise ValueError("--max-neighbors must be positive.")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive.")
    if args.min_neighbors is not None:
        if args.min_neighbors < 1:
            raise ValueError("--min-neighbors must be positive.")
        args.max_neighbors = args.min_neighbors

    train_subsets, train_indices, test_indices = load_indices(
        args.train_pkl,
        args.test_pkl,
    )

    feature_dirs = resolve_feature_dirs(
        args.feature_dir,
        args.approach,
        args.seeds,
        args.student_cnn,
        args.projector_type,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_frames = []
    aggregate_output_dirs = {}

    for feature_dir in feature_dirs:
        if not feature_dir.exists():
            raise FileNotFoundError(f"Feature cache not found: {feature_dir}")

        metadata = load_feature_metadata(feature_dir)
        cache_approach = metadata.get("approach")
        if cache_approach is not None and cache_approach != args.approach:
            raise ValueError(
                f"Expected approach {args.approach}, but {feature_dir} has "
                f"approach {cache_approach}."
            )
        if args.student_cnn and metadata.get("student_cnn") not in {None, args.student_cnn}:
            raise ValueError(
                f"Expected student-cnn {args.student_cnn}, but {feature_dir} has "
                f"student-cnn {metadata.get('student_cnn')}."
            )
        if args.projector_type:
            cache_projector = metadata.get("projector_type", "mlp")
            if cache_projector != args.projector_type:
                raise ValueError(
                    f"Expected projector {args.projector_type}, but {feature_dir} has "
                    f"projector {cache_projector}."
                )

        x_test, y_test = get_data(test_indices, feature_dir)
        results = evaluate(
            train_subsets,
            x_test,
            y_test,
            feature_dir,
            max_neighbors=args.max_neighbors,
            n_jobs=args.n_jobs,
        )

        encoder_name = get_encoder_name(feature_dir)
        aggregate_encoder_name = get_aggregate_encoder_name(feature_dir)
        random_seed = metadata.get("seed")
        cost = load_encoder_cost(feature_dir)
        result_root = get_result_directory(
            args.output_dir,
            feature_dir,
            metadata,
            encoder_name,
            "knn_evaluation",
        )
        result_dir = (
            result_root / f"seed{random_seed}"
            if random_seed is not None
            else result_root
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        aggregate_output_dirs[aggregate_encoder_name] = result_root

        results.insert(0, "probe", "knn")
        results.insert(0, "aggregate_encoder", aggregate_encoder_name)
        results.insert(0, "encoder", encoder_name)
        results["random_seed"] = random_seed
        for column, value in cost.items():
            results[column] = value

        individual_summary = (
            results.groupby(["encoder", "probe", "fraction"], dropna=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                draws=("accuracy", "count"),
                mean_n_neighbors=("n_neighbors", "mean"),
                num_parameters=("num_parameters", "first"),
                flops_per_image=("flops_per_image", "first"),
                gflops_per_image=("gflops_per_image", "first"),
                inference_latency_ms=("inference_latency_ms", "first"),
                inference_latency_std_ms=("inference_latency_std_ms", "first"),
                throughput_images_per_second=(
                    "throughput_images_per_second",
                    "first",
                ),
                device=("device", "first"),
                gpu_model=("gpu_model", "first"),
                batch_size=("batch_size", "first"),
                feature_batch_size=("feature_batch_size", "first"),
                latency_input_size=("latency_input_size", "first"),
                latency_precision=("latency_precision", "first"),
            )
            .reset_index()
        )

        output_path = result_dir / "results.csv"
        results.to_csv(output_path, index=False)
        print(f"Evaluation results saved to {output_path}")

        summary_path = result_dir / "summary.csv"
        individual_summary.to_csv(summary_path, index=False)
        print(f"Label-efficiency summary saved to {summary_path}")

        combined_frames.append(results)

    combined = pd.concat(combined_frames, ignore_index=True)
    aggregate_group_names = combined["aggregate_encoder"].unique()

    for aggregate_encoder_name in aggregate_group_names:
        encoder_results = combined[
            combined["aggregate_encoder"] == aggregate_encoder_name
        ].copy()

        combined_summary = (
            encoder_results.groupby(["aggregate_encoder", "probe", "fraction"], dropna=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                observations=("accuracy", "count"),
                random_seeds=("random_seed", "nunique"),
                mean_n_neighbors=("n_neighbors", "mean"),
                num_parameters=("num_parameters", "first"),
                flops_per_image=("flops_per_image", "first"),
                gflops_per_image=("gflops_per_image", "first"),
                inference_latency_ms=("inference_latency_ms", "first"),
                inference_latency_std_ms=("inference_latency_std_ms", "first"),
                throughput_images_per_second=(
                    "throughput_images_per_second",
                    "first",
                ),
                device=("device", "first"),
                gpu_model=("gpu_model", "first"),
                batch_size=("batch_size", "first"),
                feature_batch_size=("feature_batch_size", "first"),
                latency_input_size=("latency_input_size", "first"),
                latency_precision=("latency_precision", "first"),
            )
            .reset_index()
            .rename(columns={"aggregate_encoder": "encoder"})
        )

        aggregate_dir = aggregate_output_dirs[aggregate_encoder_name] / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        combined_path = aggregate_dir / "results.csv"
        encoder_results.to_csv(combined_path, index=False)
        print(f"Combined seed/draw results saved to {combined_path}")

        combined_summary_path = aggregate_dir / "summary.csv"
        combined_summary.to_csv(combined_summary_path, index=False)
        print(f"Combined seed/draw summary saved to {combined_summary_path}")


if __name__ == "__main__":
    main()
