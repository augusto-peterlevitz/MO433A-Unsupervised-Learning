import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Attach measured encoder FLOPs and latency to feature metadata "
            "and existing probe CSVs."
        )
    )
    parser.add_argument("--costs-csv", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    return parser.parse_args()


def load_costs(path):
    costs = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            name = row["encoder"]
            if name in costs:
                raise ValueError(f"Duplicate encoder measurements for {name}.")
            costs[name] = row
    expected = {"small_cnn", "larger_cnn", "dino_vits16"}
    if set(costs) != expected:
        raise ValueError(
            f"Expected measurements for {sorted(expected)}, got {sorted(costs)}."
        )
    return costs


def get_encoder_key(metadata, feature_dir):
    if metadata.get("approach") == "teacher_encoding":
        return "dino_vits16"
    student_cnn = metadata.get("student_cnn")
    if student_cnn == "small":
        return "small_cnn"
    if student_cnn == "large":
        return "larger_cnn"
    raise ValueError(f"Cannot identify encoder architecture in {feature_dir}.")


def as_float(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "") else None


def attach_to_feature_metadata(features_dir, costs):
    metadata_paths = sorted(features_dir.rglob("metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No feature metadata found under {features_dir}.")

    for metadata_path in metadata_paths:
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        encoder = get_encoder_key(metadata, metadata_path.parent)
        cost = costs[encoder]
        if int(metadata["num_parameters"]) != int(cost["num_parameters"]):
            raise ValueError(
                f"Parameter count mismatch for {metadata_path}: "
                f"{metadata['num_parameters']} vs {cost['num_parameters']}."
            )
        metadata.update({
            "flops_per_image": int(cost["flops_per_image"]),
            "gflops_per_image": float(cost["gflops_per_image"]),
            "inference_latency_ms": float(cost["inference_latency_ms"]),
            "latency_ms": float(cost["inference_latency_ms"]),
            "latency_std_ms": float(cost["latency_std_ms"]),
            "throughput_images_per_second": float(
                cost["throughput_images_per_second"]
            ),
            "latency_batch_size": int(cost["latency_batch_size"]),
            "latency_input_size": int(cost["input_size"]),
            "latency_precision": cost["precision"],
            "latency_device": cost["device"],
            "gpu_model": cost["gpu_model"],
            "gpu_memory_mib": as_float(cost, "gpu_memory_mib"),
            "latency_warmup_iterations": int(cost["warmup_iterations"]),
            "latency_iterations": int(cost["timed_iterations"]),
            "unsupported_flop_operators": cost["unsupported_flop_operators"],
        })
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")

    return len(metadata_paths)


def attach_to_evaluation_csvs(runs_dir, costs):
    by_parameter_count = {
        int(cost["num_parameters"]): cost for cost in costs.values()
    }
    updated = 0
    for evaluation_name in ("svm_evaluation", "knn_evaluation"):
        evaluation_dir = runs_dir / evaluation_name
        if not evaluation_dir.is_dir():
            continue
        for csv_path in sorted(evaluation_dir.rglob("*.csv")):
            frame = pd.read_csv(csv_path)
            if "num_parameters" not in frame.columns:
                continue
            parameter_counts = pd.to_numeric(
                frame["num_parameters"], errors="coerce"
            )
            if parameter_counts.isna().any():
                raise ValueError(
                    f"Missing parameter counts in evaluation file {csv_path}."
                )
            unknown = sorted(
                set(parameter_counts.astype(int)) - set(by_parameter_count)
            )
            if unknown:
                raise ValueError(
                    f"Unrecognized parameter counts in {csv_path}: {unknown}."
                )
            matched = parameter_counts.astype(int).map(by_parameter_count)

            if "batch_size" in frame.columns:
                frame["feature_batch_size"] = frame["batch_size"]
            else:
                frame["feature_batch_size"] = pd.NA
            frame["flops_per_image"] = matched.map(
                lambda cost: int(cost["flops_per_image"])
            )
            frame["gflops_per_image"] = matched.map(
                lambda cost: float(cost["gflops_per_image"])
            )
            frame["inference_latency_ms"] = matched.map(
                lambda cost: float(cost["inference_latency_ms"])
            )
            frame["inference_latency_std_ms"] = matched.map(
                lambda cost: float(cost["latency_std_ms"])
            )
            frame["throughput_images_per_second"] = matched.map(
                lambda cost: float(cost["throughput_images_per_second"])
            )
            frame["latency_batch_size"] = matched.map(
                lambda cost: int(cost["latency_batch_size"])
            )
            frame["latency_input_size"] = matched.map(
                lambda cost: int(cost["input_size"])
            )
            frame["latency_precision"] = matched.map(
                lambda cost: cost["precision"]
            )
            frame["latency_device"] = matched.map(lambda cost: cost["device"])
            frame["gpu_model"] = matched.map(lambda cost: cost["gpu_model"])
            frame["gpu_memory_mib"] = matched.map(
                lambda cost: as_float(cost, "gpu_memory_mib")
            )
            frame["unsupported_flop_operators"] = matched.map(
                lambda cost: cost["unsupported_flop_operators"]
            )
            frame["device"] = frame["latency_device"]
            frame["batch_size"] = frame["latency_batch_size"]
            frame.to_csv(csv_path, index=False)
            updated += 1
    return updated


def main():
    args = parse_args()
    costs = load_costs(args.costs_csv)
    metadata_count = attach_to_feature_metadata(args.features_dir, costs)
    csv_count = attach_to_evaluation_csvs(args.runs_dir, costs)
    print(f"Updated measured costs in {metadata_count} feature metadata files.")
    print(f"Updated measured costs in {csv_count} probe CSV files.")


if __name__ == "__main__":
    main()
