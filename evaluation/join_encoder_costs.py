import argparse
import csv
import re
from pathlib import Path


COST_COLUMN_MAP = {
    "num_parameters": "num_parameters",
    "flops_per_image": "flops_per_image",
    "gflops_per_image": "gflops_per_image",
    "inference_latency_ms": "inference_latency_ms",
    "inference_latency_std_ms": "latency_std_ms",
    "throughput_images_per_second": "throughput_images_per_second",
    "batch_size": "latency_batch_size",
    "latency_input_size": "input_size",
    "latency_precision": "precision",
    "latency_warmup_iterations": "warmup_iterations",
    "latency_timed_iterations": "timed_iterations",
    "device": "device",
    "gpu_model": "gpu_model",
    "gpu_memory_mib": "gpu_memory_mib",
    "unsupported_flop_operators": "unsupported_flop_operators",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine KNN and SVM results with encoder-cost measurements."
    )
    parser.add_argument(
        "--costs-csv",
        type=Path,
        default=Path("outputs/runs/probes/encoder_costs.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/runs/probes"),
        help="Root containing knn_evaluation and svm_evaluation results.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/runs/summarized_results.csv"),
        help="Write the combined results CSV here.",
    )
    return parser.parse_args()


def load_costs(costs_path):
    with costs_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"encoder", *COST_COLUMN_MAP.values()} <= set(reader.fieldnames):
            raise ValueError(
                f"Cost CSV {costs_path} is missing required measurement columns."
            )

        costs = {}
        for line_number, row in enumerate(reader, start=2):
            encoder = row["encoder"].strip()
            if encoder not in {"small_cnn", "larger_cnn", "dino_vits16"}:
                raise ValueError(
                    f"Unexpected encoder {encoder!r} in {costs_path}:{line_number}."
                )
            if encoder in costs:
                raise ValueError(
                    f"Duplicate cost row for {encoder!r} in {costs_path}."
                )
            costs[encoder] = row

    missing = {"small_cnn", "larger_cnn", "dino_vits16"} - costs.keys()
    if missing:
        raise ValueError(
            f"Cost CSV {costs_path} has no measurements for: "
            f"{', '.join(sorted(missing))}."
        )
    return costs


def resolve_cost_encoder(row, source_path):
    candidates = [
        row.get("student_cnn", ""),
        row.get("approach", ""),
        row.get("aggregate_encoder", ""),
        row.get("encoder", ""),
        *source_path.parts,
    ]
    normalized = [str(candidate).lower() for candidate in candidates if candidate]
    combined = " ".join(normalized)

    if any(
        value in {"teacher_encoding", "dino", "dino_vits16"}
        or "dino_vits16" in value
        for value in normalized
    ):
        return "dino_vits16"

    student_size = str(row.get("student_cnn", "")).lower()
    if student_size in {"small", "small_cnn"} or "small_cnn" in combined:
        return "small_cnn"
    if (
        student_size in {"large", "larger", "large_cnn", "larger_cnn"}
        or "larger_cnn" in combined
        or "large_cnn" in combined
        or re.search(r"(^|[^a-z])large([^a-z]|$)", combined)
    ):
        return "larger_cnn"

    raise ValueError(
        f"Cannot determine encoder size for row in {source_path}: "
        f"encoder={row.get('encoder')!r}, "
        f"aggregate_encoder={row.get('aggregate_encoder')!r}, "
        f"student_cnn={row.get('student_cnn')!r}."
    )


def join_result_file(source_path, results_root, costs):
    with source_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Result CSV has no header: {source_path}")
        rows = list(reader)

    for row in rows:
        cost_encoder = resolve_cost_encoder(row, source_path.relative_to(results_root))
        measured_cost = costs[cost_encoder]
        row["cost_encoder"] = cost_encoder
        for output_column, cost_column in COST_COLUMN_MAP.items():
            row[output_column] = measured_cost[cost_column]

    for row in rows:
        row["source_csv"] = str(source_path.relative_to(results_root))
    fieldnames = list(reader.fieldnames)
    for column in ("source_csv", "cost_encoder", *COST_COLUMN_MAP):
        if column not in fieldnames:
            fieldnames.append(column)
    return rows, fieldnames


def find_result_files(results_root):
    probe_names = ("knn_evaluation", "svm_evaluation")
    if results_root.name in probe_names:
        probe_roots = [results_root]
    else:
        probe_roots = [results_root / name for name in probe_names]

    result_files = []
    for probe_root in probe_roots:
        if not probe_root.is_dir():
            continue
        for filename in ("results.csv", "summary.csv"):
            result_files.extend(probe_root.rglob(filename))
    return sorted(set(result_files))


def main():
    args = parse_args()
    if not args.costs_csv.is_file():
        raise FileNotFoundError(f"Encoder cost CSV not found: {args.costs_csv}")
    if not args.results_dir.is_dir():
        raise FileNotFoundError(f"Probe results directory not found: {args.results_dir}")

    costs = load_costs(args.costs_csv)
    result_files = find_result_files(args.results_dir)
    if not result_files:
        raise FileNotFoundError(
            f"No results.csv or summary.csv files found under "
            f"{args.results_dir}/knn_evaluation or {args.results_dir}/svm_evaluation."
        )

    combined_rows = []
    fieldnames = []
    for source_path in result_files:
        rows, source_fieldnames = join_result_file(
            source_path,
            args.results_dir,
            costs,
        )
        combined_rows.extend(rows)
        for column in source_fieldnames:
            if column not in fieldnames:
                fieldnames.append(column)
        print(f"Joined {len(rows)} rows from {source_path}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    print(
        f"Wrote {len(combined_rows)} combined probe rows from "
        f"{len(result_files)} CSV files to {args.output_csv}."
    )


if __name__ == "__main__":
    main()
