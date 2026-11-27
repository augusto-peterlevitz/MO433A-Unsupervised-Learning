import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


SEED_SUFFIX = re.compile(r"_seed(\d+)$")
REQUIRED_COLUMNS = {
    "encoder",
    "probe",
    "fraction",
    "mean_accuracy",
    "num_parameters",
    "inference_latency_ms",
    "inference_latency_std_ms",
    "device",
    "gpu_model",
    "batch_size",
    "latency_input_size",
    "latency_precision",
    "latency_warmup_iterations",
    "latency_timed_iterations",
    "source_csv",
}
FIXED_LATENCY_FIELDS = (
    "device",
    "gpu_model",
    "batch_size",
    "latency_warmup_iterations",
    "latency_timed_iterations",
)
ENCODER_METADATA_FIELDS = (
    "num_parameters",
    "inference_latency_ms",
    "inference_latency_std_ms",
    "device",
    "gpu_model",
    "batch_size",
    "latency_input_size",
    "latency_precision",
    "latency_warmup_iterations",
    "latency_timed_iterations",
)
OUTPUT_FIELDS = (
    "encoder",
    "label_fraction",
    "probe",
    "num_parameters",
    "inference_latency_ms",
    "inference_latency_std_ms",
    "latency_device",
    "gpu_model",
    "latency_batch_size",
    "latency_input_size",
    "latency_precision",
    "latency_warmup_iterations",
    "latency_timed_iterations",
    "accuracy",
    "accuracy_mean",
    "accuracy_std_over_seeds",
    "num_seeds",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-seed probe summaries into a report-ready final "
            "results table."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/runs/probes/full/summarized_results.csv"),
        help="Joined probe summary CSV containing per-seed and aggregate rows.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV path (defaults to final_results.csv beside the input).",
    )
    return parser.parse_args()


def parse_number(value, column, source, line_number, number_type=float):
    try:
        number = number_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {column} value {value!r} in {source}:{line_number}."
        ) from error
    if isinstance(number, float) and not math.isfinite(number):
        raise ValueError(
            f"Non-finite {column} value {value!r} in {source}:{line_number}."
        )
    return number


def load_seed_summaries(input_path):
    with input_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV {input_path} is missing columns: {sorted(missing)}"
            )

        grouped = defaultdict(list)
        fixed_latency_values = {}
        for line_number, row in enumerate(reader, start=2):
            if not row["mean_accuracy"].strip():
                continue

            source = row["source_csv"].replace("\\", "/")
            source_parts = Path(source).parts
            if Path(source).name != "summary.csv" or "aggregate" in source_parts:
                continue

            encoder = row["encoder"].strip()
            seed_match = SEED_SUFFIX.search(encoder)
            if seed_match:
                seed = int(seed_match.group(1))
                encoder = encoder[:seed_match.start()]
            else:
                seed = None

            metadata = {
                "num_parameters": parse_number(
                    row["num_parameters"], "num_parameters", source,
                    line_number, int,
                ),
                "inference_latency_ms": parse_number(
                    row["inference_latency_ms"], "inference_latency_ms",
                    source, line_number,
                ),
                "inference_latency_std_ms": parse_number(
                    row["inference_latency_std_ms"], "inference_latency_std_ms",
                    source, line_number,
                ),
                "device": row["device"].strip(),
                "gpu_model": row["gpu_model"].strip(),
                "batch_size": parse_number(
                    row["batch_size"], "batch_size", source, line_number, int,
                ),
                "latency_input_size": parse_number(
                    row["latency_input_size"], "latency_input_size", source,
                    line_number, int,
                ),
                "latency_precision": row["latency_precision"].strip(),
                "latency_warmup_iterations": parse_number(
                    row["latency_warmup_iterations"],
                    "latency_warmup_iterations",
                    source,
                    line_number,
                    int,
                ),
                "latency_timed_iterations": parse_number(
                    row["latency_timed_iterations"],
                    "latency_timed_iterations",
                    source,
                    line_number,
                    int,
                ),
            }
            if not metadata["device"] or not metadata["gpu_model"]:
                raise ValueError(
                    f"Missing latency device metadata in {source}:{line_number}."
                )
            if metadata["batch_size"] < 1:
                raise ValueError(
                    f"Latency batch size must be positive in {source}:{line_number}."
                )
            if (
                metadata["latency_warmup_iterations"] < 0
                or metadata["latency_timed_iterations"] < 2
            ):
                raise ValueError(
                    f"Invalid latency iteration counts in {source}:{line_number}."
                )

            for field in FIXED_LATENCY_FIELDS:
                value = metadata[field]
                if field in fixed_latency_values and fixed_latency_values[field] != value:
                    raise ValueError(
                        f"Latency {field} must be fixed across all encoders; "
                        f"found {fixed_latency_values[field]!r} and {value!r}."
                    )
                fixed_latency_values[field] = value

            key = (
                encoder,
                parse_number(row["fraction"], "fraction", source, line_number),
                row["probe"].strip(),
            )
            accuracy = parse_number(
                row["mean_accuracy"], "mean_accuracy", source, line_number,
            )
            grouped[key].append({
                "seed": seed,
                "accuracy": accuracy,
                "metadata": metadata,
                "source": source,
            })

    if not grouped:
        raise ValueError(
            f"No per-seed summary rows found in {input_path}; "
            "aggregate-only rows cannot be used to calculate standard "
            "deviation over seeds."
        )
    return grouped


def ensure_consistent_metadata(encoder, entries):
    reference = entries[0]["metadata"]
    for entry in entries[1:]:
        for field in ENCODER_METADATA_FIELDS:
            if entry["metadata"][field] != reference[field]:
                raise ValueError(
                    f"Inconsistent {field} measurements for encoder "
                    f"{encoder!r}: {reference[field]!r} vs "
                    f"{entry['metadata'][field]!r}."
                )
    return reference


def build_final_rows(grouped):
    rows = []
    for (encoder, fraction, probe), entries in sorted(grouped.items()):
        seeds = [entry["seed"] for entry in entries if entry["seed"] is not None]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                f"Duplicate per-seed summaries for {encoder}, "
                f"{fraction:.6g}, {probe}."
            )
        if seeds and len(seeds) != len(entries):
            raise ValueError(
                f"Seeded and unseeded summaries are mixed for {encoder}, "
                f"{fraction:.6g}, {probe}."
            )

        metadata = ensure_consistent_metadata(encoder, entries)
        values = [entry["accuracy"] for entry in entries]
        accuracy_mean = statistics.mean(values)
        accuracy_std = statistics.stdev(values) if len(seeds) > 1 else None
        accuracy_text = (
            f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
            if accuracy_std is not None
            else f"{accuracy_mean:.4f}"
        )
        rows.append({
            "encoder": encoder,
            "label_fraction": fraction,
            "probe": probe,
            "num_parameters": metadata["num_parameters"],
            "inference_latency_ms": metadata["inference_latency_ms"],
            "inference_latency_std_ms": metadata["inference_latency_std_ms"],
            "latency_device": metadata["device"],
            "gpu_model": metadata["gpu_model"],
            "latency_batch_size": metadata["batch_size"],
            "latency_input_size": metadata["latency_input_size"],
            "latency_precision": metadata["latency_precision"],
            "latency_warmup_iterations": metadata["latency_warmup_iterations"],
            "latency_timed_iterations": metadata["latency_timed_iterations"],
            "accuracy": accuracy_text,
            "accuracy_mean": accuracy_mean,
            "accuracy_std_over_seeds": (
                accuracy_std if accuracy_std is not None else ""
            ),
            "num_seeds": len(seeds),
        })
    return rows


def main():
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    output_path = args.output_csv or args.input_csv.with_name("final_results.csv")
    if args.input_csv.resolve() == output_path.resolve():
        raise ValueError("Input and output CSV paths must be different.")

    rows = build_final_rows(load_seed_summaries(args.input_csv))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} final result rows to {output_path}")


if __name__ == "__main__":
    main()
