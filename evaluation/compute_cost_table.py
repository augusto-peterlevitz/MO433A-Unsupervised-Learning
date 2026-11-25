import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a per-run and aggregate GPU-hour cost table."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/checkpoints"),
    )
    parser.add_argument(
        "--teacher-feature-dir",
        type=Path,
        default=Path("outputs/features/dino_vits16"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/runs/compute_costs.csv"),
    )
    parser.add_argument(
        "--small-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument(
        "--large-seeds",
        type=int,
        nargs="+",
        default=[42, 43],
    )
    return parser.parse_args()


def finite_gpu_hours(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def normalize_student_size(size):
    size = str(size).lower()
    if size in {"small", "small_cnn"}:
        return "small"
    if size in {"large", "larger", "large_cnn", "larger_cnn"}:
        return "larger"
    return size


def base_row(row, approach, student_size, projector_type, seed, scope):
    row.update({
        "scope": scope,
        "approach": approach,
        "student_size": student_size,
        "projector_type": projector_type,
        "seed": seed,
        "training_runs": 0,
        "training_gpu_hours": 0.0,
        "teacher_forward_gpu_hours": 0.0,
        "published_pretraining_gpu_hours": None,
        "published_pretraining_reference": "",
        "local_gpu_hours": 0.0,
        "total_gpu_hours": 0.0,
        "cost_status": "measured",
    })


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def collect_vicreg_rows(checkpoint_root):
    rows = []
    for metadata_path in sorted(checkpoint_root.glob("vicreg/*/seed*/training_metadata.json")):
        metadata = load_json(metadata_path)
        size = normalize_student_size(
            metadata.get("student_cnn", metadata_path.parent.parent.name)
        )
        row = {}
        base_row(row, "vicreg", size, "none", metadata.get("seed"), "seed")
        gpu_hours = finite_gpu_hours(metadata.get("training_gpu_hours"))
        if gpu_hours is None:
            row["cost_status"] = "missing_training_metadata"
        else:
            row["training_gpu_hours"] = gpu_hours
            row["local_gpu_hours"] = gpu_hours
            row["total_gpu_hours"] = gpu_hours
        row["training_runs"] = 1
        rows.append(row)

    for config_path in sorted(checkpoint_root.glob("vicreg/*/seed*/vicreg.json")):
        metadata_path = config_path.parent / "training_metadata.json"
        if metadata_path.is_file():
            continue
        metadata = load_json(config_path)
        size = normalize_student_size(
            metadata.get("student_cnn", config_path.parent.parent.name)
        )
        seed = metadata.get("seed")
        row = {}
        base_row(row, "vicreg", size, "none", seed, "seed")
        row["cost_status"] = "missing_training_metadata"
        row["training_runs"] = 1
        rows.append(row)
    return rows


def collect_distillation_rows(checkpoint_root):
    rows = []
    for metadata_path in sorted(checkpoint_root.rglob("distillation.json")):
        metadata = load_json(metadata_path)
        size = metadata.get("student_cnn")
        if size is None:
            size = metadata_path.parent.parent.name
        size = normalize_student_size(size)
        seed = metadata.get("student_seed")
        if seed is None:
            seed_text = metadata_path.parent.name
            seed = int(seed_text.removeprefix("seed")) if seed_text.startswith("seed") else ""

        row = {}
        base_row(
            row,
            "distillation",
            size,
            metadata.get("projector_type", ""),
            seed,
            "seed",
        )
        training_hours = finite_gpu_hours(metadata.get("local_training_gpu_hours"))
        teacher_hours = finite_gpu_hours(metadata.get("teacher_forward_gpu_hours"))
        if training_hours is not None:
            row["training_gpu_hours"] = training_hours
        if teacher_hours is not None:
            row["teacher_forward_gpu_hours"] = teacher_hours
        if training_hours is None or teacher_hours is None:
            row["cost_status"] = "missing_training_or_teacher_timing"
        else:
            row["local_gpu_hours"] = training_hours + teacher_hours
            row["total_gpu_hours"] = training_hours + teacher_hours
        row["training_runs"] = 1
        rows.append(row)
    return rows


def collect_teacher_references(teacher_feature_dir):
    metadata_path = teacher_feature_dir / "metadata.json"
    if not metadata_path.is_file():
        return []

    metadata = load_json(metadata_path)
    provenance = metadata.get("teacher_provenance", {})
    references = provenance.get("pretraining_compute_references", [])
    rows = []
    for reference in references:
        wall_days = finite_gpu_hours(reference.get("reported_wall_time_days"))
        gpu_count = finite_gpu_hours(reference.get("reported_gpu_count"))
        gpu_hours = (
            wall_days * gpu_count * 24.0
            if wall_days is not None and gpu_count is not None
            else None
        )
        row = {}
        base_row(row, "teacher_encoding", "none", "none", "", "published_reference")
        row["published_pretraining_gpu_hours"] = gpu_hours
        row["published_pretraining_reference"] = reference.get("run", "")
        row["cost_status"] = (
            "published_reference_not_confirmed_for_exact_checkpoint"
            if gpu_hours is not None
            else "missing_published_compute"
        )
        row["total_gpu_hours"] = gpu_hours if gpu_hours is not None else 0.0
        rows.append(row)
    return rows


def collect_random_rows(small_seeds, large_seeds):
    rows = []
    for size, seeds in (("small", small_seeds), ("larger", large_seeds)):
        for seed in seeds:
            row = {}
            base_row(row, "random_baseline", size, "none", seed, "seed")
            row["cost_status"] = "no_training"
            rows.append(row)
    return rows


def aggregate_seed_rows(seed_rows):
    aggregates = {}
    for row in seed_rows:
        key = (row["approach"], row["student_size"], row["projector_type"])
        if key not in aggregates:
            aggregates[key] = {
                "scope": "aggregate",
                "approach": row["approach"],
                "student_size": row["student_size"],
                "projector_type": row["projector_type"],
                "seed": "",
                "training_runs": 0,
                "training_gpu_hours": 0.0,
                "teacher_forward_gpu_hours": 0.0,
                "published_pretraining_gpu_hours": None,
                "published_pretraining_reference": "",
                "local_gpu_hours": 0.0,
                "total_gpu_hours": 0.0,
                "cost_status": "measured",
            }
        aggregate = aggregates[key]
        aggregate["training_runs"] += int(row["training_runs"])
        aggregate["training_gpu_hours"] += row["training_gpu_hours"]
        aggregate["teacher_forward_gpu_hours"] += row["teacher_forward_gpu_hours"]
        aggregate["local_gpu_hours"] += row["local_gpu_hours"]
        aggregate["total_gpu_hours"] += row["total_gpu_hours"]
        statuses = {other["cost_status"] for other in seed_rows if
                    (other["approach"], other["student_size"], other["projector_type"]) == key}
        if statuses != {"measured"}:
            aggregate["cost_status"] = ",".join(sorted(statuses))
    return list(aggregates.values())


def main():
    args = parse_args()
    seed_rows = []
    seed_rows.extend(collect_random_rows(args.small_seeds, args.large_seeds))
    seed_rows.extend(collect_vicreg_rows(args.checkpoint_root))
    seed_rows.extend(collect_distillation_rows(args.checkpoint_root))

    rows = seed_rows + aggregate_seed_rows(seed_rows) + collect_teacher_references(
        args.teacher_feature_dir
    )

    fieldnames = [
        "scope",
        "approach",
        "student_size",
        "projector_type",
        "seed",
        "training_runs",
        "training_gpu_hours",
        "teacher_forward_gpu_hours",
        "published_pretraining_gpu_hours",
        "published_pretraining_reference",
        "local_gpu_hours",
        "total_gpu_hours",
        "cost_status",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} cost rows to {args.output_csv}")


if __name__ == "__main__":
    main()
