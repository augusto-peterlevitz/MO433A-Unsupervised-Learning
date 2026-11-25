import argparse
import json
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


STUDENT_SIZES = {
    "small": "small_cnn",
    "large": "larger_cnn",
}
PROJECTORS = ("mlp", "linear")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize local training GPU-hours and accuracy per total "
            "GPU-hour, including published teacher-cost references."
        )
    )
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder-costs-csv",
        type=Path,
        required=True,
        help="Measured FLOPs and inference latency table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/probes/cost_analysis"),
    )
    return parser.parse_args()


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_vicreg_event_timings(tensorboard_dir):
    event_files = sorted(
        tensorboard_dir.rglob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found under {tensorboard_dir}."
        )

    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    if "train/epoch_seconds" not in tags or "train/batch_loss" not in tags:
        raise ValueError(
            f"Missing training timing scalars in {event_files[-1]}."
        )
    epoch_seconds = [
        event.value for event in accumulator.Scalars("train/epoch_seconds")
    ]
    optimizer_steps = len(accumulator.Scalars("train/batch_loss"))
    return sum(epoch_seconds), optimizer_steps, event_files[-1]


def load_training_row(
    features_dir,
    encoder_costs,
    approach,
    projector,
    student_size,
    seed_dir,
):
    size_dir = STUDENT_SIZES[student_size]
    seed = int(seed_dir.name[4:])
    config = read_json(seed_dir / "vicreg.json")
    metadata_path = seed_dir / "training_metadata.json"
    if approach == "vicreg":
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            seconds = float(metadata["training_seconds"])
            optimizer_steps = int(metadata["optimizer_steps"])
            device = metadata.get("device")
            device_evidence = "training_metadata.json"
            source = metadata_path
            gpu_model = metadata.get("gpu_model")
            trainable_parameters = metadata.get("trainable_parameter_count")
            projector_parameters = 0
        else:
            seconds, optimizer_steps, source = read_vicreg_event_timings(
                seed_dir / "tensorboard"
            )
            device = None
            device_evidence = "not recorded in VICReg run"
            gpu_model = None
            trainable_parameters = None
            projector_parameters = 0
        teacher_forward_seconds = 0.0
        teacher_forward_gpu_hours = 0.0
    else:
        metadata_path = seed_dir / "distillation.json"
        metadata = read_json(metadata_path)
        metrics = pd.read_csv(seed_dir / "metrics.csv")
        if "epoch_seconds" not in metrics or metrics.empty:
            raise ValueError(f"Missing epoch timings in {seed_dir / 'metrics.csv'}.")
        seconds = float(
            metadata.get("training_seconds", metrics["epoch_seconds"].sum())
        )
        optimizer_steps = int(metadata["optimizer_steps"])
        device = metadata.get(
            "training_device",
            metadata.get("teacher_forward_device"),
        )
        device_evidence = "distillation.json"
        gpu_model = metadata.get("gpu_model")
        projector_parameters = int(metadata["projector_parameter_count"])
        trainable_parameters = int(
            metadata.get(
                "trainable_parameter_count",
                metadata["student_parameter_count"] + projector_parameters,
            )
        )
        teacher_forward_seconds = float(
            metadata.get("teacher_forward_seconds") or 0.0
        )
        teacher_forward_gpu_hours = float(
            metadata.get("teacher_forward_gpu_hours") or 0.0
        )
        source = seed_dir / "metrics.csv"

    if device is None:
        cache_path = (
            features_dir
            / approach
            / (f"projector_{projector}" if approach == "distillation" else "")
            / size_dir
            / f"seed{seed}"
            / "metadata.json"
        )
        cache_metadata = read_json(cache_path)
        device = cache_metadata.get("device")
        device_evidence = "inferred from feature cache; training device not saved"

    is_cuda = str(device).startswith("cuda")
    gpu_hours = seconds / 3600.0 if is_cuda else None
    cost_encoder = encoder_costs[size_dir]
    return {
        "approach": approach,
        "projector": projector or "none",
        "student_size": student_size,
        "seed": seed,
        "training_epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "optimizer_steps": optimizer_steps,
        "student_parameters": int(cost_encoder["num_parameters"]),
        "projector_parameters": projector_parameters,
        "trainable_parameters": trainable_parameters,
        "training_seconds": seconds,
        "training_wall_hours": seconds / 3600.0,
        "local_training_gpu_hours": gpu_hours,
        "training_images_per_second": (
            optimizer_steps * int(config["batch_size"]) / seconds
            if seconds > 0
            else None
        ),
        "teacher_forward_seconds": teacher_forward_seconds,
        "teacher_forward_gpu_hours": teacher_forward_gpu_hours,
        "training_device": device,
        "device_evidence": device_evidence,
        "training_gpu_model": gpu_model,
        "benchmark_gpu_model": cost_encoder["gpu_model"],
        "timing_source": str(source),
    }


def collect_training_rows(checkpoints_dir, features_dir, encoder_costs):
    rows = []
    for student_size, size_dir in STUDENT_SIZES.items():
        vicreg_root = checkpoints_dir / "vicreg" / size_dir
        for seed_dir in sorted(vicreg_root.glob("seed*")):
            rows.append(load_training_row(
                features_dir,
                encoder_costs,
                "vicreg",
                None,
                student_size,
                seed_dir,
            ))

        for projector in PROJECTORS:
            distillation_root = checkpoints_dir / "distillation"
            if projector == "linear":
                distillation_root = (
                    distillation_root / "ablation" / "projector_linear"
                )
            distillation_root /= size_dir
            for seed_dir in sorted(distillation_root.glob("seed*")):
                rows.append(load_training_row(
                    features_dir,
                    encoder_costs,
                    "distillation",
                    projector,
                    student_size,
                    seed_dir,
                ))
    if not rows:
        raise FileNotFoundError(
            f"No VICReg or distillation runs found under {checkpoints_dir}."
        )
    return pd.DataFrame(rows)


def read_accuracy_summary(path):
    summary = pd.read_csv(path)
    required = {"fraction", "mean_accuracy", "std_accuracy"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return summary.set_index("fraction")


def aggregate_training_costs(training_rows):
    return (
        training_rows.groupby(
            ["approach", "projector", "student_size"],
            as_index=False,
            dropna=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            training_seconds=("training_seconds", "sum"),
            training_wall_hours=("training_wall_hours", "sum"),
            local_training_gpu_hours=("local_training_gpu_hours", "sum"),
            teacher_forward_seconds=("teacher_forward_seconds", "sum"),
            teacher_forward_gpu_hours=("teacher_forward_gpu_hours", "sum"),
            mean_seed_training_seconds=("training_seconds", "mean"),
            std_seed_training_seconds=("training_seconds", "std"),
            student_parameters=("student_parameters", "first"),
            projector_parameters=("projector_parameters", "mean"),
        )
    )


def build_efficiency_rows(
    features_dir,
    runs_dir,
    training_rows,
    encoder_costs,
):
    training_groups = aggregate_training_costs(training_rows).set_index(
        ["approach", "projector", "student_size"]
    )
    teacher_metadata = read_json(features_dir / "dino_vits16" / "metadata.json")
    references = teacher_metadata["teacher_provenance"][
        "pretraining_compute_references"
    ]
    teacher_cost_encoder = encoder_costs["dino_vits16"]
    teacher_summary = read_accuracy_summary(
        runs_dir
        / "svm_evaluation"
        / "teacher_encoding"
        / "dino_vits16"
        / "aggregate"
        / "linear"
        / "summary.csv"
    )
    output = []

    for student_size, size_dir in STUDENT_SIZES.items():
        baseline = training_groups.loc[("vicreg", "none", student_size)]
        baseline_summary = read_accuracy_summary(
            runs_dir
            / "svm_evaluation"
            / "vicreg"
            / size_dir
            / "aggregate"
            / "linear"
            / "summary.csv"
        )
        baseline_hours = float(baseline["local_training_gpu_hours"])

        for fraction, accuracy_row in baseline_summary.iterrows():
            mean_accuracy = float(accuracy_row["mean_accuracy"])
            total_hours = baseline_hours
            output.append({
                "approach": "vicreg",
                "projector": "none",
                "student_size": student_size,
                "student_parameters": int(baseline["student_parameters"]),
                "fraction": float(fraction),
                "probe": "linear_svm",
                "mean_accuracy": mean_accuracy,
                "std_accuracy": float(accuracy_row["std_accuracy"]),
                "local_training_gpu_hours": baseline_hours,
                "teacher_forward_gpu_hours": 0.0,
                "teacher_pretraining_scenario": "none",
                "teacher_pretraining_gpu_days_reference": 0.0,
                "teacher_pretraining_gpu_hours_reference": 0.0,
                "total_gpu_hours": total_hours,
                "accuracy_percentage_points_per_gpu_hour": (
                    100 * mean_accuracy / total_hours if total_hours > 0 else None
                ),
                "gap_vs_vicreg_percentage_points": 0.0,
                "incremental_gpu_hours_vs_vicreg": 0.0,
                "gap_percentage_points_per_incremental_gpu_hour": None,
            })

        for projector in PROJECTORS:
            distillation = training_groups.loc[
                ("distillation", projector, student_size)
            ]
            distillation_summary = read_accuracy_summary(
                runs_dir
                / "svm_evaluation"
                / "distillation"
                / size_dir
                / f"projector_{projector}"
                / "aggregate"
                / "linear"
                / "summary.csv"
            )
            local_hours = float(distillation["local_training_gpu_hours"])
            for reference in references:
                pretraining_days = float(reference["derived_gpu_days"])
                pretraining_hours = pretraining_days * 24
                total_hours = local_hours + pretraining_hours
                for fraction, accuracy_row in distillation_summary.iterrows():
                    baseline_accuracy = float(
                        baseline_summary.loc[fraction, "mean_accuracy"]
                    )
                    mean_accuracy = float(accuracy_row["mean_accuracy"])
                    gap_pp = 100 * (mean_accuracy - baseline_accuracy)
                    extra_hours = total_hours - baseline_hours
                    output.append({
                        "approach": "distillation",
                        "projector": projector,
                        "student_size": student_size,
                        "student_parameters": int(
                            distillation["student_parameters"]
                        ),
                        "fraction": float(fraction),
                        "probe": "linear_svm",
                        "mean_accuracy": mean_accuracy,
                        "std_accuracy": float(accuracy_row["std_accuracy"]),
                        "local_training_gpu_hours": local_hours,
                        "teacher_forward_gpu_hours": float(
                            distillation["teacher_forward_gpu_hours"]
                        ),
                        "teacher_pretraining_scenario": reference["run"],
                        "teacher_pretraining_gpu_days_reference": pretraining_days,
                        "teacher_pretraining_gpu_hours_reference": pretraining_hours,
                        "total_gpu_hours": total_hours,
                        "accuracy_percentage_points_per_gpu_hour": (
                            100 * mean_accuracy / total_hours
                        ),
                        "gap_vs_vicreg_percentage_points": gap_pp,
                        "incremental_gpu_hours_vs_vicreg": extra_hours,
                        "gap_percentage_points_per_incremental_gpu_hour": (
                            gap_pp / extra_hours if extra_hours > 0 else None
                        ),
                    })

    for reference in references:
        pretraining_days = float(reference["derived_gpu_days"])
        pretraining_hours = pretraining_days * 24
        for fraction, accuracy_row in teacher_summary.iterrows():
            mean_accuracy = float(accuracy_row["mean_accuracy"])
            output.append({
                "approach": "teacher_encoding",
                "projector": "none",
                "student_size": "teacher",
                "student_parameters": int(teacher_cost_encoder["num_parameters"]),
                "fraction": float(fraction),
                "probe": "linear_svm",
                "mean_accuracy": mean_accuracy,
                "std_accuracy": float(accuracy_row["std_accuracy"]),
                "local_training_gpu_hours": 0.0,
                "teacher_forward_gpu_hours": 0.0,
                "teacher_pretraining_scenario": reference["run"],
                "teacher_pretraining_gpu_days_reference": pretraining_days,
                "teacher_pretraining_gpu_hours_reference": pretraining_hours,
                "total_gpu_hours": pretraining_hours,
                "accuracy_percentage_points_per_gpu_hour": (
                    100 * mean_accuracy / pretraining_hours
                ),
                "gap_vs_vicreg_percentage_points": None,
                "incremental_gpu_hours_vs_vicreg": None,
                "gap_percentage_points_per_incremental_gpu_hour": None,
            })
    return pd.DataFrame(output)


def main():
    args = parse_args()
    encoder_costs_frame = pd.read_csv(args.encoder_costs_csv)
    required_cost_columns = {
        "encoder",
        "num_parameters",
        "gpu_model",
    }
    missing = required_cost_columns.difference(encoder_costs_frame.columns)
    if missing:
        raise ValueError(
            f"{args.encoder_costs_csv} is missing columns: {sorted(missing)}"
        )
    encoder_costs = encoder_costs_frame.set_index("encoder").to_dict("index")
    expected_encoders = {"small_cnn", "larger_cnn", "dino_vits16"}
    if set(encoder_costs) != expected_encoders:
        raise ValueError(
            f"Expected measurements for {sorted(expected_encoders)}, "
            f"got {sorted(encoder_costs)}."
        )

    training_rows = collect_training_rows(
        args.checkpoints_dir,
        args.features_dir,
        encoder_costs,
    )
    efficiency_rows = build_efficiency_rows(
        args.features_dir,
        args.runs_dir,
        training_rows,
        encoder_costs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.output_dir / "training_gpu_hours.csv"
    efficiency_path = args.output_dir / "accuracy_per_gpu_hour.csv"
    training_rows.to_csv(training_path, index=False)
    efficiency_rows.to_csv(efficiency_path, index=False)
    print(f"Saved per-run training cost summary to {training_path}")
    print(f"Saved accuracy-per-GPU-hour analysis to {efficiency_path}")


if __name__ == "__main__":
    main()
