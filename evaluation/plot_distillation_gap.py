import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDENT_SIZES = {
    "small": "small_cnn",
    "large": "larger_cnn",
}
PROJECTORS = ("mlp", "linear")
PAIR_KEYS = ["random_seed", "fraction", "draw"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired distillation-minus-VICReg accuracy gaps by student "
            "size, projector, and labeled-data fraction."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/runs/probes"),
        help="Root containing svm_evaluation/ and knn_evaluation/ results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/probes/distillation_gap"),
    )
    parser.add_argument(
        "--probe",
        choices=("linear", "rbf", "knn"),
        default="linear",
    )
    return parser.parse_args()


def get_result_paths(input_dir, approach, size_dir, projector, probe):
    evaluation_dir = "knn_evaluation" if probe == "knn" else "svm_evaluation"
    result_root = input_dir / evaluation_dir / approach / size_dir
    if approach == "distillation":
        result_root /= f"projector_{projector}"

    if probe == "knn":
        result_paths = sorted(result_root.glob("seed*/results.csv"))
    else:
        result_paths = sorted(result_root.glob(f"seed*/{probe}/results.csv"))
    if not result_paths:
        raise FileNotFoundError(
            f"No {probe} result files found beneath {result_root}."
        )
    return result_paths


def load_results(paths, probe, model_name):
    expected_probe = "knn" if probe == "knn" else f"{probe}_svm"
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {
            "probe",
            "fraction",
            "draw",
            "accuracy",
            "random_seed",
            "num_parameters",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if not frame["probe"].eq(expected_probe).all():
            raise ValueError(f"Unexpected probe in {path}; expected {expected_probe}.")

        model = frame[
            ["random_seed", "fraction", "draw", "accuracy", "num_parameters"]
        ].copy()
        for column in ("random_seed", "draw", "fraction", "accuracy", "num_parameters"):
            model[column] = pd.to_numeric(model[column], errors="coerce")
        if model.isna().any().any() or not np.isfinite(model.to_numpy()).all():
            raise ValueError(f"{path} contains missing or non-finite result values.")
        model["random_seed"] = model["random_seed"].astype(int)
        model["draw"] = model["draw"].astype(int)

        # Full-data draws are identical by construction; retain one per seed.
        model = model[(model["fraction"] != 1.0) | (model["draw"] == 0)]
        if model.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{path} contains duplicate seed/fraction/draw rows.")
        model = model.rename(columns={
            "accuracy": f"accuracy_{model_name}",
            "num_parameters": f"parameters_{model_name}",
        })
        frames.append(model)

    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(PAIR_KEYS).any():
        raise ValueError(f"Duplicate seed/fraction/draw rows found for {model_name}.")
    return combined


def pair_results(baseline, distillation, projector, student_size):
    paired = baseline.merge(
        distillation,
        on=PAIR_KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = paired[paired["_merge"] != "both"]
    if not unmatched.empty:
        example = unmatched[PAIR_KEYS + ["_merge"]].head(5).to_dict("records")
        raise ValueError(
            f"VICReg and {projector}-projector results are not paired for "
            f"{student_size}; unmatched examples: {example}"
        )
    paired = paired.drop(columns="_merge")

    for column in ("parameters_vicreg", "parameters_distillation"):
        if paired[column].nunique() != 1:
            raise ValueError(
                f"Expected a single backbone parameter count for {student_size}; "
                f"found {sorted(paired[column].unique())}."
            )
    baseline_parameters = int(paired["parameters_vicreg"].iloc[0])
    distillation_parameters = int(paired["parameters_distillation"].iloc[0])
    if baseline_parameters != distillation_parameters:
        raise ValueError(
            f"VICReg and distillation backbone parameter counts differ for "
            f"{student_size}: {baseline_parameters} vs {distillation_parameters}."
        )

    paired["projector"] = projector
    paired["student_size"] = student_size
    paired["student_parameters"] = baseline_parameters
    paired["gap_pp"] = 100 * (
        paired["accuracy_distillation"] - paired["accuracy_vicreg"]
    )
    return paired


def summarize_gaps(paired):
    return (
        paired.groupby(
            ["projector", "student_size", "student_parameters", "fraction"],
            as_index=False,
        )
        .agg(
            mean_gap_pp=("gap_pp", "mean"),
            std_gap_pp=("gap_pp", "std"),
            paired_observations=("gap_pp", "size"),
            random_seeds=("random_seed", "nunique"),
            mean_vicreg_accuracy=("accuracy_vicreg", "mean"),
            mean_distillation_accuracy=("accuracy_distillation", "mean"),
        )
        .sort_values(["projector", "student_parameters", "fraction"])
    )


def plot_parameter_gaps(summary, output_path, probe):
    fractions = sorted(summary["fraction"].unique())
    figure, axes = plt.subplots(
        2, 2, figsize=(11, 8), sharey=True, squeeze=False
    )
    colors = {"mlp": "tab:blue", "linear": "tab:orange"}
    parameter_counts = sorted(summary["student_parameters"].unique())

    for axis, fraction in zip(axes.flat, fractions):
        fraction_data = summary[summary["fraction"] == fraction]
        for projector in PROJECTORS:
            data = fraction_data[fraction_data["projector"] == projector].sort_values(
                "student_parameters"
            )
            axis.errorbar(
                data["student_parameters"],
                data["mean_gap_pp"],
                yerr=data["std_gap_pp"].fillna(0),
                color=colors[projector],
                marker="o",
                capsize=4,
                linewidth=2,
                label=f"{projector.upper()} projector",
            )
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.minorticks_off()
        axis.set_xticks(parameter_counts)
        axis.set_xticklabels([f"{count:,}" for count in parameter_counts])
        axis.set_title(f"{fraction:.0%} labeled data")
        axis.grid(True, which="both", linestyle="--", alpha=0.35)

    axes[1, 0].set_xlabel("Student backbone parameters")
    axes[1, 1].set_xlabel("Student backbone parameters")
    axes[0, 0].set_ylabel("C - B mean accuracy gap (percentage points)")
    axes[1, 0].set_ylabel("C - B mean accuracy gap (percentage points)")
    axes[0, 0].legend(title="Distillation ablation")
    figure.suptitle(
        f"Distillation minus VICReg gap vs student size ({probe.upper()} probe)"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_ablation_gaps(summary, output_path, probe):
    figure, axes = plt.subplots(
        2, 2, figsize=(11, 8), sharex=True, sharey=True, squeeze=False
    )
    fractions = sorted(summary["fraction"].unique())
    for row, student_size in enumerate(STUDENT_SIZES):
        for column, projector in enumerate(PROJECTORS):
            axis = axes[row, column]
            data = summary[
                (summary["student_size"] == student_size)
                & (summary["projector"] == projector)
            ].sort_values("fraction")
            parameters = int(data["student_parameters"].iloc[0])
            axis.errorbar(
                data["fraction"],
                data["mean_gap_pp"],
                yerr=data["std_gap_pp"].fillna(0),
                marker="o",
                capsize=4,
                linewidth=2,
                color="tab:blue" if projector == "mlp" else "tab:orange",
            )
            axis.axhline(0, color="black", linewidth=1, linestyle="--")
            axis.set_xscale("log")
            axis.set_xticks(fractions)
            axis.set_xticklabels([f"{fraction:.0%}" for fraction in fractions])
            axis.set_title(
                f"{student_size.title()} / {projector.upper()} "
                f"({parameters:,} parameters)"
            )
            axis.grid(True, which="both", linestyle="--", alpha=0.35)

    axes[1, 0].set_xlabel("Labeled training fraction")
    axes[1, 1].set_xlabel("Labeled training fraction")
    axes[0, 0].set_ylabel("C - B mean accuracy gap (percentage points)")
    axes[1, 0].set_ylabel("C - B mean accuracy gap (percentage points)")
    figure.suptitle(f"C - B gap by ablation setting ({probe.upper()} probe)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    paired_frames = []

    for student_size, size_dir in STUDENT_SIZES.items():
        baseline_paths = get_result_paths(
            args.input_dir, "vicreg", size_dir, None, args.probe
        )
        baseline = load_results(baseline_paths, args.probe, "vicreg")
        for projector in PROJECTORS:
            distillation_paths = get_result_paths(
                args.input_dir,
                "distillation",
                size_dir,
                projector,
                args.probe,
            )
            distillation = load_results(
                distillation_paths, args.probe, "distillation"
            )
            paired_frames.append(
                pair_results(baseline, distillation, projector, student_size)
            )

    paired = pd.concat(paired_frames, ignore_index=True)
    summary = summarize_gaps(paired)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired_path = args.output_dir / f"distillation_gap_{args.probe}_paired.csv"
    summary_path = args.output_dir / f"distillation_gap_{args.probe}_summary.csv"
    parameter_plot_path = (
        args.output_dir / f"distillation_gap_{args.probe}_by_parameters.png"
    )
    ablation_plot_path = (
        args.output_dir / f"distillation_gap_{args.probe}_by_ablation.png"
    )
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_parameter_gaps(summary, parameter_plot_path, args.probe)
    plot_ablation_gaps(summary, ablation_plot_path, args.probe)

    print(f"Saved paired gaps to {paired_path}")
    print(f"Saved gap summary to {summary_path}")
    print(f"Saved parameter-count plot to {parameter_plot_path}")
    print(f"Saved ablation plot to {ablation_plot_path}")


if __name__ == "__main__":
    main()
