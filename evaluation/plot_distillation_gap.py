import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PAIR_COLUMNS = ["random_seed", "fraction", "student_size"]
REQUIRED_COLUMNS = {
    "probe",
    "fraction",
    "mean_accuracy",
    "num_parameters",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired seed-level linear-SVM accuracy gaps between "
            "distilled and directly trained students against parameter count."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/runs/probes/svm_evaluation"),
        help="Root containing per-seed SVM summary CSVs.",
    )
    parser.add_argument(
        "--projector-type",
        choices=("mlp", "linear"),
        default="mlp",
        help="Distillation projector to compare against VICReg.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/probes/destillation"),
    )
    return parser.parse_args()


def student_size_from_path(path):
    parts = {part.lower() for part in path.parts}
    if "small_cnn" in parts:
        return "small"
    if "large_cnn" in parts or "larger_cnn" in parts:
        return "large"
    raise ValueError(f"Cannot determine student size from result path: {path}")


def load_probe_results(input_dir, projector_type):
    frames = {"vicreg": [], "distillation": []}
    for path in sorted(input_dir.rglob("summary.csv")):
        relative_parts = path.relative_to(input_dir).parts
        if "aggregate" in relative_parts:
            continue
        if "linear" not in relative_parts:
            continue
        if "vicreg" in relative_parts:
            approach = "vicreg"
        elif "distillation" in relative_parts:
            approach = "distillation"
            if f"projector_{projector_type}" not in relative_parts:
                continue
        else:
            continue

        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame[frame["probe"] == "linear_svm"].copy()
        if frame.empty:
            continue
        seed_dirs = [
            part[4:] for part in relative_parts if part.startswith("seed")
        ]
        if len(seed_dirs) != 1 or not seed_dirs[0].isdigit():
            raise ValueError(f"Cannot determine a single random seed from {path}")
        frame["student_size"] = student_size_from_path(path)
        frame["random_seed"] = int(seed_dirs[0])
        frame["fraction"] = frame["fraction"].astype(float)
        frame["mean_accuracy"] = frame["mean_accuracy"].astype(float)
        frame["num_parameters"] = frame["num_parameters"].astype(int)
        frame["approach"] = approach
        frames[approach].append(frame)

    missing_approaches = [name for name, items in frames.items() if not items]
    if missing_approaches:
        raise FileNotFoundError(
            f"No per-seed linear-SVM results found for "
            f"{', '.join(missing_approaches)} under {input_dir}."
        )
    return {
        approach: pd.concat(items, ignore_index=True)
        for approach, items in frames.items()
    }


def prepare_paired_gaps(results):
    vicreg = results["vicreg"]
    distillation = results["distillation"]

    keys = PAIR_COLUMNS
    for label, frame in (("VICReg", vicreg), ("distillation", distillation)):
        if frame.duplicated(keys).any():
            duplicates = frame.loc[frame.duplicated(keys, keep=False), keys]
            raise ValueError(
                f"Duplicate {label} result rows for pairing keys:\n"
                f"{duplicates.to_string(index=False)}"
            )

    paired = vicreg[keys + ["mean_accuracy", "num_parameters"]].merge(
        distillation[keys + ["mean_accuracy", "num_parameters"]],
        on=keys,
        how="outer",
        suffixes=("_vicreg", "_distillation"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = paired[paired["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            "VICReg and distillation results do not have matching "
            "seed/fraction/student-size rows:\n"
            f"{unmatched[keys + ['_merge']].to_string(index=False)}"
        )

    if (paired["num_parameters_vicreg"] != paired["num_parameters_distillation"]).any():
        raise ValueError(
            "VICReg and distilled students must have the same parameter count "
            "for each paired architecture."
        )

    paired["student_parameters"] = paired["num_parameters_vicreg"]
    paired["gap_pp"] = (
        paired["mean_accuracy_distillation"]
        - paired["mean_accuracy_vicreg"]
    ) * 100.0
    paired = paired.drop(columns="_merge")
    summary = (
        paired.groupby(["student_size", "student_parameters", "fraction"])
        .agg(
            mean_gap_pp=("gap_pp", "mean"),
            std_gap_pp=("gap_pp", "std"),
            paired_seeds=("gap_pp", "count"),
            mean_vicreg_accuracy=("mean_accuracy_vicreg", "mean"),
            mean_distillation_accuracy=("mean_accuracy_distillation", "mean"),
        )
        .reset_index()
    )
    return paired, summary


def print_size_comparison(summary):
    small = summary[summary["student_size"] == "small"].set_index("fraction")
    large = summary[summary["student_size"] == "large"].set_index("fraction")
    small_fractions = set(small.index)
    large_fractions = set(large.index)
    if not small_fractions or not large_fractions:
        raise ValueError("Both small and large student results are required.")
    if small_fractions != large_fractions:
        raise ValueError(
            "Small and large students must have the same label fractions; "
            f"small-only={sorted(small_fractions - large_fractions)}, "
            f"large-only={sorted(large_fractions - small_fractions)}."
        )
    fractions = sorted(small_fractions)

    print("C − B gap by student size (mean ± 1 SD across paired seeds):")
    overtakes = []
    for fraction in fractions:
        small_gap = small.loc[fraction, "mean_gap_pp"]
        large_gap = large.loc[fraction, "mean_gap_pp"]
        small_std = small.loc[fraction, "std_gap_pp"]
        large_std = large.loc[fraction, "std_gap_pp"]
        change = large_gap - small_gap
        small_spread = f"{small_std:.2f}" if pd.notna(small_std) else "n/a"
        large_spread = f"{large_std:.2f}" if pd.notna(large_std) else "n/a"

        if large_gap < 0:
            conclusion = "B has the higher mean at the larger size"
            if small_gap >= 0:
                conclusion = "B overtakes C as size increases"
                overtakes.append(fraction)
        else:
            conclusion = "C has the higher mean at the larger size"
        trend = "narrows" if change < 0 else "widens" if change > 0 else "is unchanged"
        interval = (
            "±1 SD range includes zero"
            if pd.notna(large_std) and large_gap - large_std <= 0 <= large_gap + large_std
            else "±1 SD range excludes zero"
            if pd.notna(large_std)
            else "±1 SD unavailable (one paired seed)"
        )
        print(
            f"  {fraction:.0%}: small {small_gap:+.2f} ± {small_spread} pp; "
            f"large {large_gap:+.2f} ± {large_spread} pp; "
            f"gap {trend} by {change:+.2f} pp; {conclusion} "
            f"({interval})."
        )

    if overtakes:
        labels = ", ".join(f"{fraction:.0%}" for fraction in overtakes)
        print(f"Conclusion: B overtakes C as size increases at {labels} label fractions.")
    else:
        b_already_ahead = [
            fraction
            for fraction in fractions
            if small.loc[fraction, "mean_gap_pp"] < 0
            and large.loc[fraction, "mean_gap_pp"] < 0
        ]
        if b_already_ahead:
            labels = ", ".join(f"{fraction:.0%}" for fraction in b_already_ahead)
            print(
                "Conclusion: no size-related crossover; B already has the higher "
                f"mean at both sizes for {labels} label fractions."
            )
        else:
            print(
                "Conclusion: B does not overtake C as size increases for any "
                "listed label fraction."
            )


def plot_gaps(summary, output_path, projector_type):
    fig, ax = plt.subplots(figsize=(8, 5))
    for fraction, group in summary.groupby("fraction", sort=True):
        group = group.sort_values("student_parameters")
        ax.errorbar(
            group["student_parameters"],
            group["mean_gap_pp"],
            yerr=group["std_gap_pp"].fillna(0.0),
            marker="o",
            capsize=4,
            linewidth=2,
            label=f"{fraction:.0%} labels",
        )

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    parameter_counts = sorted(summary["student_parameters"].unique())
    ax.set_xticks(parameter_counts)
    ax.set_xticklabels([f"{count:,}" for count in parameter_counts])
    ax.set_xlabel("Student backbone parameters (log scale)")
    ax.set_ylabel("C − B linear-SVM accuracy (percentage points)")
    ax.set_title(f"C − B gap by student size ({projector_type} projector)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Label fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"SVM result directory not found: {args.input_dir}")

    results = load_probe_results(args.input_dir, args.projector_type)
    paired, summary = prepare_paired_gaps(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired_path = args.output_dir / f"distillation_gap_{args.projector_type}_paired.csv"
    summary_path = args.output_dir / f"distillation_gap_{args.projector_type}_summary.csv"
    figure_path = args.output_dir / f"distillation_gap_{args.projector_type}.png"
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_gaps(summary, figure_path, args.projector_type)
    print_size_comparison(summary)

    print(f"Paired results saved to {paired_path}")
    print(f"Gap summary saved to {summary_path}")
    print(f"Gap plot saved to {figure_path}")


if __name__ == "__main__":
    main()
