import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


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
            "Plot paired-seed linear-SVM accuracy gaps by label fraction and "
            "student size."
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
        choices=("mlp", "linear", "all"),
        default="mlp",
        help="Distillation projector to compare against VICReg, or both.",
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


def projector_change_direction(small_change, large_change):
    tolerance = 1e-9
    small_direction = (
        1 if small_change > tolerance else -1 if small_change < -tolerance else 0
    )
    large_direction = (
        1 if large_change > tolerance else -1 if large_change < -tolerance else 0
    )
    if small_direction == large_direction == 1:
        return "Yes (gap increases for both sizes)"
    if small_direction == large_direction == -1:
        return "Yes (gap decreases for both sizes)"
    if small_direction == large_direction == 0:
        return "No change for either size"
    if small_direction == 0 or large_direction == 0:
        return "No (gap changes for only one size)"
    return "No (opposite directions)"


def build_ablation_grid(projector_summaries):
    combined = pd.concat(
        [
            summary.assign(projector_type=projector_type)
            for projector_type, summary in projector_summaries.items()
        ],
        ignore_index=True,
    )
    expected_cells = {
        (projector_type, student_size)
        for projector_type in ("mlp", "linear")
        for student_size in ("small", "large")
    }

    parameter_counts = {}
    for student_size in ("small", "large"):
        counts = combined.loc[
            combined["student_size"] == student_size, "student_parameters"
        ].unique()
        if len(counts) != 1:
            raise ValueError(
                f"Expected one parameter count for {student_size} students, "
                f"found {counts.tolist()}."
            )
        parameter_counts[student_size] = int(counts[0])

    rows = []
    for fraction in sorted(combined["fraction"].unique()):
        row = {
            "fraction": fraction,
            "small_parameters": parameter_counts["small"],
            "large_parameters": parameter_counts["large"],
        }
        found_cells = set()
        cell_rows = {}
        for record in combined[combined["fraction"] == fraction].to_dict("records"):
            key = (record["projector_type"], record["student_size"])
            if key in cell_rows:
                raise ValueError(
                    f"Multiple ablation summary rows for fraction {fraction} "
                    f"and cell {key}."
                )
            found_cells.add(key)
            cell_rows[key] = record
        if found_cells != expected_cells:
            raise ValueError(
                f"Incomplete ablation grid at fraction {fraction}: "
                f"missing={sorted(expected_cells - found_cells)}, "
                f"unexpected={sorted(found_cells - expected_cells)}."
            )

        for projector_type, student_size in sorted(expected_cells):
            cell = cell_rows[(projector_type, student_size)]
            prefix = f"{projector_type}_{student_size}"
            row[f"{prefix}_mean_gap_pp"] = cell["mean_gap_pp"]
            row[f"{prefix}_std_gap_pp"] = cell["std_gap_pp"]
            row[f"{prefix}_paired_seeds"] = int(cell["paired_seeds"])

        for student_size in ("small", "large"):
            change = (
                cell_rows[("linear", student_size)]["mean_gap_pp"]
                - cell_rows[("mlp", student_size)]["mean_gap_pp"]
            )
            row[f"linear_minus_mlp_{student_size}_pp"] = change

        row["same_direction_for_both_sizes"] = projector_change_direction(
            row["linear_minus_mlp_small_pp"],
            row["linear_minus_mlp_large_pp"],
        )
        rows.append(row)

    return pd.DataFrame(rows)


def write_ablation_markdown(grid, output_path):
    lines = [
        "# C-versus-B projector ablation",
        "",
        "Each cell reports mean C-minus-B linear-SVM accuracy gap ± 1 SD "
        "across paired seeds, in percentage points. Positive values favor C; "
        "negative values favor B.",
        "",
    ]
    for row in grid.to_dict("records"):
        fraction = row["fraction"]
        lines.extend(
            [
                f"## {fraction:.0%} labeled data",
                "",
                f"| Projector | Small ({row['small_parameters']:,} params) "
                f"| Large ({row['large_parameters']:,} params) |",
                "|---|---:|---:|",
            ]
        )
        for projector_type in ("mlp", "linear"):
            cells = []
            for student_size in ("small", "large"):
                prefix = f"{projector_type}_{student_size}"
                mean = row[f"{prefix}_mean_gap_pp"]
                std = row[f"{prefix}_std_gap_pp"]
                seeds = row[f"{prefix}_paired_seeds"]
                spread = f"{std:.2f}" if pd.notna(std) else "n/a"
                cells.append(f"{mean:+.2f} ± {spread} pp (n={seeds})")
            lines.append(f"| {projector_type.upper()} | {cells[0]} | {cells[1]} |")

        small_change = row["linear_minus_mlp_small_pp"]
        large_change = row["linear_minus_mlp_large_pp"]
        lines.extend(
            [
                "",
                f"Linear − MLP gap change: small {small_change:+.2f} pp; "
                f"large {large_change:+.2f} pp. Same direction for both sizes: "
                f"{row['same_direction_for_both_sizes']}.",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


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
    student_sizes = ("small", "large")
    fractions = sorted(summary["fraction"].unique())
    if not fractions:
        raise ValueError("No label fractions found in the distillation summary.")

    parameter_counts = {}
    for student_size in student_sizes:
        size_rows = summary[summary["student_size"] == student_size]
        counts = size_rows["student_parameters"].unique()
        if len(counts) != 1:
            raise ValueError(
                f"Expected one parameter count for {student_size} students, "
                f"found {counts.tolist()}."
            )
        parameter_counts[student_size] = int(counts[0])

    cell_rows = {}
    for fraction in fractions:
        for student_size in student_sizes:
            cell = summary[
                (summary["fraction"] == fraction)
                & (summary["student_size"] == student_size)
            ]
            if len(cell) != 1:
                raise ValueError(
                    f"Expected one summary row for {fraction:.0%} labels and "
                    f"{student_size} students, found {len(cell)}."
                )
            cell_rows[(fraction, student_size)] = cell.iloc[0]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_positions = range(len(fractions))
    color_map = plt.get_cmap("tab10", len(fractions))
    colors = [color_map(index) for index in range(len(fractions))]
    student_hatches = {"small": "", "large": "///"}
    bar_width = 0.34
    student_spacing = 0.38
    for fraction_index, fraction in enumerate(fractions):
        for size_index, student_size in enumerate(student_sizes):
            cell = cell_rows[(fraction, student_size)]
            ax.bar(
                fraction_index
                + (size_index - (len(student_sizes) - 1) / 2) * student_spacing,
                cell["mean_gap_pp"],
                width=bar_width,
                yerr=0.0 if pd.isna(cell["std_gap_pp"]) else cell["std_gap_pp"],
                capsize=4,
                color=colors[fraction_index],
                edgecolor="black",
                linewidth=0.7,
                hatch=student_hatches[student_size],
                error_kw={"elinewidth": 1, "capthick": 1},
            )

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([f"{fraction:.0%}" for fraction in fractions])
    ax.set_xlabel("Labeled training fraction")
    ax.set_ylabel("C − B linear-SVM accuracy (percentage points)")
    ax.set_title(f"C − B gap by label fraction ({projector_type} projector)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fraction_handles = [
        Patch(
            facecolor=colors[index],
            edgecolor="black",
            label=f"{fraction:.0%}",
        )
        for index, fraction in enumerate(fractions)
    ]
    fraction_legend = ax.legend(
        handles=fraction_handles,
        title="Label fraction",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    ax.add_artist(fraction_legend)
    student_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=student_hatches[student_size],
            label=(
                f"{student_size.title()} = "
                f"{parameter_counts[student_size]:,} parameters"
            ),
        )
        for student_size in student_sizes
    ]
    ax.legend(
        handles=student_handles,
        title="Student",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.48),
    )
    fig.subplots_adjust(left=0.12, right=0.72, bottom=0.12, top=0.9)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"SVM result directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    projector_types = (
        ("mlp", "linear")
        if args.projector_type == "all"
        else (args.projector_type,)
    )
    projector_summaries = {}
    for projector_type in projector_types:
        results = load_probe_results(args.input_dir, projector_type)
        paired, summary = prepare_paired_gaps(results)
        projector_summaries[projector_type] = summary

        paired_path = (
            args.output_dir / f"distillation_gap_{projector_type}_paired.csv"
        )
        summary_path = (
            args.output_dir / f"distillation_gap_{projector_type}_summary.csv"
        )
        figure_path = args.output_dir / f"distillation_gap_{projector_type}.png"
        paired.to_csv(paired_path, index=False)
        summary.to_csv(summary_path, index=False)
        plot_gaps(summary, figure_path, projector_type)
        print_size_comparison(summary)

        print(f"Paired results saved to {paired_path}")
        print(f"Gap summary saved to {summary_path}")
        print(f"Gap plot saved to {figure_path}")

    if args.projector_type == "all":
        grid = build_ablation_grid(projector_summaries)
        csv_path = args.output_dir / "distillation_gap_ablation_grid.csv"
        markdown_path = args.output_dir / "distillation_gap_ablation_table.md"
        grid.to_csv(csv_path, index=False)
        write_ablation_markdown(grid, markdown_path)
        print(f"Ablation grid saved to {csv_path}")
        print(f"Report-ready ablation table saved to {markdown_path}")
        for row in grid.to_dict("records"):
            print(
                f"{row['fraction']:.0%} labels: projector change is "
                f"{row['same_direction_for_both_sizes']} "
                f"(small {row['linear_minus_mlp_small_pp']:+.2f} pp; "
                f"large {row['linear_minus_mlp_large_pp']:+.2f} pp)."
            )


if __name__ == "__main__":
    main()
