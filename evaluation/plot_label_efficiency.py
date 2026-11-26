import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SVM or KNN label-efficiency curves from probe summaries."
    )
    parser.add_argument(
        "--summary",
        dest="summaries",
        nargs="+",
        type=Path,
        help="SVM summary CSV files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/runs/probes"),
        help="Directory searched for probe summary CSV files.",
    )
    parser.add_argument(
        "--probe",
        choices=("linear", "rbf", "knn"),
        default="linear",
        help="Probe to plot.",
    )
    parser.add_argument(
        "--combined-mean-figure",
        action="store_true",
        help="Combine all three seed-mean probe curves in one figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/probes/label_efficiency"),
    )
    return parser.parse_args()


def find_summaries(args):
    if args.summaries:
        return [path for path in args.summaries if path.exists()]
    if args.probe == "knn":
        summaries = set(args.input_dir.rglob("knn_evaluation_*_summary.csv"))
        summaries.update(
            path
            for path in args.input_dir.rglob("summary.csv")
            if "knn_evaluation" in path.parts
        )
        return sorted(summaries)
    summaries = set(
        args.input_dir.rglob(f"svm_evaluation_*_{args.probe}_summary.csv")
    )
    summaries.update(
        path
        for path in args.input_dir.rglob("summary.csv")
        if "svm_evaluation" in path.parts and args.probe in path.parts
    )
    return sorted(summaries)


def load_results(paths, probe="linear"):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"encoder", "fraction", "mean_accuracy", "std_accuracy"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        probe_name = "knn" if probe == "knn" else f"{probe}_svm"
        if "probe" in frame.columns:
            frame = frame[frame["probe"] == probe_name]

        frame = frame.copy()
        frame["source"] = path.name
        frames.append(frame)

    if not frames:
        raise ValueError(f"No {probe.upper()} summary CSV files found.")

    results = pd.concat(frames, ignore_index=True)
    results["fraction"] = results["fraction"].astype(float)
    results["mean_accuracy"] = results["mean_accuracy"].astype(float)
    results["std_accuracy"] = results["std_accuracy"].fillna(0.0).astype(float)
    return results.sort_values(["encoder", "fraction"])


def plot_results(results, output_path, probe="linear", mean_across_seeds=False):
    fig, ax = plt.subplots(figsize=(8, 5))

    for encoder, encoder_results in results.groupby("encoder", sort=True):
        ax.errorbar(
            encoder_results["fraction"],
            encoder_results["mean_accuracy"],
            yerr=encoder_results["std_accuracy"],
            marker="o",
            capsize=4,
            linewidth=2,
            label=encoder,
        )

    ax.set_xscale("log")
    fractions = sorted(results["fraction"].unique())
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{fraction:.0%}" for fraction in fractions])
    ax.set_xlabel("Labeled training fraction")
    probe_label = "KNN" if probe == "knn" else f"{probe.upper()}-SVM"
    ax.set_ylabel(f"{probe_label} test accuracy")
    title = f"{probe_label} label-efficiency"
    ax.set_title(
        f"{title} (mean across seeds)"
        if mean_across_seeds
        else f"{title} curves"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Encoder")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_mean_curves(output_dir):
    probes = ("knn", "linear", "rbf")
    frames = {}
    for probe in probes:
        path = output_dir / f"label_efficiency_{probe}_seed_mean_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Seed-mean summary for {probe.upper()} not found: {path}"
            )
        frame = pd.read_csv(path)
        required = {"encoder", "fraction", "mean_accuracy", "std_accuracy"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames[probe] = frame

    reference_encoders = set(frames["knn"]["encoder"])
    reference_fractions = set(frames["knn"]["fraction"])
    for probe, frame in frames.items():
        if set(frame["encoder"]) != reference_encoders:
            raise ValueError(
                f"{probe.upper()} seed-mean summary has different encoders."
            )
        if set(frame["fraction"]) != reference_fractions:
            raise ValueError(
                f"{probe.upper()} seed-mean summary has different label fractions."
            )

    encoders = sorted(reference_encoders)
    colors = plt.get_cmap("tab10")
    color_by_encoder = {
        encoder: colors(index % colors.N)
        for index, encoder in enumerate(encoders)
    }
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    probe_titles = {
        "knn": "k-NN",
        "linear": "Linear SVM",
        "rbf": "RBF SVM",
    }
    for axis, probe in zip(axes, probes):
        frame = frames[probe]
        for encoder in encoders:
            encoder_results = frame[frame["encoder"] == encoder].sort_values(
                "fraction"
            )
            axis.errorbar(
                encoder_results["fraction"],
                encoder_results["mean_accuracy"],
                yerr=encoder_results["std_accuracy"],
                marker="o",
                capsize=3,
                linewidth=2,
                color=color_by_encoder[encoder],
                label=encoder,
            )
        axis.set_xscale("log")
        axis.set_xticks(sorted(reference_fractions))
        axis.set_xticklabels(
            [f"{fraction:.0%}" for fraction in sorted(reference_fractions)]
        )
        axis.set_title(probe_titles[probe])
        axis.set_xlabel("Labeled training fraction")
        axis.set_ylabel("Test accuracy")
        axis.grid(True, which="both", linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Encoder",
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle("Label-efficiency curves (mean across seeds)", y=1.02)
    figure.tight_layout(rect=(0, 0.18, 1, 0.98))
    output_path = output_dir / "label_efficiency_mean_curves_all_probes.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def summarize_seed_results(results):
    seed_info = results["encoder"].str.extract(
        r"^(?P<base>.+)_seed(?P<seed>\d+)$"
    )
    seeded = results[seed_info["base"].notna()].copy()
    seeded["summary_encoder"] = seed_info.loc[seeded.index, "base"]
    seeded["seed"] = seed_info.loc[seeded.index, "seed"].astype(int)

    per_seed = (
        seeded.groupby(
            ["summary_encoder", "seed", "fraction"],
            as_index=False,
        )
        .agg(mean_accuracy=("mean_accuracy", "mean"))
    )
    summary = (
        per_seed.groupby(
            ["summary_encoder", "fraction"],
            as_index=False,
        )
        .agg(
            mean_accuracy=("mean_accuracy", "mean"),
            std_accuracy=("mean_accuracy", "std"),
            seed_count=("seed", "nunique"),
        )
        .rename(columns={"summary_encoder": "encoder"})
    )
    summary["std_accuracy"] = summary["std_accuracy"].fillna(0.0)

    seeded_encoders = set(summary["encoder"])
    fixed = results[
        ~seed_info["base"].notna()
        & ~results["encoder"].isin(seeded_encoders)
    ]
    if not fixed.empty:
        fixed_summary = (
            fixed.groupby(["encoder", "fraction"], as_index=False)
            .agg(mean_accuracy=("mean_accuracy", "mean"))
            .assign(std_accuracy=0.0, seed_count=0)
        )
        summary = pd.concat([summary, fixed_summary], ignore_index=True)

    return summary.sort_values(["encoder", "fraction"]).reset_index(drop=True)


def main():
    args = parse_args()
    if args.combined_mean_figure:
        output_path = plot_combined_mean_curves(args.output_dir)
        print(f"Combined seed-mean figure saved to {output_path}")
        return

    paths = find_summaries(args)
    if not paths:
        raise SystemExit(
            f"No {args.probe.upper()} summary CSVs found in "
            f"{args.input_dir} or --summary."
        )

    results = load_results(paths, args.probe)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_path = args.output_dir / f"label_efficiency_{args.probe}_summary.csv"
    results.to_csv(table_path, index=False)

    figure_path = args.output_dir / f"label_efficiency_{args.probe}_curves.png"
    plot_results(results, figure_path, args.probe)

    seed_summary = summarize_seed_results(results)
    seed_summary_path = (
        args.output_dir
        / f"label_efficiency_{args.probe}_seed_mean_summary.csv"
    )
    seed_summary.to_csv(seed_summary_path, index=False)
    seed_mean_figure_path = (
        args.output_dir
        / f"label_efficiency_{args.probe}_mean_curves.png"
    )
    plot_results(
        seed_summary,
        seed_mean_figure_path,
        args.probe,
        mean_across_seeds=True,
    )

    print(f"Combined label-efficiency table saved to {table_path}")
    print(f"Label-efficiency figure saved to {figure_path}")
    print(f"Seed-mean summary saved to {seed_summary_path}")
    print(f"Seed-mean figure saved to {seed_mean_figure_path}")


if __name__ == "__main__":
    main()
