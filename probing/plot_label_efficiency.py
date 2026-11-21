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
        default=Path("runs/probes"),
        help="Directory searched for probe summary CSV files.",
    )
    parser.add_argument(
        "--probe",
        choices=("linear", "rbf", "knn"),
        default="linear",
        help="Probe to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/probes/label_efficiency"),
    )
    return parser.parse_args()


def find_summaries(args):
    if args.summaries:
        return [path for path in args.summaries if path.exists()]
    if args.probe == "knn":
        return sorted(args.input_dir.glob("knn_evaluation_*_summary.csv"))
    return sorted(args.input_dir.glob(f"svm_evaluation_*_{args.probe}_summary.csv"))


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


def plot_results(results, output_path, probe="linear"):
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
    ax.set_title(f"{probe_label} label-efficiency curves")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Encoder")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
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

    print(f"Combined label-efficiency table saved to {table_path}")
    print(f"Label-efficiency figure saved to {figure_path}")


if __name__ == "__main__":
    main()
