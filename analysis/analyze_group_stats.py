from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze synthetic group stats and draw boxplots.")
    parser.add_argument("--metrics-csv", type=Path, default=Path("analysis/results/synthetic_v2/synthetic_metrics_v2.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results/synthetic_v2"))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("group_name")[["margin", "disagreement", "density"]]
        .agg(["mean", "std", "median", "min", "max"])
        .round(6)
    )
    summary.to_csv(out_dir / "group_wise_summary_v2.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, metric in zip(axes, ["margin", "disagreement", "density"]):
        sns.boxplot(data=df, x="group_name", y=metric, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
        ax.set_title(f"{metric} by group")
    fig.savefig(out_dir / "margin_disagreement_density_boxplots_v2.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
