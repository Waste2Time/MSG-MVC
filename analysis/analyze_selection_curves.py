from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze top-k selection quality for naive/bc-aware/trusted.")
    parser.add_argument("--metrics-csv", type=Path, default=Path("analysis/results/synthetic_v2/synthetic_metrics_v2.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results/synthetic_v2"))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    margin = df["margin"].to_numpy()
    disagreement = df["disagreement"].to_numpy()
    density = df["density"].to_numpy()
    groups = df["group"].to_numpy()

    c_mask = groups == 2
    d_mask = (groups == 3) | (groups == 4)

    rho_norm = (density - density.min()) / max((density.max() - density.min()), 1e-12)

    scores = {
        "naive": 1.0 - margin,
        "bc_aware": (1.0 - margin) * disagreement,
        "trusted_exploratory": (1.0 - margin) * disagreement * rho_norm,
    }

    rows = []
    for name, score in scores.items():
        rank = score.argsort()[::-1]
        for k_ratio in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]:
            k = max(1, int(len(df) * k_ratio))
            topk = rank[:k]
            rows.append(
                {
                    "strategy": name,
                    "k_ratio": k_ratio,
                    "precision_c": float(c_mask[topk].mean()),
                    "recall_c": float(c_mask[topk].sum() / max(c_mask.sum(), 1)),
                    "contamination_d": float(d_mask[topk].mean()),
                }
            )

    curves = pd.DataFrame(rows)
    curves.to_csv(out_dir / "selection_curves_v2.csv", index=False)

    summary = curves.groupby("strategy")[["precision_c", "recall_c", "contamination_d"]].agg(["mean", "max"]).round(6)
    summary.to_csv(out_dir / "selection_summary_v2.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    sns.lineplot(data=curves, x="k_ratio", y="precision_c", hue="strategy", marker="o", ax=axes[0])
    sns.lineplot(data=curves, x="k_ratio", y="recall_c", hue="strategy", marker="o", ax=axes[1])
    sns.lineplot(data=curves, x="k_ratio", y="contamination_d", hue="strategy", marker="o", ax=axes[2])
    axes[0].set_title("Precision@k (Group C)")
    axes[1].set_title("Recall@k (Group C)")
    axes[2].set_title("Contamination@k (Group D1+D2)")
    fig.savefig(out_dir / "selection_precision_recall_contamination_v2.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
