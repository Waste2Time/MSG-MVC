from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.metrics import set_global_seed
from analysis.synth_pipeline import SYNTH_GROUPS, SyntheticConfig, generate_synthetic, run_clustering_metrics, save_dataset_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate synthetic data with D1/D2 pseudo-hard variants.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results/synthetic_v2"))
    parser.add_argument("--boundary-threshold", type=float, default=0.85)
    parser.add_argument("--conflict-ratio", type=float, default=0.65)
    parser.add_argument("--d1-ratio", type=float, default=0.06)
    parser.add_argument("--d2-ratio", type=float, default=0.06)
    args = parser.parse_args()

    cfg = SyntheticConfig(
        seed=args.seed,
        boundary_threshold=args.boundary_threshold,
        conflict_ratio=args.conflict_ratio,
        d1_ratio=args.d1_ratio,
        d2_ratio=args.d2_ratio,
    )
    set_global_seed(cfg.seed)

    data = generate_synthetic(cfg)
    df = run_clustering_metrics(data, seed=cfg.seed, k_neighbors=cfg.k_neighbors, temp=cfg.temp)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dataset_npz(data, out_dir / "synthetic_dataset_v2.npz")

    # Metrics CSV (excluding vector columns for readability)
    scalar_cols = ["sample_idx", "group", "group_name", "label", "pred", "boundary_mask", "margin", "disagreement", "density"]
    df[scalar_cols].to_csv(out_dir / "synthetic_metrics_v2.csv", index=False)

    counts = df.groupby("group_name", as_index=False).size().rename(columns={"size": "count"})
    counts["ratio"] = counts["count"] / len(df)
    counts.to_csv(out_dir / "synthetic_group_counts_v2.csv", index=False)

    # quick checklist for next-step acceptance
    pivot = counts.set_index("group_name")
    checklist = {
        "B_count": int(pivot.loc[SYNTH_GROUPS[1], "count"]) if SYNTH_GROUPS[1] in pivot.index else 0,
        "C_count": int(pivot.loc[SYNTH_GROUPS[2], "count"]) if SYNTH_GROUPS[2] in pivot.index else 0,
    }
    pd.DataFrame([checklist]).to_csv(out_dir / "synthetic_checklist_v2.csv", index=False)


if __name__ == "__main__":
    main()
