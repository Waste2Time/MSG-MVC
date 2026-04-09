from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.metrics import compute_js_divergence
from analysis.synth_pipeline import SYNTH_GROUPS, SyntheticConfig, generate_synthetic, run_clustering_metrics


def _sanity_checks() -> pd.DataFrame:
    q_same_1 = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    q_same_2 = q_same_1.copy()
    js_same = compute_js_divergence(q_same_1, q_same_2)

    q_conf_1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q_conf_2 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    js_conf = compute_js_divergence(q_conf_1, q_conf_2)

    return pd.DataFrame(
        {
            "case": ["same_0", "same_1", "conflict_0", "conflict_1"],
            "js": [js_same[0], js_same[1], js_conf[0], js_conf[1]],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug disagreement metric for synthetic groups.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results/synthetic_v2"))
    parser.add_argument("--samples-per-group", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sanity = _sanity_checks()
    sanity.to_csv(out_dir / "debug_disagreement_sanity.csv", index=False)

    data = generate_synthetic(SyntheticConfig(seed=args.seed))
    df = run_clustering_metrics(data, seed=args.seed)

    rows = []
    rng = np.random.default_rng(args.seed)
    for gid, gname in SYNTH_GROUPS.items():
        idx = np.where(df["group"].to_numpy() == gid)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(args.samples_per_group, len(idx)), replace=False)
        for i in pick:
            row = df.iloc[int(i)]
            rows.append(
                {
                    "sample_idx": int(row["sample_idx"]),
                    "group": int(row["group"]),
                    "group_name": gname,
                    "margin": float(row["margin"]),
                    "disagreement": float(row["disagreement"]),
                    "q1": np.array2string(np.asarray(row["q1"]), precision=4),
                    "q2": np.array2string(np.asarray(row["q2"]), precision=4),
                    "q": np.array2string(np.asarray(row["q"]), precision=4),
                }
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "debug_disagreement_raw_cases.csv", index=False)

    grp = df.groupby("group_name", as_index=False)["disagreement"].agg(["mean", "std", "median"]).reset_index()
    grp.to_csv(out_dir / "debug_disagreement_group_summary.csv", index=False)


if __name__ == "__main__":
    main()
