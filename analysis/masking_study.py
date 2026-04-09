from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from analysis.metrics import compute_js_divergence, compute_margin, soft_assign_from_centers


def _run_short_refine(x1, x2, labels, boundary_mask, mask, seed: int, epochs: int = 8):
    z = 0.5 * (x1 + x2)
    n_clusters = len(np.unique(labels))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z)
    centers = km.cluster_centers_

    for ep in range(epochs):
        q1 = soft_assign_from_centers(x1, centers, temperature=1.0)
        q2 = soft_assign_from_centers(x2, centers, temperature=1.0)
        q = 0.5 * (q1 + q2)
        q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)
        assign = np.argmax(q, axis=1)

        # update centers but ignore masked samples
        for c in range(n_clusters):
            idx = (assign == c) & (~mask)
            if np.any(idx):
                centers[c] = z[idx].mean(axis=0)

    q1 = soft_assign_from_centers(x1, centers, temperature=1.0)
    q2 = soft_assign_from_centers(x2, centers, temperature=1.0)
    q = 0.5 * (q1 + q2)
    q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)
    pred = np.argmax(q, axis=1)

    b = boundary_mask.astype(bool)
    valid = b & (~mask)
    if np.sum(valid) == 0:
        return {"boundary_ari": np.nan, "boundary_margin": np.nan, "boundary_disagreement": np.nan}

    return {
        "boundary_ari": float(adjusted_rand_score(labels[valid], pred[valid])),
        "boundary_margin": float(compute_margin(q)[valid].mean()),
        "boundary_disagreement": float(compute_js_divergence(q1, q2)[valid].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Masking study for synthetic groups.")
    parser.add_argument("--dataset-npz", type=Path, default=Path("analysis/results/synthetic_v2/synthetic_dataset_v2.npz"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("analysis/results/synthetic_v2/synthetic_metrics_v2.csv"))
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results/synthetic_v2"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = np.load(args.dataset_npz)
    x1, x2 = ds["x1"], ds["x2"]
    labels = ds["y"]
    boundary_mask = ds["boundary_mask"]

    metrics = pd.read_csv(args.metrics_csv)
    groups = metrics["group"].to_numpy()

    rows = []
    group_ids = [0, 1, 2, 3, 4]

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        for gid in group_ids:
            idx = np.where(groups == gid)[0]
            if len(idx) == 0:
                continue
            k = max(1, int(len(idx) * args.mask_ratio))
            masked = rng.choice(idx, size=k, replace=False)
            mask = np.zeros(len(groups), dtype=bool)
            mask[masked] = True

            res = _run_short_refine(x1, x2, labels, boundary_mask, mask, seed=seed, epochs=8)
            res.update({"seed": seed, "masked_group": gid, "masked_count": int(k)})
            rows.append(res)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "masking_study_results_v2.csv", index=False)

    avg = out.groupby("masked_group", as_index=False)[["boundary_ari", "boundary_margin", "boundary_disagreement"]].mean()
    avg.to_csv(out_dir / "masking_study_summary_v2.csv", index=False)

    long_df = avg.melt(id_vars=["masked_group"], value_vars=["boundary_ari", "boundary_margin", "boundary_disagreement"], var_name="metric", value_name="value")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=long_df, x="masked_group", y="value", hue="metric")
    plt.title("Masking impact by group (boundary metrics)")
    plt.tight_layout()
    plt.savefig(out_dir / "masking_drop_barplot_v2.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
