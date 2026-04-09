from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from analysis.grouping import SYNTH_GROUP_NAMES, build_synthetic_oracle_groups
from analysis.metrics import (
    compute_js_divergence,
    compute_knn_density,
    compute_margin,
    set_global_seed,
    soft_assign_from_centers,
)
from analysis.plotting import save_group_boxplots, save_selection_curve, save_synthetic_scatter


@dataclass
class SynthConfig:
    seed: int = 42
    n_per_class: int = 500
    sigma: float = 0.7
    boundary_threshold: float = 0.5
    p_conflict: float = 0.4
    outlier_ratio: float = 0.06
    k_neighbors: int = 10


def _pairwise_center_margins(s: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.linalg.norm(s[:, None, :] - centers[None, :, :], axis=2)
    order = np.argsort(d, axis=1)
    d1 = d[np.arange(len(s)), order[:, 0]]
    d2 = d[np.arange(len(s)), order[:, 1]]
    return d1, d2, order


def generate_synthetic_data(cfg: SynthConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.8]], dtype=np.float64)

    labels = np.repeat(np.arange(3), cfg.n_per_class)
    semantic = np.vstack(
        [rng.normal(loc=centers[c], scale=cfg.sigma, size=(cfg.n_per_class, 2)) for c in range(3)]
    )

    d1, d2, nearest = _pairwise_center_margins(semantic, centers)
    delta_gt = d2 - d1
    is_boundary = delta_gt < cfg.boundary_threshold

    conflict_mask = np.zeros(len(semantic), dtype=bool)
    boundary_idx = np.where(is_boundary)[0]
    choose = rng.random(len(boundary_idx)) < cfg.p_conflict
    conflict_idx = boundary_idx[choose]
    conflict_mask[conflict_idx] = True

    direction = centers[nearest[:, 1]] - centers[nearest[:, 0]]
    direction = direction / np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8, None)
    alpha = rng.uniform(0.4, 0.8, size=(len(semantic), 1))
    semantic_view2 = semantic.copy()
    semantic_view2[conflict_mask] += alpha[conflict_mask] * direction[conflict_mask]

    outlier_count = int(len(semantic) * cfg.outlier_ratio)
    outlier_idx = rng.choice(len(semantic), size=outlier_count, replace=False)
    outlier_mask = np.zeros(len(semantic), dtype=bool)
    outlier_mask[outlier_idx] = True
    outlier_pts = rng.uniform(low=[-0.5, 0.6], high=[0.5, 1.6], size=(outlier_count, 2))
    semantic[outlier_mask] = outlier_pts
    semantic_view2[outlier_mask] = outlier_pts

    theta1, theta2 = np.deg2rad(25), np.deg2rad(-35)
    A1 = np.array([[1.2 * np.cos(theta1), -0.7 * np.sin(theta1)], [0.7 * np.sin(theta1), 1.1 * np.cos(theta1)]])
    A2 = np.array([[0.9 * np.cos(theta2), -1.0 * np.sin(theta2)], [0.5 * np.sin(theta2), 1.3 * np.cos(theta2)]])

    eps1 = rng.normal(0, 0.1, size=semantic.shape)
    eps2 = rng.normal(0, 0.1, size=semantic_view2.shape)

    x1 = semantic @ A1.T + eps1
    x2 = semantic_view2 @ A2.T + eps2

    groups = build_synthetic_oracle_groups(is_boundary, conflict_mask, outlier_mask)

    return {
        "x1": x1,
        "x2": x2,
        "semantic": semantic,
        "labels": labels,
        "groups": groups,
    }


def run_a1_a3(cfg: SynthConfig, out_dir: Path) -> None:
    set_global_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_data(cfg)
    x1, x2, groups = data["x1"], data["x2"], data["groups"]
    z = 0.5 * (x1 + x2)

    n_clusters = 3
    km_f = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=20)
    km_1 = KMeans(n_clusters=n_clusters, random_state=cfg.seed + 1, n_init=20)
    km_2 = KMeans(n_clusters=n_clusters, random_state=cfg.seed + 2, n_init=20)

    km_f.fit(z)
    km_1.fit(x1)
    km_2.fit(x2)

    q = soft_assign_from_centers(z, km_f.cluster_centers_, temperature=0.7)
    q1 = soft_assign_from_centers(x1, km_1.cluster_centers_, temperature=0.7)
    q2 = soft_assign_from_centers(x2, km_2.cluster_centers_, temperature=0.7)

    margin = compute_margin(q)
    disagreement = compute_js_divergence(q1, q2, q)
    density = compute_knn_density(z, k=cfg.k_neighbors)

    sample_df = pd.DataFrame(
        {
            "group": groups,
            "group_name": [SYNTH_GROUP_NAMES[g] for g in groups],
            "margin": margin,
            "disagreement": disagreement,
            "density": density,
        }
    )
    sample_df.to_csv(out_dir / "synthetic_sample_metrics.csv", index=False)

    summary_df = (
        sample_df.groupby("group_name")[["margin", "disagreement", "density"]]
        .agg(["mean", "std", "median"])
        .round(6)
    )
    summary_df.to_csv(out_dir / "synthetic_group_summary.csv")

    save_synthetic_scatter(z, groups, SYNTH_GROUP_NAMES, out_dir / "synthetic_groups_scatter.png")
    save_group_boxplots(
        sample_df,
        metrics=["margin", "disagreement", "density"],
        out_path=out_dir / "margin_disagreement_density_boxplots.png",
    )

    records = []
    n = len(sample_df)
    c_mask = groups == 2
    d_mask = groups == 3
    for strategy in ["naive", "bc_aware", "trusted"]:
        if strategy == "naive":
            score = 1.0 - margin
        elif strategy == "bc_aware":
            score = (1.0 - margin) * disagreement
        else:
            rho_norm = (density - density.min()) / np.clip(density.max() - density.min(), 1e-8, None)
            score = (1.0 - margin) * disagreement * rho_norm

        rank = np.argsort(-score)
        for k_ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            k = max(1, int(n * k_ratio))
            topk = rank[:k]
            precision = float(np.mean(c_mask[topk]))
            recall = float(np.sum(c_mask[topk]) / np.clip(np.sum(c_mask), 1, None))
            contamination = float(np.mean(d_mask[topk]))
            records.append(
                {
                    "strategy": strategy,
                    "k_ratio": k_ratio,
                    "precision_c": precision,
                    "recall_c": recall,
                    "contamination_d": contamination,
                }
            )

    curve_df = pd.DataFrame(records)
    curve_df.to_csv(out_dir / "selection_strategy_curves.csv", index=False)
    curve_df.groupby("strategy")[["precision_c", "recall_c", "contamination_d"]].mean().to_csv(
        out_dir / "selection_strategy_summary.csv"
    )

    save_selection_curve(curve_df, out_dir / "selection_precision_contamination.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic phenomenon study demo (A1/A3).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results/synthetic"))
    args = parser.parse_args()
    run_a1_a3(SynthConfig(seed=args.seed), args.out_dir)


if __name__ == "__main__":
    main()
