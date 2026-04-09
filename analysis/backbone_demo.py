from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis.grouping import PROXY_GROUP_NAMES, build_proxy_groups
from analysis.metrics import (
    compute_flip_rate,
    compute_js_divergence,
    compute_knn_density,
    compute_margin,
    set_global_seed,
    soft_assign_from_centers,
)
from analysis.plotting import save_barplot, save_proxy_distribution
from load_data import load_data


@dataclass
class BackboneConfig:
    dataset: str = "BDGP"
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 8
    pca_dim: int = 32
    rci_strength: float = 0.30
    k_neighbors: int = 10


def _prep_two_views(dataset: str, pca_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    views, y = load_data(dataset)
    if len(views) < 2:
        raise ValueError(f"Dataset {dataset} needs at least 2 views.")
    x1, x2 = views[0], views[1]
    x1 = StandardScaler().fit_transform(x1)
    x2 = StandardScaler().fit_transform(x2)

    dim1 = min(pca_dim, x1.shape[1], max(2, x1.shape[0] - 1))
    dim2 = min(pca_dim, x2.shape[1], max(2, x2.shape[0] - 1))
    x1 = PCA(n_components=dim1, random_state=0).fit_transform(x1)
    x2 = PCA(n_components=dim2, random_state=0).fit_transform(x2)

    n_clusters = len(np.unique(y))
    return x1, x2, np.asarray(y), n_clusters


def _align_dimensions(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = min(a.shape[1], b.shape[1])
    return a[:, :d], b[:, :d]


def _run_proxy_pipeline(
    x1: np.ndarray,
    x2: np.ndarray,
    n_clusters: int,
    seed: int,
    epochs: int,
    with_rci: bool,
    rci_strength: float,
    k_neighbors: int,
):
    x1, x2 = _align_dimensions(x1, x2)
    z = 0.5 * (x1 + x2)

    assignments_history = []
    q_hist = []
    q1_hist = []
    q2_hist = []

    for ep in range(epochs):
        km1 = KMeans(n_clusters=n_clusters, random_state=seed * 100 + ep + 1, n_init=10).fit(x1)
        km2 = KMeans(n_clusters=n_clusters, random_state=seed * 100 + ep + 2, n_init=10).fit(x2)
        q1 = soft_assign_from_centers(x1, km1.cluster_centers_, temperature=1.0)
        q2 = soft_assign_from_centers(x2, km2.cluster_centers_, temperature=1.0)

        q_fused = 0.5 * (q1 + q2)
        if with_rci:
            consensus = np.sqrt(np.clip(q1 * q2, 1e-12, None))
            q_fused = (1.0 - rci_strength) * q_fused + rci_strength * consensus

        q_fused = q_fused / np.clip(q_fused.sum(axis=1, keepdims=True), 1e-12, None)
        assign = np.argmax(q_fused, axis=1)

        # prototype refine (MSG-MVC style proxy): recompute fused centers by current pseudo-label
        centers = np.zeros((n_clusters, z.shape[1]))
        for c in range(n_clusters):
            idx = assign == c
            centers[c] = z[idx].mean(axis=0) if np.any(idx) else z[np.random.randint(0, len(z))]
        q_fused = soft_assign_from_centers(z, centers, temperature=1.0)

        assignments_history.append(assign)
        q_hist.append(q_fused)
        q1_hist.append(q1)
        q2_hist.append(q2)

    flips = []
    for i in range(1, len(assignments_history)):
        flips.append(compute_flip_rate(assignments_history[i - 1], assignments_history[i]))
    flip_rate = np.mean(np.vstack(flips), axis=0) if flips else np.zeros(len(z))

    q = q_hist[-1]
    q1 = q1_hist[-1]
    q2 = q2_hist[-1]
    margin = compute_margin(q)
    disagreement = compute_js_divergence(q1, q2, q)
    density = compute_knn_density(z, k=k_neighbors)
    loss_proxy = -np.sum(q * np.log(np.clip(0.5 * (q1 + q2), 1e-12, 1.0)), axis=1)
    grad_proxy = np.linalg.norm(q1 - q2, axis=1)

    return {
        "margin": margin,
        "disagreement": disagreement,
        "density": density,
        "flip_rate": flip_rate,
        "loss_proxy": loss_proxy,
        "grad_proxy": grad_proxy,
    }


def run_real_backbone_proxy(cfg: BackboneConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x1, x2, _, n_clusters = _prep_two_views(cfg.dataset, cfg.pca_dim)

    per_sample_records = []
    per_group_records = []

    for seed in cfg.seeds:
        set_global_seed(seed)
        for with_rci in [False, True]:
            run = _run_proxy_pipeline(
                x1=x1,
                x2=x2,
                n_clusters=n_clusters,
                seed=seed,
                epochs=cfg.epochs,
                with_rci=with_rci,
                rci_strength=cfg.rci_strength,
                k_neighbors=cfg.k_neighbors,
            )
            groups, th = build_proxy_groups(run["margin"], run["disagreement"], run["density"])

            tag = "with_rci" if with_rci else "without_rci"
            for i in range(len(groups)):
                per_sample_records.append(
                    {
                        "seed": seed,
                        "setting": tag,
                        "sample_idx": i,
                        "group": int(groups[i]),
                        "group_name": PROXY_GROUP_NAMES[int(groups[i])],
                        "margin": float(run["margin"][i]),
                        "disagreement": float(run["disagreement"][i]),
                        "density": float(run["density"][i]),
                        "flip_rate": float(run["flip_rate"][i]),
                        "loss_proxy": float(run["loss_proxy"][i]),
                        "grad_proxy": float(run["grad_proxy"][i]),
                    }
                )

            for gid, gname in PROXY_GROUP_NAMES.items():
                mask = groups == gid
                if not np.any(mask):
                    continue
                per_group_records.append(
                    {
                        "seed": seed,
                        "setting": tag,
                        "group": gid,
                        "group_name": gname,
                        "count": int(mask.sum()),
                        "flip_rate_mean": float(run["flip_rate"][mask].mean()),
                        "loss_proxy_mean": float(run["loss_proxy"][mask].mean()),
                        "grad_proxy_mean": float(run["grad_proxy"][mask].mean()),
                        "margin_mean": float(run["margin"][mask].mean()),
                        "disagreement_mean": float(run["disagreement"][mask].mean()),
                        "density_mean": float(run["density"][mask].mean()),
                        "margin_low_q30": th.margin_low,
                        "margin_high_q70": th.margin_high,
                        "disagree_low_q30": th.disagree_low,
                        "disagree_high_q70": th.disagree_high,
                        "density_low_q30": th.density_low,
                        "density_mid_q50": th.density_mid,
                    }
                )

    sample_df = pd.DataFrame(per_sample_records)
    group_df = pd.DataFrame(per_group_records)

    sample_df.to_csv(out_dir / "proxy_sample_metrics.csv", index=False)
    group_df.to_csv(out_dir / "proxy_group_summary_by_seed.csv", index=False)

    group_mean = (
        group_df.groupby(["setting", "group_name"])[["count", "flip_rate_mean", "loss_proxy_mean", "grad_proxy_mean"]]
        .agg(["mean", "std"])
        .round(6)
    )
    group_mean.to_csv(out_dir / "proxy_group_summary_mean_std.csv")

    # Figure 1: proxy distribution (first seed, w/o RCI)
    dist_slice = sample_df[(sample_df["seed"] == cfg.seeds[0]) & (sample_df["setting"] == "without_rci")]
    save_proxy_distribution(dist_slice["group"].to_numpy(), PROXY_GROUP_NAMES, out_dir / "proxy_group_distribution.png")

    # Figure 2: flip rate by group
    flip_df = group_df.groupby(["setting", "group_name"], as_index=False)["flip_rate_mean"].mean()
    save_barplot(flip_df, x="group_name", y="flip_rate_mean", hue="setting", out_path=out_dir / "flip_rate_by_group.png", title="Flip rate by proxy group")

    # Figure 3: loss/grad proxy by group
    lg_df = group_df.groupby(["setting", "group_name"], as_index=False)[["loss_proxy_mean", "grad_proxy_mean"]].mean()
    lg_long = lg_df.melt(id_vars=["setting", "group_name"], value_vars=["loss_proxy_mean", "grad_proxy_mean"], var_name="metric", value_name="value")
    save_barplot(lg_long, x="group_name", y="value", hue="setting", out_path=out_dir / "loss_or_grad_by_group.png", title="Loss/Grad proxies by group")

    # Figure 4: with vs without RCI comparison (boundary-conflict + pseudo-hard)
    focus = group_df[group_df["group_name"].isin(["Boundary-conflict", "Pseudo-hard"])]
    cmp_df = focus.groupby(["setting", "group_name"], as_index=False)[["flip_rate_mean", "loss_proxy_mean", "grad_proxy_mean"]].mean()
    cmp_long = cmp_df.melt(id_vars=["setting", "group_name"], value_vars=["flip_rate_mean", "loss_proxy_mean", "grad_proxy_mean"], var_name="metric", value_name="value")
    save_barplot(cmp_long, x="metric", y="value", hue="setting", out_path=out_dir / "with_without_rci_comparison.png", title="With vs Without RCI (focus groups)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-backbone proxy analysis demo.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results/backbone"))
    args = parser.parse_args()

    cfg = BackboneConfig(dataset=args.dataset, seeds=tuple(args.seeds), epochs=args.epochs)
    run_real_backbone_proxy(cfg, args.out_dir)


if __name__ == "__main__":
    main()
