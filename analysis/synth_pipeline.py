from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from analysis.metrics import compute_js_divergence, compute_knn_density, compute_margin, soft_assign_from_centers


@dataclass
class SyntheticConfig:
    seed: int = 42
    n_per_class: int = 500
    sigma: float = 0.7
    boundary_threshold: float = 0.85
    conflict_ratio: float = 0.65
    d1_ratio: float = 0.06
    d2_ratio: float = 0.06
    k_neighbors: int = 10
    temp: float = 0.7


SYNTH_GROUPS = {
    0: "A_easy_consistent",
    1: "B_boundary_consistent",
    2: "C_boundary_conflict",
    3: "D1_outlier_pseudo_hard",
    4: "D2_view_corrupted_pseudo_hard",
}


def _rotation(theta_deg: float, sx: float, sy: float) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    return np.array([[sx * np.cos(th), -sy * np.sin(th)], [sx * np.sin(th), sy * np.cos(th)]], dtype=np.float64)


def generate_synthetic(cfg: SyntheticConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.8]], dtype=np.float64)

    y = np.repeat(np.arange(3), cfg.n_per_class)
    s = np.vstack([rng.normal(loc=centers[c], scale=cfg.sigma, size=(cfg.n_per_class, 2)) for c in range(3)])

    # Oracle boundary using GT centers.
    dist = np.linalg.norm(s[:, None, :] - centers[None, :, :], axis=2)
    order = np.argsort(dist, axis=1)
    d1 = dist[np.arange(len(s)), order[:, 0]]
    d2 = dist[np.arange(len(s)), order[:, 1]]
    delta = d2 - d1
    is_boundary = delta < cfg.boundary_threshold

    # Conflict injection near boundary.
    conflict_mask = np.zeros(len(s), dtype=bool)
    b_idx = np.where(is_boundary)[0]
    conf_pick = rng.random(len(b_idx)) < cfg.conflict_ratio
    conflict_idx = b_idx[conf_pick]
    conflict_mask[conflict_idx] = True

    direction = centers[order[:, 1]] - centers[order[:, 0]]
    direction = direction / np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8, None)
    alpha = rng.uniform(0.5, 1.0, size=(len(s), 1))

    s_v2 = s.copy()
    s_v2[conflict_mask] = s_v2[conflict_mask] + alpha[conflict_mask] * direction[conflict_mask]

    # D1: low-density outliers far from main manifold.
    d1_count = int(len(s) * cfg.d1_ratio)
    d1_idx = rng.choice(np.arange(len(s)), size=d1_count, replace=False)
    d1_mask = np.zeros(len(s), dtype=bool)
    d1_mask[d1_idx] = True
    hole_candidates = np.vstack(
        [
            rng.uniform(low=[-4.8, -3.5], high=[-3.6, -2.2], size=(d1_count // 2 + 1, 2)),
            rng.uniform(low=[3.2, -3.2], high=[4.8, -1.8], size=(d1_count // 2 + 1, 2)),
            rng.uniform(low=[-0.8, 4.2], high=[0.8, 5.2], size=(d1_count // 2 + 1, 2)),
        ]
    )
    outlier_pts = hole_candidates[:d1_count]
    s[d1_mask] = outlier_pts
    s_v2[d1_mask] = outlier_pts + rng.normal(0, 0.05, size=(d1_count, 2))

    # D2: one-view corruption from normal samples.
    pool = np.where(~d1_mask)[0]
    d2_count = int(len(s) * cfg.d2_ratio)
    d2_idx = rng.choice(pool, size=d2_count, replace=False)
    d2_mask = np.zeros(len(s), dtype=bool)
    d2_mask[d2_idx] = True

    A1 = _rotation(theta_deg=22, sx=1.2, sy=0.9)
    A2 = _rotation(theta_deg=-33, sx=0.95, sy=1.25)

    x1 = s @ A1.T + rng.normal(0, 0.1, size=s.shape)
    x2 = s_v2 @ A2.T + rng.normal(0, 0.1, size=s.shape)

    # Strong single-view corruption for D2.
    x2[d2_mask] = x2[d2_mask] + rng.normal(0.0, 1.4, size=(d2_count, 2))

    groups = np.zeros(len(s), dtype=int)
    groups[is_boundary] = 1
    groups[is_boundary & conflict_mask] = 2
    groups[d1_mask] = 3
    groups[d2_mask] = 4

    boundary_oracle = is_boundary | (groups == 2)

    return {
        "x1": x1,
        "x2": x2,
        "semantic": s,
        "y": y,
        "groups": groups,
        "boundary_mask": boundary_oracle.astype(np.uint8),
    }


def run_clustering_metrics(data: dict[str, np.ndarray], seed: int, k_neighbors: int = 10, temp: float = 0.7) -> pd.DataFrame:
    x1, x2, groups = data["x1"], data["x2"], data["groups"]
    z = 0.5 * (x1 + x2)
    n_clusters = len(np.unique(data["y"]))

    km_f = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z)
    km1 = KMeans(n_clusters=n_clusters, random_state=seed + 1, n_init=20).fit(x1)
    km2 = KMeans(n_clusters=n_clusters, random_state=seed + 2, n_init=20).fit(x2)

    q = soft_assign_from_centers(z, km_f.cluster_centers_, temperature=temp)
    q1 = soft_assign_from_centers(x1, km1.cluster_centers_, temperature=temp)
    q2 = soft_assign_from_centers(x2, km2.cluster_centers_, temperature=temp)

    margin = compute_margin(q)
    disagreement = compute_js_divergence(q1, q2)  # intentionally pairwise JS per next-step guideline
    density = compute_knn_density(z, k=k_neighbors)

    df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(groups)),
            "group": groups,
            "group_name": [SYNTH_GROUPS[g] for g in groups],
            "label": data["y"],
            "margin": margin,
            "disagreement": disagreement,
            "density": density,
            "pred": np.argmax(q, axis=1),
            "boundary_mask": data["boundary_mask"],
        }
    )

    df["q"] = list(q)
    df["q1"] = list(q1)
    df["q2"] = list(q2)
    return df


def save_dataset_npz(data: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
