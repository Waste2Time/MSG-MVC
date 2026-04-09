from dataclasses import dataclass
from typing import Dict

import numpy as np


SYNTH_GROUP_NAMES = {
    0: "A_easy_consistent",
    1: "B_boundary_consistent",
    2: "C_boundary_conflict",
    3: "D_pseudo_hard",
}

PROXY_GROUP_NAMES = {
    -1: "Unassigned",
    0: "Easy-consistent",
    1: "Boundary-consistent",
    2: "Boundary-conflict",
    3: "Pseudo-hard",
}


@dataclass
class ProxyThresholds:
    margin_low: float
    margin_high: float
    disagree_low: float
    disagree_high: float
    density_low: float
    density_mid: float


def build_synthetic_oracle_groups(is_boundary: np.ndarray, conflict_mask: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
    groups = np.zeros(len(is_boundary), dtype=int)
    groups[is_boundary] = 1
    groups[is_boundary & conflict_mask] = 2
    groups[outlier_mask] = 3
    return groups


def group_stats(values: np.ndarray, groups: np.ndarray, name_map: Dict[int, str]) -> Dict[str, float]:
    out = {}
    for gid, gname in name_map.items():
        mask = groups == gid
        if mask.sum() == 0:
            continue
        out[f"{gname}_mean"] = float(values[mask].mean())
        out[f"{gname}_std"] = float(values[mask].std())
    return out


def build_proxy_groups(margin: np.ndarray, disagreement: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, ProxyThresholds]:
    th = ProxyThresholds(
        margin_low=float(np.quantile(margin, 0.30)),
        margin_high=float(np.quantile(margin, 0.70)),
        disagree_low=float(np.quantile(disagreement, 0.30)),
        disagree_high=float(np.quantile(disagreement, 0.70)),
        density_low=float(np.quantile(density, 0.30)),
        density_mid=float(np.quantile(density, 0.50)),
    )

    groups = np.full_like(margin, fill_value=-1, dtype=int)

    easy = (margin >= th.margin_high) & (disagreement <= th.disagree_low)
    b_consistent = (margin <= th.margin_low) & (disagreement <= th.disagree_low) & (density >= th.density_mid)
    b_conflict = (margin <= th.margin_low) & (disagreement >= th.disagree_high) & (density >= th.density_mid)
    pseudo = (margin <= th.margin_low) & (disagreement >= th.disagree_high) & (density <= th.density_low)

    groups[easy] = 0
    groups[b_consistent] = 1
    groups[b_conflict] = 2
    groups[pseudo] = 3
    return groups, th
