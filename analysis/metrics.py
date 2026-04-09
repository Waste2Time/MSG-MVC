import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set python/numpy/torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def soft_assign_from_centers(features: np.ndarray, centers: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert distances to soft assignments with a temperature-softmax."""
    dists = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    logits = -dists / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs


def compute_margin(q: np.ndarray) -> np.ndarray:
    """top1(q) - top2(q)."""
    sorted_q = np.sort(q, axis=1)
    return sorted_q[:, -1] - sorted_q[:, -2]


def _safe_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


def compute_js_divergence(q1: np.ndarray, q2: np.ndarray, q: Optional[np.ndarray] = None) -> np.ndarray:
    """Generalized JS divergence among (q1, q2, q). If q is None, use pairwise JS(q1, q2)."""
    if q is None:
        m = 0.5 * (q1 + q2)
        return 0.5 * _safe_kl(q1, m) + 0.5 * _safe_kl(q2, m)

    m = (q1 + q2 + q) / 3.0
    return (_safe_kl(q1, m) + _safe_kl(q2, m) + _safe_kl(q, m)) / 3.0


def compute_knn_density(z: np.ndarray, k: int = 10) -> np.ndarray:
    """rho_i = 1 / mean distance to k nearest neighbors."""
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(z) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(z)
    dists, _ = nbrs.kneighbors(z)
    mean_knn = np.mean(dists[:, 1:], axis=1)
    return 1.0 / np.clip(mean_knn, 1e-8, None)


def compute_flip_rate(assign_prev: np.ndarray, assign_now: np.ndarray) -> np.ndarray:
    """Per-sample assignment flip indicator."""
    return (assign_prev != assign_now).astype(float)
