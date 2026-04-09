from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_synthetic_scatter(z: np.ndarray, groups: np.ndarray, group_names: Dict[int, str], out_path: Path) -> None:
    _ensure_parent(out_path)
    plt.figure(figsize=(7, 6))
    for gid, gname in group_names.items():
        mask = groups == gid
        if np.sum(mask) == 0:
            continue
        plt.scatter(z[mask, 0], z[mask, 1], s=12, alpha=0.7, label=gname)
    plt.legend(fontsize=8)
    plt.title("Synthetic latent scatter by oracle group")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_group_boxplots(df: pd.DataFrame, metrics: list[str], out_path: Path, group_col: str = "group_name") -> None:
    _ensure_parent(out_path)
    ncols = len(metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df, x=group_col, y=metric, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
        ax.set_title(f"{metric} by group")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_selection_curve(curve_df: pd.DataFrame, out_path: Path) -> None:
    _ensure_parent(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    sns.lineplot(data=curve_df, x="k_ratio", y="precision_c", hue="strategy", marker="o", ax=axes[0])
    sns.lineplot(data=curve_df, x="k_ratio", y="contamination_d", hue="strategy", marker="o", ax=axes[1])
    axes[0].set_title("Precision@k for Group C")
    axes[1].set_title("Contamination@k by Group D")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_barplot(df: pd.DataFrame, x: str, y: str, hue: str, out_path: Path, title: str) -> None:
    _ensure_parent(out_path)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_proxy_distribution(groups: np.ndarray, group_names: Dict[int, str], out_path: Path) -> None:
    _ensure_parent(out_path)
    labels = [group_names[g] for g in groups]
    s = pd.Series(labels).value_counts().reset_index()
    s.columns = ["group", "count"]
    plt.figure(figsize=(8, 4))
    sns.barplot(data=s, x="group", y="count")
    plt.xticks(rotation=20, ha="right")
    plt.title("Proxy group distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
