"""Plotting helpers. All figures are written to reports/figures/ as PNG."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
    }
)

COLORS = {"obs": "#222222", "glm_a": "#1f77b4", "glm_b": "#2ca02c", "gbm": "#d62728", "other": "#7f7f7f"}


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def severity_tail_plots(amounts: np.ndarray, thresholds: list[float], chosen: float | None, path: Path) -> Path:
    """Mean excess plot, log-log empirical survival, and cumulative share of the largest claims."""
    x = np.sort(np.asarray(amounts, dtype=float))
    n = len(x)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # Mean excess e(u) = E[X - u | X > u] at every order statistic from the median up to
    # the 20th largest claim (beyond that the average rests on too few claims)
    tail_sum = np.cumsum(x[::-1])[::-1]  # sum of x[i:]
    idx = np.arange(n // 2, n - 20)
    us = x[idx]
    me = (tail_sum[idx + 1] / (n - idx - 1)) - us
    ax = axes[0]
    ax.plot(us / 1000, me / 1000, color=COLORS["obs"], lw=1)
    ax.set_xlabel("threshold u (thousands)")
    ax.set_ylabel("mean excess E[X-u | X>u] (thousands)")
    ax.set_title("Mean excess plot")

    ax = axes[1]
    surv = 1.0 - np.arange(n) / n
    ax.loglog(x, surv, ".", ms=2, color=COLORS["obs"])
    ax.set_xlabel("claim amount (log)")
    ax.set_ylabel("P(X > x) (log)")
    ax.set_title("Log-log empirical survival")

    ax = axes[2]
    desc = x[::-1]
    k = np.arange(1, n + 1)
    ax.semilogx(k / n * 100, np.cumsum(desc) / desc.sum() * 100, color=COLORS["obs"])
    ax.set_xlabel("largest claims, % of claim count (log)")
    ax.set_ylabel("% of total claim amount")
    ax.set_title("Concentration of claim amount")

    for u in thresholds:
        style = dict(color=COLORS["gbm"], lw=1.2) if u == chosen else dict(color=COLORS["other"], lw=0.6, ls=":")
        axes[0].axvline(u / 1000, **style)
        axes[1].axvline(u, **style)
    return save(fig, path)


def bar_line(table: pd.DataFrame, x: str, bar: str, lines: dict, path: Path, title: str, ylabel: str) -> Path:
    """Exposure bars (right axis) with one or more line series (left axis) by level of x."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax2 = ax.twinx()
    pos = np.arange(len(table))
    ax2.bar(pos, table[bar], color="#dddddd", zorder=0)
    ax2.set_ylabel(bar)
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for label, (col, color) in lines.items():
        ax.plot(pos, table[col], marker="o", ms=3, color=color, label=label)
    ax.set_xticks(pos)
    ax.set_xticklabels(table[x].astype(str), rotation=60, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    return save(fig, path)
