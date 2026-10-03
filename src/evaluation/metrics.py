"""Model evaluation metrics.

Deviance convention (used everywhere in this repo): the mean Poisson unit deviance per
policy, computed on claim COUNTS with predicted counts mu = exposure x predicted
frequency, unweighted:
    D = (2 / n) * sum[ y * log(y / mu) - (y - mu) ]
This equals the exposure-weighted deviance on frequencies divided by n, because the
Poisson deviance is homogeneous of degree one. No exposure weight is applied on top.
"""
import numpy as np
import pandas as pd


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def gamma_deviance(y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted mean Gamma unit deviance (weights = claim counts for average severities)."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)
    d = 2.0 * (-np.log(y / mu) + (y - mu) / mu)
    return float(np.sum(w * d) / np.sum(w))


def lorenz_gini(y: np.ndarray, exposure: np.ndarray, score: np.ndarray) -> float:
    """Gini index from the exposure-weighted Lorenz curve: policies sorted by predicted
    frequency (ascending), x = cumulative exposure share, y = cumulative claim share.
    Gini = 1 - 2 * area under the Lorenz curve. Higher = better risk ranking."""
    order = np.argsort(score, kind="mergesort")
    e, c = np.asarray(exposure, float)[order], np.asarray(y, float)[order]
    x = np.concatenate([[0], np.cumsum(e) / e.sum()])
    l = np.concatenate([[0], np.cumsum(c) / c.sum()])
    return float(1 - 2 * np.trapezoid(l, x))


def equal_exposure_bins(score: np.ndarray, exposure: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin index 0..n_bins-1 by sorted score, each bin holding ~equal exposure."""
    order = np.argsort(score, kind="mergesort")
    cum = np.cumsum(np.asarray(exposure, float)[order])
    bins_sorted = np.minimum((cum / cum[-1] * n_bins).astype(int), n_bins - 1)
    out = np.empty_like(bins_sorted)
    out[order] = bins_sorted
    return out


def lift_table(y, exposure, pred_freq, n_bins: int = 10) -> pd.DataFrame:
    b = equal_exposure_bins(pred_freq, exposure, n_bins)
    df = pd.DataFrame({"bin": b + 1, "y": y, "e": exposure, "mu": pred_freq * exposure})
    g = df.groupby("bin").agg(exposure=("e", "sum"), observed=("y", "sum"), predicted=("mu", "sum"))
    g["observed_frequency"] = g["observed"] / g["exposure"]
    g["predicted_frequency"] = g["predicted"] / g["exposure"]
    g["ae"] = g["observed"] / g["predicted"]
    return g.reset_index()


def double_lift_table(y, exposure, pred_a, pred_b, n_bins: int = 10) -> pd.DataFrame:
    """Sort by ratio pred_a / pred_b; per bin, observed and each model's predicted
    frequency. The model whose predicted line tracks observed better wins where they disagree."""
    b = equal_exposure_bins(np.asarray(pred_a) / np.asarray(pred_b), exposure, n_bins)
    df = pd.DataFrame({"bin": b + 1, "y": y, "e": exposure, "a": pred_a * exposure, "b": pred_b * exposure})
    g = df.groupby("bin").agg(exposure=("e", "sum"), observed=("y", "sum"), pred_a=("a", "sum"), pred_b=("b", "sum"))
    for c in ("observed", "pred_a", "pred_b"):
        g[f"{c}_frequency"] = g[c] / g["exposure"]
    g["ae_a"] = g["observed"] / g["pred_a"]
    g["ae_b"] = g["observed"] / g["pred_b"]
    return g.reset_index()


def ae_by_factor(df: pd.DataFrame, factor: str, pred_cols: dict) -> pd.DataFrame:
    """Actual vs expected claim counts by level of a factor. pred_cols maps model name to
    a column of predicted counts."""
    agg = {"exposure": ("Exposure", "sum"), "observed": ("ClaimNb", "sum")}
    agg.update({name: (col, "sum") for name, col in pred_cols.items()})
    g = df.groupby(factor, observed=True).agg(**agg)
    for name in pred_cols:
        g[f"ae_{name}"] = g["observed"] / g[name]
    g.insert(0, "factor", factor)
    return g.rename_axis("level").reset_index()
