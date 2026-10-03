"""Stage 4 (part 1): SHAP interaction analysis of the tuned GBM.

- Subsample: stratified by claim indicator from the learn set. Size from the D007 rule:
  min(sample_size_max, budget / cost per row), cost per row scaled from the benchmark
  by the tuned model's tree count.
- Pair strength: mean over rows of |phi_ij| + |phi_ji| = 2 |phi_ij| (log-frequency scale).
- Cross-check: Friedman's H**2 for the top pairs on a smaller subsample, computed on the
  raw (log) score with centred partial dependence.
- Evidence from the data itself: two-way A/E heatmaps of observed claims against
  GLM-A out-of-fold predicted claims on the learn set. GLM-A has main effects only, so
  a systematic pattern in a two-way A/E is an interaction the GLM misses.
"""
import itertools
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.config import load_config
from src.evaluation.plots import COLORS, plt, save
from src.models.gbm import features

# Display names of the banded GLM factor for each GBM feature (for A/E heatmaps)
BANDED = {"DrivAge": "DrivAge_band", "VehAge": "VehAge_band", "VehPower": "VehPower_band",
          "BonusMalus": "BonusMalus_band", "LogDensity": "LogDensity_band", "VehGas": "VehGas",
          "VehBrand": "VehBrand_grp", "Region": "Region_grp"}


def sample_size(cfg: dict, n_trees: int) -> tuple[int, float]:
    bench = pd.read_csv(cfg["paths"]["tables"] / "shap_runtime_benchmark.csv")
    row = bench[(bench["step"] == "shap_interaction_values")].sort_values("rows").iloc[-1]
    per_row = row["seconds"] / row["rows"] * n_trees / row["n_trees"]
    n = int(min(cfg["shap"]["sample_size_max"], cfg["shap"]["runtime_budget_seconds"] / per_row))
    return n, per_row


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    claim = df["ClaimNb"] > 0
    n_claim = int(round(n * claim.mean()))
    return pd.concat([df[claim].sample(n_claim, random_state=seed), df[~claim].sample(n - n_claim, random_state=seed)])


def pair_strength(iv: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        v = iv[:, i, j] + iv[:, j, i]
        rows.append({"feature_1": names[i], "feature_2": names[j], "mean_abs_interaction": float(np.abs(v).mean()),
                     "sd_interaction": float(v.std())})
    out = pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    main = np.abs(iv[:, range(len(names)), range(len(names))]).mean(axis=0)
    out["mean_abs_main_effect_1"] = out["feature_1"].map(dict(zip(names, main)))
    out["mean_abs_main_effect_2"] = out["feature_2"].map(dict(zip(names, main)))
    return out


def friedman_h2(model: lgb.Booster, X: pd.DataFrame, f1: str, f2: str) -> float:
    """H**2_jk = sum (PD_jk - PD_j - PD_k)**2 / sum PD_jk**2, centred PDs at the sample points."""
    n = len(X)

    def pd_at(cols):
        vals = X[cols].to_numpy()
        out = np.empty(n)
        for r in range(n):
            Xr = X.copy()
            Xr[cols] = vals[r]
            out[r] = model.predict(Xr, raw_score=True).mean()
        return out - out.mean()

    pj, pk, pjk = pd_at([f1]), pd_at([f2]), pd_at([f1, f2])
    return float(np.sum((pjk - pj - pk) ** 2) / np.sum(pjk**2))


def oof_glm_a(learn: pd.DataFrame, cv_fits: list, n_folds: int) -> np.ndarray:
    pred = np.empty(len(learn))
    for k in range(n_folds):
        m = (learn["fold"] == k).to_numpy()
        pred[m] = cv_fits[k].predict(learn[m], np.log(learn.loc[m, "Exposure"].to_numpy()))
    return pred


def two_way_ae(learn: pd.DataFrame, f1: str, f2: str, pred_col: str) -> pd.DataFrame:
    g = learn.groupby([f1, f2], observed=True).agg(exposure=("Exposure", "sum"), observed=("ClaimNb", "sum"),
                                                     expected=(pred_col, "sum"))
    g["ae"] = g["observed"] / g["expected"]
    # Approximate 95% band for A/E under Poisson: 1 +/- 1.96 / sqrt(expected)
    g["ae_z"] = (g["observed"] - g["expected"]) / np.sqrt(g["expected"])
    return g.reset_index().rename(columns={f1: "level_1", f2: "level_2"}).assign(factor_1=f1, factor_2=f2)


def heatmap(t: pd.DataFrame, path, title: str, min_exposure: float):
    piv = t.pivot(index="level_1", columns="level_2", values="ae")
    expo = t.pivot(index="level_1", columns="level_2", values="exposure")
    z = t.pivot(index="level_1", columns="level_2", values="ae_z")
    piv = piv.where(expo >= min_exposure)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * piv.shape[1], 1.0 + 0.4 * piv.shape[0]))
    im = ax.imshow(np.log(piv.to_numpy(dtype=float)), cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    for (r, c), v in np.ndenumerate(piv.to_numpy(dtype=float)):
        if np.isfinite(v):
            bold = abs(z.to_numpy(dtype=float)[r, c]) > 1.96
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5, fontweight="bold" if bold else "normal")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([str(c) for c in piv.columns], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([str(i) for i in piv.index], fontsize=7)
    ax.set_xlabel(t["factor_2"].iloc[0])
    ax.set_ylabel(t["factor_1"].iloc[0])
    ax.grid(False)
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, label="log(A/E)", shrink=0.8)
    return save(fig, path)


def dependence_plot(Xs: pd.DataFrame, iv: np.ndarray, names: list[str], f1: str, f2: str, path):
    i, j = names.index(f1), names.index(f2)
    y = iv[:, i, j] + iv[:, j, i]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(Xs)) if Xs[f1].nunique() < 30 else 0
    sc = ax.scatter(Xs[f1] + jitter, y, c=Xs[f2], s=3, cmap="viridis", alpha=0.5)
    fig.colorbar(sc, ax=ax, label=f2)
    ax.axhline(0, color=COLORS["other"], lw=0.8)
    ax.set_xlabel(f1)
    ax.set_ylabel(f"SHAP interaction {f1} x {f2}")
    ax.set_title(f"SHAP interaction: {f1} x {f2}")
    return save(fig, path)


def run() -> pd.DataFrame:
    cfg = load_config()
    tables, figures, processed = (cfg["paths"][k] for k in ("tables", "figures", "processed"))
    with open(processed / "models" / "frequency_stage3.pkl", "rb") as fh:
        s3 = pickle.load(fh)
    model = lgb.Booster(model_str=s3["gbm_model"])
    enc = s3["gbm_encoder"]
    p = pd.read_parquet(processed / "policies_banded.parquet")
    learn = p[~p["holdout"]].copy()

    n, per_row = sample_size(cfg, model.num_trees())
    sub = stratified_sample(learn, n, cfg["seed"])
    Xs = features(sub, cfg, enc)
    names = list(Xs.columns)
    t0 = time.perf_counter()
    iv = shap.TreeExplainer(model).shap_interaction_values(Xs.to_numpy())
    secs = time.perf_counter() - t0
    additivity = float(np.abs(iv.sum(axis=(1, 2)) + shap.TreeExplainer(model).expected_value
                              - model.predict(Xs, raw_score=True)).max())
    pd.DataFrame([{"rows": n, "claim_share": float((sub["ClaimNb"] > 0).mean()), "n_trees": model.num_trees(),
                   "expected_seconds": n * per_row, "actual_seconds": secs, "max_additivity_error": additivity}]
                 ).to_csv(tables / "shap_run.csv", index=False)

    ranks = pair_strength(iv, names)
    top = ranks.head(cfg["shap"]["top_pairs"]).copy()
    hs = stratified_sample(learn, cfg["shap"]["h_stat_sample"], cfg["seed"] + 1)
    Xh = features(hs, cfg, enc)
    top["friedman_h2"] = [friedman_h2(model, Xh, a, b) for a, b in zip(top["feature_1"], top["feature_2"])]
    ranks = ranks.merge(top[["feature_1", "feature_2", "friedman_h2"]], how="left")
    ranks.to_csv(tables / "shap_interaction_ranking.csv", index=False)

    # Two-way A/E on learn, GLM-A out-of-fold
    learn["pred_glm_a_oof"] = oof_glm_a(learn, s3["cv"]["glm_a"]["fits"], cfg["split"]["n_folds"])
    aes = []
    for _, r in top.iterrows():
        f1, f2 = r["feature_1"], r["feature_2"]
        dependence_plot(Xs, iv, names, f1, f2, figures / f"shap_dependence_{f1}_x_{f2}.png")
        t = two_way_ae(learn, BANDED[f1], BANDED[f2], "pred_glm_a_oof")
        aes.append(t)
        heatmap(t, figures / f"ae_heatmap_{f1}_x_{f2}.png",
                f"Learn A/E vs GLM-A (out-of-fold): {f1} x {f2}\nbold = |z| > 1.96; blank = cell exposure < {cfg['banding']['min_band_exposure'] / 10:g}",
                cfg["banding"]["min_band_exposure"] / 10)
    pd.concat(aes).to_csv(tables / "two_way_ae_top_pairs.csv", index=False)
    two_way_summary(pd.concat(aes), cfg["banding"]["min_band_exposure"] / 10, "GLM-A").to_csv(
        tables / "two_way_ae_summary_glm_a.csv", index=False)
    return ranks


def two_way_summary(ae: pd.DataFrame, min_exposure: float, model: str) -> pd.DataFrame:
    """Per pair, over cells with at least min_exposure policy-years:
    - mean_z_squared: sum(z**2) / cells
    - z2_per_df_rc: sum(z**2) / ((rows - 1)(cols - 1)), the complete-table independence df
    - z2_per_df_incomplete: sum(z**2) / (cells - rows - cols + 1), the df when some cells are
      excluded (quasi-independence). Headline measure: it equals the rc form for a complete
      table and stays valid when sparse cells are dropped.
    Each is about 1 if the model leaves no systematic two-way pattern."""
    rows = []
    for (a, b), g in ae.groupby(["factor_1", "factor_2"], sort=False):
        g = g[g["exposure"] >= min_exposure]
        r, c, n = g["level_1"].nunique(), g["level_2"].nunique(), len(g)
        sz = float((g["ae_z"] ** 2).sum())
        rows.append({"model": model, "factor_1": a, "factor_2": b, "cells": n, "rows": r, "cols": c,
                     "cells_abs_z_gt_1_96": int((g["ae_z"].abs() > 1.96).sum()),
                     "expected_by_chance": 0.05 * n, "sum_z_squared": sz, "mean_z_squared": sz / n,
                     "df_rc": (r - 1) * (c - 1), "z2_per_df_rc": sz / ((r - 1) * (c - 1)),
                     "df_incomplete": n - r - c + 1, "z2_per_df_incomplete": sz / (n - r - c + 1)})
    return pd.DataFrame(rows)
