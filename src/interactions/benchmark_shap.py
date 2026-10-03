"""Runtime benchmark used to size the SHAP interaction subsample (config: shap.sample_size).

Fits a reference LightGBM Poisson model on the full raw data (exposure capped at 1,
counts at 4; the cleaning stage repeats these choices with logging), then times
TreeExplainer.shap_interaction_values on 1k and 5k stratified rows. Writes
reports/tables/shap_runtime_benchmark.csv. Not part of `make all`.
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm

from src.config import load_config

FEATURES = ["DrivAge", "VehAge", "VehPower", "BonusMalus", "LogDensity", "VehGas", "VehBrand", "Region"]
REFERENCE_PARAMS = dict(
    objective="poisson", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500,
    feature_fraction=0.8, lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count(),
)
N_ROUNDS = 300


def main() -> pd.DataFrame:
    cfg = load_config()
    seed = cfg["seed"]
    f = pd.read_parquet(cfg["paths"]["raw"] / "freMTPL2freq.parquet")
    f["Exposure"] = f["Exposure"].clip(upper=1.0)
    f["ClaimNb"] = f["ClaimNb"].clip(upper=4)
    X = f[["DrivAge", "VehAge", "VehPower", "BonusMalus"]].copy()
    X["LogDensity"] = np.log(f["Density"])
    X["VehGas"] = (f["VehGas"] == "Diesel").astype(int)
    # Nominal factors as numeric ranks of observed claim frequency. shap 0.51 returns
    # interaction values that do not add up to the model output for LightGBM native
    # categorical splits (checked: additivity error ~0.1 on the log scale), so the GBM
    # does not use native categoricals.
    for c in ("VehBrand", "Region"):
        g = f.groupby(c, observed=True)[["ClaimNb", "Exposure"]].sum()
        rank = (g["ClaimNb"] / g["Exposure"]).rank(method="first")
        X[c] = f[c].map(rank).astype(float)
    X = X[FEATURES]
    rows = []

    t = time.perf_counter()
    params = dict(REFERENCE_PARAMS, seed=seed, monotone_constraints=[int(c == "BonusMalus") for c in FEATURES])
    model = lgb.train(params, lgb.Dataset(X, f["ClaimNb"] / f["Exposure"], weight=f["Exposure"]), N_ROUNDS)
    rows.append({"step": "lgbm_fit_full_data", "rows": len(X), "seconds": time.perf_counter() - t})

    explainer = shap.TreeExplainer(model)
    X_num = X.astype(float)
    claim = f["ClaimNb"] > 0
    for n in (1000, 5000):
        # stratified by claim indicator, same claim share as the full data
        n_claim = int(round(n * claim.mean()))
        idx = np.concatenate([
            f.index[claim].to_series().sample(n_claim, random_state=seed).to_numpy(),
            f.index[~claim].to_series().sample(n - n_claim, random_state=seed).to_numpy(),
        ])
        t = time.perf_counter()
        iv = explainer.shap_interaction_values(X_num.loc[idx])
        secs = time.perf_counter() - t
        additivity = np.abs(iv.sum(axis=(1, 2)) + explainer.expected_value - model.predict(X.loc[idx], raw_score=True)).max()
        rows.append({"step": "shap_interaction_values", "rows": n, "seconds": secs, "max_additivity_error": additivity})

    # Evidence for the encoding decision: same model with native categoricals
    X_cat = X.assign(VehBrand=f["VehBrand"], Region=f["Region"])
    model_cat = lgb.train(params, lgb.Dataset(X_cat, f["ClaimNb"] / f["Exposure"], weight=f["Exposure"]), N_ROUNDS)
    idx = f.sample(1000, random_state=seed).index
    X_cat_num = X_cat.loc[idx].assign(**{c: X_cat.loc[idx, c].cat.codes for c in ("VehBrand", "Region")}).astype(float)
    exp_cat = shap.TreeExplainer(model_cat)
    t = time.perf_counter()
    iv = exp_cat.shap_interaction_values(X_cat_num.to_numpy())
    secs = time.perf_counter() - t
    raw = model_cat.predict(X_cat.loc[idx], raw_score=True)
    sv = exp_cat.shap_values(X_cat_num.to_numpy())
    rows.append({"step": "shap_interaction_values_native_categorical", "rows": 1000, "seconds": secs,
                 "max_additivity_error": np.abs(iv.sum(axis=(1, 2)) + exp_cat.expected_value - raw).max()})
    rows.append({"step": "shap_values_native_categorical", "rows": 1000, "seconds": np.nan,
                 "max_additivity_error": np.abs(sv.sum(axis=1) + exp_cat.expected_value - raw).max()})

    D = pd.get_dummies(f[["VehBrand", "Region"]], drop_first=True, dtype=float)
    D["log_bm"], D["const"] = np.log(f["BonusMalus"]), 1.0
    t = time.perf_counter()
    sm.GLM(f["ClaimNb"].to_numpy(), D.to_numpy(), family=sm.families.Poisson(), offset=np.log(f["Exposure"].to_numpy())).fit()
    rows.append({"step": f"statsmodels_poisson_glm_{D.shape[1]}_params", "rows": len(D), "seconds": time.perf_counter() - t})

    out = pd.DataFrame(rows)
    out["seconds_per_1k_rows"] = out["seconds"] / out["rows"] * 1000
    out["n_trees"] = N_ROUNDS
    out["num_leaves"] = REFERENCE_PARAMS["num_leaves"]
    out["cpu_count"] = os.cpu_count()
    out.to_csv(cfg["paths"]["tables"] / "shap_runtime_benchmark.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(main())
