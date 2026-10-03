"""GLM-A-mono: GLM-A with the D039 monotone effective-BonusMalus merge rule applied.

Used as the *current* tariff in the impact analysis (Stage 7). An incumbent tariff whose
no-claims scale reverses is not realistic, and comparing against it would mix the
monotonicity fix with the interaction gains. GLM-A itself stays as the statistical model
in the results table. See DECISIONS D041.
"""
import pickle

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation import metrics as M
from src.interactions.glm_revision import make_bm_monotone

BM_BAND_A = "BonusMalus_bandA"


def run() -> dict:
    cfg = load_config()
    tables, processed = cfg["paths"]["tables"], cfg["paths"]["processed"]
    p = pd.read_parquet(processed / "policies_banded.parquet")
    learn = p[~p["holdout"]].copy()
    hold = p[p["holdout"]]
    base = pd.read_csv(tables / "base_levels.csv", dtype=str).set_index("factor")["base_level"].to_dict()
    with open(processed / "models" / "frequency_stage3.pkl", "rb") as fh:
        s3 = pickle.load(fh)

    fit, cv, factors, mapping, base = make_bm_monotone(
        cfg, base, [], learn, s3["cv"]["glm_a"], s3["glm_a"], cfg["split"]["n_folds"], tables,
        col_name=BM_BAND_A, steps_file="bm_monotonicity_steps_glm_a.csv")
    order = list(dict.fromkeys(mapping[str(c)] for c in p["BonusMalus_band"].cat.categories))
    p[BM_BAND_A] = pd.Categorical(p["BonusMalus_band"].astype(str).map(mapping), categories=order, ordered=True)
    p.to_parquet(processed / "policies_banded.parquet", index=False)
    hold = p[p["holdout"]]

    y, e = hold["ClaimNb"].to_numpy(), hold["Exposure"].to_numpy()
    f = fit.predict(hold)
    comp = pd.read_csv(tables / "frequency_model_comparison.csv")
    comp = comp[comp["model"] != "GLM-A-mono (impact baseline)"]
    row = {"model": "GLM-A-mono (impact baseline)", "n_params": fit.n_params, "cv_deviance_mean": cv["mean"],
           "cv_deviance_sd": cv["sd"], **{f"cv_fold_{k}": d for k, d in enumerate(cv["fold_deviance"])},
           "holdout_deviance": M.poisson_deviance(y, f * e), "holdout_gini": M.lorenz_gini(y, e, f),
           "holdout_predicted_over_actual": float((f * e).sum() / y.sum())}
    comp = pd.concat([comp, pd.DataFrame([row])], ignore_index=True)
    comp.to_csv(tables / "frequency_model_comparison.csv", index=False)
    pd.DataFrame([{"factor": BM_BAND_A, "base_level": base[BM_BAND_A]}]).to_csv(tables / "base_levels_glm_a_mono.csv", index=False)
    with open(processed / "models" / "frequency_glm_a_mono.pkl", "wb") as fh:
        pickle.dump({"glm_a_mono": fit, "cv": cv, "factors": factors, "bm_mapping": mapping}, fh)
    return {"row": row, "mapping": mapping}


if __name__ == "__main__":
    r = run()
    print(r["row"])
    print(r["mapping"])
