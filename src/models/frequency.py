"""Stage 3: frequency models. Intercept-only, GLM-A (current tariff: banded main effects),
a diagnostic GLM with log(exposure) as a free covariate, and the LightGBM Poisson GBM.
Evaluation: CV deviance (mean, sd) on the shared folds, holdout deviance, Gini, decile
lift, double lift GLM-A vs GBM, and holdout A/E by rating factor.
"""
import pickle

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation import metrics as M
from src.evaluation.plots import COLORS, ae_grid, double_lift_plot, lift_plot
from src.models import gbm as G
from src.models.glm import FactorTerm, NumericTerm, cv_frequency, fit_glm


def load(cfg):
    p = pd.read_parquet(cfg["paths"]["processed"] / "policies_banded.parquet")
    base = pd.read_csv(cfg["paths"]["tables"] / "base_levels.csv").set_index("factor")["base_level"].to_dict()
    return p[~p["holdout"]].copy(), p[p["holdout"]].copy(), base


def glm_a_terms(cfg, base):
    return lambda: [FactorTerm(f, base[f]) for f in cfg["glm"]["factors"]]


def fit_frequency_glm(make_terms, learn):
    return fit_glm(make_terms(), learn, learn["ClaimNb"].to_numpy(), "poisson", np.log(learn["Exposure"].to_numpy()))


def holdout_metrics(name, hold, freq, n_params, cv) -> dict:
    y, e = hold["ClaimNb"].to_numpy(), hold["Exposure"].to_numpy()
    return {
        "model": name, "n_params": n_params,
        "cv_deviance_mean": cv["mean"], "cv_deviance_sd": cv["sd"],
        **{f"cv_fold_{k}": d for k, d in enumerate(cv["fold_deviance"])},
        "holdout_deviance": M.poisson_deviance(y, freq * e),
        "holdout_gini": M.lorenz_gini(y, e, freq) if np.ptp(freq) > 0 else 0.0,
        "holdout_predicted_over_actual": float((freq * e).sum() / y.sum()),
    }


def log_exposure_diagnostic(cfg, base, learn) -> pd.DataFrame:
    """GLM-A factors plus log(exposure) as a free covariate, no offset. Under the
    proportionality assumption behind the offset, its coefficient is 1."""
    learn = learn.assign(LogExposure=np.log(learn["Exposure"]))
    make = lambda: [FactorTerm(f, base[f]) for f in cfg["glm"]["factors"]] + [NumericTerm("LogExposure")]  # noqa: E731
    res = fit_glm(make(), learn, learn["ClaimNb"].to_numpy(), "poisson")
    t = res.table().set_index("term").loc["LogExposure"]
    z = 1.959963984540054
    cv_free = cv_frequency_no_offset(make, learn, cfg["split"]["n_folds"])
    return pd.DataFrame([{
        "coef_log_exposure": t["coef"], "se": t["se"],
        "ci_lower_95": t["coef"] - z * t["se"], "ci_upper_95": t["coef"] + z * t["se"],
        "z_test_coef_equals_1": (t["coef"] - 1) / t["se"],
        "cv_deviance_mean_free_exposure": cv_free["mean"], "cv_deviance_sd_free_exposure": cv_free["sd"],
        **{f"cv_fold_{k}_free_exposure": d for k, d in enumerate(cv_free["fold_deviance"])},
    }])


def exposure_by_segment(learn: pd.DataFrame, factors=("DrivAge_band", "BonusMalus_band")) -> pd.DataFrame:
    """Where is short exposure concentrated? Learn set, by level of each factor."""
    short = learn["Exposure"] < 0.25
    rows = []
    for f in factors:
        g = learn.assign(short=short).groupby(f, observed=True)
        t = g.agg(policies=("IDpol", "size"), exposure=("Exposure", "sum"), mean_exposure=("Exposure", "mean"),
                  share_policies_exposure_lt_0_25=("short", "mean"))
        fs = learn[short].groupby(f, observed=True)[["ClaimNb", "Exposure"]].sum()
        fl = learn[~short].groupby(f, observed=True)[["ClaimNb", "Exposure"]].sum()
        t["frequency_exposure_lt_0_25"] = fs["ClaimNb"] / fs["Exposure"]
        t["frequency_exposure_ge_0_25"] = fl["ClaimNb"] / fl["Exposure"]
        t.insert(0, "factor", f)
        rows.append(t.rename_axis("level").reset_index())
    all_row = pd.DataFrame([{"factor": "all", "level": "all", "policies": len(learn), "exposure": learn["Exposure"].sum(),
                             "mean_exposure": learn["Exposure"].mean(), "share_policies_exposure_lt_0_25": short.mean(),
                             "frequency_exposure_lt_0_25": learn.loc[short, "ClaimNb"].sum() / learn.loc[short, "Exposure"].sum(),
                             "frequency_exposure_ge_0_25": learn.loc[~short, "ClaimNb"].sum() / learn.loc[~short, "Exposure"].sum()}])
    return pd.concat(rows + [all_row], ignore_index=True)


def cv_frequency_no_offset(make_terms, learn, n_folds):
    devs = []
    for k in range(n_folds):
        tr, va = learn[learn["fold"] != k], learn[learn["fold"] == k]
        res = fit_glm(make_terms(), tr, tr["ClaimNb"].to_numpy(), "poisson")
        devs.append(M.poisson_deviance(va["ClaimNb"].to_numpy(), res.predict(va)))
    devs = np.array(devs)
    return {"fold_deviance": devs, "mean": devs.mean(), "sd": devs.std(ddof=1)}


def run() -> dict:
    cfg = load_config()
    tables, figures, processed = (cfg["paths"][k] for k in ("tables", "figures", "processed"))
    models_dir = processed / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    learn, hold, base = load(cfg)
    n_folds = cfg["split"]["n_folds"]
    rows = []

    # Intercept only
    cv0 = cv_frequency(lambda: [], learn, n_folds)
    glm0 = fit_frequency_glm(lambda: [], learn)
    rows.append(holdout_metrics("Intercept only", hold, glm0.predict(hold), glm0.n_params, cv0))

    # GLM-A
    make_a = glm_a_terms(cfg, base)
    cv_a = cv_frequency(make_a, learn, n_folds)
    glm_a = fit_frequency_glm(make_a, learn)
    f_a = glm_a.predict(hold)
    rows.append(holdout_metrics("GLM-A (main effects)", hold, f_a, glm_a.n_params, cv_a))
    glm_a.table().to_csv(tables / "glm_a_coefficients.csv", index=False)

    # Exposure proportionality diagnostic (not used for pricing)
    log_exposure_diagnostic(cfg, base, learn).to_csv(tables / "log_exposure_diagnostic.csv", index=False)
    exposure_by_segment(learn).to_csv(tables / "exposure_by_segment.csv", index=False)

    # GBM: tune, comparable CV with fixed rounds, final fit
    tuning = G.tune(learn, cfg)
    tuning.drop(columns=["fold_meta"]).to_csv(tables / "gbm_tuning.csv", index=False)
    best = tuning.loc[tuning["cv_deviance_mean"].idxmin()]
    params = {k: best[k] for k in cfg["gbm"]["grid"]}
    params["num_leaves"], params["min_data_in_leaf"] = int(params["num_leaves"]), int(params["min_data_in_leaf"])
    n_rounds = int(round(best["mean_best_iteration"]))
    cv_g = G.cv_fixed_rounds(params, n_rounds, learn, cfg)
    gbm_model, enc = G.fit_final(params, n_rounds, learn, cfg)
    f_g = G.predict_frequency(gbm_model, hold, cfg, enc)
    rows.append(holdout_metrics("GBM (LightGBM Poisson)", hold, f_g, int(gbm_model.num_trees()), cv_g))
    pd.DataFrame([{**params, "n_rounds": n_rounds, "chosen_trial": int(best["trial"])}]).to_csv(tables / "gbm_chosen.csv", index=False)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables / "frequency_models_stage3.csv", index=False)

    # Holdout diagnostics
    y, e = hold["ClaimNb"].to_numpy(), hold["Exposure"].to_numpy()
    lift_a, lift_g = M.lift_table(y, e, f_a), M.lift_table(y, e, f_g)
    lift_a.assign(model="GLM-A").to_csv(tables / "lift_glm_a_holdout.csv", index=False)
    lift_g.assign(model="GBM").to_csv(tables / "lift_gbm_holdout.csv", index=False)
    lift_plot({"GLM-A": (lift_a, COLORS["glm_a"]), "GBM": (lift_g, COLORS["gbm"])},
              figures / "lift_glm_a_gbm_holdout.png", "Holdout decile lift")
    dl = M.double_lift_table(y, e, f_g, f_a)
    dl.to_csv(tables / "double_lift_gbm_vs_glm_a_holdout.csv", index=False)
    double_lift_plot(dl, "GBM", "GLM-A", COLORS["gbm"], COLORS["glm_a"], figures / "double_lift_gbm_vs_glm_a_holdout.png",
                     "Holdout double lift: GBM vs GLM-A")
    h = hold.assign(pred_glm_a=f_a * e, pred_gbm=f_g * e)
    ae = pd.concat([M.ae_by_factor(h, f, {"glm_a": "pred_glm_a", "gbm": "pred_gbm"}) for f in cfg["glm"]["factors"]])
    ae.to_csv(tables / "ae_by_factor_holdout_stage3.csv", index=False)
    ae_grid(ae, {"GLM-A": ("ae_glm_a", COLORS["glm_a"]), "GBM": ("ae_gbm", COLORS["gbm"])},
            figures / "ae_by_factor_holdout_stage3.png", "Holdout A/E by rating factor")

    with open(models_dir / "frequency_stage3.pkl", "wb") as fh:
        pickle.dump({"glm_0": glm0, "glm_a": glm_a, "gbm_model": gbm_model.model_to_string(), "gbm_encoder": enc,
                     "gbm_params": params, "gbm_rounds": n_rounds, "cv": {"glm_0": cv0, "glm_a": cv_a, "gbm": cv_g},
                     "gbm_tuning_meta": tuning[["trial", "fold_meta"]].to_dict("records")}, fh)
    return {"comparison": comparison}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(run()["comparison"].T)
