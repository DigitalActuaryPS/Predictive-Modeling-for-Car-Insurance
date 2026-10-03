"""Stage 4 (part 2): forward selection of interaction candidates into GLM-A -> GLM-B,
under the acceptance rule in DECISIONS D018.
"""
import pickle

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.config import load_config
from src.evaluation import metrics as M
from src.evaluation.plots import COLORS, double_lift_plot, lift_plot
from src.interactions.candidates import CANDIDATES, NOT_TRANSLATED
from src.interactions.shap_interactions import BANDED, heatmap, two_way_ae, two_way_summary
from src.models import gbm as G
from src.models.glm import FactorTerm, InteractionTerm, cv_frequency, fit_glm

FLOOR_SHARE = 0.05   # D018 rule 3
MAX_ACCEPTED = 5     # D018 rule 5
LRT_ALPHA = 0.001    # D018 rule 1 (alternative)
SENSIBLE_RANGE = (0.5, 2.0)  # D018 rule 4: interaction multiplier, 1st-99th percentile of learn policies


def make_terms(cfg, base, accepted: list[str]):
    def make():
        ts = [FactorTerm(f, base[f]) for f in cfg["glm"]["factors"]]
        return ts + [InteractionTerm(name, CANDIDATES[name][1]) for name in accepted]
    return make


def interaction_multiplier(fit, df, name) -> np.ndarray:
    cols = CANDIDATES[name][1](df)
    idx = [fit.names.index(f"{name}:{c}") for c in cols]
    X = np.column_stack([np.asarray(v, float) for v in cols.values()])
    return np.exp(X @ fit.coef[idx])


def evaluate(name, current_cv, current_full, cfg, base, accepted, learn, gap) -> dict:
    make = make_terms(cfg, base, accepted + [name])
    cv = cv_frequency(make, learn, cfg["split"]["n_folds"])
    diff = current_cv["fold_deviance"] - cv["fold_deviance"]
    full = fit_glm(make(), learn, learn["ClaimNb"].to_numpy(), "poisson", np.log(learn["Exposure"].to_numpy()))
    df = full.n_params - current_full.n_params
    lr = 2 * (full.loglik - current_full.loglik)
    cols = [n for n in full.names if n.startswith(f"{name}:")]
    signs = np.array([[np.sign(f.coef[f.names.index(c)]) for c in cols] for f in cv["fits"]])
    mult = interaction_multiplier(full, learn, name)
    lo, hi = np.percentile(mult, [1, 99])
    r = {
        "candidate": name, "pair": CANDIDATES[name][0], "description": CANDIDATES[name][2], "n_params": df,
        "cv_deviance_mean": cv["mean"], "improvement_mean": diff.mean(), "improvement_sd": diff.std(ddof=1),
        "folds_improved": int((diff > 0).sum()), "lrt_stat": lr, "lrt_df": df, "lrt_p": float(chi2.sf(lr, df)),
        "share_of_gap": diff.mean() / gap, "sign_stable_all_folds": bool((signs == signs[0]).all()),
        "multiplier_p01": lo, "multiplier_p99": hi,
        "coefficients": "; ".join(f"{c.split(':', 1)[1]}={full.coef[full.names.index(c)]:.4f} (se {full.se[full.names.index(c)]:.4f})" for c in cols),
        **{f"fold_{k}_improvement": d for k, d in enumerate(diff)},
    }
    r["rule1_beyond_noise"] = bool(r["improvement_mean"] > r["improvement_sd"] or r["lrt_p"] < LRT_ALPHA)
    r["rule2_all_folds"] = r["folds_improved"] == cfg["split"]["n_folds"]
    r["rule3_materiality"] = bool(r["share_of_gap"] >= FLOOR_SHARE)
    r["rule4_stable_sensible"] = bool(r["sign_stable_all_folds"] and SENSIBLE_RANGE[0] <= lo and hi <= SENSIBLE_RANGE[1])
    r["passes"] = all(r[k] for k in ("rule1_beyond_noise", "rule2_all_folds", "rule3_materiality", "rule4_stable_sensible"))
    return r, cv, full


def run() -> dict:
    cfg = load_config()
    tables, figures, processed = (cfg["paths"][k] for k in ("tables", "figures", "processed"))
    with open(processed / "models" / "frequency_stage3.pkl", "rb") as fh:
        s3 = pickle.load(fh)
    p = pd.read_parquet(processed / "policies_banded.parquet")
    learn, hold = p[~p["holdout"]].copy(), p[p["holdout"]].copy()
    base = pd.read_csv(tables / "base_levels.csv").set_index("factor")["base_level"].to_dict()
    n_folds = cfg["split"]["n_folds"]

    cv_a, cv_g = s3["cv"]["glm_a"], s3["cv"]["gbm"]
    gap = cv_a["mean"] - cv_g["mean"]
    accepted, log_rows, steps = [], [], []
    current_cv, current_full = cv_a, s3["glm_a"]
    remaining = list(CANDIDATES)
    step = 0
    while remaining and len(accepted) < MAX_ACCEPTED:
        step += 1
        results = []
        for name in remaining:
            r, cv, full = evaluate(name, current_cv, current_full, cfg, base, accepted, learn, gap)
            r["step"] = step
            results.append((r, cv, full))
            print(f"  step {step} {name}: impr {r['improvement_mean']:.6f} (sd {r['improvement_sd']:.6f}) "
                  f"folds {r['folds_improved']} gap {r['share_of_gap']:.1%} p {r['lrt_p']:.2g} "
                  f"mult [{r['multiplier_p01']:.2f},{r['multiplier_p99']:.2f}] pass {r['passes']}", flush=True)
        log_rows.extend(r for r, _, _ in results)
        passing = [x for x in results if x[0]["passes"]]
        if not passing:
            break
        best, current_cv, current_full = max(passing, key=lambda x: x[0]["improvement_mean"])
        accepted.append(best["candidate"])
        steps.append({"step": step, "accepted": best["candidate"], "pair": best["pair"],
                      "improvement_mean": best["improvement_mean"], "share_of_gap": best["share_of_gap"],
                      "cv_deviance_after": current_cv["mean"],
                      "cumulative_share_of_gap": (cv_a["mean"] - current_cv["mean"]) / gap})
        # other variants of the same SHAP pair leave the pool
        remaining = [c for c in remaining if CANDIDATES[c][0] != best["pair"]]

    log = pd.DataFrame(log_rows)
    log.to_csv(tables / "interaction_selection_log.csv", index=False)
    pd.DataFrame(steps).to_csv(tables / "interaction_accepted.csv", index=False)
    pd.DataFrame([{"pair": k, "reason": v} for k, v in NOT_TRANSLATED.items()]).to_csv(
        tables / "interaction_not_translated.csv", index=False)

    # GLM-B
    glm_b, cv_b = current_full, current_cv
    glm_b.table().to_csv(tables / "glm_b_coefficients.csv", index=False)
    y, e = hold["ClaimNb"].to_numpy(), hold["Exposure"].to_numpy()
    f_a = s3["glm_a"].predict(hold)
    f_b = glm_b.predict(hold)
    import lightgbm as lgb
    gbm = lgb.Booster(model_str=s3["gbm_model"])
    f_g = G.predict_frequency(gbm, hold, cfg, s3["gbm_encoder"])
    f_0 = s3["glm_0"].predict(hold)

    def row(name, f, n_params, cv):
        return {"model": name, "n_params": n_params, "cv_deviance_mean": cv["mean"], "cv_deviance_sd": cv["sd"],
                **{f"cv_fold_{k}": d for k, d in enumerate(cv["fold_deviance"])},
                "holdout_deviance": M.poisson_deviance(y, f * e),
                "holdout_gini": M.lorenz_gini(y, e, f) if np.ptp(f) > 0 else 0.0,
                "holdout_predicted_over_actual": float((f * e).sum() / y.sum())}

    comp = pd.DataFrame([
        row("Intercept only", f_0, 1, s3["cv"]["glm_0"]),
        row("GLM-A (current tariff)", f_a, s3["glm_a"].n_params, cv_a),
        row("GLM-B (proposed tariff)", f_b, glm_b.n_params, cv_b),
        row("GBM (LightGBM Poisson)", f_g, int(gbm.num_trees()), cv_g),
    ])
    comp.to_csv(tables / "frequency_model_comparison.csv", index=False)
    hd = comp.set_index("model")["holdout_deviance"]
    pd.DataFrame([{
        "cv_gap_glm_a_minus_gbm": gap, "cv_gap_closed_by_glm_b": (cv_a["mean"] - cv_b["mean"]) / gap,
        "holdout_gap_glm_a_minus_gbm": hd["GLM-A (current tariff)"] - hd["GBM (LightGBM Poisson)"],
        "holdout_gap_closed_by_glm_b": (hd["GLM-A (current tariff)"] - hd["GLM-B (proposed tariff)"])
        / (hd["GLM-A (current tariff)"] - hd["GBM (LightGBM Poisson)"]),
        "n_accepted": len(accepted),
    }]).to_csv(tables / "gap_closed.csv", index=False)

    lift_b = M.lift_table(y, e, f_b)
    lift_b.assign(model="GLM-B").to_csv(tables / "lift_glm_b_holdout.csv", index=False)
    lift_a = pd.read_csv(tables / "lift_glm_a_holdout.csv")
    lift_g = pd.read_csv(tables / "lift_gbm_holdout.csv")
    lift_plot({"GLM-A": (lift_a, COLORS["glm_a"]), "GLM-B": (lift_b, COLORS["glm_b"]), "GBM": (lift_g, COLORS["gbm"])},
              figures / "lift_holdout.png", "Holdout decile lift (equal-exposure deciles)")
    for name_a, fa, ca, name_b, fb, cb, fn in [
        ("GBM", f_g, COLORS["gbm"], "GLM-B", f_b, COLORS["glm_b"], "double_lift_gbm_vs_glm_b_holdout"),
        ("GLM-B", f_b, COLORS["glm_b"], "GLM-A", f_a, COLORS["glm_a"], "double_lift_glm_b_vs_glm_a_holdout"),
    ]:
        dl = M.double_lift_table(y, e, fa, fb)
        dl.to_csv(tables / f"{fn}.csv", index=False)
        double_lift_plot(dl, name_a, name_b, ca, cb, figures / f"{fn}.png", f"Holdout double lift: {name_a} vs {name_b}")

    # Residual two-way A/E after GLM-B (out-of-fold), same pairs as before
    learn["pred_glm_b_oof"] = np.nan
    for k in range(n_folds):
        m = (learn["fold"] == k).to_numpy()
        learn.loc[m, "pred_glm_b_oof"] = cv_b["fits"][k].predict(learn[m], np.log(learn.loc[m, "Exposure"].to_numpy()))
    ranks = pd.read_csv(tables / "shap_interaction_ranking.csv").head(cfg["shap"]["top_pairs"])
    aes = []
    for _, r in ranks.iterrows():
        t = two_way_ae(learn, BANDED[r["feature_1"]], BANDED[r["feature_2"]], "pred_glm_b_oof")
        aes.append(t)
        heatmap(t, figures / f"ae_heatmap_glm_b_{r['feature_1']}_x_{r['feature_2']}.png",
                f"Learn A/E vs GLM-B (out-of-fold): {r['feature_1']} x {r['feature_2']}",
                cfg["banding"]["min_band_exposure"] / 10)
    two_way_summary(pd.concat(aes), cfg["banding"]["min_band_exposure"] / 10, "GLM-B").to_csv(
        tables / "two_way_ae_summary_glm_b.csv", index=False)

    with open(processed / "models" / "frequency_glm_b.pkl", "wb") as fh:
        pickle.dump({"glm_b": glm_b, "accepted": accepted, "cv_glm_b": cv_b}, fh)
    return {"accepted": accepted, "comparison": comp}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    out = run()
    print(out["accepted"])
    print(out["comparison"].T)
