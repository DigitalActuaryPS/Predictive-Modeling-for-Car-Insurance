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
from src.interactions.candidates import (CANDIDATES, NOT_TRANSLATED, REGION_GROUP_CANDIDATES, SELECTION_ORDER,
                                         RegionSlopeGroupTerm)
from src.interactions.shap_interactions import BANDED, heatmap, two_way_ae, two_way_summary
from src.models import gbm as G
from src.models.glm import FactorTerm, InteractionTerm, NumericTerm, cv_frequency, design, fit_glm

FLOOR_SHARE = 0.05   # D018 rule 3
MAX_ACCEPTED = 5     # D018 rule 5
LRT_ALPHA = 0.001    # D018 rule 1 (alternative)
SENSIBLE_RANGE = (0.5, 2.0)  # D018 rule 4: interaction multiplier, 1st-99th percentile of learn policies


def candidate_info(name):
    if name in CANDIDATES:
        return CANDIDATES[name][0], CANDIDATES[name][2]
    return REGION_GROUP_CANDIDATES[name][0], REGION_GROUP_CANDIDATES[name][2]


def make_term(name, cfg, base, accepted_before):
    if name in CANDIDATES:
        return InteractionTerm(name, CANDIDATES[name][1])
    k = REGION_GROUP_CANDIDATES[name][1]
    return RegionSlopeGroupTerm(k, cfg["glm"]["factors"], base, list(accepted_before))


def make_terms(cfg, base, accepted: list[str]):
    def make():
        ts = [FactorTerm(f, base[f]) for f in cfg["glm"]["factors"]]
        return ts + [make_term(name, cfg, base, accepted[:i]) for i, name in enumerate(accepted)]
    return make


def interaction_multiplier(fit, df, name) -> np.ndarray:
    term = next(t for t in fit.terms if getattr(t, "name", None) == name)
    X, cols = term.transform(df)
    idx = [fit.names.index(c) for c in cols]
    return np.exp(X @ fit.coef[idx])


def evaluate(name, current_cv, current_full, cfg, base, accepted, learn, gap) -> tuple:
    make = make_terms(cfg, base, accepted + [name])
    cv = cv_frequency(make, learn, cfg["split"]["n_folds"])
    diff = current_cv["fold_deviance"] - cv["fold_deviance"]
    full = fit_glm(make(), learn, learn["ClaimNb"].to_numpy(), "poisson", np.log(learn["Exposure"].to_numpy()))
    df = full.n_params - current_full.n_params
    lr = 2 * (full.loglik - current_full.loglik)
    cols = [n for n in full.names if n.startswith(f"{name}:")]
    signs = np.array([[np.sign(f.coef[f.names.index(c)]) if c in f.names else np.nan for c in cols] for f in cv["fits"]])
    mult = interaction_multiplier(full, learn, name)
    lo, hi = np.percentile(mult, [1, 99])
    pair, desc = candidate_info(name)
    r = {
        "candidate": name, "pair": pair, "description": desc, "n_params": df,
        "cv_deviance_mean": cv["mean"], "improvement_mean": diff.mean(), "improvement_sd": diff.std(ddof=1),
        "folds_improved": int((diff > 0).sum()), "lrt_stat": lr, "lrt_df": df, "lrt_p": float(chi2.sf(lr, df)),
        "share_of_gap": diff.mean() / gap, "sign_stable_all_folds": bool((signs == signs[0]).all()),
        "multiplier_p01": lo, "multiplier_p99": hi,
        "coefficients": "; ".join(f"{c.split(':', 1)[1]}={full.coef[full.names.index(c)]:.4f} (se {full.se[full.names.index(c)]:.4f})" for c in cols),
        **{f"fold_{k}_improvement": d for k, d in enumerate(diff)},
    }
    term = next(t for t in full.terms if getattr(t, "name", None) == name)
    if isinstance(term, RegionSlopeGroupTerm):
        r["region_groups_full_learn"] = "; ".join(
            f"group{g + 1}{' (base)' if g == term.base_group else ''}: " + ", ".join(sorted(k for k, v in term.groups.items() if v == g))
            for g in range(term.k))
        r["fold_groupings_identical"] = len({tuple(sorted(next(t for t in f.terms if getattr(t, "name", None) == name).groups.items()))
                                            for f in cv["fits"]}) == 1
    r["rule1_beyond_noise"] = bool(r["improvement_mean"] > r["improvement_sd"] or r["lrt_p"] < LRT_ALPHA)
    r["rule2_all_folds"] = r["folds_improved"] == cfg["split"]["n_folds"]
    r["rule3_materiality"] = bool(r["share_of_gap"] >= FLOOR_SHARE)
    r["rule4_stable_sensible"] = bool(r["sign_stable_all_folds"] and SENSIBLE_RANGE[0] <= lo and hi <= SENSIBLE_RANGE[1])
    r["passes"] = all(r[k] for k in ("rule1_beyond_noise", "rule2_all_folds", "rule3_materiality", "rule4_stable_sensible"))
    return r, cv, full


def choose_form(passing: list, step: int, log: list):
    """D018 rule 6 (parsimony tie-break): start from the passing form with the largest mean
    improvement; if a passing form with fewer parameters is worse by less than the sd of
    the paired fold differences between the two forms, take it instead (fewest parameters
    first)."""
    best = max(passing, key=lambda x: x[0]["improvement_mean"])
    chosen = best
    for alt in sorted(passing, key=lambda x: (x[0]["n_params"], -x[0]["improvement_mean"])):
        if alt is best or alt[0]["n_params"] >= best[0]["n_params"]:
            continue
        d = alt[1]["fold_deviance"] - best[1]["fold_deviance"]  # positive = alt worse
        log.append({"step": step, "best_by_improvement": best[0]["candidate"], "best_n_params": best[0]["n_params"],
                    "simpler_form": alt[0]["candidate"], "simpler_n_params": alt[0]["n_params"],
                    "mean_paired_difference": d.mean(), "sd_paired_difference": d.std(ddof=1),
                    "simpler_taken": bool(d.mean() < d.std(ddof=1))})
        if d.mean() < d.std(ddof=1):
            chosen = alt
            break
    return chosen


def oof(learn, cv, n_folds):
    pred = np.full(len(learn), np.nan)
    for k in range(n_folds):
        m = (learn["fold"] == k).to_numpy()
        pred[m] = cv["fits"][k].predict(learn[m], np.log(learn.loc[m, "Exposure"].to_numpy()))
    return pred


def pivot_tables(ae: pd.DataFrame, model: str, min_exposure: float, tables, order: dict) -> None:
    """A/E pivot per pair, rows and columns in band order; cells under min_exposure blank."""
    for (a, b), g in ae.groupby(["factor_1", "factor_2"], sort=False):
        g = g[g["exposure"] >= min_exposure].astype({"level_1": str, "level_2": str})
        piv = g.pivot(index="level_1", columns="level_2", values="ae")
        piv = piv.reindex(index=[x for x in order[a] if x in piv.index], columns=[x for x in order[b] if x in piv.columns])
        piv.round(3).to_csv(tables / f"two_way_ae_pivot_{model}_{a}_x_{b}.csv")


def exposure_check(glm_b, accepted, cfg, base, learn) -> pd.DataFrame:
    """Refit GLM-B's structure with log(exposure) as a free covariate (no offset) and
    compare each interaction coefficient with the offset fit."""
    terms = make_terms(cfg, base, accepted)() + [NumericTerm("LogExposure")]
    free = fit_glm(terms, learn.assign(LogExposure=np.log(learn["Exposure"])), learn["ClaimNb"].to_numpy(), "poisson")
    rows = []
    for n in glm_b.names:
        if ":" in n and n.split(":", 1)[0] in accepted:
            i, j = glm_b.names.index(n), free.names.index(n)
            rows.append({"term": n, "coef_offset": glm_b.coef[i], "se_offset": glm_b.se[i],
                         "coef_free_exposure": free.coef[j], "se_free_exposure": free.se[j],
                         "ratio_free_to_offset": free.coef[j] / glm_b.coef[i]})
    out = pd.DataFrame(rows)
    out["log_exposure_coef_in_free_fit"] = free.coef[free.names.index("LogExposure")]
    return out


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
    accepted, log_rows, steps, region_signal, tiebreaks = [], [], [], [], []
    current_cv, current_full = cv_a, s3["glm_a"]
    min_cell = cfg["banding"]["min_band_exposure"] / 10

    def region_z2(cv, label):
        t = two_way_ae(learn.assign(_p=oof(learn, cv, n_folds)), "BonusMalus_band", "Region_grp", "_p")
        sm = two_way_summary(t, min_cell, label).iloc[0].to_dict()
        region_signal.append({"model": label, "mean_z_squared": sm["mean_z_squared"],
                              "z2_per_df_incomplete": sm["z2_per_df_incomplete"], "z2_per_df_rc": sm["z2_per_df_rc"],
                              "cells_abs_z_gt_1_96": sm["cells_abs_z_gt_1_96"], "cells": sm["cells"]})

    region_z2(cv_a, "GLM-A")
    for step, spec in enumerate(SELECTION_ORDER, start=1):
        if len(accepted) >= MAX_ACCEPTED:
            break
        if spec.get("requires") and spec["requires"] not in accepted:
            log_rows.append({"step": step, "candidate": "/".join(spec["variants"]), "pair": spec["pair"],
                             "passes": False, "not_tested_reason": f"prerequisite {spec['requires']} not accepted"})
            continue
        results = []
        for name in spec["variants"]:
            r, cv, full = evaluate(name, current_cv, current_full, cfg, base, accepted, learn, gap)
            r["step"] = step
            results.append((r, cv, full))
            print(f"  step {step} {name}: impr {r['improvement_mean']:.6f} (sd {r['improvement_sd']:.6f}) "
                  f"folds {r['folds_improved']} gap {r['share_of_gap']:.1%} p {r['lrt_p']:.2g} "
                  f"mult [{r['multiplier_p01']:.2f},{r['multiplier_p99']:.2f}] pass {r['passes']}", flush=True)
        log_rows.extend(r for r, _, _ in results)
        passing = [x for x in results if x[0]["passes"]]
        if not passing:
            continue
        best, current_cv, current_full = choose_form(passing, step, tiebreaks)
        accepted.append(best["candidate"])
        steps.append({"step": step, "accepted": best["candidate"], "pair": best["pair"],
                      "improvement_mean": best["improvement_mean"], "share_of_gap": best["share_of_gap"],
                      "cv_deviance_after": current_cv["mean"],
                      "cumulative_share_of_gap": (cv_a["mean"] - current_cv["mean"]) / gap})
        region_z2(current_cv, f"after {best['candidate']}")

    # Diagnostic only (not a selection step): does the BonusMalus x Region signal survive once
    # the density interaction is in? Run when density was not accepted, so the question is
    # still answered.
    if "bm_x_density" not in accepted:
        pre_region = [a for a in accepted if not a.startswith("region_")]
        if pre_region != accepted:
            region_z2(cv_frequency(make_terms(cfg, base, pre_region + ["bm_x_density"]), learn, n_folds),
                      "diagnostic: model before region term + bm_x_density")
        region_z2(cv_frequency(make_terms(cfg, base, accepted + ["bm_x_density"]), learn, n_folds),
                  "diagnostic: final model + bm_x_density (not accepted)")

    log = pd.DataFrame(log_rows)
    log.to_csv(tables / "interaction_selection_log.csv", index=False)
    pd.DataFrame(steps).to_csv(tables / "interaction_accepted.csv", index=False)
    pd.DataFrame(region_signal).to_csv(tables / "region_bm_signal_by_step.csv", index=False)
    pd.DataFrame(tiebreaks).to_csv(tables / "interaction_parsimony_tiebreaks.csv", index=False)
    summ = pd.read_csv(tables / "two_way_ae_summary_glm_a.csv")
    nt = []
    for pair, (f1, f2) in NOT_TRANSLATED.items():
        r = summ[(summ.factor_1 == f1) & (summ.factor_2 == f2)].iloc[0]
        nt.append({"pair": pair, "mean_z_squared_vs_glm_a": r.mean_z_squared, "z2_per_df_incomplete": r.z2_per_df_incomplete,
                   "z2_per_df_rc": r.z2_per_df_rc, "cells": r.cells,
                   "cells_abs_z_gt_1_96": r.cells_abs_z_gt_1_96, "expected_by_chance": r.expected_by_chance,
                   "reason": "no excess over Poisson noise in the raw two-way A/E against GLM-A; rejected without a GLM test"})
    pd.DataFrame(nt).to_csv(tables / "interaction_not_translated.csv", index=False)

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
    two_way_summary(pd.concat(aes), min_cell, "GLM-B").to_csv(tables / "two_way_ae_summary_glm_b.csv", index=False)
    order = {c: [str(x) for x in learn[c].cat.categories] for c in cfg["glm"]["factors"]}
    pivot_tables(pd.concat(aes), "glm_b", min_cell, tables, order)
    pivot_tables(pd.read_csv(tables / "two_way_ae_top_pairs.csv"), "glm_a", min_cell, tables, order)

    # Does the B12 (and every other accepted) interaction survive with exposure as a free covariate?
    exposure_check(glm_b, accepted, cfg, base, learn).to_csv(tables / "interaction_exposure_check.csv", index=False)
    short = learn["Exposure"] < 0.25
    b12 = learn["VehBrand_grp"].astype(str) == "B12"
    pd.DataFrame([{"group": g, "policies": int(m.sum()), "mean_exposure": learn.loc[m, "Exposure"].mean(),
                   "share_policies_exposure_lt_0_25": short[m].mean(),
                   "share_bm_gt_50": (learn.loc[m, "BonusMalus"] > 50).mean(),
                   "share_vehage_le_1": (learn.loc[m, "VehAge"] <= 1).mean()}
                  for g, m in (("B12", b12), ("other brands", ~b12))]).to_csv(tables / "b12_profile.csv", index=False)

    with open(processed / "models" / "frequency_glm_b.pkl", "wb") as fh:
        pickle.dump({"glm_b": glm_b, "accepted": accepted, "cv_glm_b": cv_b}, fh)
    return {"accepted": accepted, "comparison": comp}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    out = run()
    print(out["accepted"])
    print(out["comparison"].T)
