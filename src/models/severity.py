"""Stage 5: attritional severity (Gamma GLM, log link) and burning cost.

- Unit: one row per policy with at least one claim. Response = average attritional
  severity, i.e. the sum of claim amounts capped at the large-loss threshold divided
  by the number of claim records. Prior weight = number of claim records (all records,
  not the frequency cap; DECISIONS D009).
- Factor selection: forward selection over the frequency factors on the shared CV
  folds, by weighted Gamma deviance. Severity has ~5% of the frequency model's
  observations, so it is kept simpler (DECISIONS D030).
- Large losses: the excess over the threshold, as a share of capped learn losses, is a
  flat load (DECISIONS D010).
- Burning cost per policy-year = frequency (GLM) x attritional severity x (1 + load).
"""
import pickle

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation import metrics as M
from src.evaluation.plots import COLORS, plt, save
from src.models.glm import FactorTerm, fit_glm


def severity_frame(policies: pd.DataFrame, claims: pd.DataFrame, threshold: float) -> pd.DataFrame:
    c = claims.assign(capped=claims["ClaimAmount"].clip(upper=threshold))
    agg = c.groupby("IDpol").agg(n_claims=("ClaimAmount", "size"), capped_total=("capped", "sum"))
    out = policies.merge(agg, left_on="IDpol", right_index=True, how="inner")
    out["avg_sev"] = out["capped_total"] / out["n_claims"]
    return add_severity_features(out)


def add_severity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Severity-only coarse grouping of BonusMalus: 50 (floor), 51-99 (bonus), 100+ (malus)."""
    df = df.copy()
    df["BonusMalus_sev3"] = pd.Categorical(
        np.select([df["BonusMalus"] == 50, df["BonusMalus"] < 100], ["50", "51-99"], "100+"), categories=["50", "51-99", "100+"])
    return df


SEV_EXTRA_FACTORS = {"BonusMalus_sev3": "50"}  # coarse severity-only grouping, base 50
CREDIBLE_SHARE = 0.5  # sense check: share of non-base levels whose 95% CI excludes 1


def fit_severity(factors: list, base: dict, df: pd.DataFrame):
    return fit_glm([FactorTerm(f, base[f]) for f in factors], df, df["avg_sev"].to_numpy(), "gamma",
                   weights=df["n_claims"].to_numpy())


def cv_severity(factors, base, sev, n_folds) -> dict:
    devs, fits = [], []
    for k in range(n_folds):
        tr, va = sev[sev["fold"] != k], sev[sev["fold"] == k]
        res = fit_severity(factors, base, tr)
        devs.append(M.gamma_deviance(va["avg_sev"].to_numpy(), res.predict(va), va["n_claims"].to_numpy()))
        fits.append(res)
    devs = np.array(devs)
    return {"fold_deviance": devs, "mean": devs.mean(), "sd": devs.std(ddof=1), "fits": fits}


def forward_select(candidates: list, base: dict, sev: pd.DataFrame, n_folds: int) -> tuple[list, pd.DataFrame]:
    """Add the factor with the largest mean paired CV improvement while it improves in
    every fold, the mean improvement exceeds the sd of the paired differences, and the
    relativities are credible (at least half of the non-base levels have a 95% CI that
    excludes 1 on the full-learn fit). Only one form of BonusMalus may enter."""
    chosen, rows = [], []
    current = cv_severity([], base, sev, n_folds)
    step = 0
    while True:
        step += 1
        results = []
        bm_in = any(c.startswith("BonusMalus") for c in chosen)
        for f in [c for c in candidates if c not in chosen and not (bm_in and c.startswith("BonusMalus"))]:
            cv = cv_severity(chosen + [f], base, sev, n_folds)
            diff = current["fold_deviance"] - cv["fold_deviance"]
            t = fit_severity(chosen + [f], base, sev).table()
            t = t[t["term"].str.startswith(f"{f}[")]
            credible = float(((t["rel_lower_95"] > 1) | (t["rel_upper_95"] < 1)).mean())
            r = {"step": step, "factor": f, "cv_deviance_mean": cv["mean"], "improvement_mean": diff.mean(),
                 "improvement_sd": diff.std(ddof=1), "folds_improved": int((diff > 0).sum()),
                 "relative_improvement": diff.mean() / current["mean"], "share_levels_ci_excludes_1": credible}
            r["passes_cv"] = bool(r["folds_improved"] == n_folds and r["improvement_mean"] > r["improvement_sd"])
            r["passes"] = bool(r["passes_cv"] and credible >= CREDIBLE_SHARE)
            results.append((r, cv))
        rows.extend(r for r, _ in results)
        passing = [x for x in results if x[0]["passes"]]
        if not passing:
            break
        best, current = max(passing, key=lambda x: x[0]["improvement_mean"])
        chosen.append(best["factor"])
    return chosen, pd.DataFrame(rows)


def residual_diagnostics(sev: pd.DataFrame, pred: np.ndarray, claims: pd.DataFrame, threshold: float, figures, tables, label: str):
    """Gamma deviance residuals given the IRSA fixed-amount mass points: distribution,
    calibration by predicted decile, and the residual pattern for mass-point policies."""
    y, w = sev["avg_sev"].to_numpy(), sev["n_claims"].to_numpy()
    d = 2 * (-np.log(y / pred) + (y - pred) / pred)
    r_dev = np.sign(y - pred) * np.sqrt(np.maximum(d, 0) * w)
    fixed = claims["ClaimAmount"].value_counts().head(3).index
    single_fixed = (sev["n_claims"] == 1).to_numpy() & np.isin(np.round(y, 2), np.round(fixed.to_numpy(), 2))
    dec = M.equal_exposure_bins(pred, w, 10)
    cal = pd.DataFrame({"decile": dec + 1, "obs": y * w, "pred": pred * w, "w": w, "fixed": single_fixed * w}).groupby("decile").sum()
    cal["observed_mean_severity"] = cal["obs"] / cal["w"]
    cal["predicted_mean_severity"] = cal["pred"] / cal["w"]
    cal["ae"] = cal["obs"] / cal["pred"]
    cal["share_single_fixed_amount_claims"] = cal["fixed"] / cal["w"]
    cal.reset_index().to_csv(tables / f"severity_calibration_{label}.csv", index=False)

    summary = pd.DataFrame([{
        "dataset": label, "policies": len(y), "claims": int(w.sum()),
        "share_policies_single_fixed_amount_claim": float(single_fixed.mean()),
        "deviance_residual_mean": float(r_dev.mean()), "deviance_residual_sd": float(r_dev.std()),
        "deviance_residual_skew": float(pd.Series(r_dev).skew()),
        "share_abs_residual_gt_3": float((np.abs(r_dev) > 3).mean()),
        "residual_mean_fixed_amount_policies": float(r_dev[single_fixed].mean()),
        "residual_mean_other_policies": float(r_dev[~single_fixed].mean()),
        "calibration_ae_min": float(cal["ae"].min()), "calibration_ae_max": float(cal["ae"].max()),
        "fixed_amounts_used": "; ".join(f"{v:.2f}" for v in fixed),
    }])
    summary.to_csv(tables / f"severity_residual_summary_{label}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(r_dev[~single_fixed], bins=80, color=COLORS["glm_a"], alpha=0.7, label="other policies", density=True)
    axes[0].hist(r_dev[single_fixed], bins=80, color=COLORS["gbm"], alpha=0.6, label="single fixed-amount claim", density=True)
    axes[0].set_xlabel("Gamma deviance residual")
    axes[0].legend(frameon=False)
    axes[0].set_title("Deviance residuals")
    axes[1].scatter(np.log(pred), r_dev, s=2, alpha=0.3, color=COLORS["obs"])
    axes[1].axhline(0, color=COLORS["other"])
    axes[1].set_xlabel("log predicted severity")
    axes[1].set_ylabel("deviance residual")
    axes[1].set_title("Residual vs fitted")
    axes[2].plot(cal.index, cal["observed_mean_severity"], "o-", color=COLORS["obs"], label="observed")
    axes[2].plot(cal.index, cal["predicted_mean_severity"], "s--", color=COLORS["glm_b"], label="predicted")
    axes[2].set_xlabel("decile of predicted severity (claim-weighted)")
    axes[2].set_title("Calibration")
    axes[2].legend(frameon=False)
    save(fig, figures / f"severity_residuals_{label}.png")
    return summary


def burning_cost_reconciliation(pol: pd.DataFrame, freq_pred: np.ndarray, sev_pred: np.ndarray, load: float,
                                label: str, exclude_idpol=None) -> dict:
    """Modelled = sum exposure x frequency x severity x (1 + load); actual = all claim amounts."""
    m = np.ones(len(pol), bool) if exclude_idpol is None else ~pol["IDpol"].isin(exclude_idpol).to_numpy()
    e = pol["Exposure"].to_numpy()[m]
    modelled = float((e * freq_pred[m] * sev_pred[m] * (1 + load)).sum())
    actual = float(pol["ClaimAmount"].to_numpy()[m].sum())
    return {"dataset": label, "policies": int(m.sum()), "modelled_burning_cost": modelled, "actual_losses": actual,
            "actual_over_modelled": actual / modelled,
            "actual_attritional": float(pol["ClaimAmountCapped"].to_numpy()[m].sum()),
            "modelled_attritional": float((e * freq_pred[m] * sev_pred[m]).sum()),
            "actual_claims": float(pol["ClaimNbRecorded"].to_numpy()[m].sum()),
            "modelled_claims": float((e * freq_pred[m]).sum())}


def run() -> dict:
    cfg = load_config()
    tables, figures, processed = (cfg["paths"][k] for k in ("tables", "figures", "processed"))
    u = cfg["cleaning"]["large_loss_threshold"]
    n_folds = cfg["split"]["n_folds"]
    pol = add_severity_features(pd.read_parquet(processed / "policies_banded.parquet"))
    claims = pd.read_parquet(processed / "claims.parquet")
    base = pd.read_csv(tables / "base_levels.csv").set_index("factor")["base_level"].to_dict()
    with open(processed / "models" / "frequency_stage3.pkl", "rb") as fh:
        s3 = pickle.load(fh)
    with open(processed / "models" / "frequency_glm_b.pkl", "rb") as fh:
        glm_b = pickle.load(fh)["glm_b"]

    learn_claims = claims[~claims["holdout"]]
    sev = severity_frame(pol, claims, u)
    sev_l, sev_h = sev[~sev["holdout"]], sev[sev["holdout"]]

    base = {**base, **SEV_EXTRA_FACTORS}
    factors, sel = forward_select(cfg["glm"]["factors"] + list(SEV_EXTRA_FACTORS), base, sev_l, n_folds)
    sel.to_csv(tables / "severity_selection_log.csv", index=False)
    cv_sev = cv_severity(factors, base, sev_l, n_folds)
    cv_0 = cv_severity([], base, sev_l, n_folds)
    sev_model = fit_severity(factors, base, sev_l)
    sev_model.table().to_csv(tables / "severity_coefficients.csv", index=False)
    sev_0 = fit_severity([], base, sev_l)
    yh, wh = sev_h["avg_sev"].to_numpy(), sev_h["n_claims"].to_numpy()
    pd.DataFrame([
        {"model": "Severity intercept only", "factors": "", "n_params": 1, "cv_gamma_deviance_mean": cv_0["mean"],
         "cv_gamma_deviance_sd": cv_0["sd"], "holdout_gamma_deviance": M.gamma_deviance(yh, sev_0.predict(sev_h), wh),
         "dispersion": sev_0.scale},
        {"model": "Severity GLM", "factors": ", ".join(factors), "n_params": sev_model.n_params,
         "cv_gamma_deviance_mean": cv_sev["mean"], "cv_gamma_deviance_sd": cv_sev["sd"],
         "holdout_gamma_deviance": M.gamma_deviance(yh, sev_model.predict(sev_h), wh), "dispersion": sev_model.scale},
    ]).to_csv(tables / "severity_model_comparison.csv", index=False)

    # Residual diagnostics given IRSA mass points (learn out-of-fold, and holdout)
    oof = np.empty(len(sev_l))
    for k in range(n_folds):
        m = (sev_l["fold"] == k).to_numpy()
        oof[m] = cv_sev["fits"][k].predict(sev_l[m])
    diag = pd.concat([
        residual_diagnostics(sev_l, oof, learn_claims, u, figures, tables, "learn_oof"),
        residual_diagnostics(sev_h, sev_model.predict(sev_h), learn_claims, u, figures, tables, "holdout"),
    ])
    diag.to_csv(tables / "severity_residual_summary.csv", index=False)

    # Large-loss load and burning cost
    load = float(learn_claims["ClaimAmount"].sub(u).clip(lower=0).sum() / learn_claims["ClaimAmount"].clip(upper=u).sum())
    learn, hold = pol[~pol["holdout"]], pol[pol["holdout"]]
    largest = claims.loc[claims["ClaimAmount"].idxmax(), "IDpol"]
    rec = []
    for model_name, fm in (("GLM-A", s3["glm_a"]), ("GLM-B", glm_b)):
        for lbl, d, excl in (("learn", learn, None), ("holdout", hold, None), ("holdout excl. largest claim", hold, [largest])):
            r = burning_cost_reconciliation(d, fm.predict(d), sev_model.predict(d), load, lbl, excl)
            rec.append({"frequency_model": model_name, **r})
    rec = pd.DataFrame(rec)
    rec.insert(1, "large_loss_load", load)
    rec.to_csv(tables / "burning_cost_reconciliation.csv", index=False)

    sens = threshold_sensitivity(cfg, pol, claims, glm_b, base, factors)
    sens.to_csv(tables / "large_loss_sensitivity.csv", index=False)

    with open(processed / "models" / "severity.pkl", "wb") as fh:
        pickle.dump({"severity": sev_model, "factors": factors, "load": load, "threshold": u}, fh)
    return {"factors": factors, "load": load, "reconciliation": rec, "sensitivity": sens}


def threshold_sensitivity(cfg, pol, claims, glm_b, base, factors) -> pd.DataFrame:
    """Large-loss load and holdout reconciliation at alternative thresholds, refitting the
    attritional severity (same factors) at each one."""
    rows = []
    learn_claims = claims[~claims["holdout"]]
    learn, hold = pol[~pol["holdout"]], pol[pol["holdout"]]
    largest = claims.loc[claims["ClaimAmount"].idxmax(), "IDpol"]
    for u in cfg["cleaning"]["sensitivity_thresholds"]:
        sev = severity_frame(pol, claims, u)
        model = fit_severity(factors, base, sev[~sev["holdout"]])
        load = float(learn_claims["ClaimAmount"].sub(u).clip(lower=0).sum() / learn_claims["ClaimAmount"].clip(upper=u).sum())
        r_l = burning_cost_reconciliation(learn, glm_b.predict(learn), model.predict(learn), load, "learn")
        r_h = burning_cost_reconciliation(hold, glm_b.predict(hold), model.predict(hold), load, "holdout")
        r_x = burning_cost_reconciliation(hold, glm_b.predict(hold), model.predict(hold), load, "x", [largest])
        rows.append({"threshold": u, "large_loss_load": load,
                     "learn_claims_above": int((learn_claims["ClaimAmount"] > u).sum()),
                     "learn_actual_over_modelled": r_l["actual_over_modelled"],
                     "holdout_actual_over_modelled": r_h["actual_over_modelled"],
                     "holdout_excl_largest_actual_over_modelled": r_x["actual_over_modelled"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    out = run()
    print(out["factors"], out["load"])
    print(out["reconciliation"].T)
    print(out["sensitivity"])
