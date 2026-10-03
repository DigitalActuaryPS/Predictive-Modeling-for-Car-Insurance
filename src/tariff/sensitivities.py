"""Stage 6 sensitivities.

1. Exposure control: GLM-B refitted with log(exposure) as a free covariate, scored at
   exposure = 1, rebased to the same learn total; relativities side by side with the
   offset fit (DECISIONS D022).
2. BonusMalus:
   a. factor importance in GLM-A (relativity range; deviance increase on removal, learn and CV);
   b. French CRM sensitivity: GLM-B with log(BonusMalus/100) as an offset and no fitted
      BonusMalus terms or BonusMalus interactions;
   c. free coefficient on log(BonusMalus/100);
   d. BonusMalus distribution by driver age (who can reach the floor of 50);
   e. a generated paragraph for LIMITATIONS and README.
"""
import pickle

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.evaluation import metrics as M
from src.interactions.glm_revision import make_terms
from src.models.glm import FactorTerm, NumericTerm, cv_frequency, fit_glm

Z = 1.959963984540054
FLAG = 0.05  # owner: flag relativity moves above 5%


def rel_compare(a, b, name_a, name_b, prefixes=None) -> pd.DataFrame:
    rows = []
    for n in a.names[1:]:
        if n not in b.names or (prefixes and not n.startswith(prefixes)):
            continue
        ra, rb = np.exp(a.coef[a.names.index(n)]), np.exp(b.coef[b.names.index(n)])
        rows.append({"term": n, f"relativity_{name_a}": ra, f"relativity_{name_b}": rb, "ratio": rb / ra,
                     "flag_move_gt_5pct": abs(rb / ra - 1) > FLAG})
    return pd.DataFrame(rows)


def cv_custom(make_terms_fn, learn, n_folds, offset_fn):
    devs = []
    for k in range(n_folds):
        tr, va = learn[learn["fold"] != k], learn[learn["fold"] == k]
        res = fit_glm(make_terms_fn(), tr, tr["ClaimNb"].to_numpy(), "poisson", offset_fn(tr))
        devs.append(M.poisson_deviance(va["ClaimNb"].to_numpy(), res.predict(va, offset_fn(va))))
    devs = np.array(devs)
    return {"mean": devs.mean(), "sd": devs.std(ddof=1), "fold_deviance": devs}


def run() -> dict:
    cfg = load_config()
    tables, processed = cfg["paths"]["tables"], cfg["paths"]["processed"]
    n_folds = cfg["split"]["n_folds"]
    pol = pd.read_parquet(processed / "policies_banded.parquet")
    learn = pol[~pol["holdout"]].copy()
    learn["LogExposure"] = np.log(learn["Exposure"])
    learn["LogBM100"] = np.log(learn["BonusMalus"] / 100.0)
    base = pd.read_csv(tables / "base_levels.csv", dtype=str).set_index("factor")["base_level"].to_dict()
    base.update(pd.read_csv(tables / "base_levels_glm_b.csv", dtype=str).set_index("factor")["base_level"].to_dict())
    with open(processed / "models" / "frequency_glm_b.pkl", "rb") as fh:
        b = pickle.load(fh)
    with open(processed / "models" / "frequency_stage3.pkl", "rb") as fh:
        s3 = pickle.load(fh)
    with open(processed / "models" / "severity.pkl", "rb") as fh:
        sv = pickle.load(fh)
    glm_b, accepted, glm_a = b["glm_b"], b["accepted"], s3["glm_a"]
    factors = cfg["glm"]["factors"]
    y, off = learn["ClaimNb"].to_numpy(), np.log(learn["Exposure"].to_numpy())
    out = {}

    # 1. Exposure control
    free = fit_glm(make_terms(cfg, base, accepted, b["factors"])() + [NumericTerm("LogExposure")], learn, y, "poisson")
    comp = rel_compare(glm_b, free, "offset", "exposure_control")
    comp.to_csv(tables / "tariff_exposure_control_relativities.csv", index=False)
    sev_l = sv["severity"].predict(learn.assign(BonusMalus_sev3=pd.Categorical(
        np.select([learn["BonusMalus"] == 50, learn["BonusMalus"] < 100], ["50", "51-99"], "100+"), categories=["50", "51-99", "100+"])))
    r_off = glm_b.predict(learn) * sev_l
    r_free = free.predict(learn.assign(LogExposure=0.0)) * sev_l  # scored at exposure = 1
    r_free *= (learn["Exposure"] * r_off).sum() / (learn["Exposure"] * r_free).sum()  # same learn total
    ratio = pd.Series(r_free / r_off, index=learn.index)
    by = []
    for f in ("DrivAge_band", "BonusMalus_bandB"):
        g = learn.assign(w=learn["Exposure"], wr=learn["Exposure"] * ratio).groupby(f, observed=True)[["w", "wr"]].sum()
        by.append(pd.DataFrame({"factor": f, "level": g.index.astype(str), "mean_premium_ratio_control_over_offset": g["wr"] / g["w"]}))
    pd.concat(by).to_csv(tables / "tariff_exposure_control_premium_ratio.csv", index=False)
    out["exposure_control"] = {"log_exposure_coef": free.coef[free.names.index("LogExposure")],
                               "levels_flagged": int(comp["flag_move_gt_5pct"].sum()), "terms_compared": len(comp),
                               "premium_ratio_p01": float(ratio.quantile(0.01)), "premium_ratio_p99": float(ratio.quantile(0.99))}

    # 2a. Factor importance in GLM-A
    cv_a = s3["cv"]["glm_a"]
    rows = []
    for f in factors:
        t = glm_a.table()
        rel = np.exp(np.concatenate([[0.0], t.loc[t["term"].str.startswith(f"{f}["), "coef"].to_numpy()]))
        others = [x for x in factors if x != f]
        make = lambda others=others: [FactorTerm(x, base[x]) for x in others]  # noqa: E731
        res = fit_glm(make(), learn, y, "poisson", off)
        cv = cv_frequency(make, learn, n_folds)
        rows.append({"factor": f, "n_levels": len(rel), "relativity_min": rel.min(), "relativity_max": rel.max(),
                     "relativity_range_max_over_min": rel.max() / rel.min(),
                     "learn_deviance_increase_on_removal": res.deviance - glm_a.deviance,
                     "df_removed": glm_a.n_params - res.n_params,
                     "cv_deviance_increase_on_removal": cv["mean"] - cv_a["mean"],
                     "cv_increase_folds_positive": int((cv["fold_deviance"] - cv_a["fold_deviance"] > 0).sum())})
    imp = pd.DataFrame(rows).sort_values("learn_deviance_increase_on_removal", ascending=False)
    imp["share_of_glm_a_vs_intercept_cv_gain"] = imp["cv_deviance_increase_on_removal"] / (s3["cv"]["glm_0"]["mean"] - cv_a["mean"])
    imp.to_csv(tables / "glm_a_factor_importance.csv", index=False)

    # 2b. CRM offset: no fitted BonusMalus terms, log(BM/100) in the offset
    no_bm = [x for x in factors if x != "BonusMalus_band"]
    make_nb = lambda: [FactorTerm(x, base[x]) for x in no_bm]  # noqa: E731
    crm_off = lambda d: np.log(d["Exposure"].to_numpy()) + np.log(d["BonusMalus"].to_numpy() / 100.0)  # noqa: E731
    crm = fit_glm(make_nb(), learn, y, "poisson", crm_off(learn))
    cmp_crm = rel_compare(glm_b, crm, "glm_b", "crm_offset", prefixes=tuple(f"{x}[" for x in no_bm))
    cmp_crm.to_csv(tables / "bm_crm_offset_relativities.csv", index=False)
    cv_crm = cv_custom(make_nb, learn, n_folds, crm_off)

    # 2c. Free coefficient on log(BM/100)
    make_fr = lambda: make_nb() + [NumericTerm("LogBM100")]  # noqa: E731
    fr = fit_glm(make_fr(), learn, y, "poisson", off)
    i = fr.names.index("LogBM100")
    cv_fr = cv_custom(make_fr, learn, n_folds, lambda d: np.log(d["Exposure"].to_numpy()))
    cv_b = b["cv_glm_b"]
    pd.DataFrame([
        {"model": "GLM-A (banded BonusMalus)", "cv_deviance_mean": cv_a["mean"], "cv_deviance_sd": cv_a["sd"]},
        {"model": "GLM-B", "cv_deviance_mean": cv_b["mean"], "cv_deviance_sd": cv_b["sd"]},
        {"model": "CRM offset: log(BM/100) offset, no BM terms", "cv_deviance_mean": cv_crm["mean"], "cv_deviance_sd": cv_crm["sd"]},
        {"model": "free log(BM/100), no other BM terms", "cv_deviance_mean": cv_fr["mean"], "cv_deviance_sd": cv_fr["sd"],
         "coef_log_bm100": fr.coef[i], "se": fr.se[i], "ci_lower_95": fr.coef[i] - Z * fr.se[i],
         "ci_upper_95": fr.coef[i] + Z * fr.se[i], "z_test_coef_equals_1": (fr.coef[i] - 1) / fr.se[i]},
    ]).to_csv(tables / "bm_crm_sensitivity.csv", index=False)

    # 2d. BonusMalus by driver age
    g = learn.groupby("DrivAge_band", observed=True)
    pd.DataFrame({
        "exposure": g["Exposure"].sum(), "min_bonusmalus": g["BonusMalus"].min(),
        "share_exposure_at_bm_50": g.apply(lambda d: d.loc[d["BonusMalus"] == 50, "Exposure"].sum() / d["Exposure"].sum(), include_groups=False),
        "share_exposure_bm_ge_100": g.apply(lambda d: d.loc[d["BonusMalus"] >= 100, "Exposure"].sum() / d["Exposure"].sum(), include_groups=False),
        "mean_bonusmalus": g.apply(lambda d: np.average(d["BonusMalus"], weights=d["Exposure"]), include_groups=False),
    }).rename_axis("DrivAge_band").reset_index().to_csv(tables / "bm_by_driver_age.csv", index=False)
    by_age = learn.groupby("DrivAge")["BonusMalus"].min()
    pd.DataFrame({"DrivAge": by_age.index, "min_bonusmalus": by_age.to_numpy()}).to_csv(tables / "bm_min_by_age.csv", index=False)

    write_note(cfg)
    return out


def write_note(cfg) -> str:
    """2e. One paragraph for LIMITATIONS and README, numbers from the tables above."""
    T = cfg["paths"]["tables"]
    imp = pd.read_csv(T / "glm_a_factor_importance.csv")
    top = imp.iloc[0]
    crm = pd.read_csv(T / "bm_crm_sensitivity.csv").set_index("model")
    fr = crm.loc["free log(BM/100), no other BM terms"]
    cmp_crm = pd.read_csv(T / "bm_crm_offset_relativities.csv")
    age = pd.read_csv(T / "bm_by_driver_age.csv").set_index("DrivAge_band")
    text = (
        f"**BonusMalus.** BonusMalus is the strongest rating factor in GLM-A: removing it raises learn deviance by "
        f"{top.learn_deviance_increase_on_removal:,.0f} and CV deviance by {top.cv_deviance_increase_on_removal:.6f}, "
        f"{100 * top.share_of_glm_a_vs_intercept_cv_gain:.0f}% of GLM-A's whole CV gain over an intercept-only model "
        f"(`glm_a_factor_importance.csv`). It is not an independent risk characteristic: it is the insurer's record of past "
        f"claims, so it partly double-counts the claims the model is predicting and depends on the bonus-malus rules in force "
        f"(endogeneity). In a GLM with the other factors, claim frequency is associated with BonusMalus at a power of "
        f"{fr.coef_log_bm100:.2f} (95% CI {fr.ci_lower_95:.2f} to {fr.ci_upper_95:.2f}), against 1 if it moved one-for-one "
        f"with the statutory coefficient. This is an association, not evidence that the scale is mispriced: BonusMalus is "
        f"entangled with driving experience, because young drivers start high and move down with experience as well as claims. "
        f"Imposing the scale as an offset (no fitted BonusMalus terms) moves {int(cmp_crm.flag_move_gt_5pct.sum())} of "
        f"{len(cmp_crm)} other relativities by more than 5%, most of all young-driver relativities, which BonusMalus otherwise "
        f"partly absorbs (`bm_crm_offset_relativities.csv`). Under the statutory scale (ASSUMPTION: Code des assurances art. "
        f"A121-1, not verified here) the floor of 50 takes over a decade of claim-free driving, so young drivers should rarely "
        f"be at 50. In the learn data {100 * age.loc['18-20', 'share_exposure_at_bm_50']:.1f}% of exposure at ages 18-20 is at 50, "
        f"against {100 * age.loc['30-34', 'share_exposure_at_bm_50']:.0f}% at 30-34 and {100 * age.loc['45-54', 'share_exposure_at_bm_50']:.0f}% "
        f"at 45-54 (`bm_by_driver_age.csv`). For most young drivers BonusMalus therefore measures experience rather than claims, "
        f"which is why the DrivAge x BonusMalus interaction exists; for the minority at 50 the recorded BonusMalus is probably "
        f"not their own record (for example a transferred or main-driver coefficient), which the data cannot confirm. "
        f"Results are not transferable to a market with a different no-claims system, "
        f"such as UK NCD."
    )
    (ROOT / "reports" / "bonus_malus_note.md").write_text(text + "\n")
    return text


if __name__ == "__main__":
    print(run())
    print(write_note(load_config()))
