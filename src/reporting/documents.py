"""Stage 8: writes README.md and LIMITATIONS.md from reports/tables (no hand-typed results).

Judgement text is fixed in this file; every number is read from a table produced by the
pipeline, so the documents cannot drift from the code.
"""
import pandas as pd

from src.config import ROOT, load_config


def pct(x, d=1, signed=False):
    return f"{100 * x:+.{d}f}%" if signed else f"{100 * x:.{d}f}%"


def load(cfg):
    T = cfg["paths"]["tables"]
    r = lambda n, **k: pd.read_csv(T / f"{n}.csv", **k)  # noqa: E731
    return {
        "src": r("data_source"), "log": r("cleaning_log"), "ll": r("large_losses_by_split").set_index("split"),
        "split": r("split_balance"), "summ": r("data_summary").set_index("dataset"),
        "comp": r("frequency_model_comparison").set_index("model"), "gap": r("gap_closed").iloc[0],
        "acc": r("interaction_accepted"), "nt": r("interaction_not_translated"),
        "sel": r("interaction_selection_log"), "sev": r("severity_model_comparison").set_index("model"),
        "sevc": r("severity_coefficients"), "rec": r("burning_cost_reconciliation"),
        "tsum": r("tariff_summary").set_index("tariff"), "vs": r("tariff_vs_gbm_summary").iloc[0],
        "drv": r("relativities_DrivAge_band"), "bm": r("relativities_BonusMalus_bandB"),
        "bridge": r("impact_bridge"), "bands": r("impact_change_bands").set_index("change_band"),
        "caps": r("impact_capping").set_index("option"), "sens": r("large_loss_sensitivity"),
        "expo": r("log_exposure_diagnostic").iloc[0], "expo_seg": r("exposure_by_segment"),
        "ctrl": r("tariff_exposure_control_premium_ratio"), "ctrl_rel": r("tariff_exposure_control_relativities"),
        "mono": r("bm_monotonicity_steps"), "ver": r("casdatasets_version_comparison"),
        "shap": r("shap_run").iloc[0], "tie": r("interaction_parsimony_tiebreaks"),
    }


def results_table(D) -> str:
    c = D["comp"]
    rows = [("Intercept only", "Intercept only"), ("GLM-A: main effects, banded", "GLM-A (current tariff)"),
            ("GLM-B: GLM-A + 3 BonusMalus interactions", "GLM-B (proposed tariff)"), ("GBM: LightGBM Poisson", "GBM (LightGBM Poisson)")]
    out = ["| Model | Parameters | CV deviance, mean (sd) | Holdout deviance | Holdout Gini |", "|---|---|---|---|---|"]
    for label, key in rows:
        r = c.loc[key]
        n = f"{int(r.n_params)} trees" if key.startswith("GBM") else f"{int(r.n_params)}"
        out.append(f"| {label} | {n} | {r.cv_deviance_mean:.6f} ({r.cv_deviance_sd:.6f}) | {r.holdout_deviance:.6f} | {r.holdout_gini:.3f} |")
    m = c.loc["GLM-A-mono (impact baseline)"]
    out += ["", f"Deviance is the mean Poisson deviance per policy on claim counts (DECISIONS D019). CV uses 5 grouped folds of "
            f"the learn set; the holdout (20%) was used only for final evaluation. The impact analysis uses GLM-A-mono as the "
            f"current tariff: GLM-A with BonusMalus 61-80 merged so premium never falls as BonusMalus rises "
            f"({int(m.n_params)} parameters, CV {m.cv_deviance_mean:.6f}, holdout {m.holdout_deviance:.6f}; D041)."]
    return "\n".join(out)


def rel_table(t: pd.DataFrame, label: str) -> str:
    out = [f"| {label} | Fitted (95% CI) | Effective |", "|---|---|---|"]
    for r in t.itertuples():
        out.append(f"| {r.level} | {r.combined_relativity:.2f} ({r.combined_lower_95:.2f}-{r.combined_upper_95:.2f}) "
                   f"| {r.effective_relativity:.2f} |")
    return "\n".join(out)


def bm_short(cfg) -> str:
    """Short form of reports/bonus_malus_note.md for the README (same numbers)."""
    T = cfg["paths"]["tables"]
    imp = pd.read_csv(T / "glm_a_factor_importance.csv").iloc[0]
    fr = pd.read_csv(T / "bm_crm_sensitivity.csv").set_index("model").loc["free log(BM/100), no other BM terms"]
    age = pd.read_csv(T / "bm_by_driver_age.csv").set_index("DrivAge_band")
    return (f"**BonusMalus.** It carries {pct(imp.share_of_glm_a_vs_intercept_cv_gain, 0)} of GLM-A's cross-validated gain "
            f"and is itself a record of past claims (endogeneity). Claim frequency is associated with the BonusMalus "
            f"coefficient to the power {fr.coef_log_bm100:.2f} (95% CI {fr.ci_lower_95:.2f}-{fr.ci_upper_95:.2f}), but "
            f"BonusMalus is entangled with driving experience: {pct(age.loc['18-20', 'share_exposure_at_bm_50'])} of "
            f"exposure at ages 18-20 is at the floor of 50 against {pct(age.loc['45-54', 'share_exposure_at_bm_50'], 0)} at "
            f"45-54. Full note in [LIMITATIONS.md](LIMITATIONS.md).")


def readme(D, cfg) -> str:
    s, c, g, vs = D["summ"], D["comp"], D["gap"], D["vs"]
    src = D["src"].set_index("file")
    lg = D["log"].set_index("step")
    acc = D["acc"]
    sel = D["sel"]
    sev_f = D["sev"].loc["Severity GLM", "factors"]
    ll = D["ll"]
    br = D["bridge"].set_index("step")
    bands = D["bands"]
    toward = sum(abs(r.observed_over_proposed - 1) < abs(r.observed_over_current - 1) for r in bands.itertuples())
    capr = D["caps"].loc[f"capped +/-{cfg['impact']['cap']:.0%}, rebalanced"]
    rec = D["rec"].set_index(["frequency_model", "dataset"])
    ctrl = D["ctrl"].set_index(["factor", "level"])["mean_premium_ratio_control_over_offset"]
    drv, bm = D["drv"], D["bm"]
    plain = {
        "young_lbm_senior_malus": "a steeper BonusMalus effect for drivers under 30 plus a loading for drivers 55+ above the floor of 50",
        "b12_x_bm_linear": "a flatter BonusMalus effect for brand B12",
        "b12_x_bm_3grp": "a flatter BonusMalus effect for brand B12",
        "region_bm_slope_2grp": "a flatter BonusMalus effect in a group of mainly urban and southern regions",
        "region_bm_slope_3grp": "BonusMalus slopes for three region groups",
    }
    descs = {r.accepted: plain.get(r.accepted, sel[(sel.step == r.step) & (sel.candidate == r.accepted)].iloc[0].description)
             for r in acc.itertuples()}

    text = f"""# fremtpl2-motor-pricing

Motor third-party liability pricing on the French MTPL data (freMTPL2, CASdatasets): frequency and severity GLMs, a GBM
benchmark, SHAP interactions used to improve the GLM, a multiplicative tariff and a dislocation analysis.

## Summary

On {int(s.loc['cleaned (all)', 'policies']):,} policies ({s.loc['cleaned (all)', 'exposure']:,.0f} policy-years), a main-effects
Poisson GLM (GLM-A) reaches holdout Gini {c.loc['GLM-A (current tariff)', 'holdout_gini']:.3f} against
{c.loc['GBM (LightGBM Poisson)', 'holdout_gini']:.3f} for a LightGBM benchmark. SHAP interaction values, checked against raw
actual-versus-expected tables, gave three BonusMalus interactions; adding them (GLM-B) closes {pct(g.cv_gap_closed_by_glm_b)}
of the cross-validated deviance gap to the GBM ({pct(g.holdout_gap_closed_by_glm_b)} on holdout). Severity is a Gamma GLM on
claims capped at {cfg['cleaning']['large_loss_threshold']:,} plus a flat large-loss load.

## Data and cleaning

| Item | Treatment | Ref. |
|---|---|---|
| Source | CASdatasets {src.loc['freMTPL2freq', 'casdatasets_version']}; claim counts reconcile with the claims file (unlike pre-2022 releases) | D001-D005 |
| Exposure | Capped at 1 year ({int(lg.loc['cap Exposure at 1.0', 'policies_affected']):,} policies); short exposures kept | D008 |
| Claim count | Capped at 4 for frequency ({int(lg.loc['cap ClaimNb at 4 (frequency only)', 'claims_affected'])} claims) | D009 |
| Large losses | Above {cfg['cleaning']['large_loss_threshold']:,}: {int(ll.loc['learn', 'large_claims'])} learn claims; excess over the threshold is {pct(ll.loc['learn', 'excess_share_of_amount'])} of claim cost; load {ll.loc['learn', 'large_load_on_capped']:.3f} | D010 |
| Split | 20% holdout + 5 CV folds, grouped by identical rating covariates | D011 |
| Factors | Bands of at least {cfg['banding']['min_band_exposure']:,} policy-years; Area dropped (banded density) | D013-D016 |

Decisions and evidence: [DECISIONS.md](DECISIONS.md).

## Frequency results

{results_table(D)}

![Holdout lift](reports/figures/lift_holdout.png)
![Holdout double lift, GBM vs GLM-B](reports/figures/double_lift_gbm_vs_glm_b_holdout.png)

## Interactions

The GBM's strongest interactions involve BonusMalus, led by driver age: a high bonus-malus coefficient means
inexperience for a young driver but recent claims for an older one. Accepted, one pair at a time (all 5 folds improve,
at least 5% of the gap, parsimony tie-break): {'; '.join(descs.values())}. Driver age x density and driver age x region
showed no pattern in the raw data and were rejected as GBM artefacts. The region x BonusMalus grouping is data-driven,
with no identified cause, and would need fairness and regulatory review before use. See
[reports/shap_interactions.md](reports/shap_interactions.md).

## Tariff

Base rate {D['tsum'].loc['GLM-B tariff (proposed)', 'base_rate']:.2f} per policy-year; severity varies only by a
three-level BonusMalus. Fitted driver-age relativities mislead on their own because young drivers' risk is carried by
BonusMalus, so the effective relativity (mean premium relative to the base level, interactions included) is shown too.

{rel_table(drv, 'Driver age')}

{rel_table(bm, 'BonusMalus')}

BonusMalus 61-80 is a single band so that premium never falls as BonusMalus rises (D039); the original claim-frequency
spike at 61-65 that required this is unexplained by the available data.

**Main trade-off of a GLM tariff.** Against a GBM-based premium on the holdout (correlation {vs.pearson_rate:.2f}), observed
loss cost is {vs.decile1_observed_over_tariff:.2f}x the tariff premium in the decile where the tariff is cheapest relative to
the GBM ({vs.decile1_observed_over_gbm:.2f}x the GBM's) and {vs.decile10_observed_over_tariff:.2f}x in the dearest
({vs.decile10_observed_over_gbm:.2f}x). Transparency costs these tails. See [reports/tariff.md](reports/tariff.md).

## Impact

| Step (revenue neutral) | Exposure moving > 5% | Mean absolute change |
|---|---|---|
""" + "\n".join(f"| {k} | {pct(r.share_exposure_moving_gt_5pct)} | {pct(r.mean_abs_change)} |" for k, r in br.iterrows()) + f"""

The interactions, not the no-claims fix, drive the movement. On the holdout the proposed premium is closer to observed
loss cost in {toward} of {len(bands)} change bands but overshoots the largest increases; a rebalanced +/-15% cap delivers
{pct(capr.share_of_premium_movement_delivered)} of the movement in year one. See
[reports/impact_analysis.md](reports/impact_analysis.md).

## Limitations

- No dates: no trending, inflation or development; claims treated as fully developed.
- Exposure is not proportional (free coefficient {D['expo'].coef_log_exposure:.2f}); treating it as a control would move
  premium {pct(ctrl.loc[('DrivAge_band', '18-20')] - 1, signed=True)} at ages 18-20 and {pct(ctrl.loc[('DrivAge_band', '75+')] - 1, signed=True)} at 75+.
- Holdout actual over modelled burning cost is {rec.loc[('GLM-B', 'holdout'), 'actual_over_modelled']:.2f}, or
  {rec.loc[('GLM-B', 'holdout excl. largest claim'), 'actual_over_modelled']:.2f} without its largest claim.
- Burning cost only; no demand, competitor or fairness analysis.
- French TPL with a statutory bonus-malus scale; not transferable to UK motor.

{bm_short(cfg)}

## Reproduce

```
git clone <this repository> && cd <repository>
pip install -r requirements.txt
make all
```

`make all` downloads the data, runs every stage and the tests; per-stage runtimes are written to `reports/tables/pipeline_runtime.csv`.

## Sources and credits

- C. Dutang and A. Charpentier, *CASdatasets*, R package (freMTPL2freq, freMTPL2sev).
- A. Noll, R. Salzmann and M. V. Wüthrich (2020), *Case study: French motor third-party liability claims*, SSRN 3164764.
- GLMs, LightGBM, SHAP (Lundberg et al.) and Friedman's H-statistic are standard methods, used as published.

[AUTHOR NAME, ROLE]
"""
    return text


def limitations(D, cfg) -> str:
    sens = D["sens"]
    sens_t = "\n".join(f"| {r.threshold:,} | {r.learn_claims_above} | {r.large_loss_load:.3f} | {r.load_min_across_folds:.3f}-{r.load_max_across_folds:.3f} "
                       f"| {pct(r.share_of_burning_cost_in_flat_load)} | {r.holdout_attritional_actual_over_modelled:.3f} |" for r in sens.itertuples())
    rec = D["rec"].set_index(["frequency_model", "dataset"])
    e, es = D["expo"], D["expo_seg"].set_index(["factor", "level"])
    ctrl = D["ctrl"].set_index(["factor", "level"])["mean_premium_ratio_control_over_offset"]
    ver = D["ver"]
    old = ver.iloc[0]
    bm_note = (ROOT / "reports" / "bonus_malus_note.md").read_text().strip()
    shap = D["shap"]
    return f"""# Limitations

*Generated by `src/reporting/documents.py`; numbers come from `reports/tables/`.*

## Data

- **No dates.** Neither file has policy or claim dates. There is no trending, claims inflation or seasonality
  adjustment, and no way to check stability over time. The only statement on the period in the CASdatasets
  documentation is: "{D['src'].iloc[0].documented_period}" (`data_source.csv`). This cannot be verified from the data.
- **Development.** Claim amounts are treated as fully developed ("seen as at a recent date" per the source); no IBNR or
  development adjustment is possible.
- **Source versions.** Results apply to CASdatasets 1.2-1. Releases before 2022 have {int(old.freq_rows):,} policies and
  {old.freq_claimnb_total:,.0f} claims on the frequency file, so published results on those releases are not directly
  comparable (DECISIONS D005).
- **Single cross-section.** One portfolio at one point in time. Interactions found here, including the GBM's, may not be
  stable over time or across insurers.
- **Fixed-amount claims.** A large share of claims are paid at fixed amounts under the French IRSA-IDA convention. The
  Gamma mean model is calibrated, but severity standard errors are approximate (D032).

## Exposure

Claims are not proportional to time on risk: with log(exposure) as a free covariate its coefficient is
{e.coef_log_exposure:.3f} (95% CI {e.ci_lower_95:.3f} to {e.ci_upper_95:.3f}). Short exposures are concentrated among
young drivers ({pct(es.loc[('DrivAge_band', '18-20'), 'share_policies_exposure_lt_0_25'])} of policies at 18-20 under 0.25
years, against {pct(es.loc[('all', 'all'), 'share_policies_exposure_lt_0_25'])} overall), consistent with cancellation after a
claim, which the data cannot confirm. The main models keep the offset. Treating exposure as a control instead would change
premium by {pct(ctrl.loc[('DrivAge_band', '18-20')] - 1, signed=True)} at ages 18-20 and {pct(ctrl.loc[('DrivAge_band', '75+')] - 1, signed=True)}
at 75+, and by {pct(ctrl.loc[('BonusMalus_bandB', '100-110')] - 1, signed=True)} at BonusMalus 100-110 (D022, D037). The overall
premium level is right for the current mix of short and full-year policies; a different mix would need re-basing.

## BonusMalus

{bm_note}

The no-claims scale in the proposed and current tariffs was made monotone by merging BonusMalus 61-80 (D039, D041). On the
learn set this is equivalent to pooling bands until the one-way claim frequency rises monotonically. The cause of the
original reversal was not investigated.

## Large losses

Burning cost is attritional cost (claims capped at {cfg['cleaning']['large_loss_threshold']:,}) plus a flat load. The load does not
vary by risk, and it is volatile. Holdout actual over modelled burning cost is
{rec.loc[('GLM-B', 'holdout'), 'actual_over_modelled']:.3f}, or {rec.loc[('GLM-B', 'holdout excl. largest claim'), 'actual_over_modelled']:.3f}
without its single largest claim. Total reconciliation does not depend on the threshold, because the load is calibrated
on the same learn losses; what changes is shown below.

| Threshold | Learn claims above | Load | Load range across 5 learn folds | Share of burning cost in flat load | Holdout attritional actual / modelled |
|---|---|---|---|---|---|
{sens_t}

## Scope

- **Burning cost only.** No expenses, commission, profit, reinsurance or investment income.
- **Static impact analysis.** No demand, conversion, retention or competitor data; policyholder reactions to price
  changes are not modelled.
- **Market.** French third-party liability, with its legal context, claims conventions and statutory bonus-malus scale.
  Not directly transferable to UK motor (comprehensive cover, NCD, different claims environment).
- **Fairness.** Region and density can act as proxies for protected characteristics, and age is a rating factor. No
  fairness testing was done beyond noting this.

## Method

- **SHAP subsample.** Interaction values were computed on {int(shap.rows):,} learn policies, not the full learn set (D023).
  The H-statistic cross-check used 1,000.
- **Interaction selection.** The rule (D018) involves owner-set thresholds: a 5% materiality floor and a cap of 5. The
  region grouping is derived from the response and should be re-estimated on new data (D029).
- **Hyperparameters.** The GBM was tuned on the same CV folds used to report it; its CV deviance is slightly optimistic
  (the holdout is unbiased; D020).
- **Assumptions.** Every assumption is marked ASSUMPTION in DECISIONS.md.
"""


def render() -> tuple[str, str]:
    cfg = load_config()
    D = load(cfg)
    return readme(D, cfg), limitations(D, cfg)


def run() -> dict:
    r, lim = render()
    (ROOT / "README.md").write_text(r)
    (ROOT / "LIMITATIONS.md").write_text(lim)
    return {"readme_words": len(r.split()), "limitations_words": len(lim.split())}


if __name__ == "__main__":
    print(run())
