"""Stage 7: impact (dislocation) analysis, current tariff (GLM-A) -> proposed tariff (GLM-B).

- Portfolio: every policy (learn + holdout). Premiums are annual rates per policy-year,
  weighted by exposure. Both tariffs are rebalanced to the same total premium before
  comparison (revenue neutral).
- Justification check on the holdout only: per change band, observed loss cost against
  current and proposed premium. Observed loss cost uses capped claims x (1 + load), the
  same basis as the tariff's attritional component plus flat load, so that a handful of
  large claims do not decide the test; the all-claims version is shown alongside.
- Capping: an illustrative +/- cap on each policy's year-one change, with and without
  re-balancing to revenue neutrality.
"""
import itertools
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.config import ROOT, load_config
from src.evaluation import metrics as M
from src.evaluation.plots import COLORS, plt, save

BAND_LABELS = ["< -20%", "-20% to -10%", "-10% to -5%", "-5% to +5%", "+5% to +10%", "+10% to +20%", "> +20%"]


def wmean(x, w):
    return float(np.sum(np.asarray(x) * np.asarray(w)) / np.sum(w))


def change_band(change: pd.Series, edges) -> pd.Series:
    return pd.cut(change, edges, labels=BAND_LABELS, right=False)


def run() -> dict:
    cfg = load_config()
    tables, figures, processed = (cfg["paths"][k] for k in ("tables", "figures", "processed"))
    ic = cfg["impact"]
    prem = pd.read_parquet(processed / "premiums.parquet")
    pol = pd.read_parquet(processed / "policies_banded.parquet")
    with open(processed / "models" / "severity.pkl", "rb") as fh:
        load = pickle.load(fh)["load"]
    df = pol.merge(prem[["IDpol", "rate_current", "rate_proposed", "rate_gbm"]], on="IDpol", validate="one_to_one")
    e = df["Exposure"].to_numpy()

    # Revenue neutrality on the whole portfolio
    total_current = float((e * df["rate_current"]).sum())
    k = total_current / float((e * df["rate_proposed"]).sum())
    df["rate_proposed_rn"] = df["rate_proposed"] * k
    df["change"] = df["rate_proposed_rn"] / df["rate_current"] - 1
    df["band"] = change_band(df["change"], ic["change_bands"])
    neutrality = {"total_current_premium": total_current, "rebalance_factor_proposed": k,
                  "total_proposed_premium_rebalanced": float((e * df["rate_proposed_rn"]).sum())}
    neutrality["revenue_ratio"] = neutrality["total_proposed_premium_rebalanced"] / total_current

    # 1. Histogram
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(np.clip(df["change"] * 100, -60, 60), bins=120, weights=e, color=COLORS["glm_b"])
    for x in (-20, -10, -5, 5, 10, 20):
        ax.axvline(x, color=COLORS["other"], lw=0.6, ls=":")
    ax.set_xlabel("premium change, proposed vs current, % (clipped at +/-60)")
    ax.set_ylabel("policy-years")
    ax.set_title("Distribution of premium change (revenue neutral)")
    save(fig, figures / "impact_change_histogram.png")

    # 2. Change bands
    g = df.groupby("band", observed=False)
    bands = pd.DataFrame({
        "policies": g.size(), "exposure": g["Exposure"].sum(),
        "mean_change": g.apply(lambda d: wmean(d["change"], d["Exposure"]) if len(d) else np.nan, include_groups=False),
    })
    bands["share_policies"] = bands["policies"] / bands["policies"].sum()
    bands["share_exposure"] = bands["exposure"] / bands["exposure"].sum()

    # 5. Justification on holdout: observed / premium by change band
    h = df[df["holdout"]].copy()
    h["obs_capped_loaded"] = h["ClaimAmountCapped"] * (1 + load)
    hg = h.assign(pc=h["Exposure"] * h["rate_current"], pp=h["Exposure"] * h["rate_proposed_rn"]).groupby("band", observed=False)
    just = pd.DataFrame({
        "holdout_exposure": hg["Exposure"].sum(), "holdout_claims": hg["ClaimNb"].sum(),
        "observed_capped_loaded": hg["obs_capped_loaded"].sum(), "observed_all_claims": hg["ClaimAmount"].sum(),
        "current_premium": hg["pc"].sum(), "proposed_premium": hg["pp"].sum(),
    })
    just["observed_over_current"] = just["observed_capped_loaded"] / just["current_premium"]
    just["observed_over_proposed"] = just["observed_capped_loaded"] / just["proposed_premium"]
    just["all_claims_over_current"] = just["observed_all_claims"] / just["current_premium"]
    just["all_claims_over_proposed"] = just["observed_all_claims"] / just["proposed_premium"]
    # approximate 95% band on observed/current from claim count (Poisson)
    just["ae_rel_halfwidth_95"] = 1.96 / np.sqrt(just["holdout_claims"].clip(lower=1))
    out = bands.join(just).rename_axis("change_band").reset_index()
    out.to_csv(tables / "impact_change_bands.csv", index=False)

    # 3. Mean change by rating factor (original bands)
    rows = []
    for f in cfg["glm"]["factors"]:
        for lvl, d in df.groupby(f, observed=True):
            hd = d[d["holdout"]]
            obs = float((hd["ClaimAmountCapped"] * (1 + load)).sum())
            rows.append({"factor": f, "level": str(lvl), "exposure": d["Exposure"].sum(),
                         "mean_change": wmean(d["change"], d["Exposure"]),
                         "share_exposure_change_gt_10pct": wmean(d["change"].abs() > 0.10, d["Exposure"]),
                         "holdout_claims": int(hd["ClaimNb"].sum()),
                         "holdout_observed_over_current": obs / float((hd["Exposure"] * hd["rate_current"]).sum()),
                         "holdout_observed_over_proposed": obs / float((hd["Exposure"] * hd["rate_proposed_rn"]).sum())})
    by_factor = pd.DataFrame(rows)
    by_factor.to_csv(tables / "impact_by_factor.csv", index=False)

    # 4. Top segments: two-factor combinations with material exposure
    seg = []
    for f1, f2 in itertools.combinations(cfg["glm"]["factors"], 2):
        for (l1, l2), d in df.groupby([f1, f2], observed=True):
            ex = d["Exposure"].sum()
            if ex >= ic["segment_min_exposure"]:
                hd = d[d["holdout"]]
                seg.append({"segment": f"{f1}={l1} & {f2}={l2}", "exposure": ex, "policies": len(d),
                            "mean_change": wmean(d["change"], d["Exposure"]),
                            "holdout_observed_over_current": float((hd["ClaimAmountCapped"] * (1 + load)).sum()
                                                                   / (hd["Exposure"] * hd["rate_current"]).sum()),
                            "holdout_claims": int(hd["ClaimNb"].sum())})
    seg = pd.DataFrame(seg)
    seg["abs_change"] = seg["mean_change"].abs()
    top = seg.sort_values("abs_change", ascending=False).head(ic["top_segments"]).drop(columns="abs_change")
    top.to_csv(tables / "impact_top_segments.csv", index=False)

    # 6. Capping / collaring
    cap = ic["cap"]
    ratio = df["rate_proposed_rn"] / df["rate_current"]
    capped = df["rate_current"] * ratio.clip(1 - cap, 1 + cap)
    kc = brentq(lambda s: float((e * df["rate_current"] * (s * ratio).clip(1 - cap, 1 + cap)).sum()) - total_current, 0.5, 2.0)
    capped_rn = df["rate_current"] * (kc * ratio).clip(1 - cap, 1 + cap)
    hm = df["holdout"].to_numpy()
    obs_h = (df.loc[hm, "ClaimAmountCapped"] * (1 + load)).to_numpy()

    def gini(rate):
        return M.lorenz_gini(obs_h, e[hm], np.asarray(rate)[hm])

    def moved(rate):
        return float(np.sum(e * np.abs(np.asarray(rate) - df["rate_current"])))

    g_cur, g_prop = gini(df["rate_current"]), gini(df["rate_proposed_rn"])
    cap_rows = []
    for name, rate in (("current", df["rate_current"]), ("proposed (uncapped)", df["rate_proposed_rn"]),
                       (f"capped +/-{cap:.0%}, not rebalanced", capped), (f"capped +/-{cap:.0%}, rebalanced", capped_rn)):
        gi = gini(rate)
        cap_rows.append({"option": name, "total_premium": float((e * rate).sum()),
                         "revenue_vs_current": float((e * rate).sum()) / total_current - 1,
                         "share_of_premium_movement_delivered": moved(rate) / moved(df["rate_proposed_rn"]),
                         "holdout_gini_capped_loss": gi,
                         "share_of_gini_improvement_retained": (gi - g_cur) / (g_prop - g_cur),
                         "share_exposure_at_cap": float(np.sum(e * (np.abs(np.asarray(rate) / df["rate_current"] - 1) >= cap - 1e-9)) / e.sum())
                         if "capped" in name else np.nan})
    caps = pd.DataFrame(cap_rows)
    caps["rebalance_factor"] = [np.nan, np.nan, np.nan, kc]
    caps.to_csv(tables / "impact_capping.csv", index=False)
    pd.DataFrame([neutrality]).to_csv(tables / "impact_revenue_neutrality.csv", index=False)

    write_markdown(cfg, out, by_factor, top, caps, neutrality)
    df[["IDpol", "holdout", "Exposure", "rate_current", "rate_proposed_rn", "change"]].assign(
        rate_capped=capped, rate_capped_rebalanced=capped_rn).to_parquet(processed / "impact.parquet", index=False)
    return {"bands": out, "caps": caps, "neutrality": neutrality}


def pct(x, signed=True):
    return f"{100 * x:+.1f}%" if signed else f"{100 * x:.1f}%"


def write_markdown(cfg, bands, by_factor, top, caps, neutrality):
    b = bands.set_index("change_band")
    within5 = b.loc["-5% to +5%", "share_exposure"]
    big = b.loc[["< -20%", "> +20%"], "share_exposure"].sum()
    up = b.loc[["+10% to +20%", "> +20%"]]
    down = b.loc[["< -20%", "-20% to -10%"]]
    fac = by_factor.copy()
    fac["abs"] = fac["mean_change"].abs()
    fac = fac[fac["exposure"] >= cfg["banding"]["min_band_exposure"]].sort_values("abs", ascending=False)
    biggest = fac.head(4)
    c = caps.set_index("option")
    cap = cfg["impact"]["cap"]
    nr, rb = c.loc[f"capped +/-{cap:.0%}, not rebalanced"], c.loc[f"capped +/-{cap:.0%}, rebalanced"]

    def oc(lbl):
        r = b.loc[lbl]
        return f"{r.observed_over_current:.2f} against current and {r.observed_over_proposed:.2f} against proposed"

    text = [
        "# Impact analysis: current tariff (GLM-A) to proposed tariff (GLM-B)",
        "",
        "*Generated by `src/impact/dislocation.py` from `reports/tables/impact_*.csv`.*",
        "",
        "## Who moves",
        "",
        f"Both tariffs raise the same total premium (ratio {neutrality['revenue_ratio']:.6f}), so this is a redistribution. "
        f"{pct(within5, False)} of exposure moves by less than 5%; {pct(big, False)} moves by more than 20%.",
        "",
        "| Change | Policies | Exposure | Mean change | Holdout observed / current | Holdout observed / proposed |",
        "|---|---|---|---|---|---|",
    ]
    for lbl, r in b.iterrows():
        text.append(f"| {lbl} | {int(r.policies):,} | {pct(r.share_exposure, False)} | {pct(r.mean_change)} | "
                    f"{r.observed_over_current:.2f} | {r.observed_over_proposed:.2f} |")
    text += [
        "",
        "Largest moves by factor level (at least 2,000 policy-years):",
        "",
        "| Level | Mean change | Holdout claims | Observed / current | Observed / proposed |",
        "|---|---|---|---|---|",
    ]
    for r in biggest.head(3).itertuples():
        text.append(f"| {r.factor} {r.level} | {pct(r.mean_change)} | {r.holdout_claims:,} | "
                    f"{r.holdout_observed_over_current:.2f} | {r.holdout_observed_over_proposed:.2f} |")
    text += [
        "",
        f"Most affected segment: {top.iloc[0].segment} ({pct(top.iloc[0].mean_change)}, {top.iloc[0].exposure:,.0f} "
        "policy-years); top 10 in `impact_top_segments.csv`.",
        "",
        "## Why, and is it supported",
        "",
        "Two changes drive the movements. The current tariff charges BonusMalus 61-65 more than 66-80, a reversal of the "
        "no-claims scale; the proposed tariff merges 61-80 so premium never falls as BonusMalus rises (DECISIONS D039). This is "
        "the largest single source of change. The proposed tariff also adds three BonusMalus interactions: by driver age, for "
        "brand B12, and for a group of mainly urban and southern regions.",
        "",
        "Observed loss cost on the holdout, which neither tariff was fitted on, is capped claims plus the flat large-loss load "
        "(all-claims ratios are in the table file). In every band except the largest increases, the proposed premium is closer "
        f"to experience than the current one. Above +20% the proposed tariff overshoots ({oc('> +20%')}), and so does the "
        "merged band at BonusMalus 66-70. Band ratios carry sampling error of roughly "
        f"+/-{100 * b['ae_rel_halfwidth_95'].min():.0f}% to +/-{100 * b['ae_rel_halfwidth_95'].max():.0f}% (95%).",
        "",
        "## Capping trade-off",
        "",
        f"A +/-{cap:.0%} cap on year-one change, not re-balanced, changes revenue by {pct(nr.revenue_vs_current)}. Re-balanced to "
        f"revenue neutrality (scale {rb.rebalance_factor:.4f} before capping), it delivers {pct(rb.share_of_premium_movement_delivered, False)} "
        f"of the intended movement and keeps {pct(rb.share_of_gini_improvement_retained, False)} of the holdout Gini improvement; "
        f"{pct(rb.share_exposure_at_cap, False)} of exposure sits at the cap. The largest moves add little ranking power on the "
        "holdout, consistent with the overshoot above +20%. Holdout Gini is noisy, so this supports a cap rather than proves one.",
        "",
        "## Before implementation",
        "",
        "- Retention and elasticity: the analysis is static; large increases may lapse and change the mix. No demand data.",
        "- Competitor position: large decreases may already be below market; large increases may not be achievable.",
        "- Regulatory and fairness review: region and density can proxy protected characteristics; the age interactions "
        "move young and older drivers. No fairness testing was done.",
        "- BonusMalus drives most movements and is itself a record of claims (LIMITATIONS).",
        "- A multi-year transition for policies at the cap.",
    ]
    (ROOT / "reports" / "impact_analysis.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    r = run()
    print(r["bands"].round(3).to_string())
    print(r["caps"].round(4).to_string())
    print(r["neutrality"])
