"""Stage 6: multiplicative tariff from GLM-B (frequency) and the severity GLM.

Tariff rate per policy-year = base rate x product of factor relativities x interaction
multipliers x (1 + large-loss load), rebased so that the learn set's total tariff premium
(rate x exposure) equals its total recorded claim amount. By construction the tariff
rate equals rebase x GLM-B frequency x severity x (1 + load).

Outputs: reports/tables/relativities_*.csv, reports/tariff.md, tariff vs GBM comparison,
and the GLM-A tariff used as the "current" premium in Stage 7.
"""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.evaluation import metrics as M
from src.evaluation.plots import COLORS, plt, save
from src.models import gbm as G
from src.models.severity import add_severity_features

Z = 1.959963984540054
def _sev3(lvl: str) -> str:
    """Severity group of a (possibly merged) BonusMalus band label, by its lower bound."""
    lo = int(lvl.split("-")[0].rstrip("+"))
    return "50" if lo == 50 else ("100+" if lo >= 100 else "51-99")


SEV_MAP = {"BonusMalus_band": ("BonusMalus_sev3", _sev3), "BonusMalus_bandB": ("BonusMalus_sev3", _sev3)}


def load_models(cfg):
    P = cfg["paths"]["processed"] / "models"
    with open(P / "frequency_stage3.pkl", "rb") as fh:
        s3 = pickle.load(fh)
    with open(P / "frequency_glm_b.pkl", "rb") as fh:
        b = pickle.load(fh)
    with open(P / "severity.pkl", "rb") as fh:
        sv = pickle.load(fh)
    with open(P / "frequency_glm_a_mono.pkl", "rb") as fh:
        s3["glm_a_mono"] = pickle.load(fh)["glm_a_mono"]
    return s3, b, sv


class Tariff:
    def __init__(self, freq, sev, load, learn):
        self.freq, self.sev, self.load = freq, sev, load
        modelled = float((learn["Exposure"] * self.raw_rate(learn)).sum())
        self.rebase = float(learn["ClaimAmount"].sum() / modelled)
        f0, s0 = np.exp(freq.coef[0]), np.exp(sev.coef[0])
        self.base_rate = f0 * s0 * (1 + load) * self.rebase
        self.base_frequency, self.base_severity = f0, s0

    def raw_rate(self, df):
        return self.freq.predict(df) * self.sev.predict(df) * (1 + self.load)

    def rate(self, df):
        return self.raw_rate(df) * self.rebase


def _coef(model, term):
    if term in model.names:
        i = model.names.index(term)
        return model.coef[i], model.se[i]
    return 0.0, 0.0


def factor_table(f: str, learn: pd.DataFrame, freq, sev, base_level, rate: np.ndarray) -> pd.DataFrame:
    g = learn.assign(_wr=learn["Exposure"] * rate).groupby(f, observed=True).agg(
        exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"), wr=("_wr", "sum"))
    g["observed_frequency"] = g["claims"] / g["exposure"]
    g["mean_rate"] = g["wr"] / g["exposure"]
    base_mean_rate = g.loc[g.index.astype(str) == str(base_level), "mean_rate"].iloc[0]
    rows = []
    for lvl, r in g.iterrows():
        lvl = str(lvl)
        bf, sf = _coef(freq, f"{f}[{lvl}]")
        if f in SEV_MAP:
            sev_f, mapper = SEV_MAP[f]
            bs, ss = _coef(sev, f"{sev_f}[{mapper(lvl)}]")
        else:
            bs, ss = _coef(sev, f"{f}[{lvl}]")
        sc = np.sqrt(sf**2 + ss**2)  # ASSUMPTION: frequency and severity estimates independent
        rows.append({"factor": f, "level": lvl, "is_base": lvl == str(base_level), "exposure": r["exposure"],
                     "claims": int(r["claims"]), "observed_frequency": r["observed_frequency"],
                     "freq_relativity": np.exp(bf), "freq_lower_95": np.exp(bf - Z * sf), "freq_upper_95": np.exp(bf + Z * sf),
                     "sev_relativity": np.exp(bs), "sev_lower_95": np.exp(bs - Z * ss), "sev_upper_95": np.exp(bs + Z * ss),
                     "combined_relativity": np.exp(bf + bs), "combined_lower_95": np.exp(bf + bs - Z * sc),
                     "combined_upper_95": np.exp(bf + bs + Z * sc),
                     # mix-dependent: mean tariff rate in the level / in the base level (learn), includes
                     # interactions and correlated factors
                     "effective_relativity": r["mean_rate"] / base_mean_rate})
    return pd.DataFrame(rows)


def interaction_tables(freq, learn: pd.DataFrame, accepted: list) -> dict:
    """Evaluate each accepted interaction's multiplier (with delta-method 95% CI) on a grid
    of representative policies. BonusMalus at each band's learn exposure-weighted mean."""
    rep_bm = (learn.groupby("BonusMalus_bandB", observed=True)
              .apply(lambda d: np.average(d["BonusMalus"], weights=d["Exposure"]), include_groups=False).round().astype(int))
    cov = freq.cov_unscaled * freq.scale
    out = {}
    for name in accepted:
        term = next(t for t in freq.terms if getattr(t, "name", None) == name)
        if name.startswith("young_lbm"):
            grid = [{"DrivAge": a, "age_group": lbl, "BonusMalus": int(bm), "BonusMalus_band": band}
                    for lbl, a in (("<30", 25), ("30-54", 40), ("55+", 60)) for band, bm in rep_bm.items()]
            keys = ["age_group", "BonusMalus_band", "BonusMalus"]
        elif name.startswith("b12"):
            grid = [{"VehBrand_grp": "B12", "BonusMalus": int(bm), "BonusMalus_band": band} for band, bm in rep_bm.items()]
            keys = ["BonusMalus_band", "BonusMalus"]
        elif name.startswith("region"):
            regions = sorted(term.groups)
            grid = [{"Region_grp": r, "slope_group": term.groups[r] + 1, "BonusMalus": int(bm), "BonusMalus_band": band}
                    for r in regions for band, bm in rep_bm.items()]
            keys = ["slope_group", "BonusMalus_band", "BonusMalus"]
        else:
            continue
        gdf = pd.DataFrame(grid)
        if "VehBrand_grp" not in gdf:
            gdf["VehBrand_grp"] = "B1"
        gdf["LogDensity"] = 5.0
        X, cols = term.transform(gdf)
        idx = [freq.names.index(c) for c in cols]
        eta = X @ freq.coef[idx]
        se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov[np.ix_(idx, idx)], X))
        gdf["multiplier"], gdf["lower_95"], gdf["upper_95"] = np.exp(eta), np.exp(eta - Z * se), np.exp(eta + Z * se)
        t = gdf[keys + ["multiplier", "lower_95", "upper_95"]].drop_duplicates(keys)
        if name.startswith("region"):
            members = pd.DataFrame({"Region_grp": regions, "slope_group": [term.groups[r] + 1 for r in regions],
                                    "is_base_group": [term.groups[r] == term.base_group for r in regions]})
            out[f"{name}_membership"] = members
        out[name] = t.reset_index(drop=True)
    return out


def tariff_vs_gbm(hold, rate_t, rate_g, load, figures, tables):
    """Correlation and double lift (sorted by tariff / GBM) against observed loss cost, where
    observed = capped claim amounts x (1 + load), to keep large-loss noise out of the ranking test."""
    e = hold["Exposure"].to_numpy()
    obs = hold["ClaimAmountCapped"].to_numpy() * (1 + load)
    b = M.equal_exposure_bins(rate_t / rate_g, e, 10)
    df = pd.DataFrame({"bin": b + 1, "e": e, "obs": obs, "t": rate_t * e, "g": rate_g * e, "n": hold["ClaimNb"].to_numpy()})
    t = df.groupby("bin").sum()
    for c in ("obs", "t", "g"):
        t[f"{c}_per_exposure"] = t[c] / t["e"]
    t["observed_frequency"] = t["n"] / t["e"]
    t = t.rename(columns={"e": "exposure", "obs": "observed_loss_cost", "t": "tariff_premium", "g": "gbm_premium"}).reset_index()
    t.to_csv(tables / "double_lift_tariff_vs_gbm_holdout.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t["bin"], t["obs_per_exposure"], "o-", color=COLORS["obs"], label="observed (capped x (1+load))")
    ax.plot(t["bin"], t["t_per_exposure"], "s--", color=COLORS["glm_b"], label="tariff (GLM-B)")
    ax.plot(t["bin"], t["g_per_exposure"], "D--", color=COLORS["gbm"], label="GBM-based premium")
    ax.set_xlabel("decile of tariff / GBM premium ratio")
    ax.set_ylabel("loss cost per policy-year")
    ax.set_xticks(t["bin"])
    ax.legend(frameon=False)
    ax.set_title("Holdout double lift: tariff vs GBM-based premium")
    save(fig, figures / "double_lift_tariff_vs_gbm_holdout.png")
    summary = {
        "pearson_rate": float(np.corrcoef(rate_t, rate_g)[0, 1]),
        "pearson_log_rate": float(np.corrcoef(np.log(rate_t), np.log(rate_g))[0, 1]),
        "spearman_rate": float(pd.Series(rate_t).rank().corr(pd.Series(rate_g).rank())),
        "share_policies_ratio_outside_0_9_1_1": float(((rate_t / rate_g < 0.9) | (rate_t / rate_g > 1.1)).mean()),
        "holdout_gini_capped_loss_tariff": M.lorenz_gini(obs, e, rate_t),
        "holdout_gini_capped_loss_gbm": M.lorenz_gini(obs, e, rate_g),
        "decile1_observed_over_tariff": float(t["observed_loss_cost"].iloc[0] / t["tariff_premium"].iloc[0]),
        "decile1_observed_over_gbm": float(t["observed_loss_cost"].iloc[0] / t["gbm_premium"].iloc[0]),
        "decile10_observed_over_tariff": float(t["observed_loss_cost"].iloc[-1] / t["tariff_premium"].iloc[-1]),
        "decile10_observed_over_gbm": float(t["observed_loss_cost"].iloc[-1] / t["gbm_premium"].iloc[-1]),
    }
    pd.DataFrame([summary]).to_csv(tables / "tariff_vs_gbm_summary.csv", index=False)
    return summary


def fmt_table(t: pd.DataFrame, cols: dict) -> str:
    head = "| " + " | ".join(cols.values()) + " |"
    sep = "|" + "---|" * len(cols)
    lines = [head, sep]
    for _, r in t.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:,.0f}" if c == "exposure" else f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run() -> dict:
    cfg = load_config()
    tables, figures, processed = (cfg["paths"][k] for k in ("tables", "figures", "processed"))
    pol = add_severity_features(pd.read_parquet(processed / "policies_banded.parquet"))
    learn, hold = pol[~pol["holdout"]], pol[pol["holdout"]]
    base = pd.read_csv(tables / "base_levels.csv", dtype=str).set_index("factor")["base_level"].to_dict()
    base.update(pd.read_csv(tables / "base_levels_glm_b.csv", dtype=str).set_index("factor")["base_level"].to_dict())
    s3, b, sv = load_models(cfg)
    glm_a, glm_b, accepted, sev, load = s3["glm_a"], b["glm_b"], b["accepted"], sv["severity"], sv["load"]

    tb = Tariff(glm_b, sev, load, learn)
    ta = Tariff(s3["glm_a_mono"], sev, load, learn)  # current tariff for Stage 7 (D041)
    ta_stat = Tariff(glm_a, sev, load, learn)  # GLM-A as fitted (non-monotone BonusMalus); bridge step only
    gbm = lgb.Booster(model_str=s3["gbm_model"])
    gbm_freq = lambda d: G.predict_frequency(gbm, d, cfg, s3["gbm_encoder"])  # noqa: E731
    gbm_raw = lambda d: gbm_freq(d) * sev.predict(d) * (1 + load)  # noqa: E731
    gbm_rebase = float(learn["ClaimAmount"].sum() / (learn["Exposure"] * gbm_raw(learn)).sum())

    # Relativity tables (remove stale files from earlier factor structures first)
    for old in tables.glob("relativities_*.csv"):
        old.unlink()
    md_tables, all_rel = [], []
    rate_learn = tb.rate(learn)
    for f in b["factors"]:
        t = factor_table(f, learn, glm_b, sev, base[f], rate_learn)
        t.to_csv(tables / f"relativities_{f}.csv", index=False)
        all_rel.append(t)
        md_tables.append((f, t))
    inter = interaction_tables(glm_b, learn, accepted)
    for name, t in inter.items():
        t.to_csv(tables / f"relativities_interaction_{name}.csv", index=False)

    # Rebase and premium summary
    summ = pd.DataFrame([{
        "tariff": name, "base_rate": t.base_rate, "base_frequency": t.base_frequency, "base_severity": t.base_severity,
        "large_loss_load": load, "rebase_factor": t.rebase,
        "learn_premium": float((learn["Exposure"] * t.rate(learn)).sum()), "learn_actual_losses": float(learn["ClaimAmount"].sum()),
        "holdout_premium": float((hold["Exposure"] * t.rate(hold)).sum()), "holdout_actual_losses": float(hold["ClaimAmount"].sum()),
    } for name, t in (("GLM-B tariff (proposed)", tb), ("GLM-A-mono tariff (current)", ta),
                      ("GLM-A tariff (statistical model, not used as current)", ta_stat))])
    summ["learn_premium_over_actual"] = summ["learn_premium"] / summ["learn_actual_losses"]
    summ.to_csv(tables / "tariff_summary.csv", index=False)

    # Tariff vs GBM (holdout)
    rate_t, rate_g = tb.rate(hold), gbm_raw(hold) * gbm_rebase
    vs = tariff_vs_gbm(hold, rate_t, rate_g, load, figures, tables)

    # Policy-level premiums for Stage 7 (all policies)
    prem = pol[["IDpol", "holdout", "fold", "Exposure", "ClaimNb", "ClaimAmount", "ClaimAmountCapped"]].copy()
    prem["rate_current"] = ta.rate(pol)
    prem["rate_glm_a"] = ta_stat.rate(pol)
    prem["rate_proposed"] = tb.rate(pol)
    prem["rate_gbm"] = gbm_raw(pol) * gbm_rebase
    prem.to_parquet(processed / "premiums.parquet", index=False)

    write_markdown(tb, md_tables, inter, accepted, summ, vs, cfg)
    with open(processed / "models" / "tariff.pkl", "wb") as fh:
        pickle.dump({"proposed": tb, "current": ta, "gbm_rebase": gbm_rebase}, fh)
    return {"summary": summ, "vs_gbm": vs}


def write_markdown(tb, md_tables, inter, accepted, summ, vs, cfg):
    s = summ.iloc[0]
    out = [
        "# Proposed tariff (GLM-B frequency x severity GLM)",
        "",
        "*Generated by `src/tariff/relativities.py`. Relativities are fitted on the learn set; exposure and claims are learn-set totals.*",
        "",
        f"- Base rate (per policy-year, all factors at base level): **{s.base_rate:,.2f}** = base frequency "
        f"{s.base_frequency:.5f} x base attritional severity {s.base_severity:,.2f} x (1 + large-loss load {s.large_loss_load:.4f}) "
        f"x rebase factor {s.rebase_factor:.5f}.",
        f"- Rebasing: learn premium / learn recorded losses = {s.learn_premium_over_actual:.6f}.",
        "- Premium = base rate x product of the factor relativities (combined column) x interaction multipliers, per policy-year.",
        "- 95% CIs come from GLM standard errors. Combined CIs assume frequency and severity estimates are independent (ASSUMPTION).",
        "- Severity uses only a 3-level BonusMalus (50, 51-99, 100+); every other factor has severity relativity 1.",
        "- **Effective** = mean tariff rate in the level / mean tariff rate in the base level, on the learn portfolio mix. "
        "It includes interactions and correlated factors. Read the DrivAge table with it: young drivers carry high "
        "BonusMalus (most cannot yet have reached the floor of 50), so much of their risk is priced through BonusMalus "
        "and the young-driver BonusMalus slope, and the fitted DrivAge relativities alone understate the young-driver premium.",
        "",
    ]
    cols = {"level": "Level", "exposure": "Exposure", "claims": "Claims", "observed_frequency": "Obs. freq",
            "freq_relativity": "Freq rel", "freq_lower_95": "Freq lo", "freq_upper_95": "Freq hi",
            "sev_relativity": "Sev rel", "combined_relativity": "Combined", "combined_lower_95": "Comb lo",
            "combined_upper_95": "Comb hi", "effective_relativity": "Effective"}
    for f, t in md_tables:
        out += [f"## {f}", "", fmt_table(t, cols), ""]
    out += ["## Interactions", "",
            "Multipliers apply on top of the main-effect relativities. BonusMalus-slope terms are exact per policy "
            "as (BonusMalus / 50) ** coefficient; the tables evaluate them at each band's exposure-weighted mean BonusMalus.", ""]
    for name, t in inter.items():
        out += [f"### {name}", "", fmt_table(t, {c: c for c in t.columns}), ""]
    out += ["## Tariff vs GBM (holdout)", "",
            f"Correlation of policy rates: Pearson {vs['pearson_rate']:.3f}, on log rates {vs['pearson_log_rate']:.3f}, "
            f"Spearman {vs['spearman_rate']:.3f}. Share of policies where tariff and GBM-based premium differ by more than 10%: "
            f"{100 * vs['share_policies_ratio_outside_0_9_1_1']:.1f}%. In the double lift (sorted by tariff / GBM), observed capped "
            f"loss cost over premium in the lowest decile is {vs['decile1_observed_over_tariff']:.3f} for the tariff and "
            f"{vs['decile1_observed_over_gbm']:.3f} for the GBM; in the highest decile {vs['decile10_observed_over_tariff']:.3f} "
            f"and {vs['decile10_observed_over_gbm']:.3f}. See `reports/figures/double_lift_tariff_vs_gbm_holdout.png`.", ""]
    (ROOT / "reports" / "tariff.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    r = run()
    print(r["summary"].T)
    print(r["vs_gbm"])
