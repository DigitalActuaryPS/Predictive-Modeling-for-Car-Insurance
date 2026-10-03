"""Stage 2: rating factor banding and grouping.

Bands are fixed cut points in config.yaml (chosen from the one-way tables written here)
and validated against a minimum learn-set exposure per level. Base level of every
factor = its highest-exposure level on the learn set.
"""
import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.plots import COLORS, bar_line
from src.models.glm import FactorTerm, NumericTerm, SplineTerm, cv_frequency

BANDED = ["DrivAge", "VehAge", "VehPower", "BonusMalus", "LogDensity"]


def band_labels(edges: list[float], integer: bool) -> list[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if integer:
            labels.append(f"{lo}" if hi - lo == 1 else f"{lo}-{hi - 1}")
        else:
            labels.append(f"{lo:g}-{hi:g}")
    labels.append(f"{edges[-1]}+" if integer else f"{edges[-1]:g}+")
    return labels


def band(x: pd.Series, edges: list[float]) -> pd.Series:
    integer = all(float(e).is_integer() for e in edges) and pd.api.types.is_integer_dtype(x)
    labels = band_labels(edges, integer)
    bins = [-np.inf] + list(edges[1:]) + [np.inf]
    if x.min() < edges[0]:
        raise ValueError(f"{x.name}: values below first edge {edges[0]}")
    return pd.cut(x, bins=bins, labels=labels, right=False).astype(pd.CategoricalDtype(labels, ordered=True))


def nearest_frequency_merge(df: pd.DataFrame, col: str, min_exposure: float) -> dict:
    """Map each level of col below min_exposure (learn set) to the adequately sized level
    with the closest one-way claim frequency. Returns {level: target} for merged levels."""
    g = df.groupby(col, observed=True)[["ClaimNb", "Exposure"]].sum()
    g["freq"] = g["ClaimNb"] / g["Exposure"]
    small, large = g[g["Exposure"] < min_exposure], g[g["Exposure"] >= min_exposure]
    return {lvl: (large["freq"] - f).abs().idxmin() for lvl, f in small["freq"].items()}


class Banding:
    """Fitted on the learn set; applied unchanged to any data."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["banding"]
        self.brand_map: dict = {}
        self.base: dict = {}

    def fit(self, learn: pd.DataFrame) -> "Banding":
        self.brand_map = nearest_frequency_merge(learn, "VehBrand", self.cfg["min_band_exposure"])
        out = self.transform(learn)
        for f in self.factor_columns():
            expo = out.groupby(f, observed=True)["Exposure"].sum()
            self.base[f] = expo.idxmax()
        return self

    def factor_columns(self) -> list[str]:
        return [f"{c}_band" for c in BANDED] + ["VehBrand_grp", "Region_grp", "VehGas"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in BANDED:
            out[f"{c}_band"] = band(out[c], self.cfg["edges"][c])
        brand = out["VehBrand"].astype(str).replace(self.brand_map)
        out["VehBrand_grp"] = pd.Categorical(brand, categories=sorted(brand.unique(), key=lambda s: int(s[1:])))
        region = out["Region"].astype(str).replace(self.cfg["region_groups"])
        out["Region_grp"] = pd.Categorical(region, categories=sorted(region.unique()))
        out["VehGas"] = pd.Categorical(out["VehGas"].astype(str), categories=["Diesel", "Regular"])
        return out

    def exposure_check(self, learn: pd.DataFrame) -> pd.DataFrame:
        """Every level of every factor, learn exposure, and whether it meets the minimum."""
        out = self.transform(learn)
        rows = []
        for f in self.factor_columns():
            g = out.groupby(f, observed=False).agg(
                policies=("IDpol", "size"), exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"))
            for lvl, r in g.iterrows():
                rows.append({"factor": f, "level": str(lvl), "policies": int(r["policies"]),
                             "exposure": r["exposure"], "claims": r["claims"],
                             "is_base": lvl == self.base[f],
                             "meets_minimum": r["exposure"] >= self.cfg["min_band_exposure"]})
        return pd.DataFrame(rows)


def one_way(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Learn-set one-way: exposure, claims, frequency and A/E against the overall frequency."""
    overall = df["ClaimNb"].sum() / df["Exposure"].sum()
    g = df.groupby(col, observed=True).agg(exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"))
    g["frequency"] = g["claims"] / g["exposure"]
    g["ae_vs_overall"] = g["frequency"] / overall
    return g.rename_axis("level").reset_index()


def area_vs_density(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Area", observed=True)["LogDensity"].agg(["min", "max", "mean"])
    g["overlaps_next"] = g["max"] > g["min"].shift(-1)
    g["exposure"] = df.groupby("Area", observed=True)["Exposure"].sum()
    return g.rename(columns=lambda c: f"log_density_{c}" if c in ("min", "max", "mean") else c).reset_index()


def compare_bands_vs_splines(learn: pd.DataFrame, b: Banding, cfg: dict) -> pd.DataFrame:
    """CV deviance of the main-effects GLM with each of DrivAge / BonusMalus banded vs
    as a natural spline (several df) vs (BonusMalus only) log-linear. Same folds for all.
    Paired fold differences are against the all-banded model."""
    factors = cfg["glm"]["factors"]
    learn = learn.assign(LogBonusMalus=np.log(learn["BonusMalus"]))

    def terms_with(replace: str | None, new_term_fn):
        def make():
            ts = [FactorTerm(f, b.base[f]) for f in factors if f != replace]
            return ts + ([new_term_fn()] if replace else [])
        return make

    variants = {"all banded": terms_with(None, None)}
    for k in cfg["banding"]["spline_df"]["DrivAge"]:
        variants[f"DrivAge spline df={k}"] = terms_with("DrivAge_band", lambda k=k: SplineTerm("DrivAge", k))
    for k in cfg["banding"]["spline_df"]["BonusMalus"]:
        variants[f"BonusMalus spline df={k}"] = terms_with("BonusMalus_band", lambda k=k: SplineTerm("BonusMalus", k))
    variants["BonusMalus log-linear"] = terms_with("BonusMalus_band", lambda: NumericTerm("LogBonusMalus"))

    n_folds = cfg["split"]["n_folds"]
    results = {name: cv_frequency(make, learn, n_folds) for name, make in variants.items()}
    ref = results["all banded"]["fold_deviance"]
    rows = []
    for name, r in results.items():
        diff = ref - r["fold_deviance"]  # positive = variant better
        rows.append({
            "variant": name, "n_params": r["fits"][0].n_params,
            "cv_deviance_mean": r["mean"], "cv_deviance_sd": r["sd"],
            "improvement_vs_banded_mean": diff.mean(), "improvement_vs_banded_sd": diff.std(ddof=1) if name != "all banded" else 0.0,
            "folds_improved": int((diff > 0).sum()),
            **{f"fold_{k}_deviance": d for k, d in enumerate(r["fold_deviance"])},
        })
    return pd.DataFrame(rows)


def run(compare_splines: bool = True) -> Banding:
    cfg = load_config()
    tables, figures, processed = (cfg["paths"][k] for k in ("tables", "figures", "processed"))
    policies = pd.read_parquet(processed / "policies.parquet")
    learn = policies[~policies["holdout"]]

    # Fine one-ways on raw values (evidence for the cut points)
    fine = learn.assign(LogDensity_fine=(learn["LogDensity"] * 4).round() / 4)
    for c in ["DrivAge", "VehAge", "VehPower", "BonusMalus", "LogDensity_fine", "Region", "VehBrand", "Area"]:
        one_way(fine, c).to_csv(tables / f"oneway_raw_{c}.csv", index=False)
    area_vs_density(learn).to_csv(tables / "area_vs_density.csv", index=False)

    b = Banding(cfg).fit(learn)
    check = b.exposure_check(learn)
    check.to_csv(tables / "band_exposure_check.csv", index=False)
    pd.DataFrame([{"level": k, "merged_into": v} for k, v in b.brand_map.items()]).to_csv(tables / "vehbrand_merges.csv", index=False)
    pd.DataFrame([{"factor": k, "base_level": str(v)} for k, v in b.base.items()]).to_csv(tables / "base_levels.csv", index=False)
    if not check["meets_minimum"].all():
        bad = check.loc[~check["meets_minimum"], ["factor", "level", "exposure"]]
        raise ValueError(f"levels below minimum exposure:\n{bad}")

    banded = b.transform(policies)
    learn_b = banded[~banded["holdout"]]
    for f in b.factor_columns():
        t = one_way(learn_b, f)
        t.to_csv(tables / f"oneway_band_{f}.csv", index=False)
        bar_line(t, "level", "exposure", {"observed frequency": ("frequency", COLORS["obs"])},
                 figures / f"oneway_{f}.png", f"Learn one-way: {f}", "claim frequency")
    banded.to_parquet(processed / "policies_banded.parquet", index=False)
    if compare_splines:
        compare_bands_vs_splines(learn_b, b, cfg).to_csv(tables / "band_vs_spline_cv.csv", index=False)
    return b


if __name__ == "__main__":
    b = run()
    print(b.base)
    print(b.brand_map)
