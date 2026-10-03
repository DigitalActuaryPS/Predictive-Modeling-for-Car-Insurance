"""GLM-implementable interaction candidates derived from the SHAP ranking and confirmed
(or not) by two-way A/E. Each candidate is a function of the policy data returning named
columns; lbm = log(BonusMalus / 50) is 0 at the BonusMalus floor of 50.

Kept as module-level functions so fitted models can be pickled.
"""
import numpy as np
import pandas as pd


def _lbm(df):
    return np.log(df["BonusMalus"].to_numpy(float) / 50.0)


def _age(df):
    return df["DrivAge"].to_numpy()


def age_x_lbm_2grp(df):
    a, l = _age(df), _lbm(df)
    return {"lbm*age<30": l * (a < 30), "lbm*age55+": l * (a >= 55)}


def age_x_lbm_4grp(df):
    a, l = _age(df), _lbm(df)
    return {"lbm*age18-24": l * (a < 25), "lbm*age25-29": l * ((a >= 25) & (a < 30)),
            "lbm*age55-64": l * ((a >= 55) & (a < 65)), "lbm*age65+": l * (a >= 65)}


def young_lbm_senior_malus(df):
    a, l = _age(df), _lbm(df)
    return {"lbm*age<30": l * (a < 30), "age55+*BM>50": ((a >= 55) & (df["BonusMalus"].to_numpy() > 50)).astype(float)}


def b12_x_lbm(df):
    return {"lbm*B12": _lbm(df) * (df["VehBrand_grp"].astype(str).to_numpy() == "B12")}


def density_x_lbm(df):
    # centred at 5, the middle of the base LogDensity band 4.5-5.5
    return {"lbm*(LogDensity-5)": _lbm(df) * (df["LogDensity"].to_numpy() - 5.0)}


def region_x_lbm(df):
    reg = df["Region_grp"].astype(str).to_numpy()
    l = _lbm(df)
    return {f"lbm*{r}": l * (reg == r) for r in sorted(set(reg)) if r != "Centre"}


def b12_x_newcar(df):
    return {"B12*VehAge<=1": ((df["VehBrand_grp"].astype(str).to_numpy() == "B12") & (df["VehAge"].to_numpy() <= 1)).astype(float)}


def b12_x_vehage_grp(df):
    b12 = df["VehBrand_grp"].astype(str).to_numpy() == "B12"
    va = df["VehAge"].to_numpy()
    return {"B12*VehAge<=1": (b12 & (va <= 1)).astype(float), "B12*VehAge8-13": (b12 & (va >= 8) & (va <= 13)).astype(float)}


def young_x_highpower(df):
    return {"age<30*VehPower>=7": ((_age(df) < 30) & (df["VehPower"].to_numpy() >= 7)).astype(float)}


def bm_x_density(df):
    # numeric BonusMalus x numeric log density, centred at the base levels (BM 50; log density 5,
    # the middle of the base band 4.5-5.5)
    return {"(BM-50)*(LogDensity-5)": (df["BonusMalus"].to_numpy(float) - 50.0) * (df["LogDensity"].to_numpy() - 5.0)}


def b12_x_bm_linear(df):
    return {"B12*(BM-50)": (df["VehBrand_grp"].astype(str).to_numpy() == "B12") * (df["BonusMalus"].to_numpy(float) - 50.0)}


def b12_x_bm_3grp(df):
    b12 = df["VehBrand_grp"].astype(str).to_numpy() == "B12"
    bm = df["BonusMalus"].to_numpy()
    return {"B12*BM51-99": (b12 & (bm > 50) & (bm < 100)).astype(float), "B12*BM100+": (b12 & (bm >= 100)).astype(float)}


class RegionSlopeGroupTerm:
    """BonusMalus x Region in coarse form. On the data it is fitted on (a CV training set,
    or the full learn set), it fits the current model plus one log(BM/50) slope per region
    group, orders regions by that residual slope and cuts them into k groups of roughly
    equal exposure. It then carries log(BM/50) x group indicators, with the
    largest-exposure group as base. The grouping never sees the fold it is scored on."""

    def __init__(self, k: int, factors: list, base: dict, accepted: list):
        self.k, self.factors, self.base, self.accepted = k, factors, base, accepted
        self.name = f"region_bm_slope_{k}grp"
        self.groups: dict = {}
        self.base_group = None
        self.slopes: dict = {}

    def fit(self, df):
        from src.models.glm import FactorTerm, InteractionTerm, fit_glm

        terms = [FactorTerm(f, self.base[f]) for f in self.factors]
        terms += [InteractionTerm(a, CANDIDATES[a][1]) for a in self.accepted]
        terms += [InteractionTerm("region_x_lbm", region_x_lbm)]
        res = fit_glm(terms, df, df["ClaimNb"].to_numpy(), "poisson", np.log(df["Exposure"].to_numpy()))
        regions = sorted(df["Region_grp"].astype(str).unique())
        self.slopes = {r: (0.0 if r == "Centre" else res.coef[res.names.index(f"region_x_lbm:lbm*{r}")]) for r in regions}
        expo = df.groupby(df["Region_grp"].astype(str))["Exposure"].sum()
        order = sorted(regions, key=lambda r: self.slopes[r])
        cum = np.cumsum([expo[r] for r in order]) / expo.sum()
        self.groups = {r: min(int(c * self.k - 1e-9), self.k - 1) for r, c in zip(order, cum)}
        gexpo = pd.Series({g: sum(expo[r] for r in order if self.groups[r] == g) for g in range(self.k)})
        self.base_group = int(gexpo.idxmax())
        return self

    def transform(self, df):
        g = df["Region_grp"].astype(str).map(self.groups).to_numpy()
        l = _lbm(df)
        cols = {f"lbm*slopegroup{j + 1}of{self.k}": l * (g == j) for j in range(self.k) if j != self.base_group}
        return np.column_stack(list(cols.values())), [f"{self.name}:{c}" for c in cols]


# name -> (SHAP pair, function, short description)
CANDIDATES = {
    "age_x_lbm_2grp": ("DrivAge x BonusMalus", age_x_lbm_2grp, "log(BM/50) slope shifts for age <30 and 55+ (2 params)"),
    "age_x_lbm_4grp": ("DrivAge x BonusMalus", age_x_lbm_4grp, "log(BM/50) slope shifts for 18-24, 25-29, 55-64, 65+ (4 params)"),
    "young_lbm_senior_malus": ("DrivAge x BonusMalus", young_lbm_senior_malus, "log(BM/50) slope shift for age <30; step for age 55+ with BM > 50 (2 params)"),
    "b12_x_lbm": ("BonusMalus x VehBrand", b12_x_lbm, "log(BM/50) slope shift for brand B12 (1 param)"),
    "density_x_lbm": ("BonusMalus x LogDensity", density_x_lbm, "log(BM/50) slope varies linearly with log density (1 param)"),
    "region_x_lbm": ("BonusMalus x Region", region_x_lbm, "log(BM/50) slope by region group, Centre base (17 params)"),
    "b12_x_newcar": ("VehAge x VehBrand", b12_x_newcar, "B12 with vehicle age 0-1 (1 param)"),
    "b12_x_vehage_grp": ("VehAge x VehBrand", b12_x_vehage_grp, "B12 with vehicle age 0-1, and with 8-13 (2 params)"),
    "young_x_highpower": ("DrivAge x VehPower", young_x_highpower, "age <30 with VehPower 7+ (1 param)"),
    "bm_x_density": ("BonusMalus x LogDensity", bm_x_density, "(BM - 50) x (log density - 5), numeric x numeric (1 param)"),
    "b12_x_bm_linear": ("BonusMalus x VehBrand", b12_x_bm_linear, "B12 x (BM - 50) (1 param)"),
    "b12_x_bm_3grp": ("BonusMalus x VehBrand", b12_x_bm_3grp, "B12 x BM groups 51-99 and 100+, base 50 (2 params)"),
}

# Region slope-group candidates are built per fit (see RegionSlopeGroupTerm)
REGION_GROUP_CANDIDATES = {
    "region_bm_slope_2grp": ("BonusMalus x Region", 2, "log(BM/50) slope by 2 region groups formed on residual slope (1 param)"),
    "region_bm_slope_3grp": ("BonusMalus x Region", 3, "log(BM/50) slope by 3 region groups formed on residual slope (2 params)"),
}

# Owner-specified forward selection order (one step per pair; variants within a step compete)
SELECTION_ORDER = [
    {"pair": "DrivAge x BonusMalus", "variants": ["young_lbm_senior_malus"]},
    {"pair": "BonusMalus x LogDensity", "variants": ["bm_x_density"]},
    {"pair": "BonusMalus x VehBrand", "variants": ["b12_x_bm_linear", "b12_x_bm_3grp"]},
    # Originally conditional on bm_x_density being accepted. Condition dropped by the owner on
    # evidence: the region signal survives with density in (DECISIONS D029).
    {"pair": "BonusMalus x Region", "variants": ["region_bm_slope_2grp", "region_bm_slope_3grp"]},
    {"pair": "VehAge x VehBrand", "variants": ["b12_x_newcar"]},
]

# Top SHAP pairs rejected without a GLM test; reasons are filled from two_way_ae_summary_glm_a.csv
NOT_TRANSLATED = {
    "DrivAge x LogDensity": ("DrivAge_band", "LogDensity_band"),
    "DrivAge x VehPower": ("DrivAge_band", "VehPower_band"),
    "DrivAge x Region": ("DrivAge_band", "Region_grp"),
}
