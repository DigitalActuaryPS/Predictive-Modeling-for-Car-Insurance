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
}

# Top SHAP pairs with no candidate, and why (from two_way_ae_summary_glm_a.csv)
NOT_TRANSLATED = {
    "DrivAge x LogDensity": "two-way A/E vs GLM-A shows no pattern (mean z^2 below 1)",
    "DrivAge x Region": "two-way A/E vs GLM-A shows no pattern (mean z^2 below 1)",
}
