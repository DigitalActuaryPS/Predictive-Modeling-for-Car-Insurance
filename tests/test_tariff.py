import glob

import numpy as np
import pandas as pd


def test_base_level_relativity_is_one(cfg):
    files = glob.glob(str(cfg["paths"]["tables"] / "relativities_*.csv"))
    factor_files = [f for f in files if "interaction" not in f]
    assert len(factor_files) == len(cfg["glm"]["factors"])
    for f in factor_files:
        t = pd.read_csv(f)
        base = t[t["is_base"]]
        assert len(base) == 1, f
        for c in ("freq_relativity", "sev_relativity", "combined_relativity", "effective_relativity"):
            assert np.isclose(base[c].iloc[0], 1.0), (f, c)


def test_tariff_reproduces_learn_burning_cost(cfg):
    s = pd.read_csv(cfg["paths"]["tables"] / "tariff_summary.csv")
    assert (abs(s["learn_premium"] / s["learn_actual_losses"] - 1) < 0.001).all()


def test_policy_premiums_rebased(cfg):
    p = pd.read_parquet(cfg["paths"]["processed"] / "premiums.parquet")
    learn = p[~p["holdout"]]
    actual = learn["ClaimAmount"].sum()
    for c in ("rate_current", "rate_proposed", "rate_gbm", "rate_glm_a"):
        assert abs((learn["Exposure"] * learn[c]).sum() / actual - 1) < 0.001, c


def test_effective_bonusmalus_relativities_never_decrease(cfg):
    # Customer-facing NCD must not reverse (DECISIONS D039)
    t = pd.read_csv(cfg["paths"]["tables"] / "relativities_BonusMalus_bandB.csv")
    assert (np.diff(t["effective_relativity"].to_numpy()) >= -1e-9).all()


def test_glm_a_mono_is_monotone_and_reported(cfg):
    steps = pd.read_csv(cfg["paths"]["tables"] / "bm_monotonicity_steps_glm_a.csv")
    assert bool(steps["monotone"].iloc[-1])
    comp = pd.read_csv(cfg["paths"]["tables"] / "frequency_model_comparison.csv")
    assert "GLM-A-mono (impact baseline)" in set(comp["model"])
