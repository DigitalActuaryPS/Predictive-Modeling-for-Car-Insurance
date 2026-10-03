import pickle

import numpy as np
import pandas as pd


def test_burning_cost_reconciles_on_learn(cfg):
    rec = pd.read_csv(cfg["paths"]["tables"] / "burning_cost_reconciliation.csv")
    learn = rec[rec["dataset"] == "learn"]
    assert (abs(learn["actual_over_modelled"] - 1) < 0.01).all()


def test_large_loss_load_matches_learn_claims(cfg, processed):
    with open(cfg["paths"]["processed"] / "models" / "severity.pkl", "rb") as fh:
        sev = pickle.load(fh)
    c = processed["claims"]
    c = c[~c["holdout"]]
    u = cfg["cleaning"]["large_loss_threshold"]
    expected = (c["ClaimAmount"] - u).clip(lower=0).sum() / c["ClaimAmount"].clip(upper=u).sum()
    assert np.isclose(sev["load"], expected)


def test_holdout_reconciliation_reported_with_and_without_largest_claim(cfg):
    rec = pd.read_csv(cfg["paths"]["tables"] / "burning_cost_reconciliation.csv")
    assert {"holdout", "holdout excl. largest claim"} <= set(rec["dataset"])


def test_sensitivity_thresholds(cfg):
    sens = pd.read_csv(cfg["paths"]["tables"] / "large_loss_sensitivity.csv")
    assert sorted(sens["threshold"]) == sorted(cfg["cleaning"]["sensitivity_thresholds"])
    assert sens["large_loss_load"].is_monotonic_decreasing
