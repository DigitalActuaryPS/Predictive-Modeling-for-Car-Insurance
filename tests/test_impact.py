import pandas as pd


def test_impact_revenue_neutral(cfg):
    p = pd.read_parquet(cfg["paths"]["processed"] / "impact.parquet")
    cur = (p["Exposure"] * p["rate_current"]).sum()
    prop = (p["Exposure"] * p["rate_proposed_rn"]).sum()
    assert abs(prop / cur - 1) < 0.001


def test_capped_rebalanced_option_revenue_neutral(cfg):
    p = pd.read_parquet(cfg["paths"]["processed"] / "impact.parquet")
    cur = (p["Exposure"] * p["rate_current"]).sum()
    assert abs((p["Exposure"] * p["rate_capped_rebalanced"]).sum() / cur - 1) < 0.001
    ratio = p["rate_capped_rebalanced"] / p["rate_current"]
    cap = cfg["impact"]["cap"]
    assert ratio.min() >= 1 - cap - 1e-9 and ratio.max() <= 1 + cap + 1e-9


def test_change_bands_cover_portfolio(cfg):
    b = pd.read_csv(cfg["paths"]["tables"] / "impact_change_bands.csv")
    assert abs(b["share_exposure"].sum() - 1) < 1e-9
