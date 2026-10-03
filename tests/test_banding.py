import pandas as pd


def test_every_level_meets_minimum_exposure(cfg):
    check = pd.read_csv(cfg["paths"]["tables"] / "band_exposure_check.csv")
    assert check["meets_minimum"].all()
    assert (check["exposure"] >= cfg["banding"]["min_band_exposure"]).all()


def test_base_level_is_highest_exposure(cfg):
    check = pd.read_csv(cfg["paths"]["tables"] / "band_exposure_check.csv")
    for factor, g in check.groupby("factor"):
        assert g.loc[g["is_base"], "level"].tolist() == [g.loc[g["exposure"].idxmax(), "level"]], factor


def test_bands_cover_every_policy(cfg):
    p = pd.read_parquet(cfg["paths"]["processed"] / "policies_banded.parquet")
    assert not p[cfg["glm"]["factors"]].isna().any().any()
