import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.glm import FactorTerm, design, fit_glm


def _glm_a_inputs(cfg):
    p = pd.read_parquet(cfg["paths"]["processed"] / "policies_banded.parquet")
    learn = p[~p["holdout"]].reset_index(drop=True)
    base = pd.read_csv(cfg["paths"]["tables"] / "base_levels.csv").set_index("factor")["base_level"].to_dict()
    return learn, [FactorTerm(f, base[f]) for f in cfg["glm"]["factors"]]


def test_irls_matches_statsmodels_on_glm_a(cfg):
    learn, terms = _glm_a_inputs(cfg)
    off = np.log(learn["Exposure"].to_numpy())
    ours = fit_glm(terms, learn, learn["ClaimNb"].to_numpy(), "poisson", off)
    X, _ = design(ours.terms, learn)
    ref = sm.GLM(learn["ClaimNb"].to_numpy(), X, family=sm.families.Poisson(), offset=off).fit(tol=1e-12)
    np.testing.assert_allclose(ours.coef, ref.params, rtol=1e-6)
    np.testing.assert_allclose(ours.se, ref.bse, rtol=1e-6)
    np.testing.assert_allclose(ours.deviance, ref.deviance, rtol=1e-9)


def test_gamma_irls_matches_statsmodels():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({"f": pd.Categorical(rng.choice(list("abc"), n)), "x": rng.normal(size=n)})
    w = rng.integers(1, 4, n).astype(float)
    mu = np.exp(7 + 0.3 * (df["f"] == "b") - 0.2 * (df["f"] == "c") + 0.1 * df["x"])
    y = rng.gamma(shape=2 * w, scale=mu / (2 * w))
    from src.models.glm import NumericTerm

    ours = fit_glm([FactorTerm("f", "a"), NumericTerm("x")], df, y, "gamma", weights=w)
    X, _ = design(ours.terms, df)
    ref = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log()), var_weights=w).fit(tol=1e-12)
    np.testing.assert_allclose(ours.coef, ref.params, rtol=1e-6)
    np.testing.assert_allclose(ours.se, ref.bse, rtol=1e-6)
