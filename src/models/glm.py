"""Log-link GLMs (Poisson frequency, Gamma severity) fitted by IRLS in numpy.

statsmodels' GLM solves each IRLS step by a pseudo-inverse of the full n x p weighted
design (23 s for one 33-parameter fit on 678k rows, reports/tables/shap_runtime_benchmark.csv).
Forward selection needs ~100 fits, so this module solves the normal equations
X'WX b = X'Wz by Cholesky instead. Coefficients agree with statsmodels to 1e-6 relative
tolerance (tests/test_glm.py).

Design matrices are built from terms. Each term is fitted on training data (base
levels, spline knots) and applied unchanged to other data.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln

from src.evaluation.metrics import gamma_deviance, poisson_deviance


# ---------------------------------------------------------------- design terms
class FactorTerm:
    """Treatment-coded categorical; base level fixed at construction (highest learn exposure)."""

    def __init__(self, col: str, base):
        self.col, self.base = col, base
        self.levels: list = []

    def fit(self, df: pd.DataFrame) -> "FactorTerm":
        cats = df[self.col].cat.categories if isinstance(df[self.col].dtype, pd.CategoricalDtype) else sorted(df[self.col].unique())
        present = set(df[self.col].unique())
        self.levels = [lvl for lvl in cats if lvl != self.base and lvl in present]
        return self

    def transform(self, df: pd.DataFrame):
        x = df[self.col].to_numpy()
        return np.column_stack([(x == lvl).astype(float) for lvl in self.levels]), [f"{self.col}[{lvl}]" for lvl in self.levels]


class NumericTerm:
    def __init__(self, col: str):
        self.col = col

    def fit(self, df):
        return self

    def transform(self, df):
        return df[[self.col]].to_numpy(float), [self.col]


class SplineTerm:
    """Natural cubic regression spline (patsy cr), centred, knots from training quantiles."""

    def __init__(self, col: str, df_spline: int):
        self.col, self.k = col, df_spline
        self.info = None

    def fit(self, df):
        self.info = dmatrix(f"cr(x, df={self.k}, constraints='center') - 1", {"x": df[self.col].to_numpy(float)}).design_info
        return self

    def transform(self, df):
        (m,) = build_design_matrices([self.info], {"x": df[self.col].to_numpy(float)})
        return np.asarray(m), [f"cr({self.col},{self.k})[{i}]" for i in range(m.shape[1])]


class InteractionTerm:
    """Columns produced by a function of the data (used for Stage 4 candidates)."""

    def __init__(self, name: str, fn):
        self.name, self.fn = name, fn

    def fit(self, df):
        return self

    def transform(self, df):
        cols = self.fn(df)  # dict of name -> array
        return np.column_stack([np.asarray(v, float) for v in cols.values()]), [f"{self.name}:{k}" for k in cols]


def design(terms: list, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    mats, names = [np.ones((len(df), 1))], ["Intercept"]
    for t in terms:
        m, n = t.transform(df)
        mats.append(m)
        names.extend(n)
    return np.hstack(mats), names


# ---------------------------------------------------------------- IRLS fitter
@dataclass
class GLMResult:
    family: str
    names: list
    coef: np.ndarray
    cov_unscaled: np.ndarray
    scale: float
    deviance: float
    loglik: float
    n: int
    n_iter: int
    terms: list = field(repr=False, default_factory=list)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov_unscaled) * self.scale)

    @property
    def n_params(self) -> int:
        return len(self.coef)

    def table(self) -> pd.DataFrame:
        z = 1.959963984540054
        return pd.DataFrame({
            "term": self.names, "coef": self.coef, "se": self.se,
            "relativity": np.exp(self.coef),
            "rel_lower_95": np.exp(self.coef - z * self.se), "rel_upper_95": np.exp(self.coef + z * self.se),
        })

    def linear_predictor(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = design(self.terms, df)
        return X @ self.coef

    def predict(self, df: pd.DataFrame, offset: np.ndarray | None = None) -> np.ndarray:
        eta = self.linear_predictor(df)
        return np.exp(eta if offset is None else eta + offset)


def fit_glm(terms: list, df: pd.DataFrame, y: np.ndarray, family: str = "poisson",
            offset: np.ndarray | None = None, weights: np.ndarray | None = None,
            tol: float = 1e-10, max_iter: int = 50) -> GLMResult:
    """Log-link GLM by IRLS. Poisson: y counts, offset log(exposure). Gamma: y average
    severity, weights = claim counts (prior weights)."""
    terms = [t.fit(df) for t in terms]
    X, names = design(terms, df)
    y = np.asarray(y, float)
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    pw = np.ones(n) if weights is None else np.asarray(weights, float)

    # Start from the intercept-only fit
    ybar = np.sum(pw * y) / np.sum(pw * np.exp(off)) if family == "poisson" else np.sum(pw * y) / np.sum(pw)
    beta = np.zeros(p)
    beta[0] = np.log(ybar)
    eta = X @ beta + off
    dev_old = np.inf
    for it in range(1, max_iter + 1):
        mu = np.exp(eta)
        w = pw * mu if family == "poisson" else pw.copy()  # log link: w = pw * mu**2 / V(mu)
        z = eta - off + (y - mu) / mu
        XtW = X.T * w
        c = cho_factor(XtW @ X)
        beta = cho_solve(c, XtW @ z)
        eta = X @ beta + off
        mu = np.exp(eta)
        dev = _deviance(family, y, mu, pw)
        if abs(dev - dev_old) <= tol * (abs(dev) + 0.1):
            break
        dev_old = dev
    else:
        raise RuntimeError(f"IRLS did not converge in {max_iter} iterations")

    w = pw * mu if family == "poisson" else pw
    cov = cho_solve(cho_factor((X.T * w) @ X), np.eye(p))
    if family == "poisson":
        scale = 1.0
        loglik = float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
    else:
        scale = float(np.sum(pw * (y - mu) ** 2 / mu**2) / (n - p))  # Pearson estimate of dispersion
        loglik = np.nan
    return GLMResult(family, names, beta, cov, scale, dev, loglik, n, it, terms)


def _deviance(family: str, y, mu, pw) -> float:
    if family == "poisson":
        return poisson_deviance(y, mu) * len(y)
    return gamma_deviance(y, mu, pw) * np.sum(pw)


# ---------------------------------------------------------------- cross-validation
def cv_frequency(make_terms, learn: pd.DataFrame, n_folds: int, **fit_kw) -> dict:
    """Fit on four folds, score Poisson deviance on the fifth. make_terms() returns fresh
    (unfitted) terms. Returns per-fold deviances and per-fold fit results."""
    devs, fits = [], []
    for k in range(n_folds):
        tr, va = learn[learn["fold"] != k], learn[learn["fold"] == k]
        res = fit_glm(make_terms(), tr, tr["ClaimNb"].to_numpy(), "poisson", np.log(tr["Exposure"].to_numpy()), **fit_kw)
        mu = res.predict(va, np.log(va["Exposure"].to_numpy()))
        devs.append(poisson_deviance(va["ClaimNb"].to_numpy(), mu))
        fits.append(res)
    devs = np.array(devs)
    return {"fold_deviance": devs, "mean": devs.mean(), "sd": devs.std(ddof=1), "fits": fits}
