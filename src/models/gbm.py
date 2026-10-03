"""LightGBM Poisson frequency model.

- Objective: Poisson, label = ClaimNb / Exposure, sample weight = Exposure. The weighted
  Poisson loss w * (mu - y log mu) with y = N / w equals the count likelihood with mean
  w * mu, so this is a Poisson model for counts with exposure as the time at risk.
- VehBrand and Region enter as frequency ranks fitted on training rows only (D006).
- Tuning: random search under 5-fold CV. For scoring fold k, early stopping uses fold
  (k + 1) mod 5 and the model trains on the remaining three folds, so the scoring fold
  never drives early stopping. The holdout is never used.
- Final model: all learn rows, number of rounds = mean best iteration of the chosen
  configuration across the five tuning folds.
"""
import itertools
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import poisson_deviance


class FrequencyRankEncoder:
    """Replace each level of a nominal column by the rank of its observed claim frequency
    in the data it was fitted on. Unseen levels get the median rank."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.maps: dict = {}
        self.fit_index: pd.Index | None = None

    def fit(self, df: pd.DataFrame) -> "FrequencyRankEncoder":
        self.fit_index = df.index
        for c in self.cols:
            g = df.groupby(df[c].astype(str))[["ClaimNb", "Exposure"]].sum()
            self.maps[c] = (g["ClaimNb"] / g["Exposure"]).rank(method="first").to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.cols:
            m = self.maps[c]
            out[c] = df[c].astype(str).map(m).fillna(np.median(list(m.values()))).astype(float)
        return out


def features(df: pd.DataFrame, cfg: dict, encoder: FrequencyRankEncoder) -> pd.DataFrame:
    X = encoder.transform(df)[cfg["gbm"]["features"]].copy()
    X["VehGas"] = (df["VehGas"].astype(str) == "Diesel").astype(float)
    return X.astype(float)


def base_params(cfg: dict) -> dict:
    feats = cfg["gbm"]["features"]
    return {
        "objective": "poisson",
        "metric": "poisson",
        "monotone_constraints": [1 if f in cfg["gbm"]["monotone_increasing"] else 0 for f in feats],
        "seed": cfg["seed"],
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": os.cpu_count(),
        "verbose": -1,
    }


def _dataset(df: pd.DataFrame, cfg: dict, enc: FrequencyRankEncoder, reference=None) -> lgb.Dataset:
    return lgb.Dataset(features(df, cfg, enc), label=df["ClaimNb"] / df["Exposure"], weight=df["Exposure"],
                       reference=reference, free_raw_data=False)


def predict_frequency(model: lgb.Booster, df: pd.DataFrame, cfg: dict, enc: FrequencyRankEncoder) -> np.ndarray:
    return model.predict(features(df, cfg, enc))


def search_space(cfg: dict) -> list[dict]:
    grid = cfg["gbm"]["grid"]
    all_combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(all_combos), size=cfg["gbm"]["n_trials"], replace=False)
    return [all_combos[i] for i in sorted(idx)]


def cv_config(params: dict, learn: pd.DataFrame, cfg: dict) -> dict:
    """Five folds: train on 3, early-stop on 1, score on 1."""
    n_folds = cfg["split"]["n_folds"]
    devs, iters, meta = [], [], []
    for k in range(n_folds):
        es = (k + 1) % n_folds
        tr = learn[~learn["fold"].isin([k, es])]
        es_df, va = learn[learn["fold"] == es], learn[learn["fold"] == k]
        enc = FrequencyRankEncoder(cfg["gbm"]["rank_encoded"]).fit(tr)
        dtr = _dataset(tr, cfg, enc)
        model = lgb.train(
            {**base_params(cfg), **params}, dtr, num_boost_round=cfg["gbm"]["max_rounds"],
            valid_sets=[_dataset(es_df, cfg, enc, reference=dtr)],
            callbacks=[lgb.early_stopping(cfg["gbm"]["early_stopping_rounds"], verbose=False)],
        )
        mu = predict_frequency(model, va, cfg, enc) * va["Exposure"].to_numpy()
        devs.append(poisson_deviance(va["ClaimNb"].to_numpy(), mu))
        iters.append(model.best_iteration)
        meta.append({"score_fold": k, "early_stopping_fold": es,
                     "encoder_fitted_on_folds": sorted(learn.loc[enc.fit_index, "fold"].unique().tolist())})
    devs = np.array(devs)
    return {"fold_deviance": devs, "mean": devs.mean(), "sd": devs.std(ddof=1), "best_iterations": iters, "meta": meta}


def cv_fixed_rounds(params: dict, n_rounds: int, learn: pd.DataFrame, cfg: dict) -> dict:
    """Comparable CV for reporting: train on 4 folds (same as the GLMs), fixed rounds, score on the 5th."""
    n_folds = cfg["split"]["n_folds"]
    devs, preds = [], pd.Series(np.nan, index=learn.index)
    for k in range(n_folds):
        tr, va = learn[learn["fold"] != k], learn[learn["fold"] == k]
        enc = FrequencyRankEncoder(cfg["gbm"]["rank_encoded"]).fit(tr)
        model = lgb.train({**base_params(cfg), **params}, _dataset(tr, cfg, enc), num_boost_round=n_rounds)
        f = predict_frequency(model, va, cfg, enc)
        preds.loc[va.index] = f
        devs.append(poisson_deviance(va["ClaimNb"].to_numpy(), f * va["Exposure"].to_numpy()))
    devs = np.array(devs)
    return {"fold_deviance": devs, "mean": devs.mean(), "sd": devs.std(ddof=1), "oof_frequency": preds}


def fit_final(params: dict, n_rounds: int, learn: pd.DataFrame, cfg: dict) -> tuple[lgb.Booster, FrequencyRankEncoder]:
    enc = FrequencyRankEncoder(cfg["gbm"]["rank_encoded"]).fit(learn)
    model = lgb.train({**base_params(cfg), **params}, _dataset(learn, cfg, enc), num_boost_round=n_rounds)
    return model, enc


def tune(learn: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for i, params in enumerate(search_space(cfg)):
        r = cv_config(params, learn, cfg)
        rows.append({"trial": i, **params, "cv_deviance_mean": r["mean"], "cv_deviance_sd": r["sd"],
                     "mean_best_iteration": float(np.mean(r["best_iterations"])),
                     "best_iterations": r["best_iterations"], "fold_meta": r["meta"],
                     **{f"fold_{k}_deviance": d for k, d in enumerate(r["fold_deviance"])}})
        print(f"  trial {i}: {params} -> {r['mean']:.6f} ({r['sd']:.6f}), iters {r['best_iterations']}", flush=True)
    return pd.DataFrame(rows)
