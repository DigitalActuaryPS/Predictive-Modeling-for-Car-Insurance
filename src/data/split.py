"""Holdout and cross-validation fold assignment.

Rows that share every rating covariate form one group, and a group is never split
across holdout / folds. The holdout is drawn first; the five CV folds partition the
remaining (learn) groups. Every model in the project uses these folds.
"""
import numpy as np
import pandas as pd


def group_ids(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    return df.groupby(covariates, observed=True, sort=False).ngroup().to_numpy()


def assign_splits(df: pd.DataFrame, covariates: list[str], holdout_share: float, n_folds: int, seed: int) -> pd.DataFrame:
    """Return a frame with columns group_id, holdout (bool) and fold (0..n_folds-1, -1 for holdout)."""
    rng = np.random.default_rng(seed)
    gid = group_ids(df, covariates)
    n_groups = gid.max() + 1
    # A random uniform per group: groups below the holdout share go to holdout
    u = rng.random(n_groups)
    holdout_group = u < holdout_share
    # Learn groups are dealt round-robin into folds in a random order
    fold_group = np.full(n_groups, -1)
    learn_groups = np.flatnonzero(~holdout_group)
    fold_group[rng.permutation(learn_groups)] = np.arange(len(learn_groups)) % n_folds
    return pd.DataFrame({"group_id": gid, "holdout": holdout_group[gid], "fold": fold_group[gid]}, index=df.index)


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows, exposure, claims and frequency by fold (holdout shown as fold -1)."""
    g = df.groupby("fold").agg(
        policies=("IDpol", "size"),
        groups=("group_id", "nunique"),
        exposure=("Exposure", "sum"),
        claims=("ClaimNb", "sum"),
    )
    g["frequency"] = g["claims"] / g["exposure"]
    g["share_of_policies"] = g["policies"] / g["policies"].sum()
    g.index = g.index.map(lambda f: "holdout" if f == -1 else f"fold_{f}")
    return g.reset_index()
