import pickle

import numpy as np
import pandas as pd


def _glm_b(cfg):
    with open(cfg["paths"]["processed"] / "models" / "frequency_glm_b.pkl", "rb") as fh:
        return pickle.load(fh)


def test_accepted_interactions_pass_every_rule(cfg):
    log = pd.read_csv(cfg["paths"]["tables"] / "interaction_selection_log.csv")
    acc = pd.read_csv(cfg["paths"]["tables"] / "interaction_accepted.csv")
    assert len(acc) <= 5
    for r in acc.itertuples():
        row = log[(log.step == r.step) & (log.candidate == r.accepted)].iloc[0]
        assert row.passes and row.folds_improved == 5 and row.share_of_gap >= 0.05


def test_glm_b_balances_on_learn(cfg, banded):
    learn, _ = banded
    glm_b = _glm_b(cfg)["glm_b"]
    pred = glm_b.predict(learn, np.log(learn["Exposure"].to_numpy())).sum()
    assert abs(pred / learn["ClaimNb"].sum() - 1) < 0.005


def test_glm_b_terms_are_glm_a_plus_accepted(cfg):
    b = _glm_b(cfg)
    names = b["glm_b"].names
    for acc in b["accepted"]:
        assert any(n.startswith(f"{acc}:") for n in names)
