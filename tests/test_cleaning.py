import numpy as np


def test_exposure_in_unit_interval(processed, cfg):
    e = processed["policies"]["Exposure"]
    assert (e > 0).all() and (e <= cfg["cleaning"]["exposure_cap"]).all()


def test_claim_count_within_cap(processed, cfg):
    n = processed["policies"]["ClaimNb"]
    assert n.min() >= 0 and n.max() <= cfg["cleaning"]["claim_count_cap"]


def test_claim_records_preserved(processed, raw):
    # The count cap applies to the frequency model only; every claim record is kept for severity
    assert len(processed["claims"]) == len(raw["sev"])
    assert np.isclose(processed["claims"]["ClaimAmount"].sum(), raw["sev"]["ClaimAmount"].sum())
    assert (processed["policies"]["ClaimNbRecorded"].sum()) == len(raw["sev"])


def test_capped_plus_excess_equals_amount(processed, cfg):
    c = processed["claims"]
    u = cfg["cleaning"]["large_loss_threshold"]
    assert (c["ClaimAmountCapped"] <= u).all()
    assert np.allclose(c["ClaimAmountCapped"] + c["ClaimAmountExcess"], c["ClaimAmount"])


def test_groups_not_split_across_folds(processed):
    p = processed["policies"]
    assert (p.groupby("group_id")["fold"].nunique() == 1).all()


def test_holdout_and_folds_partition(processed, cfg):
    p = processed["policies"]
    assert set(p.loc[p["holdout"], "fold"]) == {-1}
    assert set(p.loc[~p["holdout"], "fold"]) == set(range(cfg["split"]["n_folds"]))
    share = p["holdout"].mean()
    assert abs(share - cfg["split"]["holdout_share"]) < 0.01


def test_claims_inherit_policy_split(processed):
    c, p = processed["claims"], processed["policies"].set_index("IDpol")
    assert (c["fold"].to_numpy() == p.loc[c["IDpol"], "fold"].to_numpy()).all()
