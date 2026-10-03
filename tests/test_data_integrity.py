import pandas as pd
import pytest

from src.data.download import parse_idpol
from src.data.integrity import reconciliation


def test_reverse_reconciliation_no_orphan_claims(raw):
    rec = reconciliation(raw["freq"], raw["sev"])
    assert rec["sev_rows_idpol_not_in_freq"] == 0


def test_forward_reconciliation_claimnb_matches_sev(raw):
    rec = reconciliation(raw["freq"], raw["sev"])
    assert rec["policies_claimnb_ne_sev_count"] == 0


@pytest.mark.parametrize("name", ["freMTPL2freq", "freMTPL2sev"])
def test_idpol_integer_and_no_parse_collisions(raw, name):
    df = raw["freq"] if name == "freMTPL2freq" else raw["sev"]
    meta = raw["meta"]["files"][name]
    assert pd.api.types.is_integer_dtype(df["IDpol"])
    assert df["IDpol"].nunique() == meta["idpol_raw_unique_labels"] == meta["idpol_parsed_unique"]


def test_idpol_unique_in_freq(raw):
    assert raw["freq"]["IDpol"].is_unique


def test_parse_idpol_handles_r_scientific_notation():
    parsed, n_raw = parse_idpol(pd.Series(["1e+05", "100001", "3"]))
    assert parsed.tolist() == [100000, 100001, 3]
    assert n_raw == 3


def test_parse_idpol_detects_collision():
    with pytest.raises(ValueError, match="collision"):
        parse_idpol(pd.Series(["100000", "1e+05"]))


def test_parse_idpol_rejects_non_integer():
    with pytest.raises(ValueError, match="not exact integers"):
        parse_idpol(pd.Series(["1.5", "2"]))
