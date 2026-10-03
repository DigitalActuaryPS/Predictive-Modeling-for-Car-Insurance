import json

import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw(cfg):
    path = cfg["paths"]["raw"]
    if not (path / "source_metadata.json").exists():
        pytest.fail("raw data missing: run `make data` (or `make all`) first")
    return {
        "freq": pd.read_parquet(path / "freMTPL2freq.parquet"),
        "sev": pd.read_parquet(path / "freMTPL2sev.parquet"),
        "meta": json.loads((path / "source_metadata.json").read_text()),
    }
