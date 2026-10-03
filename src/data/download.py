"""Fetch freMTPL2freq / freMTPL2sev from CASdatasets and cache them as parquet.

The .rda files are read with pyreadr, but factor levels are NOT mapped by pyreadr.
pyreadr 0.5.7 maps factor codes to labels with DataFrame.replace, which is quadratic
in the number of levels; IDpol is stored as a factor with ~678k levels, so the
standard read_r call does not finish in reasonable time. Instead we parse the file,
take the factor labels off each table before conversion (so pyreadr returns the raw
integer codes), and build the categoricals in pandas with Categorical.from_codes.
"""
import hashlib
import json
import urllib.request
from datetime import date

import pandas as pd
from pyreadr._pyreadr_parser import PyreadrParser

from src.config import load_config


def read_rda(path) -> dict[str, pd.DataFrame]:
    """Read every data.frame in an .rda file; factors become pandas categoricals."""
    parser = PyreadrParser()
    parser.parse(str(path).encode("utf-8"))
    out = {}
    for table in parser.table_data:
        labels = table.value_labels
        table.value_labels = {}  # disables pyreadr's slow label replacement
        df = table.convert_to_dataframe().reset_index(drop=True)
        for col_idx, mapping in labels.items():
            col = df.columns[col_idx]
            # R factor codes are 1-based; librdata delivers labels keyed 1..K
            categories = [mapping[k] for k in sorted(mapping)]
            codes = df[col].to_numpy().astype("int64") - 1
            df[col] = pd.Categorical.from_codes(codes, categories=categories)
        out[table.name] = df
    return out


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_int_id(s: pd.Series) -> pd.Series:
    # IDpol factor labels are R-formatted numbers, some in scientific notation ("1e+05")
    return pd.to_numeric(s.astype(str)).round().astype("int64")


def download(force: bool = False) -> dict:
    cfg = load_config()
    raw = cfg["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    meta_path = raw / "source_metadata.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text())

    meta = {"source": cfg["data"]["source"], "retrieved": date.today().isoformat(), "files": {}}
    with urllib.request.urlopen(cfg["data"]["description_url"]) as resp:
        desc = resp.read().decode()
    meta["casdatasets_version"] = next(
        line.split(":", 1)[1].strip() for line in desc.splitlines() if line.startswith("Version:")
    )
    for name, url in cfg["data"]["urls"].items():
        rda = raw / f"{name}.rda"
        if force or not rda.exists():
            urllib.request.urlretrieve(url, rda)
        df = read_rda(rda)[name]
        df["IDpol"] = _to_int_id(df["IDpol"])
        df.to_parquet(raw / f"{name}.parquet", index=False)
        meta["files"][name] = {"url": url, "sha256": _sha256(rda), "rows": len(df), "columns": list(df.columns)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(download(force=True), indent=2))
