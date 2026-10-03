"""Compare the current CASdatasets release of freMTPL2 with earlier releases in the
CASdatasets git history. Writes reports/tables/casdatasets_version_comparison.csv.

Run with `make versions`. Not part of `make all` because it downloads ~45 MB of
historical files that the pipeline itself does not use.
"""
import bz2
import gzip
import lzma
import struct
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.download import parse_idpol, read_rda
from src.data.integrity import reconciliation


class _XDRReader:
    """Minimal reader for R's XDR serialisation (RDX2/RDX3), enough for data.frames of
    int / double / logical / character / factor columns. Used only as a fallback for
    historical releases whose freq file has a column carrying a dim attribute, which
    pyreadr rejects ("matrix, array or table object with more than one vector").
    """

    NIL, SYM, LIST, CHAR, LGL, INT, REAL, STR, VEC = 254, 1, 2, 9, 10, 13, 14, 16, 19
    ALTREP, REF = 238, 255

    def __init__(self, raw: bytes):
        self.b, self.pos, self.refs = raw, 0, []

    def _int(self) -> int:
        (v,) = struct.unpack_from(">i", self.b, self.pos)
        self.pos += 4
        return v

    def _vec(self, fmt: str, n: int) -> np.ndarray:
        size = np.dtype(fmt).itemsize * n
        out = np.frombuffer(self.b, dtype=fmt, count=n, offset=self.pos)
        self.pos += size
        return out

    def _attrs(self, has_attr: bool) -> dict:
        return self.item() if has_attr else {}

    def item(self):
        flags = self._int()
        typ, has_attr, has_tag = flags & 0xFF, bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if typ == self.NIL:
            return None
        if typ == self.REF:
            idx = flags >> 8 or self._int()
            return self.refs[idx - 1]
        if typ == self.SYM:
            name = self.item()
            self.refs.append(name)
            return name
        if typ == self.CHAR:
            n = self._int()
            if n == -1:
                return None
            s = self.b[self.pos : self.pos + n].decode("utf-8", "replace")
            self.pos += n
            return s
        if typ == self.LIST:  # pairlist -> dict (tag -> value)
            out = {}
            while True:
                attr = self._attrs(has_attr)
                tag = self.item() if has_tag else None
                out[tag] = self.item()
                del attr
                nxt = self._int()
                if nxt & 0xFF == self.NIL:
                    return out
                has_attr, has_tag = bool(nxt & (1 << 9)), bool(nxt & (1 << 10))
        if typ == self.ALTREP:
            info, state, attr = self.item(), self.item(), self.item()
            cls = list(info.values())[0] if isinstance(info, dict) else info[0]
            if cls in ("compact_intseq", "compact_realseq"):
                n, start, step = (float(x) for x in state)
                return np.arange(start, start + step * n, step)[: int(n)]
            if cls == "deferred_string":
                return [None if v is None else str(v) for v in list(state.values())[0]]
            if str(cls).startswith("wrap_"):
                return state[0]
            raise ValueError(f"unsupported ALTREP class {cls}")
        if typ in (self.INT, self.LGL, self.REAL):
            n = self._int()
            data = self._vec(">f8" if typ == self.REAL else ">i4", n)
            return _RVector(data, self._attrs(has_attr))
        if typ in (self.STR, self.VEC):
            n = self._int()
            data = [self.item() for _ in range(n)]
            return _RVector(data, self._attrs(has_attr))
        raise ValueError(f"unsupported SEXP type {typ}")


class _RVector:
    def __init__(self, data, attrs):
        self.data, self.attrs = data, attrs or {}

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]


def _column(v: "_RVector") -> pd.Series:
    if isinstance(v.attrs.get("levels"), _RVector):  # factor
        levels = list(v.attrs["levels"].data)
        return pd.Series(pd.Categorical.from_codes(np.asarray(v.data, dtype="int64") - 1, categories=levels))
    data = np.asarray(v.data)
    if data.dtype.kind in "iuf":
        data = data.astype(data.dtype.newbyteorder("="))  # dim attribute ignored: 1-d columns only
    return pd.Series(data)


def read_rda_xdr(path) -> dict[str, pd.DataFrame]:
    raw = open(path, "rb").read()
    for opener in (gzip.decompress, lzma.decompress, bz2.decompress):
        try:
            raw = opener(raw)
            break
        except Exception:
            continue
    if raw[:5] not in (b"RDX2\n", b"RDX3\n") or raw[5:7] != b"X\n":
        raise ValueError("not an XDR .rda file")
    r = _XDRReader(raw)
    r.pos = 7
    version = r._int()
    r._int(), r._int()
    if version == 3:
        n = r._int()
        r.pos += n
    out = {}
    for name, obj in r.item().items():
        cols = list(obj.attrs["names"].data)
        out[name] = pd.DataFrame({c: _column(v) for c, v in zip(cols, obj.data)})
    return out


def _load(url: str, name: str, tmp: Path) -> tuple[pd.DataFrame, str]:
    path = tmp / f"{name}.rda"
    urllib.request.urlretrieve(url, path)
    try:
        df, reader = read_rda(path)[name], "pyreadr"
    except Exception:
        df, reader = read_rda_xdr(path)[name], "xdr_fallback"
    df["IDpol"], _ = parse_idpol(df["IDpol"])
    return df, reader


def compare() -> pd.DataFrame:
    cfg = load_config()
    hist = cfg["data"]["history"]
    rows, frames = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for rel in hist["releases"]:
            base = f"{hist['raw_base']}/{rel['commit']}/{rel['dir']}"
            freq, freq_reader = _load(f"{base}/freMTPL2freq.rda", "freMTPL2freq", Path(tmp))
            sev, sev_reader = _load(f"{base}/freMTPL2sev.rda", "freMTPL2sev", Path(tmp))
            row = {
                "commit": rel["commit"][:7],
                "date": rel["date"],
                "commit_message": rel["description"],
                "reader": f"freq:{freq_reader} sev:{sev_reader}",
            }
            row.update(reconciliation(freq, sev))
            row["freq_exposure_total"] = float(freq["Exposure"].sum())
            row["sev_amount_total"] = float(sev["ClaimAmount"].sum())
            row["n_regions"] = int(freq["Region"].nunique())
            rows.append(row)
            frames[rel["commit"][:7]] = (freq[["IDpol", "ClaimNb", "Exposure"]], sev)
    out = pd.DataFrame(rows)
    current = out.iloc[-1]
    out["freq_rows_minus_current"] = out["freq_rows"] - current["freq_rows"]
    out["sev_rows_minus_current"] = out["sev_rows"] - current["sev_rows"]
    path = cfg["paths"]["tables"] / "casdatasets_version_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    _release_diff(frames, hist["releases"][0]["commit"][:7], hist["releases"][-1]["commit"][:7], cfg)
    return out


def _release_diff(frames: dict, old_key: str, new_key: str, cfg: dict) -> pd.DataFrame:
    """Policy-level diff between the oldest and the current release."""
    (f_old, s_old), (f_new, _) = frames[old_key], frames[new_key]
    m = f_old.merge(f_new, on="IDpol", how="left", suffixes=("_old", "_new"), indicator=True)
    dropped = m["_merge"] == "left_only"
    kept = m[~dropped]
    old_sev_count = s_old.groupby("IDpol").size().reindex(kept["IDpol"], fill_value=0).to_numpy()
    changed = kept["ClaimNb_old"] != kept["ClaimNb_new"]
    out = pd.DataFrame(
        [
            {"metric": "policies_in_old_not_in_current", "value": int(dropped.sum())},
            {"metric": "claimnb_of_dropped_policies_old", "value": float(m.loc[dropped, "ClaimNb_old"].sum())},
            {"metric": "exposure_of_dropped_policies", "value": float(m.loc[dropped, "Exposure_old"].sum())},
            {"metric": "policies_in_current_not_in_old", "value": int((~f_new["IDpol"].isin(f_old["IDpol"])).sum())},
            {"metric": "kept_policies_with_claimnb_revised", "value": int(changed.sum())},
            {"metric": "revised_policies_where_new_claimnb_equals_old_sev_count",
             "value": int((kept.loc[changed, "ClaimNb_new"].to_numpy() == old_sev_count[changed.to_numpy()]).sum())},
            {"metric": "kept_policies_claimnb_increased", "value": int((kept["ClaimNb_new"] > kept["ClaimNb_old"]).sum())},
            {"metric": "kept_policies_exposure_changed", "value": int((kept["Exposure_old"] != kept["Exposure_new"]).sum())},
        ]
    )
    out.insert(0, "comparison", f"{old_key} -> {new_key}")
    out.to_csv(cfg["paths"]["tables"] / "casdatasets_release_diff.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(compare().T)
