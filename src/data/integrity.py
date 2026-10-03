"""Integrity checks on the raw freq/sev tables. Used by cleaning, tests and the version comparison."""
import pandas as pd


def reconciliation(freq: pd.DataFrame, sev: pd.DataFrame) -> dict:
    """Two-way reconciliation of freq.ClaimNb against claim records in sev."""
    n_sev = sev.groupby("IDpol").size()
    in_freq = sev["IDpol"].isin(freq["IDpol"])
    counts = freq.set_index("IDpol")["ClaimNb"]
    recount = n_sev.reindex(counts.index, fill_value=0)
    diff = counts - recount
    return {
        "freq_rows": len(freq),
        "freq_idpol_unique": bool(freq["IDpol"].is_unique),
        "freq_claimnb_total": float(counts.sum()),
        "sev_rows": len(sev),
        "sev_idpol_distinct": int(sev["IDpol"].nunique()),
        "sev_rows_idpol_not_in_freq": int((~in_freq).sum()),
        "sev_amount_total_idpol_not_in_freq": float(sev.loc[~in_freq, "ClaimAmount"].sum()),
        "policies_claimnb_ne_sev_count": int((diff != 0).sum()),
        "claims_in_freq_without_sev_record": float(diff.clip(lower=0).sum()),
        "sev_records_beyond_freq_claimnb": float((-diff).clip(lower=0).sum()),
    }
