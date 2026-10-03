"""Stage 1: integrity checks, cleaning, holdout / CV split and severity tail analysis.

Outputs
- data/processed/policies.parquet : one row per policy, cleaned, with split columns and
  per-policy claim amounts (total, attritional capped, large excess)
- data/processed/claims.parquet   : one row per claim with policy split columns
- reports/tables/: cleaning_log, data_summary, exposure_profile, claim_count_profile,
  split_balance, severity_tail_thresholds, severity_top_claims
- reports/figures/severity_tail.png
"""
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.download import download
from src.data.integrity import reconciliation
from src.data.split import assign_splits, balance_table
from src.evaluation.plots import severity_tail_plots

RATING_COLUMNS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"]


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    download()
    raw = cfg["paths"]["raw"]
    return pd.read_parquet(raw / "freMTPL2freq.parquet"), pd.read_parquet(raw / "freMTPL2sev.parquet")


def check_integrity(freq: pd.DataFrame, sev: pd.DataFrame) -> dict:
    rec = reconciliation(freq, sev)
    if not rec["freq_idpol_unique"]:
        raise ValueError("IDpol not unique in freMTPL2freq")
    if rec["sev_rows_idpol_not_in_freq"]:
        raise ValueError(f"{rec['sev_rows_idpol_not_in_freq']} claim rows without a policy (see DECISIONS D003)")
    if freq[RATING_COLUMNS + ["Exposure", "ClaimNb"]].isna().any().any():
        raise ValueError("missing values in rating columns")
    return rec


def exposure_profile(df: pd.DataFrame) -> pd.DataFrame:
    edges = [0, 1 / 365, 7 / 365, 0.05, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0, np.inf]
    labels = ["(0,1d]", "(1d,7d]", "(7d,0.05]", "(0.05,0.1]", "(0.1,0.25]", "(0.25,0.5]",
              "(0.5,0.75]", "(0.75,1)", "1", ">1"]
    band = pd.cut(df["Exposure"], edges, labels=labels, right=True)
    t = df.groupby(band, observed=False).agg(
        policies=("IDpol", "size"), exposure=("Exposure", "sum"), claims=("ClaimNb", "sum")
    )
    t["frequency"] = t["claims"] / t["exposure"]
    t["share_policies"] = t["policies"] / t["policies"].sum()
    t["share_exposure"] = t["exposure"] / t["exposure"].sum()
    return t.rename_axis("exposure_band").reset_index()


def claim_count_profile(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby("ClaimNb").agg(
        policies=("IDpol", "size"),
        exposure=("Exposure", "sum"),
        mean_exposure=("Exposure", "mean"),
        median_drivage=("DrivAge", "median"),
        median_bonusmalus=("BonusMalus", "median"),
    )
    t["claims"] = t.index * t["policies"]
    t["claims_per_policy_year"] = t["claims"] / t["exposure"]
    return t.reset_index()


def severity_tail_table(amounts: np.ndarray, candidates: list[float], folds: np.ndarray) -> pd.DataFrame:
    """Threshold candidates on learn claims. Fold columns show how much the attritional
    mean severity and the large-loss load move between the five learn folds."""
    x = np.asarray(amounts, dtype=float)
    total = x.sum()
    rows = []
    for u in candidates:
        per_fold = []
        for k in np.unique(folds):
            xf = x[folds == k]
            per_fold.append((np.minimum(xf, u).mean(), np.clip(xf - u, 0, None).sum() / np.minimum(xf, u).sum()))
        per_fold = np.array(per_fold)
        above = x > u
        excess = np.clip(x - u, 0, None).sum()
        capped = np.minimum(x, u).sum()
        rows.append({
            "threshold": u,
            "quantile_of_claims": float((x <= u).mean()),
            "claims_above": int(above.sum()),
            "share_claims_above": float(above.mean()),
            "share_amount_from_claims_above": float(x[above].sum() / total),
            "share_amount_in_excess": float(excess / total),
            "large_load_on_capped": float(excess / capped),
            "mean_excess": float(x[above].mean() - u) if above.any() else np.nan,
            "attritional_mean": float(np.minimum(x, u).mean()),
            "attritional_mean_cv_across_folds": float(per_fold[:, 0].std(ddof=1) / per_fold[:, 0].mean()),
            "load_min_across_folds": float(per_fold[:, 1].min()),
            "load_max_across_folds": float(per_fold[:, 1].max()),
            "load_cv_across_folds": float(per_fold[:, 1].std(ddof=1) / per_fold[:, 1].mean()),
        })
    return pd.DataFrame(rows)


def top_claims_table(amounts: np.ndarray) -> pd.DataFrame:
    x = np.sort(np.asarray(amounts, dtype=float))[::-1]
    total = x.sum()
    rows = [{"top_n": k, "share_of_claim_count": k / len(x), "share_of_amount": x[:k].sum() / total, "smallest_in_top": x[k - 1]}
            for k in (1, 5, 10, 25, 50, 100, 250, 500) if k <= len(x)]
    return pd.DataFrame(rows)


def fixed_amount_table(amounts: pd.Series, top: int = 10) -> pd.DataFrame:
    vc = amounts.value_counts().head(top)
    return pd.DataFrame({"amount": vc.index, "claims": vc.to_numpy(), "share_of_claims": vc.to_numpy() / len(amounts)})


def summarise(df: pd.DataFrame, label: str, count_col: str = "ClaimNb") -> dict:
    out = {
        "dataset": label,
        "policies": len(df),
        "exposure": df["Exposure"].sum(),
        "claims": df[count_col].sum(),
        "frequency": df[count_col].sum() / df["Exposure"].sum(),
    }
    if "ClaimAmount" in df:
        out["claim_amount"] = df["ClaimAmount"].sum()
        out["mean_severity"] = df["ClaimAmount"].sum() / df["ClaimNbRecorded"].sum()
    return out


def run() -> dict:
    cfg = load_config()
    c = cfg["cleaning"]
    tables, figures, processed = (cfg["paths"][k] for k in ("tables", "figures", "processed"))
    for p in (tables, figures, processed):
        p.mkdir(parents=True, exist_ok=True)

    freq, sev = load_raw(cfg)
    rec = check_integrity(freq, sev)
    log = []

    # Claim counts: recount from sev and require agreement (D005)
    recount = sev.groupby("IDpol").size()
    df = freq.copy()
    df["ClaimNbRecorded"] = df["IDpol"].map(recount).fillna(0).astype("int64")
    n_mismatch = int((df["ClaimNbRecorded"] != df["ClaimNb"]).sum())
    if n_mismatch:
        raise ValueError(f"{n_mismatch} policies with ClaimNb != sev record count (see DECISIONS D005)")
    log.append({"step": "recount ClaimNb from sev", "policies_affected": n_mismatch, "exposure_affected": 0.0, "claims_affected": 0})

    # Profiles on the raw values, before any change (evidence for D008, D009)
    exposure_profile(df.assign(ClaimNb=df["ClaimNbRecorded"])).to_csv(tables / "exposure_profile.csv", index=False)
    claim_count_profile(df.assign(ClaimNb=df["ClaimNbRecorded"])).to_csv(tables / "claim_count_profile.csv", index=False)

    # Exposure cap (D008)
    df["ExposureRaw"] = df["Exposure"]
    over = df["Exposure"] > c["exposure_cap"]
    df["Exposure"] = df["Exposure"].clip(upper=c["exposure_cap"])
    log.append({"step": f"cap Exposure at {c['exposure_cap']}", "policies_affected": int(over.sum()),
                "exposure_affected": float((df["ExposureRaw"] - df["Exposure"]).sum()),
                "claims_affected": int(df.loc[over, "ClaimNbRecorded"].sum())})

    # Claim count cap, frequency model only (D009)
    above = df["ClaimNbRecorded"] > c["claim_count_cap"]
    df["ClaimNb"] = df["ClaimNbRecorded"].clip(upper=c["claim_count_cap"])
    log.append({"step": f"cap ClaimNb at {c['claim_count_cap']} (frequency only)", "policies_affected": int(above.sum()),
                "exposure_affected": 0.0, "claims_affected": int((df["ClaimNbRecorded"] - df["ClaimNb"]).sum())})

    # Derived features used downstream
    df["LogDensity"] = np.log(df["Density"])

    # Split (D011)
    s = cfg["split"]
    df = df.join(assign_splits(df, s["group_covariates"], s["holdout_share"], s["n_folds"], cfg["seed"]))
    balance_table(df).to_csv(tables / "split_balance.csv", index=False)

    # Claims with policy split columns
    claims = sev.merge(df[["IDpol", "holdout", "fold"]], on="IDpol", how="left", validate="many_to_one")

    # Severity tail on learn claims only (D010); the holdout is not used for any choice
    learn_amounts = claims.loc[~claims["holdout"], "ClaimAmount"].to_numpy()
    learn_folds = claims.loc[~claims["holdout"], "fold"].to_numpy()
    u = c["large_loss_threshold"]
    severity_tail_table(learn_amounts, c["large_loss_candidates"], learn_folds).to_csv(tables / "severity_tail_thresholds.csv", index=False)
    top_claims_table(learn_amounts).to_csv(tables / "severity_top_claims.csv", index=False)
    fixed_amount_table(claims.loc[~claims["holdout"], "ClaimAmount"]).to_csv(tables / "severity_fixed_amounts.csv", index=False)
    severity_tail_plots(learn_amounts, c["large_loss_candidates"], u, figures / "severity_tail.png")

    # Per-claim and per-policy amounts
    if u is not None:
        claims["ClaimAmountCapped"] = claims["ClaimAmount"].clip(upper=u)
        claims["ClaimAmountExcess"] = claims["ClaimAmount"] - claims["ClaimAmountCapped"]
        claims["IsLarge"] = claims["ClaimAmount"] > u
        split = np.where(claims["holdout"], "holdout", "learn")
        ll = claims.groupby(split).agg(
            claims=("ClaimAmount", "size"),
            claim_amount=("ClaimAmount", "sum"),
            large_claims=("IsLarge", "sum"),
            attritional_capped_amount=("ClaimAmountCapped", "sum"),
            excess_amount=("ClaimAmountExcess", "sum"),
            largest_claim=("ClaimAmount", "max"),
        )
        ll["excess_share_of_amount"] = ll["excess_amount"] / ll["claim_amount"]
        ll["large_load_on_capped"] = ll["excess_amount"] / ll["attritional_capped_amount"]
        ll.rename_axis("split").reset_index().assign(threshold=u).to_csv(tables / "large_losses_by_split.csv", index=False)
    agg_cols = [col for col in ("ClaimAmount", "ClaimAmountCapped", "ClaimAmountExcess", "IsLarge") if col in claims]
    per_policy = claims.groupby("IDpol")[agg_cols].sum()
    df = df.join(per_policy, on="IDpol")
    df[agg_cols] = df[agg_cols].fillna(0)

    # Summary table
    summary = pd.DataFrame([
        summarise(freq.assign(ClaimNbRecorded=freq["ClaimNb"], ClaimAmount=freq["IDpol"].map(sev.groupby("IDpol")["ClaimAmount"].sum()).fillna(0)), "raw"),
        summarise(df, "cleaned (all)"),
        summarise(df[~df["holdout"]], "learn"),
        summarise(df[df["holdout"]], "holdout"),
    ])
    summary.to_csv(tables / "data_summary.csv", index=False)
    pd.DataFrame(log).to_csv(tables / "cleaning_log.csv", index=False)

    df.to_parquet(processed / "policies.parquet", index=False)
    claims.to_parquet(processed / "claims.parquet", index=False)
    return {"policies": df, "claims": claims, "reconciliation": rec, "summary": summary, "log": pd.DataFrame(log)}


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    out = run()
    print(out["log"])
    print(out["summary"])
