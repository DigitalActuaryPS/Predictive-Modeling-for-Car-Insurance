# Decisions log

One entry per judgement call, written when the call was made. Each entry gives the
decision, the alternatives, the evidence (a file in `reports/` produced by code in
this repo) and the rationale. Assumptions are marked **ASSUMPTION**.

## Working rules

- Work in stages. At the end of each stage: run all code, run the tests, commit, and
  push the working branch (`claude/sweet-thompson-thk7jp`) to `origin`.
- The repository is public by the owner's decision. Do not merge to `main`, change
  repository visibility, or create releases or tags without the owner's explicit
  approval.
- Every number in README, DECISIONS and narrative files comes from a file written by
  code in this repo (`reports/tables/`, `reports/figures/`, `reports/*.md`).
- Raw and processed data are never committed (`data/` is git-ignored).

---

## D001 Data source and provenance (pre-Stage 1)

- **Decision.** Use freMTPL2freq and freMTPL2sev from the CASdatasets GitHub repository
  (`dutangc/CASdatasets`, master branch, package version 1.2-1), read from `.rda`.
- **Alternatives.** OpenML (`fetch_openml`, ids 41214 / 41215). Not reachable from the
  build environment (the network proxy rejects openml.org), so the ids and the OpenML
  contents are **not verified** in this repo.
- **Evidence.** `reports/tables/data_source.csv`: retrieval date 2026-10-03, freq 677,991
  rows, sev 26,444 rows, SHA-256 of both files. The same hashes match the files at
  CASdatasets commit `8a37c3c` (2024-09-17), the most recent commit touching them.
- **Rationale.** CASdatasets is the maintained source cited by the literature. The
  hashes make a later upstream change detectable.

## D002 Reading .rda without pyreadr's factor mapping (pre-Stage 1)

- **Decision.** Parse with `pyreadr`'s parser, remove factor labels before pyreadr
  converts the table, and build categoricals with `pd.Categorical.from_codes`
  (`src/data/download.py::read_rda`).
- **Alternatives.** `pyreadr.read_r` as documented. Version 0.5.7 has no option to skip
  label mapping, and its `DataFrame.replace` mapping slows down sharply as the number
  of levels grows. IDpol is a factor with 677,991 levels, and the call did not finish
  in over 10 minutes. The polars output path would avoid it, but adds a dependency.
- **Evidence.** The output matches the label-mapped read value for value on both files.
  Download and read take a few seconds.
- **Rationale.** No new dependency, and no monkeypatching. The parser class is pyreadr
  internals, which is acceptable because pyreadr is pinned in `requirements.txt`.

## D003 Reverse reconciliation: claims without a policy (pre-Stage 1)

- **Decision.** Require zero sev rows whose IDpol is absent from freq. Checked in
  `src/data/integrity.py::reconciliation` during cleaning and in
  `tests/test_data_integrity.py`.
- **Evidence.** `reports/tables/casdatasets_version_comparison.csv`, column
  `sev_rows_idpol_not_in_freq`: 0 in the current release (commit `8a37c3c`).
- **Treatment.** None needed. Had it been non-zero, orphan claims would have been
  excluded from severity and reported, because they cannot be attached to rating
  factors or exposure.

## D004 IDpol integrity (pre-Stage 1)

- **Decision.** IDpol is stored in the `.rda` files as a factor whose labels are
  R-formatted numbers, some in scientific notation (for example `1e+05`). Parse labels
  to int64 and fail if any label is not an exact integer, or if two distinct raw
  labels map to the same integer (`src/data/download.py::parse_idpol`).
- **Evidence.** `reports/tables/data_source.csv`. For freq: 677,991 raw labels and
  677,991 parsed integers, unique per row. For sev: 24,944 raw labels and 24,944 parsed
  integers. Unit tests cover scientific notation, collisions and non-integer labels.

## D005 ClaimNb reconciliation and comparison with earlier releases (pre-Stage 1)

- **Finding.** In CASdatasets 1.2-1, `ClaimNb` in freq equals the number of sev records
  for every policy. 0 policies mismatch, and both files total 26,444 claims. The
  mismatch reported in the literature for this dataset is **absent in this release**.
- **Evidence.** `src/data/versions.py` (run with `make versions`) reads every distinct
  version of the two files in the CASdatasets git history.
  `reports/tables/casdatasets_version_comparison.csv`:
  - Releases from 2017-01-19 to 2020-11-16 (commits `c49cbbb`, `9aa07b9`, `4511e99`)
    have 678,013 freq rows with ClaimNb totalling 36,102. They have 26,639 sev rows, of
    which 195 (amount 788,714.18) have no matching policy. 9,117 policies have ClaimNb
    different from their sev record count, and 9,658 claims have no sev record.
  - Commit `531aa23` (2022-06-23, "update to remove inconsistencies") is the first
    release where the files reconcile. It has 677,991 freq rows (22 fewer) and 26,444
    sev rows (195 fewer).
  - `reports/tables/casdatasets_release_diff.csv` (oldest to current): 22 policies were
    dropped (22 claims, 16.61 policy-years). ClaimNb was revised for 9,095 retained
    policies, always downwards, and every revised value equals that policy's sev record
    count in the old release. No exposures changed.
  - The 2024-09-17 commit ("bug fix in Region") produces the same values as the
    2022-06-23 release once both are decoded.
  - The 2020-11-16 file shows 21 Region levels against 22 in other releases. This is
    not investigated further because that release is not used.
- **Decision.** Use ClaimNb as delivered. Cleaning still recounts claims from sev and
  asserts equality, so a source change cannot silently reintroduce the mismatch.
- **Comparison with other published versions.** The pre-2022 CASdatasets layout (678,013
  rows, ClaimNb 36,102) is, as far as this repo can tell, the version most published
  analyses used. Whether a particular mirror such as OpenML 41214 matches it cannot be
  checked here (D001). **ASSUMPTION:** results in this repo are comparable to
  published results only up to this 22-row and 9,658-claim difference. Frequency levels
  in pre-2022 analyses will be higher than here.

## D006 GBM encoding of nominal factors (decided before Stage 3, for Stage 4)

- **Decision.** The GBM will not use LightGBM native categorical handling. VehBrand and
  Region enter as numeric ranks of observed claim frequency. The ranking is computed
  within each CV training set (the four training folds), and on the full learn set for
  the final model. It is never computed on a validation fold, an early-stopping fold or
  the holdout. A test checks this (Stage 3, `tests/test_gbm.py`).
- **Alternatives.** Native categoricals, which is the usual best practice. One-hot
  encoding, which adds 31 features and makes the interaction matrix about 20 times
  larger.
- **Evidence.** In shap 0.51, `TreeExplainer.shap_interaction_values` is wrong for
  LightGBM native categorical splits. In `reports/tables/shap_runtime_benchmark.csv`,
  column `max_additivity_error`, interaction values for a native-categorical model miss
  the raw score by up to 0.208 on the log scale over 1,000 rows. Plain SHAP values for
  the same model miss by 1.5e-14, which is correct. With frequency-rank encoding,
  interaction values miss by 1.1e-14.
- **Rationale.** Stage 4 depends on correct interaction values. Ranking levels by
  frequency lets ordered splits isolate any group of levels whose frequencies are
  adjacent, which recovers most of what native categorical splits offer. Fitting the
  ranking on training data only keeps the CV evaluation fold clean.

## D007 SHAP interaction subsample size rule (pre-Stage 1, applied in Stage 4)

- **Decision.** Subsample size = min(20,000, 900 s / expected seconds per row). The cost
  per row comes from the benchmark, scaled linearly by the tuned model's tree count.
- **Evidence.** `reports/tables/shap_runtime_benchmark.csv`, a reference model with 300
  trees and 31 leaves on 4 CPUs: 13.72 s for 1,000 rows and 66.27 s for 5,000 rows,
  13.25 s per 1,000 rows at the larger size. Runtime is linear in rows.
- **Rationale.** At this cost, 20,000 rows on a model of the reference size takes about
  4.4 minutes (20 x 13.25 s). A tuned model with more trees would exceed a 15-minute stage budget, so
  the size is computed from the evidence rather than fixed at 20,000.
- **Also from the benchmark.** One statsmodels Poisson GLM fit with 33 parameters on all
  677,991 rows takes 23.14 s. Stage 3 to 4 forward selection needs on the order of 100
  fits, so the GLM fitter needs to be faster than statsmodels' default IRLS (see the
  Stage 3 entries).

---

## Stage 1: Data and cleaning

## D008 Exposure: cap at 1 year, keep short exposures

- **Decision.** Cap Exposure at 1.0 policy-year. Keep every policy with exposure above
  zero, including very short ones.
- **Evidence.** `reports/tables/exposure_profile.csv` (raw values) and
  `reports/tables/cleaning_log.csv`.
  - 1,224 policies have exposure above 1, totalling 1,363.34 policy-years. Their raw
    frequency is 0.0396, the lowest of any band; full-year policies (exposure exactly 1)
    run at 0.0548. Capping removes 139.34 policy-years and affects 54 claims.
  - Frequency falls steadily as exposure rises: 1.089 for 1 to 7 days (10,498 policies,
    92.75 policy-years), 0.244 for 7 days to 0.05, 0.143 for 0.05 to 0.1, and 0.0548 for
    full-year policies. Policies of 1 day or less (3,105) run at 0.353.
- **Alternatives.**
  - Drop exposures above 1. This loses 1,224 valid policies over what is most likely a
    recording issue.
  - Drop short exposures, for example under 7 days. Policies of 7 days or less are
    13,603 policies but only 101.25 policy-years (0.03% of exposure), so dropping them
    barely changes the fit. It would, however, hide a real feature of the data.
- **Rationale.**
  - **ASSUMPTION:** policies are annual contracts, and exposure above 1 is a recording or
    aggregation artefact. The source documentation says policies are "observed mostly on
    one year".
  - High frequency at short exposure is consistent with mid-term cancellation after a
    claim, such as a write-off, where the claim causes the short exposure. Removing these
    policies does not remove the effect; the effect means claim counts are not
    proportional to exposure. The offset `log(exposure)` still assumes proportionality.
    This is noted in LIMITATIONS rather than fixed with an exposure-band rating factor,
    because a tariff cannot rate on a duration that is unknown at inception.

## D009 Claim count cap at 4 (frequency model only)

- **Decision.** Cap ClaimNb at 4 for the frequency models. Severity keeps every claim
  record, including claims beyond the 4th on a policy.
- **Evidence.** `reports/tables/claim_count_profile.csv` and `cleaning_log.csv`.
  - 5 policies have exactly 4 claims.
  - 8 policies have 5 to 16 claims: 71 claims on 2.38 policy-years. Their median
    DrivAge is 52 to 59.5 and median BonusMalus 50, the best possible level. That is
    implausible for genuine repeat claimants, whose BonusMalus rises with claims, and
    looks like fleet or data artefacts.
  - The cap removes 39 claims from frequency, 0.15% of 26,444.
- **Alternatives.**
  - No cap. A handful of rows with 9 to 16 claims on under 0.4 policy-years would get
    large weight in Poisson fits, the GBM in particular.
  - Drop the 8 policies. This loses their claim amounts, which individually look
    ordinary, from severity.
  - Cap at 3. This would remove 5 more plausible 4-claim policies.
- **Consequence.** Modelled claim counts sit 39 below recorded counts, so
  frequency x severity understates recorded losses by about that share. The tariff is
  rebased to actual learn-set losses in Stage 6, which absorbs this. 4 matches the cap
  used by Noll, Salzmann and Wüthrich.

## D010 Large-loss threshold: 20,000

- **Decision.** Treat claims above 20,000 as large. The attritional severity model uses
  amounts capped at 20,000, and the excess above 20,000 is spread as a flat load. The
  threshold is chosen on learn claims only.
- **Evidence.** `reports/figures/severity_tail.png`,
  `reports/tables/severity_tail_thresholds.csv`, `severity_top_claims.csv` and
  `large_losses_by_split.csv`.
  - **Tail shape.** The log-log survival curve is roughly linear above a few thousand,
    a heavy Pareto-type tail. The mean excess rises steeply up to about 20,000, then at a
    lower, roughly linear slope out to about 80,000. Above that it rests on few claims.
  - **Concentration.** The largest learn claim (1,403,057.40) is 3.1% of the learn claim
    amount. The top 10 claims make up 12.6% and the top 100 make up 28.6%.
  - **At 20,000.** This is the 99.19th percentile of learn claims, with 170 claims (0.81%)
    above it. Those claims carry 32.7% of the amount, and the excess above 20,000 is
    25.1% of the amount. The load on capped losses is 0.336.
  - **Stability against neighbouring thresholds.** The attritional mean varies 3.2%
    across folds at 20,000, against 2.6% at 10,000 and 5.6% at 50,000. The load itself is
    volatile at every threshold: at 20,000 it ranges from 0.158 to 0.497 across the five
    learn folds.
- **Alternatives.**
  - 10,000 makes the attritional mean slightly more stable, but pools 30.7% of the amount
    into a load that does not vary by risk.
  - 50,000 or above leaves 66 or fewer learn claims to set the load, and the attritional
    mean becomes noticeably less stable.
- **Rationale.** 20,000 sits at the visible change in the mean excess slope. It keeps
  three quarters of the claim amount in the risk-rated attritional model, and leaves 170
  learn claims to estimate the load.
- **Holdout warning (not used for the choice).** The holdout contains the largest claim in
  the data (4,075,400.56). The holdout excess is 42.1% of its amount, and its load on
  capped losses is 0.727 against 0.336 on learn. Holdout burning-cost reconciliation in
  Stage 5 will therefore show modelled below actual, driven by this one claim.
- **ASSUMPTION:** claim amounts are fully developed and in consistent money terms. There
  are no dates to trend or develop them (see LIMITATIONS).

## D011 Holdout and CV split: grouped by identical rating covariates

- **Decision.** Rows sharing all nine rating covariates form one group: Area, VehPower,
  VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density and Region. A random 20% of
  groups is the untouched holdout. The remaining groups are dealt at random into 5 CV
  folds. Every model and every selection step uses these folds.
- **Evidence.** `reports/tables/split_balance.csv`.
  - The holdout holds 20.08% of policies.
  - Each fold holds about 16.0% of policies and about 84,565 groups.
  - Fold frequencies range from 0.0722 to 0.0745; the holdout is 0.0745.
- **Alternatives.**
  - Plain random rows. Possibly the same risk split across rows would then sit on both
    sides of a split, which leaks information and flatters CV scores, the GBM's
    especially.
  - Stratify on claim count. With about 4,200 claims per fold, random group assignment
    already balances frequency, so stratification adds complexity for no visible gain.
- **ASSUMPTION:** rows with identical covariates may be the same risk recorded across
  several rows. The data cannot confirm this. Grouping guards against the leakage at
  almost no cost, because the largest group has 22 rows.

## D012 Fixed-amount claims (IRSA-IDA convention)

- **Finding.** In `reports/tables/severity_fixed_amounts.csv`, three amounts make up
  37.2% of learn claims: 1,204.00 (18.2%), 1,128.12 (11.6%) and 1,172.00 (7.7%). The
  CASdatasets documentation says some amounts are fixed under the French IRSA-IDA
  claims convention between insurers.
- **Decision.** Keep these claims as recorded. A Gamma GLM estimates the mean severity
  by risk group, and it remains consistent for the mean even though the claim
  distribution is lumpy and not Gamma.
- **Consequence.** Severity relativities will be flat for many factors, because much of
  the claim amount is a convention rate rather than a cost driven by the risk. This is
  noted for Stage 5 and LIMITATIONS.

---

## Stage 2: Banding and features

## D013 Minimum exposure per level: 2,000 learn policy-years

- **Decision.** Every level of every rating factor must carry at least 2,000
  policy-years in the learn set (`banding.min_band_exposure`). This is checked in code,
  and the stage fails if any level is short.
- **Evidence.** `reports/tables/band_exposure_check.csv`: all 79 levels pass. The
  smallest are DrivAge 18-20 (2,061.7 policy-years, 469 claims), BonusMalus 111+
  (2,135.3, 815 claims), Nord-Est small (2,391.2) and Haute-Normandie (2,559.5).
- **Rationale.** At the portfolio frequency of about 0.073, 2,000 policy-years gives
  roughly 150 claims. The standard error of a log relativity is then about
  1/sqrt(150) = 0.08, which is about the coarsest a tariff cell should rest on. That is
  0.7% of the 286,515 learn policy-years.
- **ASSUMPTION:** a single threshold for all factors is adequate. It is judged against
  the frequency model, because severity uses the same factors with fewer observations
  (Stage 5 handles this by keeping severity simpler).

## D014 Band cut points

- **Decision.** Cut points are in `config.yaml` (`banding.edges`), chosen from the raw
  learn one-ways (`reports/tables/oneway_raw_*.csv`) subject to D013. Final one-ways are
  in `reports/tables/oneway_band_*.csv` and `reports/figures/oneway_*.png`.
  - **DrivAge:** 18-20, 21-22, 23-24, 25-26, 27-29, 30-34, 35-44, 45-54, 55-64, 65-74,
    75+. Narrow bands under 30, where frequency falls steeply from 0.281 at 18 to about
    0.07 by 30. Ages 18-20 are merged because age 18 alone has only 170.8 policy-years.
    The 45-54 band captures the mid-life hump, plausibly children driving a parent's car,
    which is not tested here.
  - **VehAge:** 0, 1, 2-4, 5-7, 8-10, 11-13, 14-16, 17-19, 20+. The effect is mild.
    20+ is merged because single ages above 22 are sparse.
  - **VehPower:** each value 4 to 11, and 12+.
  - **BonusMalus:** 50, 51-55, 56-60, 61-65, 66-70, 71-75, 76-80, 81-90, 91-99,
    100-110, 111+. 50 holds 62.8% of learn exposure (179,975.6 policy-years). Above 100,
    D013 forces the two bands 100-110 and 111+.
  - **LogDensity:** 0-2.5, then steps of 1.0 up to 9.5, then 9.5+.
- **Alternatives.** Data-driven merging of adjacent bands by statistical tests. Not used:
  it tunes the bands to noise in the learn data, and fixed cut points are easier to
  defend and to carry in a tariff.

## D015 Area dropped; density used as banded log(Density)

- **Decision.** Drop Area. Use banded log(Density) instead.
- **Evidence.** In `reports/tables/area_vs_density.csv`, each Area level covers a
  contiguous, non-overlapping range of log(Density): A is 0 to 3.91, B 3.91 to 4.61,
  and so on up to F at 9.21 to 10.20 (`overlaps_next` is False for every level). Area
  is therefore a six-band version of log(Density).
- **Rationale.** Including both duplicates the same information and makes the density
  relativities unstable through collinearity. Banding log(Density) more finely, with 9
  levels, keeps the information Area discards.

## D016 Region and VehBrand grouping

- **Region (geographic).** Merge sparse old regions with their neighbours (D013):
  - Alsace, Champagne-Ardenne and Franche-Comte into "Nord-Est small" (2,391.2
    policy-years).
  - Auvergne and Limousin into one group (3,738.2).
  - Corse into Provence-Alpes-Cotes-D'Azur, as "PACA + Corse" (30,142.0).
  The other 15 regions stay separate. Geography was chosen over grouping by observed
  frequency because it keeps the groups explainable and does not tune them to the
  learn-set response. Evidence: `band_exposure_check.csv`.
- **VehBrand (by frequency).** Brand labels are anonymised, B1 to B14, so there is no
  natural neighbour. B14 (1,819.8 policy-years) falls short and is merged in code into
  the brand with the nearest learn one-way frequency, which is B12
  (`reports/tables/vehbrand_merges.csv`). This is the only factor grouped using the
  response, and it moves 1,820 policy-years into a level of 51,746.
- **Base levels.** In `reports/tables/base_levels.csv`: DrivAge 45-54, VehAge 2-4,
  VehPower 6, BonusMalus 50, LogDensity 4.5-5.5, VehBrand B1, Region Centre, VehGas
  Regular. Each is its factor's highest-exposure level on learn, and
  `tests/test_banding.py` checks this. For VehBrand, B1 has 76,339.3 policy-years
  against B2's 75,915.6, so the margin is narrow and the base choice has no practical
  consequence. B12 has the most learn **policies** (135,689 against B1's 130,053), but
  its policies are shorter, so its exposure is lower (53,565.6). Base levels follow
  exposure, not policy count.

## D017 Bands rather than splines for DrivAge and BonusMalus

- **Decision.** Keep bands for both factors in GLM-A and GLM-B.
- **Evidence.** `reports/tables/band_vs_spline_cv.csv`, the main-effects Poisson GLM
  with 5-fold CV on the shared folds. "Improvement" is the all-banded model's fold
  deviance minus the variant's, so positive means the variant is better. The sd is
  across the five paired fold differences.
  - **All banded:** CV deviance 0.239260 (sd 0.003246), 72 parameters.
  - **DrivAge spline, df 8:** improves by 0.000169 (sd 0.000074) and is better in 5 of
    5 folds. With df 6 the gain is 0.000129 (sd 0.000079), also 5 of 5. With df 4 the
    spline is worse by 0.000532.
  - **BonusMalus spline, df 3, 5 or 7:** worse in every fold, by 0.001591, 0.000647
    and 0.000347. Log-linear BonusMalus is worse by 0.001423.
- **Rationale.**
  - **BonusMalus.** Bands win outright. The relationship is not smooth: the 61-65 band
    has a frequency of 0.136, above both neighbours (`oneway_band_BonusMalus_band.csv`).
  - **DrivAge.** The spline gain is consistent but tiny: 0.07% of deviance, against a
    fold-to-fold sd of 0.003246 in deviance levels. It comes from smoothing within the
    young bands. Bands are kept because:
    - GLM-A represents an incumbent banded tariff.
    - Stage 4 interactions such as young driver x vehicle power are simpler and easier
      to explain on bands.
    - A gain of this size is small next to the GLM-to-GBM gap that Stage 4 targets.
  - The spline result is kept in the table so the trade-off is visible.
- **Implication for Stage 4.** The sd of fold deviance levels (0.0032) is about 40 times
  the sd of paired fold differences (0.00007 to 0.0001). "Improvement beyond one CV
  standard deviation" is therefore read as the mean paired improvement exceeding the
  sd of the paired fold differences. Measured against the sd of deviance levels, no
  plausible single effect could pass, so that reading would make the test meaningless.
  This interpretation is a judgement call.

## D018 Acceptance rule for Stage 4 interactions (set before running Stage 4)

An interaction candidate is accepted into GLM-B only if **all** of the following hold.
Each is evaluated on the shared 5 CV folds, against the current model in forward
selection.

1. **Beyond noise.** The mean paired fold improvement in Poisson deviance (current minus
   candidate) exceeds the sd of the five paired differences (D017), **or** a likelihood
   ratio test on the full learn set is significant at p < 0.001.
2. **Every fold.** Improvement is positive in all 5 folds.
3. **Materiality floor.** The mean paired improvement is at least 2% of the CV deviance
   gap between GLM-A and the GBM. **ASSUMPTION:** 2% is a judgement threshold set by the
   project owner. It screens out effects that pass statistical tests only because the
   learn set has 541,840 rows but are too small to justify an extra tariff table.
4. **Stable and sensible.** Coefficient signs agree across the five fold fits, and the
   relativities are monotone or explainable, with no level outside a plausible range.
   This is checked from the fold-fit tables and recorded per candidate.

Each accepted interaction's share of the gap closed,
(improvement / (GLM-A CV deviance - GBM CV deviance)), is reported per interaction and
cumulatively.
