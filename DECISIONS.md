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
3. **Materiality floor.** The mean paired improvement is at least **5%** of the CV deviance
   gap between GLM-A and the GBM. The gap is 0.001578 (Stage 3,
   `frequency_models_stage3.csv`), so the floor is about 0.0000789. **ASSUMPTION:** 5%
   is a judgement threshold set by the project owner. It was raised from an initial 2%.
   It screens out effects that pass statistical tests only because the learn set is
   large (541,840 rows) but are too small to justify an extra tariff table.
4. **Stable and sensible.** Coefficient signs agree across the five fold fits, and the
   relativities are monotone or explainable, with no level outside a plausible range.
   This is checked from the fold-fit tables and recorded per candidate.

5. **Cap.** At most 5 interactions are accepted. **ASSUMPTION:** this limit was set by the
   owner to keep the tariff carryable. Forward selection stops when no candidate passes
   rules 1 to 4, or when 5 have been accepted.

6. **Parsimony tie-break** (added by the owner after Stage 4). Within one step, take the
   passing form with the largest mean improvement, unless a passing form with fewer
   parameters is worse by less than the sd of the paired fold differences between the
   two forms. In that case take the simpler form. The log is
   `interaction_parsimony_tiebreaks.csv`.

Each accepted interaction's share of the gap closed,
(improvement / (GLM-A CV deviance - GBM CV deviance)), is reported per interaction and
cumulatively.

---

## Stage 3: Frequency models

## D019 Deviance convention

- **Decision.** Every model is scored with the mean Poisson unit deviance per policy,
  computed on claim counts. Predicted counts are exposure x predicted frequency, and no
  weights are applied (`src/evaluation/metrics.py::poisson_deviance`). This is
  identical to the exposure-weighted deviance on frequencies divided by the number of
  policies, because Poisson deviance is homogeneous of degree one. Exposure weights are
  never applied on top of the count deviance.
- **Gini.** Policies are sorted by predicted frequency, and the Lorenz curve plots
  cumulative exposure share against cumulative claim share; Gini = 1 - 2 x the area
  under that curve. It measures ranking only, so it ignores calibration.

## D020 GBM design and tuning

- **Features.** Raw DrivAge, VehAge, VehPower, BonusMalus and LogDensity; VehGas as 0/1;
  VehBrand and Region ungrouped (11 and 22 levels) and frequency-rank encoded per D006.
  Area is excluded (D015). The GBM sees the unbanded data, so it is a fair upper
  benchmark for what the rating information can support.
- **Objective.** Poisson, with label ClaimNb / Exposure and weight Exposure (see the
  module docstring for why this equals a count model with exposure as time at risk).
  Squared-error loss is never used.
- **CV scheme.** To score fold k, the model trains on three folds and early-stops on fold
  (k+1) mod 5, so the scoring fold never drives early stopping. The frequency-rank
  encoder is fitted on the three training folds only (`tests/test_gbm.py`). The holdout
  is not touched.
- **Search.** 24 random draws without replacement from a 324-point grid: learning rate
  {0.03, 0.05, 0.1}, num_leaves {15, 31, 63}, min_data_in_leaf {200, 500, 1000, 2000},
  feature_fraction {0.6, 0.8, 1.0} and lambda_l2 {0, 1, 10}. Early-stopping patience is
  100 rounds.
- **Evidence.** In `reports/tables/gbm_tuning.csv`, CV deviance across all 24 trials
  ranges from 0.237720 to 0.238125. The search is insensitive within this grid. The
  best is trial 3: learning rate 0.03, 15 leaves, min_data_in_leaf 1000,
  feature_fraction 0.8, lambda_l2 0 (`gbm_chosen.csv`).
- **Final rounds.** The final model uses 565 rounds, the mean best iteration of trial 3
  across the five tuning folds. **ASSUMPTION:** this round count, found with three
  folds of training data, is not rescaled for the larger training sets used afterwards.
  More data usually supports slightly more rounds, so the final GBM is if anything
  slightly under-fitted, which is the conservative direction for the benchmark.
- **Reported GBM CV.** To match the GLMs, the reported GBM CV deviance retrains on four
  folds with the fixed 565 rounds and scores the fifth: 0.237682 (sd 0.003109). The
  hyperparameters were chosen on the same folds, so this figure is slightly optimistic.
  The holdout deviance (0.240744) is the unbiased comparison.

## D021 Monotone constraint: GBM increasing in BonusMalus

- **Decision.** The GBM's prediction must not decrease as BonusMalus rises, holding the
  other features fixed. A test checks this on a 50 to 230 grid for 200 policies.
- **Rationale.** BonusMalus is the insurer's own claims-history score: higher means
  worse history. A price that falls as history worsens cannot be explained to
  customers or a regulator, and it breaks the incentive the BonusMalus system exists to
  create.
- **Evidence the constraint costs little.**
  - The banded learn one-way (`reports/tables/oneway_band_BonusMalus_band.csv`) rises
    from 0.0510 at 50 to 0.2199 at 100-110 and 0.3817 at 111+.
  - Above 100, single values with more than 50 policy-years range from 0.264 to 0.480 in
    frequency (`oneway_raw_BonusMalus.csv`), on thin exposure. All values above 100
    together total 2,867.4 policy-years.
  - The one exception to a rising trend is a local spike at 61-65: 0.1365, against
    0.0823 at 56-60 and 0.0948 at 66-70. An unconstrained model would fit such spikes.
  - The GLM keeps BonusMalus as an unconstrained banded factor, and Stage 6 reviews the
    tariff's BonusMalus relativities against this.

## D022 Exposure proportionality (diagnostic; main approach unchanged)

- **Test.** Refit GLM-A with log(exposure) as a free covariate instead of an offset.
- **Evidence** (`reports/tables/log_exposure_diagnostic.csv`). The coefficient is 0.641
  (se 0.0098, 95% CI 0.622 to 0.660), and the z-test of coefficient = 1 gives -36.7. As
  a free covariate, CV deviance is 0.237128 (sd 0.003151), against 0.239260 for GLM-A
  with the offset.
- **Interpretation.** Claims grow far less than proportionally with time on risk, so
  short-exposure policies claim at a much higher annual rate (D008). The likely cause is
  informative exposure: policies cancelled mid-term after a claim, such as a write-off,
  end up with short exposure because of the claim. Lapse dates and reasons are not in
  the data, so this is not verified.
- **Why the main approach is unchanged.**
  - The free exposure coefficient is not a rating factor. Each policy is quoted for a
    full year before its exposure is known.
  - GLM-A, GLM-B and the GBM all treat exposure as proportional (offset or weight). The
    GLM-to-GBM comparison is therefore like for like.
  - The project owner asked to be told before any change in approach.
- **Material consequence.** Exposure alone lowers CV deviance by 0.002132 (0.239260
  minus 0.237128). That is more than the whole GLM-A to GBM gap of 0.001578 (0.239260
  minus 0.237682). A large share of the remaining deviance therefore reflects
  short-exposure policies the offset cannot represent, not missing risk
  differentiation. Stage 4's "gap closed" percentages are measured against the
  GLM-A to GBM gap only, which is the correct like-for-like denominator, but they should
  not be read as a share of all explainable deviance.
- **Pricing effect.** The offset model's annual rate averages over the current mix of
  short and full-year exposures. Rebasing the tariff to actual learn losses (Stage 6)
  fixes the overall level for that mix. A book with more full-year policies would be
  overcharged by an amount this data cannot quantify without lapse information.
  Recorded in LIMITATIONS.

### D022 addendum: where short exposure is concentrated

`reports/tables/exposure_by_segment.csv` (learn set):
- **Share of policies with exposure under 0.25 years.** 31.9% overall. 43.8% at DrivAge
  18-20, falling steadily to 18.0% at 75+. By BonusMalus, 27.2% at 50 against 48.0% at
  100-110.
- **Frequency on short policies.** Within most bands, short-exposure policies claim at
  about twice the annual rate of the rest: 0.1373 against 0.0690 overall, and 0.4147
  against 0.2001 at DrivAge 18-20. BonusMalus 111+ is the exception, at 0.3554 against
  0.3841.
- **Implication.** Short exposure is concentrated in segments that already carry high
  relativities. Under the offset, part of the short-exposure excess can be absorbed
  into the young-driver and high-BonusMalus relativities. Stage 6 measures this by
  refitting GLM-B with log(exposure) as a control variable and comparing the
  relativities.

---

## Stage 4: SHAP interactions and GLM revision

## D023 SHAP subsample: 20,000 learn policies

- **Decision.** Apply the D007 rule with the tuned model's 565 trees. The expected cost of
  499 s for 20,000 rows is under the 900 s budget, so the size is the cap of 20,000.
  Rows are stratified by claim indicator from the learn set; the holdout is not used.
- **Evidence.** `reports/tables/shap_run.csv`: 20,000 rows, claim share 0.0367, 194 s
  actual, and max additivity error 2.0e-14. The run beat its estimate because the tuned
  trees have 15 leaves against 31 in the benchmark, and the rule scales by tree count
  only, which errs on the safe side.
- **Trade-off.** Mean absolute interaction values are averages over 20,000 rows, and the
  top pairs are separated by clear margins (`shap_interaction_ranking.csv`). The
  Friedman H cross-check uses 1,000 rows, because its cost grows with the square of
  the sample size.

## D024 Which SHAP pairs were taken forward

- **Ranking.** In `reports/tables/shap_interaction_ranking.csv`, the top 8 pairs by mean
  |SHAP interaction| are:
  1. DrivAge x BonusMalus, 0.075
  2. BonusMalus x VehBrand, 0.047
  3. BonusMalus x Region, 0.046
  4. BonusMalus x LogDensity, 0.036
  5. DrivAge x LogDensity, 0.022
  6. VehAge x VehBrand, 0.022
  7. DrivAge x VehPower, 0.016
  8. DrivAge x Region, 0.016
- **H-statistic.** Friedman's H² agrees on the leader (0.047). It ranks VehAge x VehBrand
  second (0.045).
- **Confirmation against raw data.** Each pair gets a two-way table of observed claims
  against GLM-A out-of-fold predictions on the learn set, using cells with at least 200
  policy-years. The headline statistic follows D030. The ordered tables are
  `two_way_ae_pivot_glm_a_*.csv` and `two_way_ae_pivot_glm_b_*.csv`, including DrivAge x
  BonusMalus. Values below are from `two_way_ae_summary_glm_a.csv` and
  `two_way_ae_summary_glm_b.csv`.

| Pair | vs GLM-A: sum z²/df, excluded-cell df (headline) | vs GLM-A: sum z²/(r-1)(c-1) | vs GLM-A: mean z² per cell | vs GLM-A: abs(z) > 1.96, cells (expected) | vs GLM-B: headline |
|---|---|---|---|---|---|
| DrivAge x BonusMalus | 5.79 | 3.88 | 4.41 | 33 / 88 (4.4) | 3.30 |
| BonusMalus x VehBrand | 3.58 | 2.82 | 2.79 | 16 / 91 (4.5) | 1.18 |
| BonusMalus x Region | 3.88 | 2.21 | 3.01 | 25 / 125 (6.2) | 2.01 |
| BonusMalus x LogDensity | 2.20 | 1.96 | 1.74 | 14 / 90 (4.5) | 1.57 |
| DrivAge x LogDensity | 0.82 | 0.69 | 0.64 | 1 / 87 (4.4) | 0.85 |
| VehAge x VehBrand | 1.80 | 1.62 | 1.41 | 11 / 83 (4.2) | 1.77 |
| DrivAge x VehPower | 1.52 | 1.12 | 1.15 | 5 / 78 (3.9) | 1.49 |
| DrivAge x Region | 0.99 | 0.64 | 0.79 | 2 / 137 (6.9) | 1.16 |

- **Rejected without a GLM test (owner decision).** Reasons, from
  `interaction_not_translated.csv`:
  - **DrivAge x LogDensity:** 0.82, with 1 significant cell against 4.4 expected.
  - **DrivAge x Region:** 0.99, with 2 significant cells against 6.9 expected.

  Both are consistent with no interaction.
- **DrivAge x VehPower: caveat.** The owner also rejected this pair untested. Under the
  df-adjusted statistic it scores 1.52 (5 significant cells against 3.9 expected),
  above the two pairs with no interaction. The per-cell figure (1.15) understated it,
  and "no excess over noise" is weaker for this pair. It was tested in the superseded
  first run as `young_x_highpower` (age under 30 x VehPower 7+), which reached at most
  2.1% of the gap: improvement in every fold, but below the 5% floor. The rejection
  therefore stands on materiality.

## D025 Candidate forms and owner-set selection order

- **Order.** Candidates are defined in `src/interactions/candidates.py` (`SELECTION_ORDER`).
  Each step tests one pair against the current model under D018, and accepts the best
  passing form, if any.
  1. **DrivAge x BonusMalus**, as `young_lbm_senior_malus`. A log(BM/50) slope shift for
     age under 30, plus a step for age 55+ with BonusMalus above 50.
  2. **BonusMalus x LogDensity**, as `bm_x_density`. (BM - 50) x (log density - 5): numeric
     times numeric, centred at the base levels. One parameter.
  3. **B12 x BonusMalus**, in two forms: `b12_x_bm_linear`, B12 x (BM - 50); and
     `b12_x_bm_3grp`, B12 x BonusMalus 51-99 and 100+, with base 50.
  4. **BonusMalus x Region**, in coarse form: `region_bm_slope_2grp` and
     `region_bm_slope_3grp`. The method is in D029.
  5. **VehAge 0-1 x B12**, as `b12_x_newcar`.
- **ASSUMPTION (scale logic).** The dataset's BonusMalus follows the French statutory
  scale. The coefficient starts at 1.00 for a new driver (100 here), falls 5% per
  claim-free year down to 0.50 (50 here), and rises with claims (Code des assurances,
  art. A121-1). The source documentation states only that values below 100 are bonus
  and above 100 are malus. Under that scale, BonusMalus partly measures experience for
  young drivers, whereas above 50 it signals recent claims for drivers over 55
  (`two_way_ae_pivot_glm_a_DrivAge_band_x_BonusMalus_band.csv`). Ages 65-74 have A/E of
  1.29 to 1.60 at BonusMalus 51-75, but 0.90 at 50. Ages 25-26 have A/E of 0.47 to 0.76
  at BonusMalus 56-70.

## D026 Rule 4 ("sensible") made operational: ASSUMPTION

- **Decision.** A candidate passes rule 4 if its coefficient signs are the same in all
  five fold fits, and if its interaction multiplier stays within 0.5 to 2.0 between the
  1st and 99th percentile of learn policies. The multiplier is exp of the interaction
  terms only, computed on the full-learn fit.
- **ASSUMPTION:** the 0.5 to 2.0 range is my own judgement threshold, not one set by the
  owner. It guards against a single interaction term doubling or halving a policy's
  price on top of its main effects.
- **Where it binds.**
  - In the final run it decided nothing. Every candidate rejected there failed rule 2 or
    rule 3, and the accepted forms' multipliers range from 0.55 to 1.98
    (`interaction_selection_log.csv`).
  - In the superseded first run it rejected two age x BonusMalus variants (99th
    percentiles of 2.01 and 2.19). Neither was among the forms the owner later specified.

## D027 Forward selection result: three interactions accepted

All figures are from `reports/tables/interaction_selection_log.csv`,
`interaction_accepted.csv`, `interaction_parsimony_tiebreaks.csv`, `gap_closed.csv` and
`frequency_model_comparison.csv`. The GLM-A to GBM CV gap is 0.001578, so the 5% floor
is about 0.0000789.

| Step | Candidate | Parameters | Share of gap | Folds improved | Outcome |
|---|---|---|---|---|---|
| 1 | young_lbm_senior_malus | 2 | 16.9% | 5 | Accepted |
| 2 | bm_x_density | 1 | 3.9% | 5 | Rejected: below the 5% floor. Passes every other rule |
| 3 | b12_x_bm_linear | 1 | 18.2% | 5 | **Accepted by the parsimony tie-break** |
| 3 | b12_x_bm_3grp | 2 | 20.4% | 5 | Passed. Not taken |
| 4 | region_bm_slope_2grp | 1 | 10.9% | 5 | **Accepted by the parsimony tie-break** |
| 4 | region_bm_slope_3grp | 2 | 11.3% | 5 | Passed. Not taken |
| 5 | b12_x_newcar | 1 | 2.4% | 3 | Rejected: only 3 of 5 folds, and below the floor |

- **Tie-breaks.**
  - At step 3, the 3-group form beat the linear form by a mean of 0.0000336, against a
    paired sd of 0.0000494.
  - At step 4, the 3-group form beat the 2-group form by 0.0000067, against a paired sd
    of 0.0000508.
  - Both are within noise, so the simpler form was taken each time. The owner asked for
    the region switch, and the same rule also switched B12 to the linear form.
- **Final GLM-B interaction coefficients** (`glm_b_coefficients.csv`; se in brackets,
  relativity in square brackets):
  - lbm x age under 30: +0.907 (0.108)
  - age 55+ x BonusMalus above 50: +0.289 (0.035) [1.335]
  - B12 x (BonusMalus - 50): -0.01078 (0.00111), a multiplier of 0.989 per BonusMalus
    point above 50 for B12
  - lbm x region slope group 1: -0.522 (0.053)

**Result.**

| Model | Parameters | CV deviance (sd) | Holdout deviance | Holdout Gini |
|---|---|---|---|---|
| GLM-A | 72 | 0.239260 (0.003246) | 0.242651 | 0.294 |
| GLM-B | 76 | 0.238534 (0.003023) | 0.241963 | 0.302 |
| GBM | 565 trees | 0.237682 (0.003109) | 0.240744 | 0.316 |

Gap closed: 46.0% on CV and 36.1% on holdout. The cap of 5 did not bind.

**Superseded runs.**
- Run 1 was best-improvement-first over 9 candidates. It closed 36.1% on CV.
- Run 2 was owner-ordered, with region conditional on density, and closed 37.3%.
- Run 3 tested region unconditionally with no tie-break. It took the 3-group forms and
  closed 49.0%.
- The code implements only the current run.

## D028 B12 interaction under free exposure: ASSUMPTION on materiality

- **Test.** Refit GLM-B's structure with log(exposure) as a free covariate instead of the
  offset (`reports/tables/interaction_exposure_check.csv`).
- **Evidence.**
  - The log(exposure) coefficient in the free fit is 0.648.
  - The B12 x (BM - 50) coefficient moves by a ratio of 0.855.
  - B12 policies are shorter: mean exposure 0.39 against 0.57, and 40.2% under 0.25 years
    against 29.1%. They are also much newer: 46.3% of B12 vehicles are aged 0-1, against
    9.9% for other brands (`b12_profile.csv`).
- **ASSUMPTION:** "shrinks materially" means a coefficient ratio below 0.75. This is my
  own threshold, not one set by the owner.
- **Finding.** About 14.5% of the B12 effect is associated with short exposure, and the
  rest persists. The short-exposure hypothesis is not supported as the main driver. The
  B12 profile (newer cars, shorter policies) fits a fleet, lease or new-car channel, but
  brand labels are anonymised and this is not verifiable.
- **Other terms.** The age terms move by ratios of 0.933 and 1.055, and the region term
  by 0.974.

## D029 BonusMalus x Region: condition dropped on evidence, and result

- **Original condition.** The owner set step 4 to run only if BonusMalus x LogDensity
  (step 2) was accepted.
- **Why it was dropped.** Step 2 was rejected (3.9%). As a diagnostic, density was added
  to the pre-region model anyway: the BonusMalus x Region residual statistic moved only
  from 2.97 to 2.76 (`region_bm_signal_by_step.csv`, headline measure, D030). The signal
  is not explained by density. The owner dropped the condition on that evidence, and
  step 4 was tested under D018 as normal.
- **Method.** Inside each CV training set, fit the current model plus one log(BM/50)
  slope per region group (Centre as base). Order the regions by that slope, and cut
  them into groups of equal exposure, with the largest-exposure group as base. The
  scoring fold never informs the grouping. Groupings differ slightly between folds
  (`fold_groupings_identical` is False), but the coefficient signs are stable.
- **Accepted 2-group form, full-learn grouping** (`interaction_selection_log.csv`):
  - **Group 1, flatter BonusMalus slope, coefficient -0.522:** Aquitaine,
    Haute-Normandie, Ile-de-France, Languedoc-Roussillon, Lorraine, Midi-Pyrenees,
    Nord-Est small, Nord-Pas-de-Calais, PACA + Corse, Picardie, Rhone-Alpes.
  - **Group 2, base:** Auvergne + Limousin, Basse-Normandie, Bourgogne, Bretagne,
    Centre, Pays-de-la-Loire, Poitou-Charentes.
- **Result.** 10.9% of the gap, better in 5 of 5 folds. The residual statistic falls from
  2.97 to 2.01, and adding density afterwards gives 2.03.
- **Interpretation.** The BonusMalus penalty is steepest in the rural west and centre.
  The data does not show why. The grouping is response-derived, so it should be
  re-estimated on new data before use.

## D030 Two-way interaction statistic: df for tables with excluded cells

- **Decision.** The headline two-way statistic is sum(z²) / df, with df = cells - rows -
  cols + 1, computed over cells with at least 200 policy-years. This is the
  independence df for a table with some cells excluded, known as quasi-independence.
  For a complete table it equals (rows - 1)(cols - 1).
- **Evidence.** In `two_way_ae_summary_glm_a.csv`:

| Pair | Per cell | Divided by (r-1)(c-1) | Divided by excluded-cell df |
|---|---|---|---|
| DrivAge x LogDensity | 0.64 | 0.69 | 0.82 |
| DrivAge x Region | 0.79 | 0.64 | 0.99 |

  - The owner's (r-1)(c-1) form moves DrivAge x LogDensity towards 1, but it moves
    DrivAge x Region further away. That table has 137 retained cells, but
    (11 - 1)(18 - 1) = 170: dividing by the full-table df counts the excluded cells as
    if they carried information.
  - The excluded-cell df brings both pairs with no interaction to 0.82 and 0.99, close
    to the value of 1 expected with no interaction.
- **Rationale.** Under the owner's condition ("use the df-adjusted figure if the null
  pairs move closer to 1"), the excluded-cell form qualifies and the (r-1)(c-1) form
  does not. All three statistics are kept in the tables, for transparency.

---

## Stage 5: Severity and burning cost

## D031 Severity model: Gamma GLM with a 3-level BonusMalus only

- **Set-up.** One row per policy with at least one claim. The response is average
  attritional severity: claim amounts capped at 20,000, divided by the number of claim
  records. The weight is the number of claim records. Gamma family, log link, on the
  same CV folds. Learn has 19,893 policies and 21,071 claims; the holdout has 5,051
  and 5,373 (`severity_residual_summary.csv`).
- **Selection.** Forward selection over the frequency factors, plus a severity-only
  coarse BonusMalus (50, 51-99, 100+). A factor enters only if:
  - it improves Gamma CV deviance in all 5 folds;
  - the mean paired improvement exceeds the paired sd;
  - **ASSUMPTION (sense check):** at least half of its non-base levels have a 95% CI
    that excludes 1. This is my own credibility threshold.
  Only one form of BonusMalus may enter. Evidence: `severity_selection_log.csv`.
  - **Banded BonusMalus (11 levels).** It passes the CV rules (improvement 0.001675, sd
    0.000875, 5 of 5 folds) but fails the sense check. Only 30% of its levels are
    credibly different from 1, and the relativities are not monotone.
  - **Coarse BonusMalus (3 levels).** Improvement 0.001807, sd 0.000778, 5 of 5 folds,
    and both levels credible. Accepted.
  - **Nothing else enters.** Every other factor improves in at most 3 of 5 folds, in
    both rounds.
- **Result.** In `severity_model_comparison.csv` and `severity_coefficients.csv`:

| Severity model | CV Gamma deviance (sd) | Holdout Gamma deviance |
|---|---|---|
| Intercept only | 1.038046 (0.038359) | 0.987231 |
| GLM, coarse BonusMalus | 1.036239 (0.038337) | 0.986332 |

  The relativities are 1.045 for BonusMalus 51-99 (95% CI 1.001 to 1.091) and 1.159 for
  100+ (1.078 to 1.246). The dispersion estimate is 2.307.
- **Why severity is simpler than frequency.**
  - **Volume.** 19,893 claim policies against 541,840 policies.
  - **Noise.** The fold-to-fold sd of severity CV deviance is about 3.7% of its mean,
    against 1.3% to 1.4% for the frequency GLMs (GLM-B 0.003053 / 0.238488; GLM-A
    0.003246 / 0.239260).
  - **Fixed-amount claims.** 37.2% of claims are paid at one of three fixed amounts
    (D012), which damps any risk signal in severity.
  - **The first banded fit was rejected.** Holdout deviance got worse with banded
    BonusMalus (0.988201 against 0.987231 for intercept only, in the first fit before
    the sense check was added), which suggests overfitting.
- **Alternative not taken.** A severity GBM, which the brief makes optional. With this
  little signal in severity, it would not change the tariff.

## D032 Gamma residual check given the IRSA fixed amounts

- **Evidence.** `severity_residual_summary.csv` and
  `reports/figures/severity_residuals_learn_oof.png`. The learn figures are
  out-of-fold.
  - **Fixed-amount policies.** 35.9% of learn claim policies have a single claim at one
    of the three fixed amounts. Their deviance residuals form a spike near -0.3. Their
    mean residual is -0.299, against -0.373 for all other policies.
  - **Shape.** Residual skew is 1.29, and 1.6% of absolute residuals exceed 3.
  - **Calibration.** Claim-weighted deciles of predicted severity have A/E from 0.947 to
    1.046 out-of-fold on learn, and 0.867 to 1.086 on holdout, where each decile holds
    about 537 claims.
- **Finding.** The distribution is clearly not Gamma: it has a mass point. That affects
  the variance assumption, so the severity standard errors and CIs are approximate,
  even though they are scaled by the Pearson dispersion. The mean model is calibrated,
  and fixed-amount policies show no systematic residual offset against other policies.
  The residuals are therefore not "clearly poor" for a mean model, and no separate
  mass-point model is proposed.

## D033 Burning cost and reconciliation

- **Definition.** Burning cost per policy = exposure x frequency (GLM) x attritional
  severity x (1 + load). The load is 0.3359: the learn excess over 20,000 divided by
  learn capped losses.
- **Reconciliation.** Actual is all recorded claim amounts. From
  `burning_cost_reconciliation.csv`:

| Frequency model | Learn | Holdout | Holdout excl. largest claim (4,075,400.56) |
|---|---|---|---|
| GLM-A | 1.0007 | 1.3036 | 0.9420 |
| GLM-B | 1.0007 | 1.3030 | 0.9416 |

  Figures are actual / modelled.
- **Learn.** Learn reconciles within 0.07%. The small excess comes from:
  - the frequency count cap: modelled claims are 21,056 against 21,071 recorded (D009);
  - the Gamma GLM, which is not exactly balanced in total.
  Stage 6 rebases the tariff to learn actual.
- **Holdout.**
  - The attritional part reconciles: modelled 8.44m against actual 8.49m excluding the
    largest claim.
  - The gap is all in large losses. Including the 4.08m claim, actual is 30% above
    modelled. Excluding it, actual is 6% below, because the other holdout large claims
    come to less than the flat load allows for.
  - This is large-loss volatility in a 72k policy-year holdout, not a model bias. The
    split was not altered (owner instruction).
- **Sensitivity** (`large_loss_sensitivity.csv`).
  - Total reconciliation does not depend on the threshold, by construction: the load is
    calibrated on the same learn losses. The table therefore reports what does change.

| Threshold | Learn claims above | Load | Load range across 5 learn folds | Share of burning cost in the flat load | Holdout attritional actual / modelled |
|---|---|---|---|---|---|
| 10,000 | 388 | 0.442 | 0.236 to 0.601 | 30.7% | 1.006 |
| 20,000 | 170 | 0.336 | 0.159 to 0.497 | 25.1% | 1.008 |
| 50,000 | 66 | 0.223 | 0.102 to 0.399 | 18.2% | 1.021 |

  - A higher threshold puts less of the cost into the unrated flat load, but makes the
    rated part less stable.

---

## Stage 6: Tariff

## D034 Tariff construction and rebasing

- **Structure.** Rate per policy-year = base rate x factor relativities x interaction
  multipliers. Frequency relativities come from GLM-B. Severity relativities come from
  the severity GLM, used for BonusMalus only: each BonusMalus band is mapped to its
  3-level severity group. The (1 + load) factor is folded into the base rate. The
  tariff is rebased so that learn premium equals learn recorded losses.
- **Evidence** (`tariff_summary.csv`).
  - The base rate is 108.48. That is base frequency 0.05236 x base severity 1,549.70 x
    1.3359 x rebase 1.00069.
  - Learn premium / learn losses = 1.000000. The test requires this within 0.1%.
  - The GLM-A tariff (base rate 113.41) is built the same way, as the current premium
    for Stage 7.
- **Confidence intervals.** 95% CIs come from GLM standard errors. Severity standard
  errors are approximate (D032). **ASSUMPTION:** the combined CIs treat the frequency and
  severity estimates as independent.
- **Interactions.** Interaction multipliers are tabulated at each band's
  exposure-weighted mean BonusMalus, with delta-method CIs. The tariff applies them per
  policy as (BM / 50)^coefficient, or as a per-point factor for B12.

## D035 Effective relativities added to the tariff tables

- **Why.** The fitted DrivAge relativities in GLM-B are below 1 for young drivers: 0.575
  at 18-20 (`relativities_DrivAge_band.csv`). Young drivers have high BonusMalus (mean
  94.0 at 18-20, `bm_by_driver_age.csv`), and their risk is priced mainly through the
  BonusMalus relativities and the young-driver BonusMalus slope. Read on its own, the
  fitted column would mislead a reviewer.
- **Decision.** Every factor table adds an "effective" relativity: the mean tariff rate
  in the level, divided by the mean in the base level, on the learn mix. For DrivAge
  18-20 it is 3.312, against a fitted 0.575. It depends on the portfolio mix and is
  shown for interpretation only. It is not a tariff parameter.

## D036 Tariff against GBM: what the tariff leaves on the table

From `tariff_vs_gbm_summary.csv` and
`reports/figures/double_lift_tariff_vs_gbm_holdout.png`, on the holdout. The GBM-based
premium is GBM frequency x the same severity x (1 + load), rebased to learn losses.
Observed loss cost is capped amounts x (1 + load).

- **Correlation.** Pearson 0.941 on rates (0.951 on log rates), Spearman 0.942.
- **Disagreement.** 51.2% of policies are priced more than 10% apart.
- **Lowest decile of tariff / GBM** (tariff cheapest relative to the GBM): observed /
  premium is 1.248 for the tariff and 0.917 for the GBM.
- **Highest decile:** 0.660 for the tariff and 0.909 for the GBM.
- **Gini.** Holdout Gini on capped loss is 0.327 for the tariff and 0.338 for the GBM.

The GBM is better calibrated exactly where the two disagree. That is the remaining
54.0% of the CV deviance gap the 76-parameter tariff does not capture.

## D037 Exposure control: GLM-B with log(exposure) as a free covariate

- **Method.** Same structure as GLM-B, with log(exposure) as a covariate instead of an
  offset. Policies are scored at exposure = 1 and rebased to the same learn total
  (`tariff_exposure_control_relativities.csv`,
  `tariff_exposure_control_premium_ratio.csv`). The fitted coefficient is 0.648.
- **Evidence.** 16 of 75 relativities move by more than 5%:
  - BonusMalus 66+ falls by 6% to 15% (100-110: 5.918 to 5.038).
  - VehAge 0 falls from 0.952 to 0.834.
  - B12 falls from 0.873 to 0.801.
  - Six region levels fall by 5.5% to 12.1%.
  - The young-driver BonusMalus slope multiplier falls from 2.478 to 2.332.
- **Premium effect.** Exposure-weighted mean premium ratio (control / offset):
  - 0.915 at DrivAge 18-20, rising steadily to 1.105 at 75+;
  - 0.875 at BonusMalus 100-110, against 1.050 at 50.
  - The 1st to 99th percentile of the policy ratio is 0.791 to 1.151.
- **Reading.** Under the offset, part of the short-exposure claim excess (D022) is
  priced into young-driver and high-BonusMalus relativities, the segments where short
  exposure is concentrated. Treating exposure as a control would move about 8% to 12%
  of premium from the youngest and highest-BonusMalus policies to older and
  claim-free ones.
- **Decision.** This is material. The main approach is unchanged, per the owner (offset
  retained, D022). The comparison is reported in the README and LIMITATIONS.
- **ASSUMPTION:** scoring at exposure = 1 is the correct basis for an annual quote.

## D038 BonusMalus: importance, French CRM sensitivity, and age

- **(a) Largest effect in GLM-A** (`glm_a_factor_importance.csv`). BonusMalus: its
  relativity range (max / min) is 8.47. Removing it raises learn deviance by 4,416 on 10
  df and CV deviance by 0.008125, worse in all 5 folds. That is 63% of GLM-A's CV gain
  over intercept only. Next is DrivAge: range 1.95, learn deviance +699, 9.7% of the
  gain. Every other factor is 3.5% or less.
- **(b) CRM offset** (`bm_crm_offset_relativities.csv`, `bm_crm_sensitivity.csv`). GLM-B
  is refitted with log(BM/100) as an offset and no fitted BonusMalus terms or
  BonusMalus interactions.
  - 27 of 61 other relativities move by more than 5%. Young-driver relativities rise
    2.0 to 3.2 times at ages under 30 (18-20: 0.575 to 1.818). Dense-area relativities
    (log density 6.5+) rise 5.7% to 9.1%. Eleven regions fall by 5.3% to 15.2%, and
    Bourgogne rises 6.8%.
  - CV deviance is 0.242636, worse than GLM-A at 0.239260.
- **(c) Free coefficient on log(BM/100)**, with no other BonusMalus terms: 2.074 (se
  0.031, 95% CI 2.013 to 2.136; z against 1 = 34.1). Claim frequency rises about with
  the square of the CRM coefficient, so much more steeply than a premium that moves
  one-for-one with it. CV deviance is 0.240683. The banded form is better.
- **(d) DrivAge x BonusMalus.** Shares of exposure at BonusMalus 50
  (`bm_by_driver_age.csv`):
  - 6.1% at ages 18-20, and 2.5% to 4.9% across ages 21-29;
  - 25.4% at 30-34, and 79.5% at 45-54.

  Mean BonusMalus is 94.0 at 18-20 against 53.5 at 45-54.
  - **ASSUMPTION:** BonusMalus follows the statutory scale: start at 100, down 5% a year
    without claims, floor 50 (Code des assurances art. A121-1). That rule is cited from
    general knowledge and not verified in this environment. Under it the floor takes
    over a decade of claim-free driving.
  - **Interpretation.** For most young drivers, BonusMalus measures years of experience
    rather than claims. A high BonusMalus is normal for them, and the same level carries
    a different risk at 25 and at 65. That is the DrivAge x BonusMalus interaction.
  - **Contradiction of the stated premise.** The premise that young drivers cannot reach
    50 is not fully supported: a minority of young drivers do sit at 50. That
    contradicts the scale as stated, which suggests the recorded BonusMalus is not
    always the driver's own record (for example a transferred or main-driver
    coefficient). This cannot be checked from the data.
- **(e)** A paragraph for LIMITATIONS and README is generated into
  `reports/bonus_malus_note.md` by `src/tariff/sensitivities.py`.

## D039 Non-monotone BonusMalus relativity at 61-65 (flagged, not changed)

- **Finding.** The GLM-B frequency relativity for BonusMalus 61-65 is 3.036 (95% CI 2.861
  to 3.222). That is above 66-70 (2.287, CI 2.118 to 2.469), 71-75 (2.701) and 76-80
  (2.738), and level with 81-90 (3.004) (`relativities_BonusMalus_band.csv`). This
  reproduces the learn one-way spike (D021).
- **Options.**
  1. Keep it, because it is credible: the CI does not overlap 66-70.
  2. Constrain BonusMalus relativities to be monotone, or merge 61-65 with neighbouring
     bands.
  3. Investigate which BonusMalus values within 61-65 drive it. Not done.
- **Decision.** Left as fitted, and flagged for owner review. A tariff that charges more
  at 61-65 than at 66-80 would need a business justification before filing.
