# Decisions log

One entry per judgement call, written when the call was made. Each entry gives the
decision, the alternatives, the evidence (a file in `reports/` produced by code in
this repo) and the rationale. Assumptions are marked **ASSUMPTION**.

## Working rules

- Work in stages. At the end of each stage: run all code, run the tests, commit, and
  push the working branch (`claude/sweet-thompson-thk7jp`) to `origin`.
- The GitHub repository stays **private**. Do not merge to `main`, change repository
  visibility, or create releases or tags without the owner's explicit approval.
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
  Region enter as numeric ranks of observed claim frequency, with the ranking fitted
  on the training data of each fit (each CV training set, and the full learn set for
  the final model).
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
