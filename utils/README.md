## Utils directory

Pure functions the numbered scripts share:

| Module | What it holds |
|---|---|
| `common.py` | every tunable constant (study window, 7am, burn-in, round-trip rule, min unit beds (env `CLIF_MIN_UNIT_BEDS`), COVID dates, exposure, outcome and placebo windows, POA share, min cell size), config, per-site paths, small-cell suppression |
| `census.py` | `merge_round_trips` (OR/CT/dialysis trips folded into the ICU stay), ICU blocks, unit stays, rolling census, daily 7am strain, `capacities` per unit-year, at hospital or unit level |
| `patient_days.py` | cohort, landmark rows (`expand_patient_days`, `assign_unit` with `incumbent`), `own_admissions`, `add_exposure` (other patients' admissions over [s, s+24h) and the future/past placebo, prior and hospital-wide variants), `add_outcomes` (HDHD-7/30, in-hospital and any-source death, eligibility, positive controls), COVID flag |
| `sofa.py` | `assign_windows` (each event keyed to its one landmark window before clifpy), `rescore_sofa` (fixes to clifpy's cardiovascular/respiratory scoring and total; cardiovascular is NA when a pressor ran but no dose could be weight-normalised), vasopressor list |
| `models.py` | `fit_logit` and `fit_ols` with two-way clustered SEs (Newton, then an L-BFGS refit from its solution if it did not converge), spline contrasts, `random_effects` pooling (fixed effect alongside), `e_value` |
| `figures.py` | Will's per-unit figures, monthly strain, forest plot, spline curve |
