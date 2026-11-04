## Code directory

Numbered scripts. `uv run python run_pipeline.py` runs 01–05 in order from the repo root
(`--from <step>` restarts later); each also runs alone with `uv run python code/<script>.py`. Each reads
`config/config.json` and writes to
the site's two output folders: `output/intermediate_phi/<site>/` (row-level, never shared) and
`output/final_no_phi/<site>_upload_to_box/` (aggregates, n ≥ 11). See [`../output/README.md`](../output/README.md).

| Script | Reads | Writes (upload folder unless noted) |
|---|---|---|
| `01_census.py` | `hospitalization`, `adt` (stops if `hospital_id` has nulls: a null key silently empties the census); stitches hospitalizations within 6 h into encounters (clifpy) | `capacity_by_{unit,hospital}.csv` (one row per unit-year / hospital-year), `monthly_strain_by_{unit,hospital}.csv`, `census_qc.csv`, `monthly_admit_strain_by_unit.png`; encounters (first admission, last discharge and its category), the hospitalization-to-encounter map, ICU blocks, unit stays, rolling census and daily strain to `intermediate_phi/` |
| `02_strain_figures.py` | step 1 | Will's per-unit figures (`census_week.png`, `daily_admit_strain.png`) to `intermediate_phi/<site>/figures/` only: they show daily counts under 11. Optional arg: week start `YYYY-MM-DD` |
| `03_patient_days.py` | `patient`, `hospital_diagnosis` (mapped to encounters), step 1 (encounters in place of `hospitalization`) | `attrition.csv` (ICU stays, patients, landmarks per step; outcome split; a text row naming the Charlson/COVID code source, with the share flagged POA and the share in ICD10CM); landmark rows `patient_days.parquet` to `intermediate_phi/` |
| `04_sofa.py` | vitals, labs, patient_assessments, respiratory_support, medication_admin_continuous (read with polars, scored with clifpy per landmark window) | `sofa_landmark.parquet` and the window-keyed `sofa_input/` tables to `intermediate_phi/` |
| `05_models.py` | steps 3–4 | `model_estimates.csv`, `unit_year_gate.csv` (positive control per unit-year), `table1.csv`, `spline_primary.png` |
| `06_pool.py` | every site's `model_estimates.csv` under `output/final_no_phi/*_upload_to_box/` (or `CLIF_POOL_DIR`), skipping `demo` and `mimic`; coordinating site only, not run by `run_pipeline.py` | `final_no_phi/pooled/`: `pooled_estimates.csv`, `forest_primary.png` |
| `07_abstract_figures.py` | every site's `unit_year_gate.csv` and `model_estimates.csv`, and step 6; coordinating site only | `final_no_phi/pooled/abstract_figure.{png,pdf}` (systems and units anonymised) |

Shared functions live in [`../utils/`](../utils): `common.py` (all tunable constants, paths, suppression),
`encounters.py`, `census.py`, `patient_days.py`, `sofa.py`, `models.py`, `figures.py`.

### Definitions

- **Unit**: `location_name` within `hospital_id`. **Unit stay**: consecutive ADT rows in one unit (bed moves
  within 1h merged). An OR, radiology or dialysis trip (`location_category` procedural, radiology, dialysis)
  with the patient back in the same unit within 12h is folded into the stay: the bed was held, so the return
  is not an admission (`merge_round_trips`). An ICU-to-ICU transfer is an admission to the receiving unit,
  flagged `from_icu`. ICU rows without a `location_name` count in the hospital census but in no unit (01 warns
  above 2%). An ICU stay with any row lacking `out_dttm` has no known end and is not in the cohort.
- **Strain day**: local 7am to 7am. **Capacity**: per unit and calendar year, the 99th percentile of the 7am
  census over mornings with anyone present (max census and Corbett's 95th percentile without the COVID surge
  are reported too). A unit-year under 4 beds gets no capacity and contributes no landmarks (D6 is a unit
  rule; a hospital-year needs only 1 bed). `CLIF_MIN_UNIT_BEDS` overrides the 4 (the demo test sets 1).
- **Landmark**: a 7am *s* at which the patient is already in a unit (the unit stay began before *s*). One row
  per ICU stay per landmark; the arrival day is not one. Landmarks run from the 60-day census burn-in to
  8 days before the data end (31 days for `hdhd_30d`, else NA).
- **Exposure**: other patients' admissions to the patient's unit over [*s*, *s*+24h) ÷ that year's capacity
  (`admit_strain`); the patient's own bounce-back is not counted. Also `admit_strain_max` (max-census
  capacity), `prior_admit_strain` ([*s*−24h, *s*)), `placebo_admit_strain` ([*s*+8d, *s*+9d)),
  `placebo_past_admit_strain` ([*s*−8d, *s*−7d)), `census_strain` (occupancy at *s*), and hospital-wide
  `hosp_admit_strain`, `hosp_census_strain`. v4: `wocc_strain` (other incumbents of the unit on IMV or a
  vasopressor at *s* ÷ capacity, A2); `ed_strain` (hospital ED arrivals over [*s*, *s*+24h) ÷ hospital ICU
  capacity: demand the ICU's own bed-freeing cannot cause, A3), with `prior_ed_strain` and `placebo_ed_strain`
  over [*s*−24h, *s*) and [*s*+8d, *s*+9d).
- **Outcome**: in-hospital death (discharged expired) or hospice discharge in [*s*+24h, *s*+8d), from hospital
  records across the patient's hospitalizations (`hdhd_7d`, "HDHD-7"). A row with such an event inside
  [*s*, *s*+24h) is excluded (`eligible`). Secondaries: `hdhd_30d`, `death_7d` (in-hospital death only),
  `death_any_7d` (adds `patient.death_dttm`, whose completeness differs by site), `hdhd_48h` (the next day),
  `hdhd_hosp` (any time to the end of this encounter; NA while it is still open). Positive-control outcomes:
  `icu_discharge_24h`, `unit_exit_24h` (left inside the exposure window). v4 process outcomes (B1), NA outside
  their denominators: `readmit_72h` and `after_hours_discharge` (18:00–06:00) among landmarks discharged from
  the ICU inside the exposure window; `new_limitation_48h` (a limitation order in [*s*, *s*+48h)) among
  landmarks not limited at *s*. A limitation is any `code_status_category` but Full, Presume Full and Other;
  orders with no category are ignored, orders at the same time count as a limitation if any of them is one,
  and only this encounter's orders can be in force at *s* (code status resets at admission).
- **Model**: logistic regression, `outcome ~ admit_strain (per +0.1) + age (spline) + sex + Charlson + COVID +
  SOFA components over [s−24h, s) (categorical, with a "not measured" level) + on_imv + on_pressor (any
  invasive ventilation / vasopressor in [s−24h, s), 0/1; vasopressors are norepinephrine, epinephrine,
  dopamine, vasopressin, phenylephrine and angiotensin) + occupancy at s + admissions in [s−24h, s) +
  hospital-wide occupancy + ICU day (capped at 7) + month + day of week + unit × year`, SEs clustered two
  ways (patient, and unit-week). Charlson and COVID come from present-on-admission codes when at least 20%
  of the site's codes carry `poa_present`, else from all codes (clifpy's Charlson reads ICD10CM codes only;
  03 warns when under half are). The cardiovascular SOFA component is "not measured" when a vasopressor
  ran but no dose could be weight-normalised (no weight charted, or a drug clifpy does not score: vasopressin,
  phenylephrine, angiotensin). Fitted first, before any mortality model:
  positive controls (ICU discharge, unit exit inside the exposure window) and balance tests (baseline SOFA,
  age, Charlson, IMV regressed on the exposure by OLS). Then: primary, future-strain placebo (`placebo10`
  added), past-strain placebo (`placebo_past10`), admissions × occupancy (occupancy centred at 90%),
  obligate-ICU subgroup (on_imv or on_pressor), HDHD-30, in-hospital death only, any-source death,
  max-census capacity, hospital-wide strain, no SOFA, patient-only clustering, next-day and
  whole-encounter outcomes; a spline of the primary for
  shape (if it fails on few distinct exposure values, the linear rows stay and a `p75_vs_p25` row carries
  `status` `spline_failed: …`). Unit-years whose outcome never varies are dropped from a fit. Newton is
  followed by an L-BFGS refit from its solution when statsmodels reports non-convergence; `status` is the
  final fit's. A model that fails or doesn't converge is recorded in `status` and the others still run.
  `model_estimates.csv` is unrounded, `n_obs`/`n_events` are integers; `scale` is `log_odds` or `linear`
  (balance tests, first-stage t²). v4: the A1 gate fits the positive control per unit-year
  (`unit_year_gate.csv`, ≥ 11 discharges; validated when the OR's lower bound > 1) and refits the primary on
  validated unit-years; A2 weighted occupancy, a hinge above 92.5% occupancy, an occupancy spline, and
  occupancy × support; A3 the ED-arrival reduced form, two-stage residual inclusion (2SRI: the first-stage
  residual of admissions on ED arrivals enters the outcome model) and an ED placebo; B1 the process outcomes;
  B3 the obligate-ICU subgroup for weighted occupancy and the IV. The 2SRI SEs are the second stage's
  clustered SEs, not corrected for the estimated first-stage residual (small with first-stage t² of 97–3,039;
  bootstrap both stages by unit-week if the IV becomes a headline).
- **Pooling** (`06_pool.py`): common effect (inverse variance) is the primary pool; random effects with
  Paule-Mandel τ² and a t interval on k−1 df beside it; E-value for the pooled odds ratio and its CI bound.
  Pooled terms: `per10`, `placebo10`, `placebo_past10`, `placebo_ed10`, `per10:occ_c10`, `occ_hinge10`,
  `wocc10`, `per10:wocc10`.

The demo (`clif_demo/`) is MIMIC-derived and its dates are shifted per patient, so its census and capacity
are meaningless: the demo run checks plumbing only. For the same reason **MIMIC cannot be used as a site
for this project**, and 06 and 07 skip `demo` and `mimic`.
