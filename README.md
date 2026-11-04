# CLIF ICU Strain and Mortality

## CLIF VERSION

2.1

## Objective

Determine the association of ICU strain with mortality. Strain is measured per ICU unit from the ADT
table: for each patient already in a unit at 7am, how many other patients were admitted to that unit
over the next 24 hours, relative to the unit's capacity that year. The outcome is in-hospital death or
hospice discharge over the following 7 days (from 24 hours to 8 days after that 7am), taken from
hospital records only so it is the same at every site. A secondary, acuity-weighted strain counts each other
patient as 1 + invasive ventilation + vasopressor + CRRT, over the admissions and over the patients present at 7am.

## Required CLIF tables and fields

1. **adt**: `hospitalization_id`, `hospital_id` (no nulls: step 01 stops on one), `in_dttm`, `out_dttm`,
   `location_category`, `location_type`, `location_name` — the rolling census, unit capacity, and
   admissions per day.
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`,
   `age_at_admission`, `discharge_category` — adult cohort, in-hospital death, hospice discharge.
3. **patient**: `patient_id`, `death_dttm`, `sex_category` — demographics; linked death for a secondary outcome.
4. **hospital_diagnosis**: `hospitalization_id`, `diagnosis_code`, `diagnosis_code_format`, and
   `poa_present` when the site has it — COVID (ICD-10 U07.1) and the Charlson index
   (`clifpy.calculate_cci`, which reads ICD10CM codes only), from present-on-admission codes when at
   least 20% of codes are flagged.
5. **vitals** (`recorded_dttm`, `vital_category`, `vital_value`), **labs** (`lab_result_dttm`,
   `lab_category`, `lab_value`, `lab_value_numeric`), **patient_assessments** (`recorded_dttm`,
   `assessment_category`, `numerical_value`, `categorical_value`), **respiratory_support**
   (`recorded_dttm`, `device_category`, `mode_category`, `fio2_set`, `lpm_set`, `tidal_volume_set`,
   `resp_rate_set`), **medication_admin_continuous** (`admin_dttm`, `med_category`, `med_dose`,
   `med_dose_unit`) — SOFA over the 24h before each 7am (`clifpy` `compute_sofa_polars`), and
   whether the patient was on invasive ventilation or a vasopressor in that window.
6. **code_status** (optional: `patient_id`, `start_dttm`, `code_status_category`) — new limitation orders,
   a process outcome; without it that outcome is NA.
7. **crrt_therapy** (optional: `hospitalization_id`, `recorded_dttm`) — CRRT in the acuity weights, with invasive
   ventilation and vasopressors from the tables above; without it the weights use those two only.

## Cohort identification

Hospitalizations of one patient that start within 6 hours of the previous discharge (an ED encounter
and the admission after it, a transfer between a system's hospitals) are first stitched into one
encounter with `clifpy`'s `stitch_encounters`; the encounter keeps the first admission and the last
discharge and its category, and every step below works on encounters.
Adults (`age_at_admission >= 18`) with an ICU stay (consecutive `adt` rows with
`location_category == 'icu'`, merged across gaps of ≤1 hour; OR, radiology and dialysis trips with the
patient back in the same unit within 12 hours are folded into the stay) whose ICU admission falls in
the study window (2019-2024). One row per ICU stay per 7am at which the patient is already in the unit
(a landmark), from the 60-day census burn-in to 8 days before the data end. Rows in a unit-year under
4 beds, or with a death or hospice discharge in the 24 hours after the landmark, are excluded.

## Expected Results

Each site uploads the contents of `output/final_no_phi/<site_name>_upload_to_box/`:

- `capacity_by_unit.csv`, `capacity_by_hospital.csv` — one row per unit-year / hospital-year: capacity
  (99th percentile of the 7am census; max census; 95th percentile without the COVID surge), strain
  days, hours above capacity.
- `monthly_strain_by_unit.csv`, `monthly_strain_by_hospital.csv`, `monthly_admit_strain_by_unit.png` —
  monthly admissions and mean strain.
- `census_qc.csv`, `attrition.csv` — ADT counts behind the census (rows folded into round trips,
  unit-years without capacity); cohort flow in ICU stays, patients and landmarks, the outcome split
  into in-hospital death and hospice discharge, and a text row saying whether Charlson and COVID came
  from present-on-admission or all codes (with the share flagged POA and the share in ICD10CM).
- `unit_year_gate.csv` — per unit-year, the positive control (do admissions displace incumbents?) and
  whether it validates the unit-year.
- `table1.csv`, `model_estimates.csv`, `spline_primary.png` — cohort description; estimates (unrounded,
  per +0.1 admission strain and per interquartile range) for the exposure checks, the primary model,
  the future and past placebos, the secondary outcomes and the sensitivity models; the primary's
  spline curve.

## Detailed Instructions for running the project

### 1. Configure `config/config.json`

Follow [`config/README.md`](config/README.md): set `site_name`, the path to your CLIF tables,
`filetype` and `timezone`. `data_directory` defaults to the bundled [`clif_demo/`](clif_demo)
dataset, so the pipeline runs out of the box before you point it at real site data. The demo is
MIMIC-derived and its dates are shifted per patient, so its census and capacity are meaningless: a
demo run checks plumbing only. For the same reason **MIMIC cannot be used as a site for this project**,
and pooling skips it.

### 2. Set up the project environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the project root:
```
uv sync
```
`uv.lock` pins every package for macOS, Linux and Windows. On Windows, run the same commands in PowerShell from
the project folder; write `data_directory` with forward slashes (`"C:/CLIF/tables"`) or doubled backslashes; a
console that is not UTF-8 shows clifpy's log symbols as `?`, which is harmless.

### 3. Run the pipeline

```
uv run python run_pipeline.py                            # config/config.json, every step
uv run python run_pipeline.py --from 04_sofa             # resume after a failed step
```
It runs these steps in order and stops at the first failure:
```
01_census          unit stays (round trips folded in), rolling census, capacity per unit-year, daily 7am strain
02_strain_figures  per-unit census week + daily admission strain (local only)
03_patient_days    landmark rows: exposure over [s, s+24h), outcomes over [s+24h, s+8d), covariates at s
04_sofa            SOFA over [s-24h, s) per landmark (memory-heavy: chunked clifpy calls)
05_models          exposure checks, primary, placebos, secondary and sensitivity models, Table 1, spline figure
```
Each step can also be run alone: `uv run python code/<step>.py`. Definitions and outputs per script:
[`code/README.md`](code/README.md). When it finishes, upload `output/final_no_phi/<site_name>_upload_to_box/`
(aggregates only; never `output/intermediate_phi/`).

### 4. Pool (coordinating site only)

Copy each site's upload folder under `output/final_no_phi/`, then
```
uv run python code/06_pool.py
```
writes `output/final_no_phi/pooled/pooled_estimates.csv` and `forest_primary.png`: common-effect
(inverse variance) pool as the primary, random effects with a t interval beside it, E-values.
