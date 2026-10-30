# CLIF ICU Strain and Mortality

## CLIF VERSION

2.1

## Objective

Determine the association of ICU strain with mortality. Strain is measured per ICU unit and day
from the ADT table: how many patients are in the unit at 7am and how many new patients arrived over
the previous 24 hours, each relative to the unit's capacity. The outcome is death within 7 days of
each patient-day in the ICU. 

## Required CLIF tables and fields

1. **adt**: `hospitalization_id`, `hospital_id`, `in_dttm`, `out_dttm`, `location_category`,
   `location_type`, `location_name` — the rolling census, unit capacity, and admissions per day.
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`,
   `age_at_admission`, `discharge_category` — adult cohort, in-hospital death, hospice discharge.
3. **patient**: `patient_id`, `death_dttm`, `sex_category` — death date, demographics.
4. **hospital_diagnosis**: `hospitalization_id`, `diagnosis_code`, `diagnosis_code_format` — COVID
   admission (ICD-10 U07.1) and the Charlson index (`clifpy.calculate_cci`).
5. **vitals**, **labs**, **patient_assessments**, **respiratory_support**,
   **medication_admin_continuous** — SOFA in the first 24h of ICU (`clifpy` `compute_sofa_scores`).

## Cohort identification

Adults (`age_at_admission >= 18`) with an ICU stay (consecutive `adt` rows with
`location_category == 'icu'`, merged across gaps of ≤1 hour) whose ICU admission falls in the study
window (2018-2024). One observation per patient per 7am-to-7am day in the ICU.

## Expected Results

Each site uploads the contents of `output/final_no_phi/<site_name>_upload_to_box/`:

- `capacity.csv` — ICU capacity per hospital (max census; 95th percentile of the 7am census with and
  without the COVID surge) and the number of strain days.
- `census_qc.csv` — ADT row counts behind the census.
- `monthly_strain.csv`, `strain_monthly.png` — monthly mean and peak strain.
- Model estimates, Table 1 and spline figures are added by later scripts.


## Detailed Instructions for running the project

### 1. Configure `config/config.json`

Follow [`config/README.md`](config/README.md): set `site_name`, the path to your CLIF tables,
`filetype` and `timezone`. `data_directory` defaults to the bundled [`clif_demo/`](clif_demo)
dataset, so the pipeline runs out of the box before you point it at real site data.

### 2. Set up the project environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the project root:
```
uv sync
```

### 3. Run the pipeline, in order

```
uv run python code/01_census.py     # rolling census, capacity, daily 7am strain
```

