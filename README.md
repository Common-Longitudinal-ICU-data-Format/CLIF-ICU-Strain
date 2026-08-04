# CLIF ICU Strain — Monthly Ventilation and ARF Subtype Volume

## CLIF VERSION

2.1

## Objective

An exploratory, single-site analysis of ICU respiratory-failure workload, answering three
questions:

1. What is the average number of **newly mechanically ventilated** patients per month?
2. What is the average number of **acute respiratory failure (ARF) cases, by physiologic
   subtype** (hypoxemic, hypercapnic, mixed), per month?
3. What is the **typical time from first meeting ARF criteria to ICU admission**, stratified by
   subtype — including cases where that time is negative, i.e. the patient already met criteria
   before ICU transfer?

All three are computed over a single calendar year (2024 at the reference site — the most recent
complete year in that site's data).

**These are ARF subtypes, not ARDS.** The New Global Definition of ARDS requires bilateral
opacities on chest imaging and exclusion of cardiogenic pulmonary edema; CLIF has no imaging
table, so ARDS cannot be ascertained here. Deliverable 2's counts are physiologic ARF subtypes
only and must not be read as validated ARDS cases.

## Required CLIF tables and fields

1. **patient**: `patient_id` — the identifier other CLIF tables key off; not read directly by this
   pipeline (age comes from `hospitalization`), but required for standard CLIF orchestration.
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`,
   `age_at_admission` — the cohort spine; supplies the adult-age filter and the hospitalization
   window boundaries.
3. **adt**: `hospitalization_id`, `in_dttm`, `out_dttm`, `location_category` — defines ICU stay
   boundaries; consecutive `icu` rows are collapsed into contiguous blocks to find each
   hospitalization's index ICU stay.
4. **labs**: `hospitalization_id`, `lab_result_dttm`, `lab_category`, `lab_value_numeric` —
   arterial blood gas values (`po2_arterial`, `pco2_arterial`, `ph_arterial`) drive the
   hypercapnic criterion and the P/F ratio.
5. **vitals**: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value` — SpO2
   drives the room-air hypoxemia rule and the S/F fallback ratio.
6. **respiratory_support**: `hospitalization_id`, `recorded_dttm`, `device_category`, `fio2_set` —
   device state (room air vs. IMV vs. other support) and FiO2 for the P/F and S/F ratios; run
   through `clifpy.process_resp_support_waterfall` before use.

For Python users, the [clifpy](https://common-longitudinal-icu-data-format.github.io/clifpy/)
package provides the outlier handling and respiratory-support-waterfall utilities this project
relies on. See the
[clifpy user guide](https://common-longitudinal-icu-data-format.github.io/clifpy/user-guide/) for
detailed documentation.

## Cohort identification

- **Adults**: `age_at_admission >= 18`.
- **Index ICU stay**: the *first* contiguous ICU block per hospitalization (consecutive `adt` rows
  with `location_category == 'icu'`, merged across gaps of ≤1 hour).
- **Analysis year**: `icu_admit_dttm` falls in the configured analysis year
  (`icu_strain.definitions.ANALYSIS_YEAR`, 2024 at the reference site).
- **Deliberately NOT ARF-filtered.** The cohort built by `01_cohort_identification.py` is every
  adult with an index ICU stay in the analysis year, regardless of respiratory status — this is
  the denominator for deliverable 1 (ventilation volume). Deliverables 2 and 3 operate on the
  subset of this cohort that additionally meets ARF criteria within the search window
  (`admission_dttm` through `icu_admit_dttm + 24h`); the attrition table in the HTML report shows
  both counts.

This intentionally keeps more patients than the source manuscript's research cohort (which
additionally excludes ICU LOS <24h, missing ABG, missing continuous pulse oximetry, and repeat
admissions) — those exclusions suit an exposure-outcome study, not a workload estimate.

## Expected Results

Three deliverables, written to [`output/final_no_phi/`](output/README.md):

1. **`monthly_imv.csv`** — hospitalizations with any IMV during the index ICU stay
   (`n_imv_any`) vs. a genuine new in-stay IMV initiation (`n_imv_new`), by month.
2. **`monthly_arf_subtypes.csv`** — hospitalization counts by ARF subtype (hypoxemic, hypercapnic,
   mixed, total), by month of ICU admission.
3. **`time_to_icu_by_subtype.csv`** — median/IQR of signed hours from first meeting ARF criteria
   to ICU admission, by subtype, plus the percent with a negative delta (met criteria before ICU
   arrival).

Supporting outputs feeding the HTML report: `attrition.csv`, `quality_checks.csv`,
`outlier_summary.csv`, and the report itself, `icu_strain_report.html`.

> [!WARNING]
> **Never upload patient-level data to Box.** Only **aggregate** results may be placed in
> [`output/final_no_phi/`](output/README.md) and shared with the project PI / consortium:
> - No `patient_id` or any row-level / individual patient records.
> - Minimum cell size **n ≥ 10** for every reported statistic (prevents re-identification).
> - No raw `.csv` / `.parquet` data files.
>
> See [`output/README.md`](output/README.md) for the file-naming convention and
> [`guides/primer.md`](guides/primer.md) for the full data-security rules.

## Detailed Instructions for running the project

### 1. Configure `config/config.json`

Follow the instructions in [`config/README.md`](config/README.md) to set your site name, the path
to your CLIF tables, and the file type. `data_directory` defaults to the bundled
[`clif_demo/`](clif_demo) dataset, so the pipeline runs out of the box before you point it at real
site data.

### 2. Set up the project environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the project root
install the project's dependencies:
```
uv sync
```
Run project code inside the managed environment with `uv run python code/<script>.py`.

### 3. Run the pipeline, in order

```
uv run python code/01_cohort_identification.py    # cohort spine -> output/intermediate_phi/
uv run python code/02_project_quality_checks.py    # data-availability QC -> output/final_no_phi/
uv run python code/03_outlier_handling.py          # clifpy outlier handling + resp waterfall
uv run python code/04_project_analysis.py          # ARF classification + the 3 deliverables
uv run python code/05_generate_report.py           # self-contained HTML report
```
`01`–`03` read `config/config.json`; `04` and `05` read only files already written under
`output/` by the earlier steps. Final results, including `icu_strain_report.html`, land in
[`output/final_no_phi`](output/README.md) — remember the data-security rules above before sharing
anything.

### 4. Run the tests

```
uv run pytest
```
Includes an end-to-end smoke test (`tests/test_end_to_end_demo.py`) that runs the full `01`→`05`
pipeline against `clif_demo/` to catch import errors, schema drift, and empty-cohort edge cases
that hand-built unit fixtures can't.

## Interpreting the results

- **The S/F (SpO2/FiO2) fallback is uncapped and not part of the source manuscript's definition.**
  It applies only when a hospitalization has no arterial blood gas anywhere in the search window,
  and it carries no SpO2 ≤97% ceiling (unlike the New Global Definition's S/F criterion). Above
  97% saturation the SpO2/FiO2 ratio loses physiologic meaning, so this fallback may
  over-classify hypoxemia. Treat the hypoxemic count as an upper bound, not a precise estimate.
- **Roughly half the cohort has no arterial blood gas.** Hypercapnia is definitionally a
  PaCO2/pH criterion and cannot be established from pulse oximetry, so hospitalizations without an
  ABG cannot be classified hypercapnic or mixed even if they are physiologically so. The
  hypercapnic and mixed counts are therefore a **floor**, not a complete count — read them
  alongside the ABG-availability rate in `quality_checks.csv`, which the HTML report surfaces
  next to the subtype table for exactly this reason.
- **These are ARF subtypes, not ARDS** — see Objective above. Do not report deliverable 2's
  counts as ARDS incidence.

## Example Repositories
* [CLIF Variation in Ventilation](https://github.com/ingra107/clif_vent_variation) for R
* [CLIF Eligibility for mobilization](https://github.com/kaveriC/CLIF-eligibility-for-mobilization) for Python
