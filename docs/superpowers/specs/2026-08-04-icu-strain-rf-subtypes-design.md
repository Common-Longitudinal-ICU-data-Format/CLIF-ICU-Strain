# ICU Strain — Monthly Ventilation and Respiratory Failure Subtype Volume

**Date:** 2026-08-04
**Status:** Approved design, pending implementation plan
**Site:** ucmc (CLIF 2.1.0, parquet, US/Central)

## Purpose

Exploratory site-level analysis answering three questions about ICU workload:

1. Average number of **newly mechanically ventilated** patients per month.
2. Average number of **acute respiratory failure (ARF) cases by physiologic subtype** per month.
3. **Typical time from first meeting ARF criteria to ICU admission**, stratified by subtype.

All three are computed over calendar year **2024** — the most recent complete year in the site
dataset, which spans 2018-01-01 to 2024-12-31.

## Scope decisions

These were settled during design and are binding on the implementation.

### ARF subtypes, not ARDS

The original framing asked for ARDS counts under the New Global Definition. That definition
requires bilateral opacities on chest imaging and exclusion of cardiogenic pulmonary edema.
CLIF has no imaging table and no adjudicated cardiac-etiology field, so ARDS cannot be
ascertained here without NLP over radiology reports, which is out of scope.

Deliverable 2 therefore counts the three **physiologic ARF subtypes** defined in
`references/main_text_12272025.docx` (Methods, "Patient Population and ARF Clinical Definition").
Output labels must say "ARF subtype" and must not say "ARDS", so that downstream readers do not
mistake these counts for validated ARDS cases.

### Operational census, not the manuscript research cohort

The manuscript applies exclusions (ICU LOS <24h, missing ABG, missing continuous pulse oximetry,
missing demographics, first ICU admission only) that reduced 665,074 ICU admissions to 166,152 —
roughly a 75% loss. Those exclusions are correct for an exposure-outcome study and wrong for a
workload estimate.

This analysis keeps only: **age ≥18** and **met ARF criteria within the search window**. In
exchange for dropping the ABG requirement, hypoxemia may also be established from pulse oximetry
alone (see S/F fallback below). Hypercapnia still requires an arterial blood gas because it is
definitionally a PaCO₂/pH criterion; ABG availability is therefore reported alongside the
hypercapnic and mixed counts so the reader can see how supply-limited those counts are.

### Counting unit: hospitalization

One row per `hospitalization_id`, using the **first** contiguous ICU stay in that hospitalization
as the index stay. Patients with multiple hospitalizations contribute multiple rows; ICU bouncebacks
within one hospitalization do not.

### Search window: hospital admission → ICU admission + 24h

The manuscript only looks 24h forward from ICU admission. Under that window the deliverable-3
interval is always between 0 and 24 hours and always positive, which cannot answer a question
phrased as time *to* ICU admission.

This analysis scans from `hospitalization.admission_dttm` through `icu_admit_dttm + 24h`. The
reported interval is therefore signed, and **negative values are the finding of interest**: they
mean the patient already met ARF criteria in the ED or on the ward before ICU transfer.

## Data

Source: `/Users/kavenchhikara/Library/CloudStorage/Box-Box/RCLIF_data/03-CLIF-2.1/2.1.0/`

| Table | Fields used | Notes |
|---|---|---|
| `hospitalization` | `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission` | Cohort spine |
| `adt` | `hospitalization_id`, `in_dttm`, `out_dttm`, `location_category` | ICU stay boundaries |
| `labs` | `hospitalization_id`, `lab_result_dttm`, `lab_category`, `lab_value_numeric` | ABG |
| `vitals` | `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value` | SpO₂ |
| `respiratory_support` | `hospitalization_id`, `recorded_dttm`, `device_category`, `fio2_set` | FiO₂, IMV, room air |

### Verified category values at this site

- `adt.location_category`: `icu` (58,211 rows), plus `ward`, `ed`, `procedural`, `l&d`, `other`.
- `labs.lab_category`: `po2_arterial` (744,927), `pco2_arterial` (699,895), `ph_arterial` (788,190).
- `vitals.vital_category`: `spo2` (16,624,186).
- `respiratory_support.device_category`: Title Case — `IMV`, `Room Air`, `Nasal Cannula`,
  `High Flow NC`, `NIPPV`, `CPAP`, `Trach Collar`, `Face Mask`, `Other`, and **3,236,541 NULL**
  out of ~7M rows.

### Two data traps that must be handled

**Venous and capillary blood gases.** `pco2_venous` (297,195 rows) and `ph_venous` (215,931 rows)
exist alongside the arterial categories. A venous PCO₂ ≥45 is physiologically normal. Filtering on
a loose pattern like `pco2` would substantially overcount hypercapnic ARF. Lab filters must match
the arterial categories **exactly**.

**NULL `device_category`.** Nearly half of `respiratory_support` rows carry a NULL device. Neither
raw filtering nor case-normalizing (`device_category.lower() == ...`) is safe on this column. All
device and FiO₂ logic must run against the output of `clifpy.process_resp_support_waterfall`, which
forward-fills device and FiO₂ and resolves the device hierarchy.

**Mixed-unit and impossible FiO₂.** `fio2_set` is predominantly fractional (median 0.40) but ranges
from **−1.0 to 88,880**, with 660 rows above 1.5. Since P/F and S/F are ratios with FiO₂ in the
denominator, a mis-scaled or negative FiO₂ silently produces a qualifying ratio. Normalize before
use: values in `(1.0, 100]` are percentages and divide by 100; values `<= 0` or `> 100` are set to
NULL. A NULL FiO₂ disqualifies that observation from the P/F and S/F criteria rather than defaulting
to any assumed value. Room-air observations carry FiO₂ 0.21 as imputed by the waterfall.

## Approach

clifpy for loading and cleaning, DuckDB for joins and criteria evaluation.

`ClifOrchestrator` loads the five tables; `apply_outlier_handling` cleans labs, vitals, and
respiratory support; `process_resp_support_waterfall` produces the forward-filled device/FiO₂
timeline. Criteria evaluation and the as-of joins then run in DuckDB against those cleaned frames.

Rejected alternatives:

- **Pure DuckDB SQL over parquet.** Faster and more transparent, but hand-rolls the respiratory
  support waterfall. That is precisely the step the consortium wrote clifpy to standardize, and a
  local reimplementation would not be portable to other CLIF sites.
- **Full `create_wide_dataset`.** The natural clifpy idiom, but it pivots the entire 1 GB labs
  table when only three lab categories are needed. Disproportionate memory cost for this analysis.

## Components

Follows the existing `code/templates/Python/` numbering already in the repo.

### `code/01_cohort_identification.py`

Builds the cohort spine.

- Load `hospitalization`, `adt`.
- Collapse `adt` into contiguous ICU blocks: consecutive rows with `location_category == 'icu'`
  merged when the gap between one row's `out_dttm` and the next row's `in_dttm` is ≤ 1 hour.
- Index ICU stay = earliest such block per `hospitalization_id`.
- `icu_admit_dttm` = block start; `icu_discharge_dttm` = block end.
- Filter to `age_at_admission >= 18` and `icu_admit_dttm` in calendar 2024.

Output: `output/intermediate_phi/cohort.parquet` — one row per hospitalization.

**This cohort is not ARF-filtered.** It is the denominator for deliverable 1, which counts
ventilated patients regardless of respiratory failure subtype. Deliverables 2 and 3 operate on the
subset that additionally meets ARF criteria. The attrition table in the report shows both.

### `code/02_project_quality_checks.py`

Determines whether the monthly series reflects clinical reality or documentation gaps. Runs before
analysis; its findings are surfaced in the HTML report.

Per month of 2024, within the cohort's search windows:
- % of hospitalizations with ≥1 SpO₂, ≥1 arterial blood gas, ≥1 non-null FiO₂.
- `device_category` NULL rate before vs. after the waterfall.
- Count of ICU blocks merged by the 1-hour gap rule (sanity check on that threshold).

Output: `output/final_no_phi/quality_checks.csv`.

### `code/03_outlier_handling.py`

Applies `clifpy.apply_outlier_handling` to labs, vitals, and respiratory support, and records how
many values were nulled per variable.

This must run **before** criteria evaluation. An SpO₂ recorded as 0 or a PaO₂ of 9999 would
otherwise manufacture false hypoxemia. The count of removed values goes into the report so the
effect is visible rather than silent.

Output: `output/final_no_phi/outlier_summary.csv`.

### `code/04_project_analysis.py`

Classifies ARF and produces the three deliverables.

#### ARF criteria

Evaluated at every timestamped observation from `admission_dttm` through `icu_admit_dttm + 24h`.
FiO₂ and device come from the waterfall timeline, as-of joined to each lab and vital observation.

**Hypoxemic ARF** — any one of:
- `spo2 < 90` while device is `Room Air`
- `po2_arterial <= 60` while device is `Room Air`
- P/F ≤ 300, where `pf_ratio = po2_arterial / fio2_set`, at any FiO₂
- S/F ≤ 315, where `sf_ratio = spo2 / fio2_set`, at any FiO₂ — **fallback only**, applied when the
  hospitalization has no arterial blood gas anywhere in the window

**Hypercapnic ARF** — both, on the same arterial specimen draw:
- `pco2_arterial >= 45`
- `ph_arterial < 7.35`

Paired by identical `lab_result_dttm`; if the two analytes are recorded at slightly different
timestamps, pair within a ±5 minute tolerance.

**Mixed ARF** — the hospitalization meets hypoxemic criteria at some point in the window and
hypercapnic criteria at some point in the window. The two need not be simultaneous.

**Subtype assignment** is the union over the whole window: a patient hypoxemic in the ED and
hypercapnic in the ICU is classified mixed. `first_arf_dttm` is the earliest qualifying observation
of any type.

#### Deliverable 1 — monthly new IMV

Count hospitalizations whose **first** `IMV` device observation (post-waterfall) falls within the
index ICU stay. Binned by **month of IMV initiation**, not month of ICU admission, since the strain
question is about when the ventilator workload lands.

Reported as a 12-row monthly series plus mean and SD across months.

#### Deliverable 2 — monthly ARF subtype volume

Counts of hospitalizations by subtype (hypoxemic, hypercapnic, mixed, and total), binned by month
of ICU admission. Reported alongside per-month ABG availability, since hypercapnic and mixed counts
are bounded by it.

#### Deliverable 3 — time from ARF criteria to ICU admission

`delta_hours = first_arf_dttm - icu_admit_dttm`, in hours, signed.

By subtype: median, IQR (25th/75th), and the percentage with `delta_hours < 0`, i.e. patients who
already met criteria before ICU arrival. Distribution rendered as a histogram with a reference line
at zero.

Outputs: `output/final_no_phi/monthly_imv.csv`, `monthly_arf_subtypes.csv`,
`time_to_icu_by_subtype.csv`.

### `code/05_generate_report.py`

Generates a single self-contained HTML report from the CSVs produced by scripts 02–04. No database
access; it reads only aggregate outputs, which keeps it trivially PHI-safe.

Contents:

1. **Header** — site name, CLIF version, analysis window, generation date, row counts.
2. **Definitions** — the exact ARF criteria as implemented, stated as thresholds and rendered
   from the same constants the analysis uses, so the report cannot drift from the code. Explicitly
   states that these are ARF subtypes and not ARDS, and why.
3. **Cohort attrition** — hospitalizations → adults → with ICU stay in 2024 → meeting ARF criteria.
4. **Deliverable 1** — monthly new-IMV table and bar chart.
5. **Deliverable 2** — monthly subtype table and stacked bar chart.
6. **Deliverable 3** — summary table and signed-delta histogram by subtype.
7. **Data quality** — ABG/SpO₂/FiO₂ availability by month, outliers removed, waterfall NULL rates.
8. **Limitations** — the S/F fallback is a local construction not present in the manuscript;
   hypercapnia is limited by ABG ordering practice; ARDS was not ascertained.

Charts are embedded as inline SVG or base64 PNG so the file is a single portable artifact. Styling
must render in both light and dark.

Output: `output/final_no_phi/icu_strain_report.html`.

## Definitions constants

The criteria thresholds live in one module imported by both `04` and `05`, so the report's
"Definitions" section is generated from the same values used in the computation:

```
SPO2_HYPOXEMIA        = 90     # %,  strict less-than, room air
PAO2_HYPOXEMIA        = 60     # mmHg, less-than-or-equal, room air
PF_RATIO_THRESHOLD    = 300    # less-than-or-equal, any FiO2
SF_RATIO_THRESHOLD    = 315    # less-than-or-equal, fallback when no ABG
PACO2_HYPERCAPNIA     = 45     # mmHg, greater-than-or-equal
PH_HYPERCAPNIA        = 7.35   # strict less-than
ARF_WINDOW_HOURS      = 24     # hours after ICU admission
ICU_BLOCK_GAP_HOURS   = 1      # ADT contiguity threshold
ABG_PAIR_TOLERANCE_MIN = 5     # minutes, pairing pco2 with ph
MIN_CELL_SIZE         = 10     # suppression threshold for reported cells
```

## Data security

- Patient-level intermediates stay in `output/intermediate_phi/`, which is gitignored.
- Only aggregates reach `output/final_no_phi/`.
- Every reported cell with **n < 10** is suppressed, per `output/README.md`. Monthly subtype counts
  are the likely place this bites — hypercapnic-only cases may be sparse in some months.
- The HTML report reads exclusively from `final_no_phi/` CSVs, so it cannot leak row-level data.

## Testing

- **Synthetic fixtures** for criteria evaluation: hand-built cases for each subtype, a case that
  qualifies only on the S/F fallback, a case with a venous gas that must *not* qualify as
  hypercapnic, a case with a NULL `device_category` resolved by the waterfall, and a case meeting
  criteria pre-ICU (negative delta).
- **`clif_demo/`** as an end-to-end smoke test — the pipeline must run start to finish and emit
  the HTML report without touching site data.
- **Invariants** on real output: subtype counts sum to the total; `first_arf_dttm` always falls
  inside the declared window; no `delta_hours` exceeds +24.

## Open risks

- The 1-hour ADT gap threshold for ICU contiguity is a judgment call. Script 02 reports how many
  blocks it merges; if that number is large the threshold should be revisited.
- The S/F ≤315 fallback has **no SpO₂ ≤97% ceiling**, unlike the New Global Definition. Above 97%
  the SpO₂/FiO₂ relationship saturates and the ratio loses meaning. Applied without the ceiling
  this fallback may over-classify hypoxemia; sensitivity to adding the ceiling should be checked
  once counts exist.
- clifpy resolves to 0.5.0 on import while pip metadata reports 0.4.9, indicating two environments.
  Pin the version before results are considered final.
