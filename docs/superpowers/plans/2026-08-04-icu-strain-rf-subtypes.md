# ICU Strain — RF Subtype Volume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a self-contained HTML report giving monthly new-IMV volume, monthly ARF subtype volume, and time-from-ARF-criteria-to-ICU-admission for the ucmc CLIF site over calendar 2024.

**Architecture:** Testable logic lives in an importable package `code/icu_strain/`; the numbered pipeline scripts `code/01_*.py` … `code/05_*.py` are thin drivers that wire config, call the package, and write outputs. clifpy handles loading, outlier handling, and the respiratory-support waterfall; DuckDB handles the as-of joins and criteria evaluation; the report script reads only aggregate CSVs.

**Tech Stack:** Python 3.10+, clifpy 0.5.0, DuckDB, pandas, matplotlib, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-04-icu-strain-rf-subtypes-design.md`

## Global Constraints

- Numbered pipeline scripts (`01_`…`05_`) begin with a digit and are therefore **not importable**. All logic that needs a test lives in `code/icu_strain/`. Scripts are drivers only.
- Lab category filters must match **exactly**: `po2_arterial`, `pco2_arterial`, `ph_arterial`. Never pattern-match `pco2` or `ph` — `pco2_venous` (297,195 rows) and `ph_venous` (215,931 rows) exist and a venous PCO₂ ≥45 is physiologically normal.
- Never filter or case-normalize `respiratory_support.device_category` directly. It is Title Case with 3,236,541 NULLs out of ~7M rows. All device/FiO₂ logic reads the output of `clifpy.process_resp_support_waterfall`.
- `fio2_set` ranges −1.0 to 88,880 with a 0.40 median. Always normalize via `normalize_fio2` before any ratio.
- Patient-level data goes only to `output/intermediate_phi/` (gitignored). Only aggregates reach `output/final_no_phi/`.
- Every reported cell with n < 10 is suppressed. Use `MIN_CELL_SIZE`.
- Output labels say "ARF subtype", never "ARDS".
- Timezone is `US/Central` from `config/config.json`. Never hardcode it.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, pytest config |
| `code/icu_strain/__init__.py` | Package marker |
| `code/icu_strain/definitions.py` | Threshold constants + human-readable criteria text |
| `code/icu_strain/physiology.py` | FiO₂ normalization, ARF criteria evaluation |
| `code/icu_strain/cohort.py` | ICU block collapsing, cohort construction |
| `code/icu_strain/aggregate.py` | Monthly rollups, small-cell suppression |
| `code/icu_strain/report.py` | HTML rendering |
| `code/01_cohort_identification.py` | Driver: build cohort |
| `code/02_project_quality_checks.py` | Driver: availability checks |
| `code/03_outlier_handling.py` | Driver: outlier handling + waterfall |
| `code/04_project_analysis.py` | Driver: classify + aggregate |
| `code/05_generate_report.py` | Driver: render HTML |
| `tests/test_*.py` | Unit tests per module |

---

### Task 1: Project scaffolding and definitions module

**Files:**
- Create: `pyproject.toml`
- Create: `code/icu_strain/__init__.py`
- Create: `code/icu_strain/definitions.py`
- Test: `tests/test_definitions.py`

**Interfaces:**
- Consumes: nothing
- Produces: module `icu_strain.definitions` exporting int/float constants `SPO2_HYPOXEMIA=90`, `PAO2_HYPOXEMIA=60`, `PF_RATIO_THRESHOLD=300`, `SF_RATIO_THRESHOLD=315`, `PACO2_HYPERCAPNIA=45`, `PH_HYPERCAPNIA=7.35`, `ARF_WINDOW_HOURS=24`, `ICU_BLOCK_GAP_HOURS=1`, `ABG_PAIR_TOLERANCE_MIN=5`, `MIN_CELL_SIZE=10`, `ANALYSIS_YEAR=2024`, `MIN_AGE=18`; string constants `ABG_PO2='po2_arterial'`, `ABG_PCO2='pco2_arterial'`, `ABG_PH='ph_arterial'`; and `CRITERIA_TEXT: dict[str, str]` with keys `hypoxemic`, `hypercapnic`, `mixed`, `not_ards`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "clif-icu-strain"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "clifpy==0.5.0",
    "duckdb>=1.0",
    "pandas>=2.0",
    "pyarrow>=14",
    "matplotlib>=3.8",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["code"]
```

`pythonpath = ["code"]` is what lets `import icu_strain` resolve in tests without installing the package.

- [ ] **Step 2: Write the failing test**

Create `tests/test_definitions.py`:

```python
from icu_strain import definitions as d


def test_thresholds_match_manuscript():
    assert d.SPO2_HYPOXEMIA == 90
    assert d.PAO2_HYPOXEMIA == 60
    assert d.PF_RATIO_THRESHOLD == 300
    assert d.SF_RATIO_THRESHOLD == 315
    assert d.PACO2_HYPERCAPNIA == 45
    assert d.PH_HYPERCAPNIA == 7.35


def test_abg_categories_are_arterial_only():
    cats = {d.ABG_PO2, d.ABG_PCO2, d.ABG_PH}
    assert cats == {"po2_arterial", "pco2_arterial", "ph_arterial"}
    assert not any("venous" in c for c in cats)


def test_criteria_text_covers_all_subtypes():
    for key in ("hypoxemic", "hypercapnic", "mixed", "not_ards"):
        assert key in d.CRITERIA_TEXT
        assert len(d.CRITERIA_TEXT[key]) > 40


def test_report_never_calls_this_ards():
    for key in ("hypoxemic", "hypercapnic", "mixed"):
        assert "ARDS" not in d.CRITERIA_TEXT[key]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_definitions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'icu_strain'`

- [ ] **Step 4: Create the package and definitions**

Create empty `code/icu_strain/__init__.py`.

Create `code/icu_strain/definitions.py`:

```python
"""Threshold constants and criteria prose for the ICU strain ARF analysis.

Both the analysis (04) and the HTML report (05) import from here so the
documented criteria cannot drift from the implemented ones.

Source: references/main_text_12272025.docx, Methods, "Patient Population and
ARF Clinical Definition".
"""

# --- ARF physiologic thresholds ---
SPO2_HYPOXEMIA = 90          # %, strict less-than, room air only
PAO2_HYPOXEMIA = 60          # mmHg, less-than-or-equal, room air only
PF_RATIO_THRESHOLD = 300     # PaO2/FiO2, less-than-or-equal, any FiO2
SF_RATIO_THRESHOLD = 315     # SpO2/FiO2, less-than-or-equal, no-ABG fallback
PACO2_HYPERCAPNIA = 45       # mmHg, greater-than-or-equal
PH_HYPERCAPNIA = 7.35        # strict less-than

# --- Windows and tolerances ---
ARF_WINDOW_HOURS = 24        # hours after ICU admission
ICU_BLOCK_GAP_HOURS = 1      # ADT contiguity threshold
ABG_PAIR_TOLERANCE_MIN = 5   # minutes, pairing pco2 with ph

# --- Cohort ---
ANALYSIS_YEAR = 2024
MIN_AGE = 18

# --- Reporting ---
MIN_CELL_SIZE = 10           # suppress any reported cell below this

# --- CLIF category values (exact matches; venous/capillary deliberately absent) ---
ABG_PO2 = "po2_arterial"
ABG_PCO2 = "pco2_arterial"
ABG_PH = "ph_arterial"
SPO2_VITAL = "spo2"
ICU_LOCATION = "icu"
IMV_DEVICE = "IMV"
ROOM_AIR_DEVICE = "Room Air"

CRITERIA_TEXT = {
    "hypoxemic": (
        f"SpO2 < {SPO2_HYPOXEMIA}% or PaO2 <= {PAO2_HYPOXEMIA} mmHg while on room air, "
        f"OR PaO2/FiO2 <= {PF_RATIO_THRESHOLD} at any FiO2, "
        f"OR SpO2/FiO2 <= {SF_RATIO_THRESHOLD} as a fallback when the hospitalization "
        "has no arterial blood gas anywhere in the window."
    ),
    "hypercapnic": (
        f"PaCO2 >= {PACO2_HYPERCAPNIA} mmHg AND arterial pH < {PH_HYPERCAPNIA}, "
        f"measured on the same arterial draw (paired within "
        f"{ABG_PAIR_TOLERANCE_MIN} minutes). Venous and capillary gases are excluded."
    ),
    "mixed": (
        "Meets hypoxemic criteria at some point in the window and hypercapnic criteria "
        "at some point in the window. The two need not be simultaneous."
    ),
    "not_ards": (
        "These are physiologic acute respiratory failure subtypes, NOT ARDS. The New "
        "Global Definition of ARDS requires bilateral opacities on chest imaging and "
        "exclusion of cardiogenic pulmonary edema; CLIF contains no imaging table, so "
        "ARDS cannot be ascertained here. Do not read these counts as ARDS cases."
    ),
}

SF_FALLBACK_CAVEAT = (
    f"The SpO2/FiO2 <= {SF_RATIO_THRESHOLD} fallback is a local construction, not part "
    "of the manuscript's ARF definition. It carries no SpO2 <= 97% ceiling, so above "
    "97% saturation the ratio loses meaning and hypoxemia may be over-classified."
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv sync && uv run pytest tests/test_definitions.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock code/icu_strain/ tests/test_definitions.py
git commit -m "feat: add project scaffolding and ARF definition constants"
```

---

### Task 2: FiO₂ normalization

**Files:**
- Create: `code/icu_strain/physiology.py`
- Test: `tests/test_physiology_fio2.py`

**Interfaces:**
- Consumes: `icu_strain.definitions`
- Produces: `normalize_fio2(s: pd.Series) -> pd.Series` returning float FiO₂ as a fraction in `[0.21, 1.0]`, or `NaN` where uninterpretable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_physiology_fio2.py`:

```python
import pandas as pd
import numpy as np
from icu_strain.physiology import normalize_fio2


def test_fractions_pass_through():
    out = normalize_fio2(pd.Series([0.21, 0.40, 1.0]))
    assert list(out) == [0.21, 0.40, 1.0]


def test_percentages_are_rescaled():
    out = normalize_fio2(pd.Series([21.0, 40.0, 100.0]))
    assert list(out) == [0.21, 0.40, 1.0]


def test_impossible_values_become_nan():
    out = normalize_fio2(pd.Series([-1.0, 0.0, 88880.0, 101.0]))
    assert out.isna().all()


def test_boundary_one_is_treated_as_fraction():
    # 1.0 is 100% oxygen, not 1%. Must not be rescaled to 0.01.
    assert normalize_fio2(pd.Series([1.0])).iloc[0] == 1.0


def test_below_room_air_fraction_is_nan():
    # 0.10 is below the fraction of oxygen in air; not physically deliverable.
    assert np.isnan(normalize_fio2(pd.Series([0.10])).iloc[0])


def test_nan_stays_nan():
    assert normalize_fio2(pd.Series([np.nan])).isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_physiology_fio2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'icu_strain.physiology'`

- [ ] **Step 3: Write minimal implementation**

Create `code/icu_strain/physiology.py`:

```python
"""ARF criteria evaluation over the CLIF physiologic timeline."""

import numpy as np
import pandas as pd

from . import definitions as d


def normalize_fio2(s: pd.Series) -> pd.Series:
    """Coerce fio2_set to a fraction in [0.21, 1.0].

    The raw column at ucmc ranges -1.0 to 88880 with a 0.40 median: mostly
    fractions, some percentages, some garbage. FiO2 is the denominator of both
    P/F and S/F, so an unnormalized value silently manufactures a qualifying
    ratio. Anything uninterpretable becomes NaN, which disqualifies the
    observation rather than defaulting it.
    """
    v = pd.to_numeric(s, errors="coerce").astype(float)
    # (1, 100] are percentages. 1.0 itself is 100% oxygen, so it stays a fraction.
    v = v.where(~((v > 1.0) & (v <= 100.0)), v / 100.0)
    # Outside physically deliverable range -> unusable.
    v = v.where((v >= 0.21) & (v <= 1.0), np.nan)
    return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_physiology_fio2.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add code/icu_strain/physiology.py tests/test_physiology_fio2.py
git commit -m "feat: add FiO2 normalization with mixed-unit and outlier handling"
```

---

### Task 3: ICU block collapsing and cohort construction

**Files:**
- Create: `code/icu_strain/cohort.py`
- Test: `tests/test_cohort.py`

**Interfaces:**
- Consumes: `icu_strain.definitions`
- Produces:
  - `collapse_icu_blocks(adt: pd.DataFrame, gap_hours: float = 1) -> pd.DataFrame` with columns `hospitalization_id`, `icu_admit_dttm`, `icu_discharge_dttm`, `block_index` (0-based, ordered by time), `n_adt_rows_merged`.
  - `build_cohort(hospitalization: pd.DataFrame, adt: pd.DataFrame, year: int, min_age: int) -> pd.DataFrame` with columns `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`, `icu_admit_dttm`, `icu_discharge_dttm`, `window_start`, `window_end`. One row per hospitalization, index (first) ICU block only.

`window_start` = `admission_dttm`; `window_end` = `icu_admit_dttm + ARF_WINDOW_HOURS`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cohort.py`:

```python
import pandas as pd
import pytest
from icu_strain.cohort import collapse_icu_blocks, build_cohort


def _adt(rows):
    return pd.DataFrame(rows, columns=["hospitalization_id", "in_dttm", "out_dttm", "location_category"]).assign(
        in_dttm=lambda x: pd.to_datetime(x["in_dttm"]),
        out_dttm=lambda x: pd.to_datetime(x["out_dttm"]),
    )


def test_adjacent_icu_rows_merge_into_one_block():
    adt = _adt([
        ("H1", "2024-03-01 00:00", "2024-03-01 06:00", "icu"),
        ("H1", "2024-03-01 06:00", "2024-03-01 12:00", "icu"),
    ])
    out = collapse_icu_blocks(adt, gap_hours=1)
    assert len(out) == 1
    assert out.iloc[0]["icu_admit_dttm"] == pd.Timestamp("2024-03-01 00:00")
    assert out.iloc[0]["icu_discharge_dttm"] == pd.Timestamp("2024-03-01 12:00")
    assert out.iloc[0]["n_adt_rows_merged"] == 2


def test_gap_beyond_threshold_splits_blocks():
    adt = _adt([
        ("H1", "2024-03-01 00:00", "2024-03-01 06:00", "icu"),
        ("H1", "2024-03-01 10:00", "2024-03-01 12:00", "icu"),
    ])
    out = collapse_icu_blocks(adt, gap_hours=1).sort_values("block_index")
    assert len(out) == 2
    assert list(out["block_index"]) == [0, 1]


def test_non_icu_rows_are_ignored():
    adt = _adt([
        ("H1", "2024-03-01 00:00", "2024-03-01 06:00", "ed"),
        ("H1", "2024-03-01 06:00", "2024-03-01 12:00", "ward"),
    ])
    assert len(collapse_icu_blocks(adt)) == 0


def test_intervening_ward_row_does_not_bridge_icu_blocks():
    # A ward stay between two ICU rows is a real transfer out; the 1h gap rule
    # must not merge across it even though the timestamps are contiguous.
    adt = _adt([
        ("H1", "2024-03-01 00:00", "2024-03-01 06:00", "icu"),
        ("H1", "2024-03-01 06:00", "2024-03-01 06:30", "ward"),
        ("H1", "2024-03-01 06:30", "2024-03-01 12:00", "icu"),
    ])
    out = collapse_icu_blocks(adt, gap_hours=1)
    assert len(out) == 2


def test_build_cohort_keeps_first_block_and_filters():
    hosp = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "hospitalization_id": ["H1", "H2", "H3"],
        "admission_dttm": pd.to_datetime(["2024-03-01 00:00", "2024-03-02 00:00", "2023-06-01 00:00"]),
        "discharge_dttm": pd.to_datetime(["2024-03-10 00:00", "2024-03-12 00:00", "2023-06-10 00:00"]),
        "age_at_admission": [55, 17, 70],
    })
    adt = _adt([
        ("H1", "2024-03-01 04:00", "2024-03-03 00:00", "icu"),
        ("H1", "2024-03-05 00:00", "2024-03-06 00:00", "icu"),
        ("H2", "2024-03-02 04:00", "2024-03-04 00:00", "icu"),   # minor
        ("H3", "2023-06-01 04:00", "2023-06-03 00:00", "icu"),   # wrong year
    ])
    out = build_cohort(hosp, adt, year=2024, min_age=18)
    assert list(out["hospitalization_id"]) == ["H1"]
    assert out.iloc[0]["icu_admit_dttm"] == pd.Timestamp("2024-03-01 04:00")
    assert out.iloc[0]["window_start"] == pd.Timestamp("2024-03-01 00:00")
    assert out.iloc[0]["window_end"] == pd.Timestamp("2024-03-02 04:00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cohort.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'icu_strain.cohort'`

- [ ] **Step 3: Write minimal implementation**

Create `code/icu_strain/cohort.py`:

```python
"""Cohort construction: ICU stay boundaries and eligibility."""

import pandas as pd

from . import definitions as d


def collapse_icu_blocks(adt: pd.DataFrame, gap_hours: float = d.ICU_BLOCK_GAP_HOURS) -> pd.DataFrame:
    """Merge consecutive ICU ADT rows into contiguous blocks.

    A new block starts when the previous location was not ICU, or when the gap
    from the previous ICU row's out_dttm exceeds gap_hours. Sorting includes
    non-ICU rows so that an intervening ward stay correctly breaks the chain.
    """
    df = adt.sort_values(["hospitalization_id", "in_dttm"]).copy()
    df["_is_icu"] = df["location_category"].str.lower() == d.ICU_LOCATION

    prev_out = df.groupby("hospitalization_id")["out_dttm"].shift()
    prev_icu = df.groupby("hospitalization_id")["_is_icu"].shift(fill_value=False)
    gap = (df["in_dttm"] - prev_out).dt.total_seconds() / 3600.0

    starts_block = (~prev_icu) | (gap > gap_hours) | gap.isna()
    df["_block"] = (starts_block & df["_is_icu"]).groupby(df["hospitalization_id"]).cumsum()

    icu = df[df["_is_icu"]]
    if icu.empty:
        return pd.DataFrame(
            columns=["hospitalization_id", "icu_admit_dttm", "icu_discharge_dttm",
                     "block_index", "n_adt_rows_merged"]
        )

    out = (
        icu.groupby(["hospitalization_id", "_block"], as_index=False)
        .agg(
            icu_admit_dttm=("in_dttm", "min"),
            icu_discharge_dttm=("out_dttm", "max"),
            n_adt_rows_merged=("in_dttm", "size"),
        )
    )
    out["block_index"] = out.groupby("hospitalization_id")["icu_admit_dttm"].rank("dense").astype(int) - 1
    return out.drop(columns=["_block"])


def build_cohort(
    hospitalization: pd.DataFrame,
    adt: pd.DataFrame,
    year: int = d.ANALYSIS_YEAR,
    min_age: int = d.MIN_AGE,
) -> pd.DataFrame:
    """One row per hospitalization: adults with an index ICU stay in `year`.

    Not ARF-filtered. This is the denominator for deliverable 1; deliverables 2
    and 3 operate on the ARF-positive subset.
    """
    blocks = collapse_icu_blocks(adt)
    index_stay = blocks[blocks["block_index"] == 0]

    cohort = hospitalization.merge(index_stay, on="hospitalization_id", how="inner")
    cohort = cohort[
        (cohort["age_at_admission"] >= min_age)
        & (cohort["icu_admit_dttm"].dt.year == year)
    ].copy()

    cohort["window_start"] = cohort["admission_dttm"]
    cohort["window_end"] = cohort["icu_admit_dttm"] + pd.Timedelta(hours=d.ARF_WINDOW_HOURS)

    return cohort[[
        "patient_id", "hospitalization_id", "admission_dttm", "discharge_dttm",
        "age_at_admission", "icu_admit_dttm", "icu_discharge_dttm",
        "window_start", "window_end",
    ]].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cohort.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add code/icu_strain/cohort.py tests/test_cohort.py
git commit -m "feat: add ICU block collapsing and cohort construction"
```

---

### Task 4: ARF criteria evaluation and subtype classification

**Files:**
- Modify: `code/icu_strain/physiology.py`
- Test: `tests/test_physiology_criteria.py`

**Interfaces:**
- Consumes: `normalize_fio2` from Task 2; cohort frame from Task 3.
- Produces:
  - `evaluate_hypoxemia(obs: pd.DataFrame) -> pd.DataFrame` — input columns `hospitalization_id`, `dttm`, `spo2`, `po2`, `fio2`, `device_category`, `has_abg`; returns rows that qualify with added column `hypoxemia_rule` in `{"room_air_spo2", "room_air_pao2", "pf_ratio", "sf_ratio"}`.
  - `evaluate_hypercapnia(labs: pd.DataFrame, tolerance_min: int) -> pd.DataFrame` — input long-format columns `hospitalization_id`, `lab_result_dttm`, `lab_category`, `lab_value_numeric`; returns qualifying rows with `dttm`.
  - `classify_arf(hypox: pd.DataFrame, hypercap: pd.DataFrame) -> pd.DataFrame` — columns `hospitalization_id`, `arf_subtype` in `{"hypoxemic","hypercapnic","mixed"}`, `first_arf_dttm`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_physiology_criteria.py`:

```python
import pandas as pd
from icu_strain.physiology import evaluate_hypoxemia, evaluate_hypercapnia, classify_arf
from icu_strain import definitions as d


def _obs(rows):
    return pd.DataFrame(
        rows,
        columns=["hospitalization_id", "dttm", "spo2", "po2", "fio2", "device_category", "has_abg"],
    ).assign(dttm=lambda x: pd.to_datetime(x["dttm"]))


def test_room_air_spo2_below_90_qualifies():
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", 88, None, 0.21, "Room Air", True)]))
    assert list(out["hypoxemia_rule"]) == ["room_air_spo2"]


def test_room_air_spo2_at_90_does_not_qualify():
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", 90, None, 0.21, "Room Air", True)]))
    assert out.empty


def test_low_spo2_on_supplemental_o2_does_not_trigger_room_air_rule():
    # 88% on 6L nasal cannula must not match the room-air rule.
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", 88, None, 0.40, "Nasal Cannula", True)]))
    assert "room_air_spo2" not in set(out["hypoxemia_rule"])


def test_pf_ratio_at_or_below_300_qualifies():
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", None, 120.0, 0.40, "IMV", True)]))
    assert list(out["hypoxemia_rule"]) == ["pf_ratio"]  # 120/0.40 = 300


def test_sf_fallback_only_applies_without_abg():
    with_abg = _obs([("H1", "2024-03-01 01:00", 90, None, 0.40, "Nasal Cannula", True)])
    without_abg = _obs([("H2", "2024-03-01 01:00", 90, None, 0.40, "Nasal Cannula", False)])
    assert evaluate_hypoxemia(with_abg).empty          # 90/0.40 = 225 but ABG exists
    assert not evaluate_hypoxemia(without_abg).empty   # fallback allowed


def test_null_fio2_disqualifies_ratio_rules():
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", 85, 50.0, None, "Nasal Cannula", False)]))
    assert out.empty


def _labs(rows):
    return pd.DataFrame(
        rows, columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"]
    ).assign(lab_result_dttm=lambda x: pd.to_datetime(x["lab_result_dttm"]))


def test_paired_arterial_gas_qualifies():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", d.ABG_PCO2, 50.0),
        ("H1", "2024-03-01 01:00", d.ABG_PH, 7.30),
    ]), tolerance_min=5)
    assert len(out) == 1


def test_high_pco2_with_normal_ph_does_not_qualify():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", d.ABG_PCO2, 50.0),
        ("H1", "2024-03-01 01:00", d.ABG_PH, 7.40),
    ]), tolerance_min=5)
    assert out.empty


def test_venous_gas_never_qualifies():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", "pco2_venous", 55.0),
        ("H1", "2024-03-01 01:00", "ph_venous", 7.30),
    ]), tolerance_min=5)
    assert out.empty


def test_gases_beyond_tolerance_do_not_pair():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", d.ABG_PCO2, 50.0),
        ("H1", "2024-03-01 01:30", d.ABG_PH, 7.30),
    ]), tolerance_min=5)
    assert out.empty


def test_classify_assigns_mixed_and_earliest_timestamp():
    hypox = pd.DataFrame({
        "hospitalization_id": ["H1", "H2"],
        "dttm": pd.to_datetime(["2024-03-01 05:00", "2024-03-02 01:00"]),
    })
    hypercap = pd.DataFrame({
        "hospitalization_id": ["H1", "H3"],
        "dttm": pd.to_datetime(["2024-03-01 02:00", "2024-03-03 01:00"]),
    })
    out = classify_arf(hypox, hypercap).set_index("hospitalization_id")
    assert out.loc["H1", "arf_subtype"] == "mixed"
    assert out.loc["H1", "first_arf_dttm"] == pd.Timestamp("2024-03-01 02:00")
    assert out.loc["H2", "arf_subtype"] == "hypoxemic"
    assert out.loc["H3", "arf_subtype"] == "hypercapnic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_physiology_criteria.py -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate_hypoxemia'`

- [ ] **Step 3: Append implementation to `code/icu_strain/physiology.py`**

```python
def evaluate_hypoxemia(obs: pd.DataFrame) -> pd.DataFrame:
    """Return observations meeting any hypoxemic ARF rule.

    `has_abg` is hospitalization-level: True if the hospitalization has any
    arterial blood gas anywhere in its window. The S/F fallback is gated on it
    so that pulse-ox-only patients are not lost, without letting S/F override
    a real gas.
    """
    df = obs.copy()
    df["fio2"] = normalize_fio2(df["fio2"])
    is_room_air = df["device_category"].fillna("") == d.ROOM_AIR_DEVICE

    rules = {
        "room_air_spo2": is_room_air & (df["spo2"] < d.SPO2_HYPOXEMIA),
        "room_air_pao2": is_room_air & (df["po2"] <= d.PAO2_HYPOXEMIA),
        "pf_ratio": (df["po2"] / df["fio2"]) <= d.PF_RATIO_THRESHOLD,
        "sf_ratio": (~df["has_abg"]) & ((df["spo2"] / df["fio2"]) <= d.SF_RATIO_THRESHOLD),
    }

    hits = []
    for name, mask in rules.items():
        m = mask.fillna(False)
        if m.any():
            hits.append(df[m].assign(hypoxemia_rule=name))
    if not hits:
        return df.head(0).assign(hypoxemia_rule=pd.Series(dtype=object))
    return pd.concat(hits, ignore_index=True)


def evaluate_hypercapnia(labs: pd.DataFrame, tolerance_min: int = d.ABG_PAIR_TOLERANCE_MIN) -> pd.DataFrame:
    """Return arterial draws with PaCO2 >= 45 AND pH < 7.35.

    Only exact arterial categories are considered; pco2_venous / ph_venous are
    never matched, since a venous PCO2 >= 45 is physiologically normal.
    """
    pco2 = labs[labs["lab_category"] == d.ABG_PCO2][
        ["hospitalization_id", "lab_result_dttm", "lab_value_numeric"]
    ].rename(columns={"lab_result_dttm": "pco2_dttm", "lab_value_numeric": "pco2"})
    ph = labs[labs["lab_category"] == d.ABG_PH][
        ["hospitalization_id", "lab_result_dttm", "lab_value_numeric"]
    ].rename(columns={"lab_result_dttm": "ph_dttm", "lab_value_numeric": "ph"})

    if pco2.empty or ph.empty:
        return pd.DataFrame(columns=["hospitalization_id", "dttm", "pco2", "ph"])

    paired = pco2.merge(ph, on="hospitalization_id", how="inner")
    within = (paired["pco2_dttm"] - paired["ph_dttm"]).abs() <= pd.Timedelta(minutes=tolerance_min)
    paired = paired[within]

    qualifies = (paired["pco2"] >= d.PACO2_HYPERCAPNIA) & (paired["ph"] < d.PH_HYPERCAPNIA)
    out = paired[qualifies].copy()
    out["dttm"] = out[["pco2_dttm", "ph_dttm"]].min(axis=1)
    return out[["hospitalization_id", "dttm", "pco2", "ph"]].reset_index(drop=True)


def classify_arf(hypox: pd.DataFrame, hypercap: pd.DataFrame) -> pd.DataFrame:
    """Assign one subtype per hospitalization from the union over the window.

    A patient hypoxemic in the ED and hypercapnic in the ICU is 'mixed'.
    first_arf_dttm is the earliest qualifying observation of any type.
    """
    a = hypox.groupby("hospitalization_id")["dttm"].min().rename("first_hypox")
    b = hypercap.groupby("hospitalization_id")["dttm"].min().rename("first_hypercap")
    j = pd.concat([a, b], axis=1)

    has_a, has_b = j["first_hypox"].notna(), j["first_hypercap"].notna()
    j["arf_subtype"] = pd.NA
    j.loc[has_a & ~has_b, "arf_subtype"] = "hypoxemic"
    j.loc[~has_a & has_b, "arf_subtype"] = "hypercapnic"
    j.loc[has_a & has_b, "arf_subtype"] = "mixed"

    j["first_arf_dttm"] = j[["first_hypox", "first_hypercap"]].min(axis=1)
    return j.reset_index()[["hospitalization_id", "arf_subtype", "first_arf_dttm"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_physiology_criteria.py -v`
Expected: 11 passed

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests from Tasks 1–4 pass

- [ ] **Step 6: Commit**

```bash
git add code/icu_strain/physiology.py tests/test_physiology_criteria.py
git commit -m "feat: add ARF criteria evaluation and subtype classification"
```

---

### Task 5: Monthly aggregation and small-cell suppression

**Files:**
- Create: `code/icu_strain/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: cohort frame (Task 3), classification frame (Task 4).
- Produces:
  - `suppress_small_cells(df: pd.DataFrame, count_cols: list[str], min_n: int) -> pd.DataFrame` — replaces counts `0 < n < min_n` with `pd.NA`. Exact zeros are kept, since zero reveals nothing about an individual.
  - `monthly_imv_counts(imv_starts: pd.DataFrame) -> pd.DataFrame` — columns `month` (`YYYY-MM`), `n_new_imv`.
  - `monthly_subtype_counts(cohort: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame` — columns `month`, `hypoxemic`, `hypercapnic`, `mixed`, `total`.
  - `time_to_icu_summary(cohort: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame` — columns `arf_subtype`, `n`, `median_hours`, `q25_hours`, `q75_hours`, `pct_before_icu`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregate.py`:

```python
import pandas as pd
from icu_strain.aggregate import (
    suppress_small_cells, monthly_imv_counts, monthly_subtype_counts, time_to_icu_summary,
)


def test_small_counts_are_suppressed_but_zero_is_kept():
    df = pd.DataFrame({"month": ["2024-01", "2024-02", "2024-03"], "n": [0, 5, 40]})
    out = suppress_small_cells(df, ["n"], min_n=10)
    assert out.loc[0, "n"] == 0
    assert pd.isna(out.loc[1, "n"])
    assert out.loc[2, "n"] == 40


def test_monthly_imv_bins_by_imv_start_month():
    imv = pd.DataFrame({
        "hospitalization_id": ["H1", "H2", "H3"],
        "imv_start_dttm": pd.to_datetime(["2024-01-15", "2024-01-20", "2024-02-02"]),
    })
    out = monthly_imv_counts(imv).set_index("month")
    assert out.loc["2024-01", "n_new_imv"] == 2
    assert out.loc["2024-02", "n_new_imv"] == 1


def test_monthly_subtypes_sum_to_total():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1", "H2", "H3"],
        "icu_admit_dttm": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"]),
    })
    classified = pd.DataFrame({
        "hospitalization_id": ["H1", "H2", "H3"],
        "arf_subtype": ["hypoxemic", "mixed", "hypoxemic"],
        "first_arf_dttm": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"]),
    })
    out = monthly_subtype_counts(cohort, classified)
    row = out[out["month"] == "2024-01"].iloc[0]
    assert row["hypoxemic"] + row["hypercapnic"] + row["mixed"] == row["total"] == 3


def test_time_to_icu_reports_negative_deltas_as_pre_icu():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1", "H2"],
        "icu_admit_dttm": pd.to_datetime(["2024-01-05 12:00", "2024-01-06 12:00"]),
    })
    classified = pd.DataFrame({
        "hospitalization_id": ["H1", "H2"],
        "arf_subtype": ["hypoxemic", "hypoxemic"],
        # H1 met criteria 4h BEFORE ICU; H2 2h after.
        "first_arf_dttm": pd.to_datetime(["2024-01-05 08:00", "2024-01-06 14:00"]),
    })
    out = time_to_icu_summary(cohort, classified).set_index("arf_subtype")
    assert out.loc["hypoxemic", "n"] == 2
    assert out.loc["hypoxemic", "median_hours"] == -1.0   # median of (-4, +2)
    assert out.loc["hypoxemic", "pct_before_icu"] == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'icu_strain.aggregate'`

- [ ] **Step 3: Write minimal implementation**

Create `code/icu_strain/aggregate.py`:

```python
"""Monthly rollups and disclosure control."""

import pandas as pd

from . import definitions as d

SUBTYPES = ["hypoxemic", "hypercapnic", "mixed"]


def suppress_small_cells(df: pd.DataFrame, count_cols: list, min_n: int = d.MIN_CELL_SIZE) -> pd.DataFrame:
    """Mask counts in (0, min_n) as NA. Exact zeros are safe to publish."""
    out = df.copy()
    for c in count_cols:
        out[c] = out[c].mask((out[c] > 0) & (out[c] < min_n), pd.NA)
    return out


def monthly_imv_counts(imv_starts: pd.DataFrame) -> pd.DataFrame:
    """Count hospitalizations by month of first IMV initiation."""
    df = imv_starts.copy()
    df["month"] = df["imv_start_dttm"].dt.to_period("M").astype(str)
    return (
        df.groupby("month", as_index=False)["hospitalization_id"]
        .nunique()
        .rename(columns={"hospitalization_id": "n_new_imv"})
        .sort_values("month")
        .reset_index(drop=True)
    )


def monthly_subtype_counts(cohort: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:
    """Counts by ARF subtype, binned by month of ICU admission."""
    df = classified.merge(
        cohort[["hospitalization_id", "icu_admit_dttm"]], on="hospitalization_id", how="inner"
    )
    df["month"] = df["icu_admit_dttm"].dt.to_period("M").astype(str)

    wide = (
        df.pivot_table(index="month", columns="arf_subtype",
                       values="hospitalization_id", aggfunc="nunique", fill_value=0)
        .reindex(columns=SUBTYPES, fill_value=0)
        .reset_index()
    )
    wide.columns.name = None
    wide["total"] = wide[SUBTYPES].sum(axis=1)
    return wide.sort_values("month").reset_index(drop=True)


def time_to_icu_summary(cohort: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:
    """Signed hours from first ARF criteria to ICU admission, by subtype.

    Negative means the patient already met criteria before ICU arrival.
    """
    df = classified.merge(
        cohort[["hospitalization_id", "icu_admit_dttm"]], on="hospitalization_id", how="inner"
    )
    df["delta_hours"] = (df["first_arf_dttm"] - df["icu_admit_dttm"]).dt.total_seconds() / 3600.0

    return (
        df.groupby("arf_subtype", as_index=False)
        .agg(
            n=("delta_hours", "size"),
            median_hours=("delta_hours", "median"),
            q25_hours=("delta_hours", lambda s: s.quantile(0.25)),
            q75_hours=("delta_hours", lambda s: s.quantile(0.75)),
            pct_before_icu=("delta_hours", lambda s: 100.0 * (s < 0).mean()),
        )
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add code/icu_strain/aggregate.py tests/test_aggregate.py
git commit -m "feat: add monthly aggregation and small-cell suppression"
```

---

### Task 6: Pipeline drivers 01–04

**Files:**
- Create: `code/01_cohort_identification.py`
- Create: `code/02_project_quality_checks.py`
- Create: `code/03_outlier_handling.py`
- Create: `code/04_project_analysis.py`
- Create: `code/icu_strain/pipeline.py`
- Test: `tests/test_pipeline_smoke.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `icu_strain.pipeline.load_tables(config: dict) -> dict[str, pd.DataFrame]` returning keys `hospitalization`, `adt`, `labs`, `vitals`, `respiratory_support` (waterfall-processed, outlier-handled); `icu_strain.pipeline.build_observation_frame(cohort, labs, vitals, resp) -> pd.DataFrame` with the columns `evaluate_hypoxemia` expects; `icu_strain.pipeline.first_imv_starts(cohort, resp) -> pd.DataFrame` with `hospitalization_id`, `imv_start_dttm`.

**Note on the observation frame:** DuckDB's `ASOF JOIN` attaches the most recent prior waterfall row (device + FiO₂) to each lab and vital observation. This is the step where the waterfall's forward-fill actually gets used.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_smoke.py`:

```python
import pandas as pd
from icu_strain.pipeline import build_observation_frame, first_imv_starts


def test_observation_frame_attaches_most_recent_support_state():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "window_start": pd.to_datetime(["2024-01-01 00:00"]),
        "window_end": pd.to_datetime(["2024-01-02 00:00"]),
    })
    vitals = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 05:00"]),
        "vital_category": ["spo2"],
        "vital_value": [88.0],
    })
    labs = pd.DataFrame(columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"])
    resp = pd.DataFrame({
        "hospitalization_id": ["H1", "H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 04:00"]),
        "device_category": ["Nasal Cannula", "Room Air"],
        "fio2_set": [0.40, 0.21],
    })
    out = build_observation_frame(cohort, labs, vitals, resp)
    row = out[out["spo2"].notna()].iloc[0]
    assert row["device_category"] == "Room Air"   # 04:00 state, not 00:00
    assert row["fio2"] == 0.21


def test_observations_outside_window_are_dropped():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "window_start": pd.to_datetime(["2024-01-01 00:00"]),
        "window_end": pd.to_datetime(["2024-01-01 12:00"]),
    })
    vitals = pd.DataFrame({
        "hospitalization_id": ["H1", "H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 06:00", "2024-01-02 06:00"]),
        "vital_category": ["spo2", "spo2"],
        "vital_value": [88.0, 85.0],
    })
    labs = pd.DataFrame(columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"])
    resp = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 00:00"]),
        "device_category": ["Room Air"],
        "fio2_set": [0.21],
    })
    out = build_observation_frame(cohort, labs, vitals, resp)
    assert len(out) == 1


def test_first_imv_start_is_earliest_within_index_icu_stay():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "icu_admit_dttm": pd.to_datetime(["2024-01-01 00:00"]),
        "icu_discharge_dttm": pd.to_datetime(["2024-01-05 00:00"]),
    })
    resp = pd.DataFrame({
        "hospitalization_id": ["H1", "H1", "H1"],
        "recorded_dttm": pd.to_datetime(["2023-12-31 00:00", "2024-01-02 00:00", "2024-01-03 00:00"]),
        "device_category": ["IMV", "IMV", "IMV"],
        "fio2_set": [0.5, 0.5, 0.5],
    })
    out = first_imv_starts(cohort, resp)
    assert len(out) == 1
    # The pre-ICU IMV row is outside the index stay and must not count.
    assert out.iloc[0]["imv_start_dttm"] == pd.Timestamp("2024-01-02 00:00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'icu_strain.pipeline'`

- [ ] **Step 3: Write `code/icu_strain/pipeline.py`**

```python
"""Loading, waterfall processing, and observation assembly."""

import duckdb
import pandas as pd

from . import definitions as d
from .physiology import normalize_fio2


def build_observation_frame(cohort, labs, vitals, resp) -> pd.DataFrame:
    """One row per lab/vital observation inside the window, with support state.

    ASOF JOIN attaches the most recent prior waterfall row, which is how the
    forward-filled device and FiO2 reach each observation.
    """
    con = duckdb.connect()
    con.register("cohort", cohort)
    con.register("labs", labs)
    con.register("vitals", vitals)
    con.register("resp", resp)

    obs = con.execute(f"""
        WITH events AS (
            SELECT hospitalization_id, recorded_dttm AS dttm,
                   CASE WHEN vital_category = '{d.SPO2_VITAL}' THEN vital_value END AS spo2,
                   NULL::DOUBLE AS po2
            FROM vitals WHERE vital_category = '{d.SPO2_VITAL}'
            UNION ALL
            SELECT hospitalization_id, lab_result_dttm AS dttm,
                   NULL::DOUBLE AS spo2,
                   CASE WHEN lab_category = '{d.ABG_PO2}' THEN lab_value_numeric END AS po2
            FROM labs WHERE lab_category = '{d.ABG_PO2}'
        ),
        windowed AS (
            SELECT e.* FROM events e
            JOIN cohort c USING (hospitalization_id)
            WHERE e.dttm >= c.window_start AND e.dttm <= c.window_end
        )
        SELECT w.hospitalization_id, w.dttm, w.spo2, w.po2,
               r.device_category, r.fio2_set AS fio2
        FROM windowed w
        ASOF LEFT JOIN resp r
          ON w.hospitalization_id = r.hospitalization_id
         AND w.dttm >= r.recorded_dttm
        ORDER BY w.hospitalization_id, w.dttm
    """).df()
    con.close()

    obs["fio2"] = normalize_fio2(obs["fio2"])
    return obs


def attach_abg_availability(obs: pd.DataFrame, labs: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Add hospitalization-level `has_abg`, gating the S/F fallback."""
    arterial = {d.ABG_PO2, d.ABG_PCO2, d.ABG_PH}
    windowed = labs.merge(
        cohort[["hospitalization_id", "window_start", "window_end"]], on="hospitalization_id"
    )
    in_window = (
        (windowed["lab_result_dttm"] >= windowed["window_start"])
        & (windowed["lab_result_dttm"] <= windowed["window_end"])
        & (windowed["lab_category"].isin(arterial))
    )
    ids = set(windowed[in_window]["hospitalization_id"])
    out = obs.copy()
    out["has_abg"] = out["hospitalization_id"].isin(ids)
    return out


def first_imv_starts(cohort: pd.DataFrame, resp: pd.DataFrame) -> pd.DataFrame:
    """Earliest IMV observation inside each hospitalization's index ICU stay."""
    df = resp[resp["device_category"] == d.IMV_DEVICE].merge(
        cohort[["hospitalization_id", "icu_admit_dttm", "icu_discharge_dttm"]],
        on="hospitalization_id", how="inner",
    )
    inside = (df["recorded_dttm"] >= df["icu_admit_dttm"]) & (df["recorded_dttm"] <= df["icu_discharge_dttm"])
    return (
        df[inside].groupby("hospitalization_id", as_index=False)["recorded_dttm"]
        .min().rename(columns={"recorded_dttm": "imv_start_dttm"})
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline_smoke.py -v`
Expected: 3 passed

- [ ] **Step 5: Write the four driver scripts**

Each driver reads `config/config.json` via the existing `utils/config.py` pattern, and must be run from the repo root.

`code/01_cohort_identification.py`:

```python
"""Build the ICU cohort for the analysis year. Run from repo root."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from clifpy import ClifOrchestrator

from icu_strain import definitions as d
from icu_strain.cohort import build_cohort

config = json.loads(Path("config/config.json").read_text())
out_dir = Path("output/intermediate_phi")
out_dir.mkdir(parents=True, exist_ok=True)

co = ClifOrchestrator(
    data_directory=config["data_directory"],
    filetype=config["filetype"],
    timezone=config["timezone"],
)
co.initialize(tables=["hospitalization", "adt"])

cohort = build_cohort(co.hospitalization.df, co.adt.df, year=d.ANALYSIS_YEAR, min_age=d.MIN_AGE)
cohort.to_parquet(out_dir / "cohort.parquet", index=False)
print(f"cohort: {len(cohort):,} hospitalizations with an index ICU stay in {d.ANALYSIS_YEAR}")
```

`code/02_project_quality_checks.py`:

```python
"""Per-month data availability. Run from repo root."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from clifpy import ClifOrchestrator

from icu_strain import definitions as d

config = json.loads(Path("config/config.json").read_text())
out_dir = Path("output/final_no_phi")
out_dir.mkdir(parents=True, exist_ok=True)

cohort = pd.read_parquet("output/intermediate_phi/cohort.parquet")

co = ClifOrchestrator(
    data_directory=config["data_directory"],
    filetype=config["filetype"],
    timezone=config["timezone"],
)
co.initialize(tables=["labs", "vitals", "respiratory_support"])

arterial = {d.ABG_PO2, d.ABG_PCO2, d.ABG_PH}
labs, vitals, resp = co.labs.df, co.vitals.df, co.respiratory_support.df

cohort["month"] = cohort["icu_admit_dttm"].dt.to_period("M").astype(str)


def _in_window(df, dttm_col, ids_col="hospitalization_id"):
    m = df.merge(cohort[[ids_col, "window_start", "window_end", "month"]], on=ids_col)
    return m[(m[dttm_col] >= m["window_start"]) & (m[dttm_col] <= m["window_end"])]


abg_ids = _in_window(labs[labs["lab_category"].isin(arterial)], "lab_result_dttm")
spo2_ids = _in_window(vitals[vitals["vital_category"] == d.SPO2_VITAL], "recorded_dttm")
fio2_ids = _in_window(resp[resp["fio2_set"].notna()], "recorded_dttm")

rows = []
for month, grp in cohort.groupby("month"):
    n = len(grp)
    ids = set(grp["hospitalization_id"])
    rows.append({
        "month": month,
        "n_hospitalizations": n,
        "pct_with_abg": 100.0 * len(ids & set(abg_ids["hospitalization_id"])) / n,
        "pct_with_spo2": 100.0 * len(ids & set(spo2_ids["hospitalization_id"])) / n,
        "pct_with_fio2": 100.0 * len(ids & set(fio2_ids["hospitalization_id"])) / n,
    })

qc = pd.DataFrame(rows)
qc["device_null_rate_raw"] = 100.0 * resp["device_category"].isna().mean()
qc.to_csv(out_dir / "quality_checks.csv", index=False)
print(qc.to_string(index=False))
```

`code/03_outlier_handling.py`:

```python
"""Apply clifpy outlier handling and the respiratory support waterfall.

Must run before criteria evaluation: an SpO2 of 0 or PaO2 of 9999 would
otherwise manufacture false hypoxemia.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from clifpy import ClifOrchestrator, apply_outlier_handling, process_resp_support_waterfall

from icu_strain import definitions as d

config = json.loads(Path("config/config.json").read_text())
phi_dir = Path("output/intermediate_phi")
out_dir = Path("output/final_no_phi")
out_dir.mkdir(parents=True, exist_ok=True)

co = ClifOrchestrator(
    data_directory=config["data_directory"],
    filetype=config["filetype"],
    timezone=config["timezone"],
)
co.initialize(tables=["labs", "vitals", "respiratory_support"])

before = {
    "labs": co.labs.df["lab_value_numeric"].notna().sum(),
    "vitals": co.vitals.df["vital_value"].notna().sum(),
    "respiratory_support": co.respiratory_support.df["fio2_set"].notna().sum(),
}

for tbl in (co.labs, co.vitals, co.respiratory_support):
    apply_outlier_handling(tbl)

after = {
    "labs": co.labs.df["lab_value_numeric"].notna().sum(),
    "vitals": co.vitals.df["vital_value"].notna().sum(),
    "respiratory_support": co.respiratory_support.df["fio2_set"].notna().sum(),
}

pd.DataFrame([
    {"table": k, "values_before": before[k], "values_after": after[k],
     "values_removed": before[k] - after[k]}
    for k in before
]).to_csv(out_dir / "outlier_summary.csv", index=False)

waterfall = process_resp_support_waterfall(co.respiratory_support.df)
waterfall.to_parquet(phi_dir / "resp_support_waterfall.parquet", index=False)

co.labs.df.to_parquet(phi_dir / "labs_clean.parquet", index=False)
co.vitals.df.to_parquet(phi_dir / "vitals_clean.parquet", index=False)
print(f"waterfall rows: {len(waterfall):,}; "
      f"device NULL rate after: {100.0 * waterfall['device_category'].isna().mean():.1f}%")
```

`code/04_project_analysis.py`:

```python
"""Classify ARF subtypes and emit the three deliverables. Run from repo root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from icu_strain import definitions as d
from icu_strain.aggregate import (
    monthly_imv_counts, monthly_subtype_counts, suppress_small_cells, time_to_icu_summary,
)
from icu_strain.physiology import classify_arf, evaluate_hypercapnia, evaluate_hypoxemia
from icu_strain.pipeline import attach_abg_availability, build_observation_frame, first_imv_starts

phi_dir = Path("output/intermediate_phi")
out_dir = Path("output/final_no_phi")
out_dir.mkdir(parents=True, exist_ok=True)

cohort = pd.read_parquet(phi_dir / "cohort.parquet")
labs = pd.read_parquet(phi_dir / "labs_clean.parquet")
vitals = pd.read_parquet(phi_dir / "vitals_clean.parquet")
resp = pd.read_parquet(phi_dir / "resp_support_waterfall.parquet")

obs = build_observation_frame(cohort, labs, vitals, resp)
obs = attach_abg_availability(obs, labs, cohort)

hypox = evaluate_hypoxemia(obs)

labs_win = labs.merge(cohort[["hospitalization_id", "window_start", "window_end"]], on="hospitalization_id")
labs_win = labs_win[
    (labs_win["lab_result_dttm"] >= labs_win["window_start"])
    & (labs_win["lab_result_dttm"] <= labs_win["window_end"])
]
hypercap = evaluate_hypercapnia(labs_win)

classified = classify_arf(hypox, hypercap)
classified.to_parquet(phi_dir / "arf_classified.parquet", index=False)

# Deliverable 1
imv = first_imv_starts(cohort, resp)
suppress_small_cells(monthly_imv_counts(imv), ["n_new_imv"]).to_csv(
    out_dir / "monthly_imv.csv", index=False)

# Deliverable 2
subtype_cols = ["hypoxemic", "hypercapnic", "mixed", "total"]
suppress_small_cells(monthly_subtype_counts(cohort, classified), subtype_cols).to_csv(
    out_dir / "monthly_arf_subtypes.csv", index=False)

# Deliverable 3.
# Suppress the whole row, not just n: publishing a median for a subtype with
# n=3 while masking n would leak more than the count it hides.
t3 = time_to_icu_summary(cohort, classified)
t3.loc[t3["n"] < d.MIN_CELL_SIZE,
       ["n", "median_hours", "q25_hours", "q75_hours", "pct_before_icu"]] = pd.NA
t3.to_csv(out_dir / "time_to_icu_by_subtype.csv", index=False)

# Attrition, for the report
pd.DataFrame([
    {"step": "adults with index ICU stay in 2024", "n": len(cohort)},
    {"step": "met any ARF criteria in window", "n": len(classified)},
    {"step": "new IMV during index ICU stay", "n": len(imv)},
]).to_csv(out_dir / "attrition.csv", index=False)

print(classified["arf_subtype"].value_counts().to_string())
```

- [ ] **Step 6: Run the drivers against site data**

```bash
uv run python code/01_cohort_identification.py
uv run python code/02_project_quality_checks.py
uv run python code/03_outlier_handling.py
uv run python code/04_project_analysis.py
```

Expected: four CSVs in `output/final_no_phi/`. Confirm subtype counts sum to the total and no `delta_hours` exceeds +24. If `pct_with_abg` is very low in `quality_checks.csv`, stop and report it — the hypercapnic counts would be supply-limited rather than clinically meaningful.

- [ ] **Step 7: Commit**

```bash
git add code/icu_strain/pipeline.py code/0[1-4]_*.py tests/test_pipeline_smoke.py
git commit -m "feat: add pipeline drivers for cohort, QC, outliers, and analysis"
```

---

### Task 7: HTML report

**Files:**
- Create: `code/icu_strain/report.py`
- Create: `code/05_generate_report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: CSVs in `output/final_no_phi/`; `CRITERIA_TEXT` and `SF_FALLBACK_CAVEAT` from `definitions`.
- Produces: `render_report(tables: dict[str, pd.DataFrame], charts: dict[str, str], meta: dict) -> str` returning a complete HTML document string. `charts` values are base64 PNG data URIs.

- [ ] **Step 1: Load the dataviz skill**

Run the `dataviz` skill before writing any chart code. It sets the palette, chart-form, and accessibility conventions for the three figures.

- [ ] **Step 2: Write the failing test**

Create `tests/test_report.py`:

```python
import pandas as pd
from icu_strain.report import render_report
from icu_strain import definitions as d


def _tables():
    return {
        "attrition": pd.DataFrame({"step": ["adults with ICU stay"], "n": [100]}),
        "monthly_imv": pd.DataFrame({"month": ["2024-01"], "n_new_imv": [42]}),
        "monthly_arf_subtypes": pd.DataFrame({
            "month": ["2024-01"], "hypoxemic": [30], "hypercapnic": [11], "mixed": [15], "total": [56],
        }),
        "time_to_icu_by_subtype": pd.DataFrame({
            "arf_subtype": ["hypoxemic"], "n": [30], "median_hours": [-2.5],
            "q25_hours": [-8.0], "q75_hours": [1.0], "pct_before_icu": [62.0],
        }),
        "quality_checks": pd.DataFrame({
            "month": ["2024-01"], "n_hospitalizations": [100],
            "pct_with_abg": [55.0], "pct_with_spo2": [99.0], "pct_with_fio2": [80.0],
        }),
        "outlier_summary": pd.DataFrame({
            "table": ["vitals"], "values_before": [100], "values_after": [98], "values_removed": [2],
        }),
    }


def test_report_is_a_complete_html_document():
    html = render_report(_tables(), charts={}, meta={"site": "ucmc", "year": 2024})
    assert html.strip().startswith("<!doctype html>")
    assert "</html>" in html


def test_report_states_these_are_not_ards():
    html = render_report(_tables(), charts={}, meta={"site": "ucmc", "year": 2024})
    assert "NOT ARDS" in html
    assert d.CRITERIA_TEXT["not_ards"][:40] in html


def test_report_embeds_the_implemented_thresholds():
    html = render_report(_tables(), charts={}, meta={"site": "ucmc", "year": 2024})
    for token in ("90", "60", "300", "315", "45", "7.35"):
        assert token in html


def test_report_carries_the_sf_fallback_caveat():
    html = render_report(_tables(), charts={}, meta={"site": "ucmc", "year": 2024})
    assert d.SF_FALLBACK_CAVEAT[:40] in html


def test_suppressed_cells_render_as_marker_not_blank():
    tables = _tables()
    tables["monthly_imv"] = pd.DataFrame({"month": ["2024-01"], "n_new_imv": [pd.NA]})
    html = render_report(tables, charts={}, meta={"site": "ucmc", "year": 2024})
    assert "&lt;10" in html or "<10" in html


def test_charts_are_embedded_inline_when_supplied():
    html = render_report(_tables(), charts={"imv": "data:image/png;base64,AAA"},
                         meta={"site": "ucmc", "year": 2024})
    assert "data:image/png;base64,AAA" in html
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'icu_strain.report'`

- [ ] **Step 4: Implement `code/icu_strain/report.py`**

Render a single self-contained document. Requirements the tests pin down:
- starts with `<!doctype html>`, closes `</html>`
- a Definitions section built from `CRITERIA_TEXT` (all four keys) and `SF_FALLBACK_CAVEAT`
- suppressed (`NA`) cells render as the literal `<10`, not blank, so a reader can tell suppression from missing data
- charts embedded from the `charts` dict as `<img src="{data_uri}">`
- CSS inline in a `<style>` block, with a `@media (prefers-color-scheme: dark)` variant
- section order: header, definitions, attrition, deliverable 1, deliverable 2, deliverable 3, data quality, limitations

Use `df.to_html(index=False, na_rep="<10", escape=False)` for tables, wrapped in a `div` with `overflow-x: auto`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: 6 passed

- [ ] **Step 6: Write `code/05_generate_report.py`**

Reads the six CSVs from `output/final_no_phi/`, builds three matplotlib figures (monthly new-IMV bars; monthly subtype stacked bars; signed-delta summary by subtype with a reference line at zero), encodes each as a base64 PNG data URI, calls `render_report`, and writes `output/final_no_phi/icu_strain_report.html`.

It must not read anything from `output/intermediate_phi/` or the site data directory. That restriction is what makes the report structurally PHI-safe.

- [ ] **Step 7: Generate and inspect the report**

```bash
uv run python code/05_generate_report.py
open output/final_no_phi/icu_strain_report.html
```

Walk every section against the spec's report contents list (items 1–8). Confirm the definitions shown match the constants in `definitions.py`.

- [ ] **Step 8: Commit**

```bash
git add code/icu_strain/report.py code/05_generate_report.py tests/test_report.py
git commit -m "feat: add self-contained HTML report generation"
```

---

### Task 8: End-to-end verification on demo data

**Files:**
- Create: `tests/test_end_to_end_demo.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything.
- Produces: no new interfaces.

- [ ] **Step 1: Write the end-to-end test**

Create `tests/test_end_to_end_demo.py`:

```python
"""Full pipeline against the bundled synthetic clif_demo dataset.

Guards against import errors, schema drift, and column-name typos that unit
tests with hand-built frames cannot catch.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "clif_demo"


@pytest.mark.skipif(not DEMO.exists(), reason="demo dataset not present")
def test_pipeline_runs_end_to_end_on_demo(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        f'{{"site_name": "demo", "data_directory": "{DEMO}", '
        f'"filetype": "parquet", "timezone": "US/Central", "output_directory": "./output"}}'
    )
    # Merge into os.environ rather than replacing it; a bare env dict drops
    # PATH and the virtualenv, and the subprocess fails for the wrong reason.
    env = {**os.environ, "CLIF_CONFIG_PATH": str(config)}
    for script in ["01_cohort_identification.py", "03_outlier_handling.py",
                   "04_project_analysis.py", "05_generate_report.py"]:
        r = subprocess.run(
            [sys.executable, f"code/{script}"], cwd=REPO, capture_output=True, text=True, env=env
        )
        assert r.returncode == 0, f"{script} failed:\n{r.stderr}"

    report = REPO / "output/final_no_phi/icu_strain_report.html"
    assert report.exists()
    assert "NOT ARDS" in report.read_text()


@pytest.mark.skipif(not DEMO.exists(), reason="demo dataset not present")
def test_subtype_counts_sum_to_total():
    df = pd.read_csv(REPO / "output/final_no_phi/monthly_arf_subtypes.csv")
    unsuppressed = df.dropna(subset=["hypoxemic", "hypercapnic", "mixed", "total"])
    sums = unsuppressed[["hypoxemic", "hypercapnic", "mixed"]].sum(axis=1)
    assert (sums == unsuppressed["total"]).all()


@pytest.mark.skipif(not DEMO.exists(), reason="demo dataset not present")
def test_no_delta_exceeds_the_declared_window():
    df = pd.read_csv(REPO / "output/final_no_phi/time_to_icu_by_subtype.csv")
    assert (df["q75_hours"] <= 24.0).all()
```

- [ ] **Step 2: Add config-path override to the drivers**

Each driver currently hardcodes `config/config.json`. Change all five to:

```python
import os
config_path = Path(os.environ.get("CLIF_CONFIG_PATH", "config/config.json"))
config = json.loads(config_path.read_text())
```

This is what lets the end-to-end test point at the demo dataset without mutating the site config.

- [ ] **Step 3: Run the end-to-end test**

Run: `uv run pytest tests/test_end_to_end_demo.py -v`
Expected: 3 passed. If the demo dataset has too few ICU stays in 2024 to produce output, adjust `ANALYSIS_YEAR` via an env override in the test rather than weakening the assertions.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 5: PHI audit before anything is shared**

Run the `clif-phi-auditor` agent over `output/final_no_phi/`. Confirm no `patient_id`, no `hospitalization_id`, no row-level records, and no unsuppressed cell below `MIN_CELL_SIZE`.

- [ ] **Step 6: Update `README.md`**

Replace the template placeholders: objective, the five required CLIF tables with rationale, cohort definition (adults, index ICU stay, 2024), the three deliverables, and the run order `01 → 02 → 03 → 04 → 05`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_end_to_end_demo.py code/0[1-5]_*.py README.md
git commit -m "test: add end-to-end demo verification and document the pipeline"
```

---

## Self-Review Notes

**Spec coverage.** Every spec section maps to a task: cohort → 3/6, quality checks → 6, outlier handling → 6, ARF criteria → 4, deliverables 1–3 → 5/6, HTML report → 7, definitions constants → 1, data security → 5 (suppression) and 8 (audit), testing → 2–8, FiO₂ normalization → 2, venous-gas trap → 4, NULL device trap → 6.

**Deviation from spec.** The spec listed `code/01`…`05` as the file structure. The plan adds the `code/icu_strain/` package because leading-digit filenames cannot be imported, so none of the spec's testing requirements would be satisfiable otherwise. The numbered scripts remain as drivers, preserving the template's convention.

**Deferred to implementation.** Task 7 Steps 4 and 6 describe the report and chart code by requirement rather than pinning full source, because the `dataviz` skill loads at Step 1 and sets palette and chart-form conventions that should govern that code. The tests in Step 2 pin the contract.
