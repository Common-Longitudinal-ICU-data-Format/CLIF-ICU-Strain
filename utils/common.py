"""Constants, config, per-site output paths, and small-cell suppression used by every script.

Every tunable analysis choice lives here so a reviewer finds them in one place.
"""

import json
import os
from pathlib import Path

import pandas as pd

# --- Cohort -------------------------------------------------------------------
MIN_AGE = 18
STUDY_START = "2019-01-01"           # ICU admissions in [start, end); the census burn-in trims the first weeks
STUDY_END = "2025-01-01"
STITCH_HOURS = 6                     # a hospitalization starting this soon after the previous discharge is the same encounter (clifpy default)
DEV_SITES = {"demo", "mimic"}        # code-test runs, never pooled: MIMIC shifts dates per patient, so its census is not real

# --- Census and strain ------------------------------------------------------------
ICU_LOCATION = "icu"
ICU_BLOCK_GAP_HOURS = 1              # ADT rows closer than this are one stay (ICU stay, or bed move within a unit)
TRANSPARENT_LOCATIONS = {"procedural", "radiology", "dialysis"}   # the ICU bed is held while the patient is there (D5)
ROUND_TRIP_HOURS = 12                # back in the same unit within this via transparent locations only: same stay (D5)
DAY_START_HOUR = 7                   # 7am-to-7am strain day (typical rounding time)
CENSUS_BURN_IN_DAYS = 60             # census undercounts until stays begun before the extract have left
COVID_SURGE = ("2020-03-01", "2022-03-01")   # [start, end); for Corbett's capacity variant and the covariate
MIN_UNIT_BEDS = int(os.environ.get("CLIF_MIN_UNIT_BEDS", 4))  # D6: smaller unit-years get no capacity; env lets the demo run
MAX_UNIT_BEDS, MAX_UNITS = 60, 30    # above these, location_name is probably a floor, not a unit
MAX_UNNAMED_ICU_PCT = 2              # ICU rows without a location_name count in the hospital census only; warn above this share
MIN_END_CENSUS_PCT = 85              # last 60 days' census vs the 60 before; lower suggests a discharged-only extract
HOSPITAL = ["hospital_id"]
UNIT = ["hospital_id", "location_name"]

# --- Covariates and models ------------------------------------------------------------
COVID_ICD10_PREFIX = "U071"          # U07.1, matched after stripping dots
ICU_DAY_CAP = 7                      # later ICU days are sparse; pool them
SPLINE_DF = 4
OCCUPANCY_CENTER = 0.9               # interaction model: admissions effect when the unit is 90% full at the landmark
EXPOSURE_HOURS = 24                  # landmark s: exposure window [s, s+24h); follow-up starts when it ends
OUTCOME_DAYS = 7                     # HDHD-7: in-hospital death or hospice discharge in [s+24h, s+8d)
OUTCOME_DAYS_LONG = 30               # HDHD-30, a key secondary
PLACEBO_OFFSET_DAYS = 8              # placebos: admissions in [s+8d, s+9d) (after the outcome window) and [s-8d, s-7d)
MIN_POA_SHARE = 0.2                  # use present-on-admission codes only if at least this share of codes is flagged POA

# --- v4: occupancy, demand-side exposure, process outcomes ------------------------------
OCCUPANCY_TIPPING = 0.925            # Kuntz 2015: the mortality slope turns positive above this share of capacity
ED_LOCATION = "ed"                   # ADT rows whose arrivals are demand that bed-freeing cannot cause
READMIT_HOURS = 72                   # ICU readmission after an ICU discharge inside the exposure window
LIMITATION_HOURS = 48                # new DNR/DNI (or any other limitation) order within this of the landmark
AFTER_HOURS = (18, 6)                # ICU discharge from 18:00 to 06:00 local
NON_LIMITATION_CODES = {"full", "presume full", "other"}   # any other code_status_category is a limitation (dnr, dnar, udnr, dnr/dni, dnar/dni, dni_only, and)
MIN_GATE_EVENTS = 11                 # a unit-year needs this many discharges to be gated (A1)

# --- Disclosure ---------------------------------------------------------------------
MIN_CELL_SIZE = 11                   # suppress any shared count below this


def load_config() -> dict:
    """config/config.json, or the file named by CLIF_CONFIG_PATH (the demo test uses this)."""
    return json.loads(Path(os.environ.get("CLIF_CONFIG_PATH", "config/config.json")).read_text())


def study_window() -> tuple:
    """(start, end) for ICU admissions: env, else the config's study_start/study_end (date-shifted data: demo, MIMIC)."""
    cfg = load_config()
    return (os.environ.get("CLIF_STUDY_START", cfg.get("study_start", STUDY_START)),
            os.environ.get("CLIF_STUDY_END", cfg.get("study_end", STUDY_END)))


def output_root() -> Path:
    return Path(os.environ.get("CLIF_OUTPUT_ROOT", "output"))


def phi_dir(site: str) -> Path:
    """Row-level, PHI-bearing intermediates for one site. Never shared."""
    path = output_root() / "intermediate_phi" / site
    path.mkdir(parents=True, exist_ok=True)
    return path


def final_dir(site: str) -> Path:
    """Disclosure-controlled aggregates: the folder a site uploads to Box."""
    path = output_root() / "final_no_phi" / f"{site}_upload_to_box"
    path.mkdir(parents=True, exist_ok=True)
    return path


def suppress_small_cells(df: pd.DataFrame, count_cols: list, min_n: int = MIN_CELL_SIZE) -> pd.DataFrame:
    """Mask counts in (0, min_n) as NA. Exact zeros are safe to publish.

    Integer columns go to nullable "Int64" first so a published 40 stays "40"
    in the CSV rather than "40.0"; float columns are masked as they are.
    """
    out = df.copy()
    for c in count_cols:
        col = out[c].astype("Int64") if pd.api.types.is_integer_dtype(out[c]) else out[c]
        out[c] = col.mask((col > 0) & (col < min_n), pd.NA)
    return out


def suppress_small_rows(df: pd.DataFrame, n_col: str, value_cols: list, min_n: int = MIN_CELL_SIZE) -> pd.DataFrame:
    """Mask n and every value column on rows where 0 < n < min_n (a median next to n=3 leaks)."""
    out = df.copy()
    small = (out[n_col] > 0) & (out[n_col] < min_n)
    for c in [n_col, *value_cols]:
        col = out[c].astype("Int64") if pd.api.types.is_integer_dtype(out[c]) else out[c]
        out[c] = col.mask(small, pd.NA)
    return out
