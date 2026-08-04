"""Per-month data-availability and duplicate-key quality metrics.

Extracted from the 02 driver so the disclosure-control step (small-cell
suppression) is unit-testable the same way 04_project_analysis.py's is,
rather than living only inside a script.
"""

import pandas as pd

from . import definitions as d
from .aggregate import suppress_small_rows

ARTERIAL = {d.ABG_PO2, d.ABG_PCO2, d.ABG_PH}

# Per-month columns derived from a single month's cohort count -- these are
# exactly what suppress_small_rows must mask together whenever that month's
# n_hospitalizations falls below MIN_CELL_SIZE. device_null_rate_cohort and
# the two dup-key rates are NOT in this list: they're single overall values
# computed over the whole (already cohort-filtered) table, not conditioned on
# any one month's small n, so masking them per-month would hide information
# without protecting anything.
PER_MONTH_VALUE_COLS = ["pct_with_abg", "pct_with_spo2", "pct_with_fio2"]


def _in_window(df: pd.DataFrame, dttm_col: str, cohort: pd.DataFrame, ids_col: str = "hospitalization_id"):
    m = df.merge(cohort[[ids_col, "window_start", "window_end", "month"]], on=ids_col)
    return m[(m[dttm_col] >= m["window_start"]) & (m[dttm_col] <= m["window_end"])]


def monthly_quality_checks(
    cohort: pd.DataFrame, labs: pd.DataFrame, vitals: pd.DataFrame, resp: pd.DataFrame
) -> pd.DataFrame:
    """Per-month ABG/SpO2/FiO2 availability, plus overall NULL-device and dup-key rates.

    `cohort`, `labs`, `vitals`, `resp` are expected already filtered to this
    project's cohort hospitalization_ids (the driver pushes that filter into
    the DuckDB scan). Small-cell suppression is applied before returning:
    n_hospitalizations and the three per-month availability percentages are
    masked together for any month with 0 < n_hospitalizations < MIN_CELL_SIZE
    -- publishing a percentage while hiding the count it was computed over
    still discloses the count, so they must be suppressed as one unit (same
    rule 04_project_analysis.py applies to its own monthly tables).
    """
    cohort = cohort.copy()
    cohort["month"] = cohort["icu_admit_dttm"].dt.to_period("M").astype(str)

    abg_ids = _in_window(labs[labs["lab_category"].isin(ARTERIAL)], "lab_result_dttm", cohort)
    spo2_ids = _in_window(vitals[vitals["vital_category"] == d.SPO2_VITAL], "recorded_dttm", cohort)
    fio2_ids = _in_window(resp[resp["fio2_set"].notna()], "recorded_dttm", cohort)

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

    # Named "_cohort", not "_raw": this is the null rate over the
    # cohort-filtered respiratory_support scan (the driver's `filters`
    # push-down), not the raw site-wide table. A consortium consumer pooling
    # this column across sites would otherwise compare a cohort-conditional
    # rate here against another site's site-wide rate under the same column
    # name.
    qc["device_null_rate_cohort"] = 100.0 * resp["device_category"].isna().mean()

    # Duplicate-key quality signal, overall (not per-month): what fraction of
    # cohort-scoped arterial lab rows (po2_arterial/pco2_arterial/ph_arterial)
    # share a (hospitalization_id, lab_result_dttm, lab_category) key with
    # another row. Reported as two SEPARATE rates because they are different
    # problems: an identical-value duplicate is a benign redundant record, but
    # a differing-value duplicate is a genuine conflict that
    # physiology.evaluate_hypercapnia must arbitrarily-but-deterministically
    # resolve (see its merge_asof tie-break). This rate determines whether
    # this site's hypercapnic/mixed counts are comparable to a site without
    # the same duplication -- collapsing the two into one number would hide
    # which problem (if either) a given site actually has.
    dup_key = ["hospitalization_id", "lab_result_dttm", "lab_category"]
    arterial_labs = labs[labs["lab_category"].isin(ARTERIAL)]
    n_arterial = len(arterial_labs)
    if n_arterial:
        group_size = arterial_labs.groupby(dup_key)["lab_value_numeric"].transform("size")
        group_nunique = arterial_labs.groupby(dup_key)["lab_value_numeric"].transform("nunique")
        is_conflicting_dup = (group_size > 1) & (group_nunique > 1)
        is_identical_dup = (group_size > 1) & (group_nunique == 1)
        pct_conflicting = 100.0 * is_conflicting_dup.sum() / n_arterial
        pct_identical = 100.0 * is_identical_dup.sum() / n_arterial
    else:
        pct_conflicting = pct_identical = 0.0
    qc["pct_arterial_dup_key_conflicting_overall"] = pct_conflicting
    qc["pct_arterial_dup_key_identical_overall"] = pct_identical

    return suppress_small_rows(qc, "n_hospitalizations", PER_MONTH_VALUE_COLS)
