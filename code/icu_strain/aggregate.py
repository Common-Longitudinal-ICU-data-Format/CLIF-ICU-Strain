"""Monthly rollups and disclosure control."""

import pandas as pd

from . import definitions as d

SUBTYPES = ["hypoxemic", "hypercapnic", "mixed"]


def suppress_small_cells(df: pd.DataFrame, count_cols: list, min_n: int = d.MIN_CELL_SIZE) -> pd.DataFrame:
    """Mask counts in (0, min_n) as NA. Exact zeros are safe to publish.

    Integer columns are cast to the nullable "Int64" dtype before masking so
    integer semantics survive both the NA mask and CSV serialization (a plain
    `.mask()` on int64 silently upcasts the whole column to float64, so a
    published count of 40 would round-trip as "40.0"). Columns that are
    already float (e.g. median_hours, pct_before_icu) are left as float.
    """
    out = df.copy()
    for c in count_cols:
        col = out[c].astype("Int64") if pd.api.types.is_integer_dtype(out[c]) else out[c]
        out[c] = col.mask((col > 0) & (col < min_n), pd.NA)
    return out


def suppress_small_rows(
    df: pd.DataFrame, n_col: str, value_cols: list, min_n: int = d.MIN_CELL_SIZE
) -> pd.DataFrame:
    """Mask an entire row (n plus every value column) when 0 < n < min_n.

    Unlike suppress_small_cells (which masks each count column
    independently, cell by cell), this is for one-row-per-group summary
    tables where a small n makes every other statistic on that row
    identifying too -- publishing a median/quartile time-to-ICU for a
    subtype with n=3 leaks more than n alone would, so the whole row goes
    dark together, not just the count.

    Exact n=0 rows are left alone (nothing to disclose), matching
    suppress_small_cells' zero-is-safe convention.

    Applies the same per-column dtype discipline as suppress_small_cells:
    integer columns are cast to nullable "Int64" before masking, so integer
    semantics survive both the NA mask and CSV serialization even when the
    mask matches zero rows (a plain `.loc[mask, cols] = pd.NA` silently
    upcasts int64 -> float64 unconditionally, so a published n of 40 would
    round-trip as "40.0" even when nothing was actually suppressed).
    """
    out = df.copy()
    mask = (out[n_col] > 0) & (out[n_col] < min_n)
    for c in [n_col, *value_cols]:
        col = out[c].astype("Int64") if pd.api.types.is_integer_dtype(out[c]) else out[c]
        out[c] = col.mask(mask, pd.NA)
    return out


def monthly_imv_counts(imv_starts: pd.DataFrame) -> pd.DataFrame:
    """Count hospitalizations by month of first IMV initiation.

    Reduces to each hospitalization's earliest imv_start_dttm before binning,
    so a hospitalization with multiple IMV-start rows spanning month
    boundaries is counted exactly once, in the month it first started IMV --
    regardless of whether the caller already de-duplicated upstream.
    """
    df = imv_starts.copy()
    df["imv_start_dttm"] = pd.to_datetime(df["imv_start_dttm"])
    first = df.groupby("hospitalization_id", as_index=False)["imv_start_dttm"].min()
    first["month"] = first["imv_start_dttm"].dt.to_period("M").astype(str)
    return (
        first.groupby("month", as_index=False)["hospitalization_id"]
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
    df["icu_admit_dttm"] = pd.to_datetime(df["icu_admit_dttm"])
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
    df["icu_admit_dttm"] = pd.to_datetime(df["icu_admit_dttm"])
    df["first_arf_dttm"] = pd.to_datetime(df["first_arf_dttm"])
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
