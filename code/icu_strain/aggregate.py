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
