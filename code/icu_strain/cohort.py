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
        return pd.DataFrame({
            "hospitalization_id": pd.Series(dtype="object"),
            "icu_admit_dttm": pd.Series(dtype="datetime64[ns]"),
            "icu_discharge_dttm": pd.Series(dtype="datetime64[ns]"),
            "block_index": pd.Series(dtype="int64"),
            "n_adt_rows_merged": pd.Series(dtype="int64"),
        })

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
