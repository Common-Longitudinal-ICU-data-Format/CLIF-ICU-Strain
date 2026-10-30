"""Shared functions for the ICU strain pipeline: constants, output paths, disclosure
control, ICU stay blocks, rolling census, daily 7am strain, and capacity.

Pure pandas, no I/O, so every function is unit-tested on hand-built frames in
tests/test_strain.py. The numbered scripts in code/ do the loading and writing.
"""

import os
from pathlib import Path

import pandas as pd

# --- Constants --------------------------------------------------------------
MIN_AGE = 18
ICU_LOCATION = "icu"
ICU_BLOCK_GAP_HOURS = 1              # ADT rows closer than this are one ICU stay
MIN_CELL_SIZE = 11                   # suppress any shared count below this (n >= 11 rule)

STUDY_START = "2019-01-01"           # first ICU admission included
STUDY_END = "2026-01-01"             # exclusive
DAY_START_HOUR = 7                   # 7am-to-7am strain day (typical rounding time)
CENSUS_BURN_IN_DAYS = 60             # census undercounts until pre-extract stays have left
CAPACITY_QUANTILE = 0.95
COVID_SURGE = ("2020-03-01", "2022-03-01")   # [start, end); confirm with PIs
PRIMARY_EXPOSURE = "census_norm_p95_nonsurge"
COVID_ICD10_PREFIX = "U071"          # U07.1, matched after stripping dots


# --- Output paths -------------------------------------------------------------
def output_root() -> Path:
    """`CLIF_OUTPUT_ROOT` lets the demo test write to a scratch root instead of output/."""
    return Path(os.environ.get("CLIF_OUTPUT_ROOT", "output"))


def phi_dir(site: str) -> Path:
    """Row-level, PHI-bearing intermediates for one site. Never shared."""
    return output_root() / "intermediate_phi" / site


def final_dir(site: str) -> Path:
    """Disclosure-controlled aggregates: the folder a site uploads to Box."""
    return output_root() / "final_no_phi" / f"{site}_upload_to_box"


# --- Disclosure control ---------------------------------------------------------
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
    """Mask n and every value column on rows where 0 < n < min_n.

    For one-row-per-group summaries, where a median published next to n=3 leaks
    more than n alone would.
    """
    out = df.copy()
    small = (out[n_col] > 0) & (out[n_col] < min_n)
    for c in [n_col, *value_cols]:
        col = out[c].astype("Int64") if pd.api.types.is_integer_dtype(out[c]) else out[c]
        out[c] = col.mask(small, pd.NA)
    return out


# --- ICU stays -------------------------------------------------------------------
def collapse_icu_blocks(adt: pd.DataFrame, gap_hours: float = ICU_BLOCK_GAP_HOURS) -> pd.DataFrame:
    """Merge consecutive ICU ADT rows into contiguous ICU stays, one row each.

    A new block starts when the previous location was not ICU, or when the gap
    from the previous ICU row's out_dttm exceeds gap_hours. Non-ICU rows stay in
    the sort so an intervening ward stay breaks the chain.
    """
    df = adt.sort_values(["hospitalization_id", "in_dttm"]).copy()
    df["_is_icu"] = df["location_category"].str.lower() == ICU_LOCATION

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
            "n_adt_rows_merged": pd.Series(dtype="int64"),
            "block_index": pd.Series(dtype="int64"),
        })

    aggs = {
        "icu_admit_dttm": ("in_dttm", "min"),
        "icu_discharge_dttm": ("out_dttm", "max"),
        "n_adt_rows_merged": ("in_dttm", "size"),
    }
    if "hospital_id" in icu:
        aggs["hospital_id"] = ("hospital_id", "first")
    out = icu.groupby(["hospitalization_id", "_block"], as_index=False).agg(**aggs)
    out["block_index"] = out.groupby("hospitalization_id")["icu_admit_dttm"].rank("dense").astype(int) - 1
    return out.drop(columns=["_block"])


# --- Census and strain --------------------------------------------------------------
def icu_intervals(adt: pd.DataFrame, adult_ids) -> pd.DataFrame:
    """Adult ICU ADT rows as non-overlapping [in_dttm, out_dttm) intervals.

    A patient holds one bed at a time, so each row is clipped at the start of
    that hospitalization's next ICU row: overlapping ADT rows would otherwise
    count one patient twice.
    """
    icu = adt[(adt["location_category"].str.lower() == ICU_LOCATION) & adt["hospitalization_id"].isin(adult_ids)]
    icu = icu.dropna(subset=["in_dttm", "out_dttm"]).sort_values(["hospitalization_id", "in_dttm"])
    next_in = icu.groupby("hospitalization_id")["in_dttm"].shift(-1)
    icu = icu.assign(out_dttm=icu["out_dttm"].where(next_in.isna() | (icu["out_dttm"] <= next_in), next_in))
    cols = ["hospitalization_id", "hospital_id", "location_type", "location_name", "in_dttm", "out_dttm"]
    return icu.loc[icu["out_dttm"] > icu["in_dttm"], cols].reset_index(drop=True)


def rolling_census(intervals: pd.DataFrame, by=("hospital_id",)) -> pd.DataFrame:
    """Census after every ICU entry or exit: one row per (*by, time)."""
    by = list(by)
    events = pd.concat([
        intervals[by + ["in_dttm"]].rename(columns={"in_dttm": "time"}).assign(delta=1),
        intervals[by + ["out_dttm"]].rename(columns={"out_dttm": "time"}).assign(delta=-1),
    ])
    # Net the deltas per instant so a same-minute bed swap never reads as two patients.
    events = events.groupby(by + ["time"], as_index=False)["delta"].sum().sort_values(by + ["time"])
    events["census"] = events.groupby(by)["delta"].cumsum()
    return events.drop(columns="delta").reset_index(drop=True)


def _seven_ams(first: pd.Timestamp, last: pd.Timestamp, hour: int) -> pd.DatetimeIndex:
    """Every local `hour`:00 from first's date to last. Built on wall-clock dates, then localized, so DST cannot shift it."""
    tz = first.tz
    wall_first, wall_last = (t.tz_localize(None) if tz else t for t in (first, last))
    days = pd.date_range(wall_first.normalize(), wall_last, freq="D") + pd.Timedelta(hours=hour)
    days = days[days <= wall_last]
    return days.tz_localize(tz) if tz else days


def daily_strain(census: pd.DataFrame, admit_times: pd.DataFrame, hour: int = DAY_START_HOUR,
                 burn_in_days: int = CENSUS_BURN_IN_DAYS) -> pd.DataFrame:
    """One row per hospital per local 7am: the census then, and ICU admissions in [previous 7am, this 7am).

    `census` is rolling_census(by=["hospital_id"]); `admit_times` has one row per
    ICU stay (hospital_id, icu_admit_dttm). Days run from the first census event
    to the last ICU admission; the first `burn_in_days` are then dropped.
    """
    last_admit = admit_times.groupby("hospital_id")["icu_admit_dttm"].max()
    snaps = pd.concat([
        pd.DataFrame({"hospital_id": h, "day_start": _seven_ams(g["time"].min(), last_admit[h], hour)})
        for h, g in census.groupby("hospital_id") if h in last_admit
    ]).sort_values("day_start")
    # merge_asof's `by` key must match dtypes exactly; clifpy loads hospital_id as nullable "string".
    snaps["hospital_id"] = snaps["hospital_id"].astype(census["hospital_id"].dtype)
    out = pd.merge_asof(snaps, census.sort_values("time"), left_on="day_start", right_on="time", by="hospital_id")

    nxt = pd.merge_asof(
        admit_times.sort_values("icu_admit_dttm"), snaps.rename(columns={"day_start": "next_7am"}),
        left_on="icu_admit_dttm", right_on="next_7am", by="hospital_id",
        direction="forward", allow_exact_matches=False,
    )
    admits = nxt.groupby(["hospital_id", "next_7am"]).size().rename("admits_24h")
    out = out.merge(admits, left_on=["hospital_id", "day_start"], right_index=True, how="left")
    out["admits_24h"] = out["admits_24h"].fillna(0).astype(int)
    out = out.sort_values(["hospital_id", "day_start"])
    out["day_end"] = out.groupby("hospital_id")["day_start"].shift(-1)

    first = out.groupby("hospital_id")["day_start"].transform("min")
    out = out[out["day_start"] >= first + pd.Timedelta(days=burn_in_days)]
    return out[["hospital_id", "day_start", "day_end", "census", "admits_24h"]].reset_index(drop=True)


def capacities(census: pd.DataFrame, daily: pd.DataFrame, q: float = CAPACITY_QUANTILE,
               surge=COVID_SURGE) -> pd.DataFrame:
    """ICU capacity per hospital: all-time max census, and the q-th percentile of the 7am census with and without the COVID surge."""
    tz = daily["day_start"].dt.tz
    start, end = (pd.Timestamp(s, tz=tz) for s in surge)
    nonsurge = daily[(daily["day_start"] < start) | (daily["day_start"] >= end)]
    caps = pd.DataFrame({
        "capacity_max": census.groupby("hospital_id")["census"].max(),
        "capacity_p95": daily.groupby("hospital_id")["census"].quantile(q),
        "capacity_p95_nonsurge": nonsurge.groupby("hospital_id")["census"].quantile(q),
        "n_days": daily.groupby("hospital_id").size(),
    })
    return caps.where(caps > 0).rename_axis("hospital_id").reset_index()  # 0 -> NA: never divide by zero


def add_normalized(daily: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    """Census over each capacity definition; admissions over the primary one."""
    out = daily.merge(caps, on="hospital_id", how="left")
    for kind in ["max", "p95", "p95_nonsurge"]:
        out[f"census_norm_{kind}"] = out["census"] / out[f"capacity_{kind}"]
    out["admits_norm"] = out["admits_24h"] / out["capacity_p95_nonsurge"]
    return out
