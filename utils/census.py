"""ICU stays, rolling census, daily 7am strain, and capacity, at hospital or unit level.

Pure pandas; every function takes `by` = common.HOSPITAL or common.UNIT so the same
code builds hospital-wide and per-unit (location_name) tables.
"""

import numpy as np
import pandas as pd

from .common import (CENSUS_BURN_IN_DAYS, COVID_SURGE, DAY_START_HOUR, HOSPITAL, ICU_BLOCK_GAP_HOURS, ICU_LOCATION,
                     MIN_UNIT_BEDS, ROUND_TRIP_HOURS, TRANSPARENT_LOCATIONS)


def is_icu(adt: pd.DataFrame) -> pd.Series:
    return adt["location_category"].str.lower().eq(ICU_LOCATION).fillna(False).astype(bool)


def merge_round_trips(adt: pd.DataFrame, hours: float = ROUND_TRIP_HOURS, transparent=TRANSPARENT_LOCATIONS) -> pd.DataFrame:
    """Fold OR / CT / dialysis round trips into the ICU row they interrupt (design v3, D5).

    An ICU row that begins within `hours` of the previous ICU row of the same hospitalization,
    in the same unit, with only transparent locations charted in between, continues that row:
    the bed was held, so the return is neither an admission nor a new ICU stay. Transparent
    rows are dropped; every other row comes back unchanged.
    """
    df = adt.sort_values(["hospitalization_id", "in_dttm"]).reset_index(drop=True)
    icu = is_icu(df)
    clear = df["location_category"].str.lower().isin(transparent).fillna(False).astype(bool)
    opaque = (~icu & ~clear).groupby(df["hospitalization_id"]).cumsum()  # ward / ED / other rows seen so far

    i = df[icu].assign(_opaque=opaque[icu])
    g = i.groupby("hospitalization_id")
    bridged = ((i["in_dttm"] - g["out_dttm"].shift()) <= pd.Timedelta(hours=hours)) \
        & i["location_name"].eq(g["location_name"].shift()) & i["hospital_id"].eq(g["hospital_id"].shift()).fillna(False).astype(bool) \
        & i["_opaque"].eq(g["_opaque"].shift())
    aggs = {col: "first" for col in adt.columns} | {"in_dttm": "min", "out_dttm": "max"}
    merged = i.groupby((~bridged).cumsum()).agg(aggs)
    return pd.concat([df[~icu & ~clear], merged]).sort_values(["hospitalization_id", "in_dttm"]).reset_index(drop=True)


def collapse_icu_blocks(adt: pd.DataFrame, gap_hours: float = ICU_BLOCK_GAP_HOURS) -> pd.DataFrame:
    """Merge consecutive ICU ADT rows into contiguous ICU stays (any unit), one row each.

    A new block starts when the previous location was not ICU, when the gap
    from the previous ICU row's out_dttm exceeds gap_hours, or when the hospital
    changes (a stitched encounter can span hospitals). Non-ICU rows stay in the
    sort so an intervening ward stay breaks the chain. A block with any row
    lacking out_dttm has no known end (NaT).
    """
    df = adt.sort_values(["hospitalization_id", "in_dttm"]).copy()
    df["_is_icu"] = is_icu(df)

    prev_out = df.groupby("hospitalization_id")["out_dttm"].shift()
    prev_icu = df.groupby("hospitalization_id")["_is_icu"].shift(fill_value=False)
    prev_hospital = df.groupby("hospitalization_id")["hospital_id"].shift()
    gap = (df["in_dttm"] - prev_out).dt.total_seconds() / 3600.0

    new_hospital = df["hospital_id"].ne(prev_hospital).fillna(True).astype(bool)  # clifpy strings compare to pyarrow bools
    starts_block = (~prev_icu) | (gap > gap_hours) | gap.isna() | new_hospital
    df["_block"] = (starts_block & df["_is_icu"]).groupby(df["hospitalization_id"]).cumsum()

    icu = df[df["_is_icu"]].assign(_open=lambda d: d["out_dttm"].isna())
    out = icu.groupby(["hospitalization_id", "_block"], as_index=False).agg(
        hospital_id=("hospital_id", "first"),
        icu_admit_dttm=("in_dttm", "min"),
        icu_discharge_dttm=("out_dttm", "max"),
        _open=("_open", "any"),
    )
    out.loc[out["_open"], "icu_discharge_dttm"] = pd.NaT  # max() skips a NaT: the stay would end at the previous row's exit
    out["block_index"] = out.groupby("hospitalization_id")["icu_admit_dttm"].rank("dense").astype(int) - 1
    return out.drop(columns=["_block", "_open"])


def icu_intervals(adt: pd.DataFrame, adult_ids) -> pd.DataFrame:
    """Adult ICU ADT rows as non-overlapping [in_dttm, out_dttm) intervals.

    A patient holds one bed at a time, so each row is clipped at the start of
    that hospitalization's next ICU row: overlapping ADT rows would otherwise
    count one patient twice. Rows without a location_name are kept (they hold
    a bed in the hospital census); drop them before any unit-level use.
    """
    icu = adt[is_icu(adt) & adt["hospitalization_id"].isin(adult_ids)]
    icu = icu.dropna(subset=["in_dttm", "out_dttm"]).sort_values(["hospitalization_id", "in_dttm"])
    next_in = icu.groupby("hospitalization_id")["in_dttm"].shift(-1)
    icu = icu.assign(out_dttm=icu["out_dttm"].where(next_in.isna() | (icu["out_dttm"] <= next_in), next_in))
    cols = ["hospitalization_id", "hospital_id", "location_type", "location_name", "in_dttm", "out_dttm"]
    return icu.loc[icu["out_dttm"] > icu["in_dttm"], cols].reset_index(drop=True)


def unit_stays(intervals: pd.DataFrame, gap_hours: float = ICU_BLOCK_GAP_HOURS) -> pd.DataFrame:
    """One row per stay in a unit (location_name): an admission to that unit.

    Consecutive rows in the same unit within gap_hours (bed moves) are one stay.
    `from_icu` marks a stay that began within gap_hours of leaving another ICU unit.
    """
    iv = intervals.sort_values(["hospitalization_id", "in_dttm"])
    g = iv.groupby("hospitalization_id")
    near = (iv["in_dttm"] - g["out_dttm"].shift()) <= pd.Timedelta(hours=gap_hours)
    same_unit = (iv["location_name"].eq(g["location_name"].shift()) & iv["hospital_id"].eq(g["hospital_id"].shift())).fillna(False).astype(bool)
    new = ~(near & same_unit)
    iv = iv.assign(from_icu=near & new, _stay=new.cumsum())
    return iv.groupby("_stay").agg(
        hospitalization_id=("hospitalization_id", "first"),
        hospital_id=("hospital_id", "first"),
        location_type=("location_type", "first"),
        location_name=("location_name", "first"),
        in_dttm=("in_dttm", "min"),
        out_dttm=("out_dttm", "max"),
        from_icu=("from_icu", "first"),
    ).reset_index(drop=True)


def rolling_census(intervals: pd.DataFrame, by=HOSPITAL) -> pd.DataFrame:
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


def daily_strain(census: pd.DataFrame, admits: pd.DataFrame, by=HOSPITAL, hour: int = DAY_START_HOUR,
                 burn_in_days: int = CENSUS_BURN_IN_DAYS) -> pd.DataFrame:
    """One row per group per local 7am: the census then, and admissions in [previous 7am, this 7am).

    `census` is rolling_census(by=by). `admits` has the `by` columns, `in_dttm`, and
    optionally `from_icu`, one row per admission. Days run from the group's first
    census event to the last admission anywhere (the data end; a far-future out_dttm
    cannot stretch it); the first `burn_in_days` are dropped.
    """
    by = list(by)
    last = admits["in_dttm"].max()
    span = census.groupby(by)["time"].min().rename("first").reset_index()
    span["day_start"] = [_seven_ams(f, last, hour) for f in span["first"]]
    snaps = span.drop(columns="first").explode("day_start").dropna(subset=["day_start"])
    snaps["day_start"] = snaps["day_start"].astype(census["time"].dtype)
    snaps = snaps.sort_values("day_start")
    out = pd.merge_asof(snaps, census.sort_values("time"), left_on="day_start", right_on="time", by=by)

    nxt = pd.merge_asof(admits.sort_values("in_dttm"), snaps.rename(columns={"day_start": "next_7am"}),
                        left_on="in_dttm", right_on="next_7am", by=by,
                        direction="forward", allow_exact_matches=False)
    aggs = {"admits_24h": ("in_dttm", "size")}
    if "from_icu" in nxt:
        aggs["transfers_24h"] = ("from_icu", "sum")
    counts = nxt.groupby(by + ["next_7am"]).agg(**aggs)
    out = out.merge(counts, left_on=by + ["day_start"], right_index=True, how="left")
    out[list(aggs)] = out[list(aggs)].fillna(0).astype(int)

    out = out.sort_values(by + ["day_start"])
    g = out.groupby(by)["day_start"]
    out["day_end"], out["day_prev"] = g.shift(-1), g.shift(1)
    out = out[out["day_start"] >= g.transform("min") + pd.Timedelta(days=burn_in_days)]
    return out[by + ["day_start", "day_end", "day_prev", "census", *aggs]].reset_index(drop=True)


def capacities(census: pd.DataFrame, daily: pd.DataFrame, by=HOSPITAL, surge=COVID_SURGE,
               min_beds: int = MIN_UNIT_BEDS) -> pd.DataFrame:
    """Capacity per group and calendar year (design v3, D6): 99th percentile of the 7am census over
    days with anyone present (primary), the max census that year, and Corbett's 95th percentile
    outside the COVID surge. A group-year under `min_beds` gets NA capacity, and so no strain:
    a one-bed "unit" is a room, and dividing by it means nothing."""
    by = list(by)
    tz = daily["day_start"].dt.tz
    start, end = (pd.Timestamp(x, tz=tz) for x in surge)
    d = daily.assign(year=daily["day_start"].dt.year)
    d = d[d["census"] > 0]
    nonsurge = d[(d["day_start"] < start) | (d["day_start"] >= end)]
    key = by + ["year"]
    seven_am = d.groupby(key)["census"]
    caps = pd.DataFrame({
        "capacity_p99": seven_am.quantile(0.99),
        "capacity_max": census.assign(year=census["time"].dt.year).groupby(key)["census"].max(),
        "capacity_p95_nonsurge": nonsurge.groupby(key)["census"].quantile(0.95),
        "n_days": seven_am.size(),
    })
    small = ~(caps["capacity_p99"] >= min_beds)  # NA (no morning with anyone present) counts as small
    caps.loc[small, ["capacity_p99", "capacity_max", "capacity_p95_nonsurge"]] = np.nan
    caps["n_days"] = caps["n_days"].fillna(0).astype(int)  # a year with census events but no occupied morning
    return caps.reset_index()


def add_normalized(daily: pd.DataFrame, caps: pd.DataFrame, by=HOSPITAL) -> pd.DataFrame:
    """Census and admissions over that year's capacity (p99 primary, max as sensitivity)."""
    out = daily.assign(year=daily["day_start"].dt.year).merge(caps, on=list(by) + ["year"], how="left")
    for suffix, cap in [("", "capacity_p99"), ("_max", "capacity_max")]:
        out[f"census_strain{suffix}"] = out["census"] / out[cap]
        out[f"admit_strain{suffix}"] = out["admits_24h"] / out[cap]
    return out


def hours_above_capacity(census: pd.DataFrame, caps: pd.DataFrame, by=HOSPITAL) -> pd.DataFrame:
    """Hours each group-year spent above its p99 capacity (e.g. bed-turnover overlap in a full unit).
    An interval that crosses New Year counts in the year it starts (at most one hour per group-year)."""
    by = list(by)
    c = census.assign(year=census["time"].dt.year).merge(caps[by + ["year", "capacity_p99"]], on=by + ["year"], how="left")
    hours = (c.groupby(by)["time"].shift(-1) - c["time"]).dt.total_seconds() / 3600
    return (c.assign(h=hours.where(c["census"] > c["capacity_p99"], 0))
             .groupby(by + ["year"], as_index=False)["h"].sum().rename(columns={"h": "hours_above_p99"}))
