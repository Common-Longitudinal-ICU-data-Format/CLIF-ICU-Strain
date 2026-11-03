"""Landmark rows for the outcome model (design v3, E2: incumbents).

One row per ICU stay per local 7am s at which the patient is already in a unit. The exposure is
that unit's admissions by other patients over [s, s+24h) as a share of its capacity; the outcome
is in-hospital death or hospice discharge over [s+24h, s+8d). Covariates are measured at or
before s.
"""

import numpy as np
import pandas as pd

from .common import COVID_ICD10_PREFIX, EXPOSURE_HOURS, MIN_AGE, OUTCOME_DAYS, OUTCOME_DAYS_LONG, PLACEBO_OFFSET_DAYS, UNIT

EXPOSURE = pd.Timedelta(hours=EXPOSURE_HOURS)


def plus_days(t: pd.Series, k: int) -> pd.Series:
    """t + k calendar days at the same wall-clock time (a 7am stays 7am across DST)."""
    tz = t.dt.tz
    out = (t.dt.tz_localize(None) if tz else t) + pd.Timedelta(days=k)
    return out.dt.tz_localize(tz) if tz else out


def icu_cohort(hosp: pd.DataFrame, blocks: pd.DataFrame, start: str, end: str, min_age: int = MIN_AGE) -> pd.DataFrame:
    """Adult ICU stays (contiguous ICU blocks, all of them, not only the first) starting in [start, end)."""
    c = hosp.merge(blocks, on="hospitalization_id")
    t = c["icu_admit_dttm"]
    lo, hi = (pd.Timestamp(x, tz=t.dt.tz) for x in (start, end))
    keep = (c["age_at_admission"] >= min_age) & (t >= lo) & (t < hi)
    c = c[keep].dropna(subset=["icu_discharge_dttm"]).reset_index(drop=True)
    return c.assign(stay_id=c["hospitalization_id"].astype(str) + "_" + c["block_index"].astype(str))


def expand_patient_days(cohort: pd.DataFrame, hosp_daily: pd.DataFrame) -> pd.DataFrame:
    """One row per ICU stay per hospital strain day [day_start, day_end) it overlaps.

    Day 0 starts at the last 7am at or before ICU admission (it is not a landmark, but it keeps
    the ICU-day count honest). Stays starting before the hospital's first strain day (census
    burn-in) are dropped.
    """
    daily = hosp_daily[["hospital_id", "day_start", "day_end", "day_prev"]].sort_values(["hospital_id", "day_start"])
    daily = daily.assign(day_num=daily.groupby("hospital_id").cumcount(),
                         # merge_asof needs identical `by` dtypes; parquet can round-trip strings differently
                         hospital_id=daily["hospital_id"].astype(cohort["hospital_id"].dtype))
    snaps = daily[["hospital_id", "day_start", "day_num"]].sort_values("day_start")
    c = pd.merge_asof(cohort.sort_values("icu_admit_dttm"), snaps.rename(columns={"day_start": "_d0", "day_num": "_first"}),
                      left_on="icu_admit_dttm", right_on="_d0", by="hospital_id")
    c = pd.merge_asof(c.sort_values("icu_discharge_dttm"), snaps.rename(columns={"day_start": "_d1", "day_num": "_last"}),
                      left_on="icu_discharge_dttm", right_on="_d1", by="hospital_id",
                      allow_exact_matches=False)  # leaving at exactly 7:00 does not open a new day
    c = c.dropna(subset=["_first", "_last"])
    n = (c["_last"] - c["_first"] + 1).astype(int)
    c, n = c[n > 0], n[n > 0]
    rows = c.loc[c.index.repeat(n)].copy()
    rows["icu_day"] = rows.groupby(level=0).cumcount()
    rows["day_num"] = rows["_first"].astype(int) + rows["icu_day"]
    rows = rows.drop(columns=["_d0", "_d1", "_first", "_last"]).merge(daily, on=["hospital_id", "day_num"])
    # A same-day ICU readmission repeats that 7am; keep the earlier stay's row.
    rows = rows.sort_values("icu_admit_dttm").drop_duplicates(["hospitalization_id", "day_start"])
    return rows.drop(columns="day_num").reset_index(drop=True)


def assign_unit(days: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """The unit (location_name) each patient is in at 7am s, and that unit stay's start and end.

    `incumbent` is True when the unit stay began before s and holds s: the row is a landmark.
    Day 0 (arrived after s) and a 7am falling in a <1h gap between ICU rows take the next unit
    stay instead, so the unit is known, but they are not incumbents.
    """
    st = (stays[["hospitalization_id", "location_name", "location_type", "in_dttm", "out_dttm", "from_icu"]]
          .rename(columns={"in_dttm": "stay_in", "out_dttm": "stay_out"}).sort_values("stay_in"))
    d = days.sort_values("day_start").reset_index(drop=True)
    back = pd.merge_asof(d, st, left_on="day_start", right_on="stay_in", by="hospitalization_id")
    fwd = pd.merge_asof(d, st, left_on="day_start", right_on="stay_in", by="hospitalization_id", direction="forward")
    use_fwd = ~(back["stay_out"] > back["day_start"])
    back["incumbent"] = ~use_fwd & (back["stay_in"] < back["day_start"])
    cols = ["location_name", "location_type", "stay_in", "stay_out", "from_icu"]
    back.loc[use_fwd, cols] = fwd.loc[use_fwd, cols]
    return back


def own_admissions(days: pd.DataFrame, stays: pd.DataFrame, lo: pd.Series, hi: pd.Series) -> pd.Series:
    """Per row, how many of the patient's own stays in the row's unit begin in [lo, hi).

    Every stay counts, not only the one holding the landmark: a bounce-back inside the window
    is the patient's own admission, not someone else's. `lo` and `hi` share `days`' index.
    """
    d = days[["hospitalization_id", "location_name"]].assign(_lo=lo, _hi=hi, _row=np.arange(len(days)))
    m = d.merge(stays[["hospitalization_id", "location_name", "in_dttm"]], on=["hospitalization_id", "location_name"])
    n = m[(m["in_dttm"] >= m["_lo"]) & (m["in_dttm"] < m["_hi"])].groupby("_row").size()
    return pd.Series(n.reindex(range(len(days)), fill_value=0).to_numpy(), index=days.index)


def add_exposure(days: pd.DataFrame, stays: pd.DataFrame, unit_daily: pd.DataFrame, hosp_daily: pd.DataFrame) -> pd.DataFrame:
    """Strain around landmark s, from step 1's unit-day and hospital-day tables.

    admit_strain: other patients' admissions to the patient's unit over [s, s+24h) / that
    year's capacity (primary); admit_strain_max: the same over max-census capacity;
    placebo_admit_strain: admissions over [s+8d, s+9d), after the outcome window;
    placebo_past_admit_strain: admissions over [s-8d, s-7d), a week before the exposure window;
    prior_admit_strain: admissions over [s-24h, s) (unit state at s); census_strain: occupancy
    at s; hosp_admit_strain, hosp_census_strain: hospital-wide over [s, s+24h) and at s
    (shortcut: the patient's own admission is not removed from the hospital count, 1 in ~20/day).
    """
    days = days.reset_index(drop=True)
    u = unit_daily.assign(hospital_id=unit_daily["hospital_id"].astype(days["hospital_id"].dtype))
    h = hosp_daily.assign(hospital_id=hosp_daily["hospital_id"].astype(days["hospital_id"].dtype))
    out = days.merge(u[UNIT + ["day_start", "census_strain", "capacity_p99", "capacity_max"]],
                     on=UNIT + ["day_start"], how="left")
    admits = u[UNIT + ["day_start", "admits_24h"]].rename(columns={"day_start": "_end"})  # counted on the day ending at _end
    s = out["day_start"]
    windows = {"": (s, out["day_end"]),
               "prior_": (out["day_prev"], s),
               "placebo_": (plus_days(s, PLACEBO_OFFSET_DAYS), plus_days(s, PLACEBO_OFFSET_DAYS + 1)),
               "placebo_past_": (plus_days(s, -PLACEBO_OFFSET_DAYS), plus_days(s, -PLACEBO_OFFSET_DAYS + 1))}
    for prefix, (lo, hi) in windows.items():
        total = out[UNIT].assign(_end=hi).merge(admits, on=UNIT + ["_end"], how="left")["admits_24h"]
        out[f"{prefix}admits"] = total.to_numpy() - own_admissions(out, stays, lo, hi).to_numpy()
        out[f"{prefix}admit_strain"] = out[f"{prefix}admits"] / out["capacity_p99"]
    out["admit_strain_max"] = out["admits"] / out["capacity_max"]
    out = out.merge(h[["hospital_id", "day_start", "census_strain"]].rename(columns={"census_strain": "hosp_census_strain"}),
                    on=["hospital_id", "day_start"], how="left")
    return out.merge(h[["hospital_id", "day_start", "admit_strain"]]
                     .rename(columns={"day_start": "day_end", "admit_strain": "hosp_admit_strain"}),
                     on=["hospital_id", "day_end"], how="left")


def _events(hosp: pd.DataFrame, kinds) -> pd.DataFrame:
    disp = hosp["discharge_category"].str.lower()
    e = hosp.loc[disp.isin(kinds) & hosp["discharge_dttm"].notna(), ["patient_id", "discharge_dttm"]]
    return e.rename(columns={"discharge_dttm": "t"})


def _next_event(days: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Time of the patient's first event at or after each row's day_start (NaT when none)."""
    d = days[["patient_id", "day_start"]].reset_index(names="_i").sort_values("day_start")
    e = events.dropna(subset=["t"])
    e = e.assign(t=e["t"].astype(d["day_start"].dtype), patient_id=e["patient_id"].astype(d["patient_id"].dtype)).sort_values("t")
    m = pd.merge_asof(d, e, left_on="day_start", right_on="t", by="patient_id", direction="forward")
    return m.set_index("_i")["t"].reindex(days.index)


def add_outcomes(days: pd.DataFrame, hosp: pd.DataFrame, patient: pd.DataFrame) -> pd.DataFrame:
    """Outcomes after the exposure window, from hospital records at the patient level (D2).

    An HDHD event is an in-hospital death (discharged Expired, at discharge_dttm) or a hospice
    discharge, in any hospitalization of the patient. hdhd_7d: event in [s+24h, s+8d);
    hdhd_30d: in [s+24h, s+31d); death_7d: in-hospital death only; death_any_7d adds
    patient.death_dttm (post-discharge linkage, whose completeness differs by site).
    `eligible` is False when an event falls inside the exposure window [s, s+24h).
    icu_discharge_24h and unit_exit_24h (left the ICU, or the unit, inside the exposure
    window) are the positive-control outcomes.
    """
    if days.empty:
        return days.assign(eligible=True, hdhd_7d=0, hdhd_30d=0, death_7d=0, death_any_7d=0,
                           icu_discharge_24h=0, unit_exit_24h=0)
    expired, hospice = _events(hosp, ["expired"]), _events(hosp, ["hospice"])
    linked = patient[["patient_id", "death_dttm"]].rename(columns={"death_dttm": "t"})
    s = days["day_start"]
    fu = s + EXPOSURE
    hdhd = _next_event(days, pd.concat([expired, hospice]))
    dead = _next_event(days, expired)
    any_ = _next_event(days, pd.concat([expired, linked]))

    def within(t, horizon):
        return ((t >= fu) & (t < s + pd.Timedelta(days=horizon + 1))).astype(int)

    return days.assign(
        eligible=~(hdhd < fu),  # event-free when follow-up starts: a death inside the exposure window is not its effect
        hdhd_7d=within(hdhd, OUTCOME_DAYS), hdhd_30d=within(hdhd, OUTCOME_DAYS_LONG),
        death_7d=within(dead, OUTCOME_DAYS), death_any_7d=within(any_, OUTCOME_DAYS),
        icu_discharge_24h=(days["icu_discharge_dttm"] < fu).astype(int),
        unit_exit_24h=(days["stay_out"] < fu).astype(int),
    )


def covid_hospitalizations(dx: pd.DataFrame):
    """Hospitalizations with ICD-10-CM U07.1 coded anywhere."""
    code = dx["diagnosis_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    return dx.loc[code.str.startswith(COVID_ICD10_PREFIX), "hospitalization_id"].unique()
