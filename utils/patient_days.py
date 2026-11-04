"""Landmark rows for the outcome model (design v3, E2: incumbents; v4 adds demand-side exposures and process outcomes).

One row per ICU stay per local 7am s at which the patient is already in a unit. The exposure is
that unit's admissions by other patients over [s, s+24h) as a share of its capacity; the outcome
is in-hospital death or hospice discharge over [s+24h, s+8d). Covariates are measured at or
before s.
"""

import numpy as np
import pandas as pd

from .common import (AFTER_HOURS, COVID_ICD10_PREFIX, EXPOSURE_HOURS, LIMITATION_HOURS, MIN_AGE, NON_LIMITATION_CODES,
                     OUTCOME_DAYS, OUTCOME_DAYS_LONG, PLACEBO_OFFSET_DAYS, READMIT_HOURS, SUPPORT_HOURS, UNIT)
from .encounters import death_times

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


def own_admissions(days: pd.DataFrame, stays: pd.DataFrame, lo: pd.Series, hi: pd.Series, weight: str | None = None) -> pd.Series:
    """Per row, how many of the patient's own stays in the row's unit begin in [lo, hi) (or their `weight` summed).

    Every stay counts, not only the one holding the landmark: a bounce-back inside the window
    is the patient's own admission, not someone else's. `lo` and `hi` share `days`' index.
    """
    d = days[["hospitalization_id", *UNIT]].assign(_lo=lo, _hi=hi, _row=np.arange(len(days)))
    cols = ["hospitalization_id", *UNIT, "in_dttm", *([weight] if weight else [])]
    m = d.merge(stays[cols], on=["hospitalization_id", *UNIT])  # a unit is hospital + name
    hit = m[(m["in_dttm"] >= m["_lo"]) & (m["in_dttm"] < m["_hi"])].groupby("_row")
    n = hit[weight].sum() if weight else hit.size()
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
    v4: ed_strain, prior_ed_strain, placebo_ed_strain: hospital ED arrivals over [s, s+24h),
    [s-24h, s) and [s+8d, s+9d) / hospital ICU capacity (demand the ICU's bed-freeing cannot cause).
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
    out = out.merge(h[["hospital_id", "day_start", "admit_strain"]]
                    .rename(columns={"day_start": "day_end", "admit_strain": "hosp_admit_strain"}),
                    on=["hospital_id", "day_end"], how="left")
    if "ed_strain" in h:
        ed = h[["hospital_id", "day_start", "ed_strain"]].rename(columns={"day_start": "_end"})
        for name, end in [("ed_strain", out["day_end"]), ("prior_ed_strain", s),
                          ("placebo_ed_strain", plus_days(s, PLACEBO_OFFSET_DAYS + 1))]:
            out[name] = out[["hospital_id"]].assign(_end=end).merge(ed, on=["hospital_id", "_end"], how="left")["ed_strain"].to_numpy()
    return out


def unit_presence(stays: pd.DataFrame, unit_daily: pd.DataFrame) -> pd.DataFrame:
    """One row per unit stay per unit strain day s (unit_daily's 7ams) with in_dttm <= s < out_dttm: who is in the unit at s.

    The same convention as the 7am census: arriving exactly at 7:00 counts, leaving exactly at 7:00 does not.
    """
    u = unit_daily[UNIT + ["day_start"]].assign(hospital_id=unit_daily["hospital_id"].astype(stays["hospital_id"].dtype))
    u = u.sort_values(UNIT + ["day_start"]).assign(_n=lambda d: d.groupby(UNIT).cumcount()).sort_values("day_start")
    st = stays[["hospitalization_id", *UNIT, "in_dttm", "out_dttm"]].reset_index(drop=True)
    c = pd.merge_asof(st.sort_values("in_dttm"), u.rename(columns={"day_start": "_d0", "_n": "_first"}),
                      left_on="in_dttm", right_on="_d0", by=UNIT, direction="forward")  # first 7am at or after arrival
    c = pd.merge_asof(c.sort_values("out_dttm"), u.rename(columns={"day_start": "_d1", "_n": "_last"}),
                      left_on="out_dttm", right_on="_d1", by=UNIT, allow_exact_matches=False)  # last 7am before leaving
    c = c.dropna(subset=["_first", "_last"])
    n = (c["_last"] - c["_first"] + 1).astype(int)
    c, n = c[n > 0], n[n > 0]
    rows = c.loc[c.index.repeat(n)].copy()
    rows["_n"] = rows["_first"].astype(int) + rows.groupby(level=0).cumcount()
    rows = rows.drop(columns=["_d0", "_d1", "_first", "_last"]).merge(u, on=UNIT + ["_n"])
    return rows[["hospitalization_id", *UNIT, "in_dttm", "out_dttm", "day_start"]].reset_index(drop=True)


def add_weighted_exposure(days: pd.DataFrame, stays: pd.DataFrame, presence: pd.DataFrame, unit_daily: pd.DataFrame) -> pd.DataFrame:
    """Acuity-weighted strain (v5): other patients' weights summed over admissions to the unit in [s, s+24h)
    (wadmits, wadmit_strain) and over patients present at s (wcensus, wcensus_strain), each / that year's capacity.

    `stays` carries `w_arrival` (the patient's weight over the first SUPPORT_HOURS of the unit stay) and
    `presence` (unit_presence rows) carries `w_present` (over the SUPPORT_HOURS before s). The patient's own
    stays and presence are left out. With every weight 1, wadmits is `admits` and wcensus is the census at s less one.
    """
    days = days.reset_index(drop=True)
    cast = {"hospital_id": days["hospital_id"].dtype}
    u = unit_daily[UNIT + ["day_start"]].astype(cast).rename(columns={"day_start": "_end"}).sort_values("_end")
    st = stays[["hospitalization_id", *UNIT, "in_dttm", "w_arrival"]].astype(cast).sort_values("in_dttm")
    nxt = pd.merge_asof(st, u, left_on="in_dttm", right_on="_end", by=UNIT, direction="forward", allow_exact_matches=False)
    arrivals = nxt.groupby(UNIT + ["_end"])["w_arrival"].sum().rename("wadmits").reset_index().rename(columns={"_end": "day_end"})
    out = days.merge(arrivals, on=UNIT + ["day_end"], how="left")
    own = own_admissions(out, st, out["day_start"], out["day_end"], weight="w_arrival")
    out["wadmits"] = out["wadmits"].fillna(0).to_numpy() - own.to_numpy()
    pres = presence.astype(cast)
    out = out.merge(pres.groupby(UNIT + ["day_start"])["w_present"].sum().rename("wcensus").reset_index(), on=UNIT + ["day_start"], how="left")
    own = out[["hospitalization_id", *UNIT, "day_start"]].merge(pres[["hospitalization_id", *UNIT, "day_start", "w_present"]],
                                                               on=["hospitalization_id", *UNIT, "day_start"], how="left")["w_present"]
    out["wcensus"] = out["wcensus"].fillna(0).to_numpy() - own.fillna(0).to_numpy()
    out["wadmit_strain"] = out["wadmits"] / out["capacity_p99"]
    out["wcensus_strain"] = out["wcensus"] / out["capacity_p99"]
    return out


def add_present_exposure(days: pd.DataFrame, stays: pd.DataFrame, unit_daily: pd.DataFrame) -> pd.DataFrame:
    """v6: other patients' admissions to the unit while the patient is still in it, over [s, min(stay_out, s+24h)),
    as a rate over the strain day / capacity (present_admit_strain), with hours_present; and the same hours one week
    earlier (placebo_present_admit_strain; NA when that day is before the unit's first strain day).

    A patient present all day gets exactly admit_strain. Nothing after the patient's own exit is counted, so the bed
    that exit frees cannot raise their exposure.
    """
    days = days.reset_index(drop=True)
    cast = {"hospital_id": days["hospital_id"].dtype}
    grid = unit_daily[UNIT + ["day_start"]].astype(cast).sort_values("day_start")
    adm = stays[["hospitalization_id", *UNIT, "in_dttm"]].astype(cast).sort_values("in_dttm")
    adm = pd.merge_asof(adm, grid.rename(columns={"day_start": "_key"}), left_on="in_dttm", right_on="_key", by=UNIT)
    adm = adm.dropna(subset=["_key"]).rename(columns={"hospitalization_id": "_other"})  # keyed by the strain day holding the admission
    s = days["day_start"]
    cut = days["stay_out"].where(days["stay_out"] < days["day_end"], days["day_end"])
    hours = (cut - s).dt.total_seconds() / 3600
    day_hours = (days["day_end"] - s).dt.total_seconds() / 3600

    def count(lo, hi):
        """Admissions by other patients to the row's unit in [lo, hi); NA when lo is not a strain day of the unit."""
        r = days[["hospitalization_id", *UNIT]].assign(_lo=lo, _hi=hi, _row=np.arange(len(days)))
        m = r.merge(adm.rename(columns={"_key": "_lo"}), on=[*UNIT, "_lo"], how="left")
        hit = m[(m["in_dttm"] >= m["_lo"]) & (m["in_dttm"] < m["_hi"]) & (m["_other"] != m["hospitalization_id"])]
        n = hit.groupby("_row").size().reindex(range(len(days)), fill_value=0).to_numpy()
        known = r[[*UNIT, "_lo"]].merge(grid.rename(columns={"day_start": "_lo"}).assign(_k=True), on=[*UNIT, "_lo"], how="left")["_k"]
        return pd.Series(np.where(known.fillna(False).to_numpy(), n, np.nan), index=days.index)

    out = days.assign(hours_present=hours, present_admits=count(s, cut))
    out["present_admit_strain"] = out["present_admits"] / hours * day_hours / out["capacity_p99"]
    lo = plus_days(s, -7)
    out["placebo_present_admit_strain"] = count(lo, lo + (cut - s)) / hours * day_hours / out["capacity_p99"]
    return out


def _events(hosp: pd.DataFrame, kinds) -> pd.DataFrame:
    disp = hosp["discharge_category"].str.lower()
    e = hosp.loc[disp.isin(kinds) & hosp["discharge_dttm"].notna(), ["patient_id", "discharge_dttm"]]
    return e.rename(columns={"discharge_dttm": "t"})


def _event_time(days: pd.DataFrame, events: pd.DataFrame, direction: str = "forward", value: str = "t") -> pd.Series:
    """`value` of the patient's first event at or after (forward) or last event strictly before (backward) each row's
    day_start; NaN when there is none."""
    d = days[["patient_id", "day_start"]].reset_index(names="_i").sort_values("day_start")
    e = events.dropna(subset=["t"])
    e = e.assign(t=e["t"].astype(d["day_start"].dtype), patient_id=e["patient_id"].astype(d["patient_id"].dtype)).sort_values("t")
    m = pd.merge_asof(d, e, left_on="day_start", right_on="t", by="patient_id", direction=direction,
                      allow_exact_matches=direction == "forward")
    return m.set_index("_i")[value].reindex(days.index)


def add_outcomes(days: pd.DataFrame, hosp: pd.DataFrame, patient: pd.DataFrame,
                 blocks: pd.DataFrame | None = None, code_status: pd.DataFrame | None = None) -> pd.DataFrame:
    """Outcomes after the exposure window, from hospital records at the patient level (D2).

    `hosp` is the stitched encounter table (step 01), so a transfer's interim discharge category is
    replaced by the encounter's final one. An HDHD event is an in-hospital death (discharged Expired,
    at discharge_dttm) or a hospice discharge, in any encounter of the patient. hdhd_7d: event in [s+24h, s+8d);
    hdhd_30d: in [s+24h, s+31d); hdhd_48h: in [s+24h, s+48h) (the next day); hdhd_hosp: at any
    time from s+24h to the end of this hospitalization (NA while it is still open); death_7d: in-hospital death only;
    death_any_7d adds patient.death_dttm (post-discharge linkage, whose completeness differs by site),
    reconciled by utils.encounters.death_times.
    `eligible` is False when an event falls inside the exposure window [s, s+24h).
    icu_discharge_24h and unit_exit_24h (left the ICU, or the unit, inside the exposure
    window) are the positive-control outcomes.
    v4 process outcomes: among landmarks discharged from the ICU inside the window, readmit_72h
    (a later ICU stay of the same hospitalization begins within 72 h of that discharge) and
    after_hours_discharge (18:00-06:00); among landmarks whose last code-status order of this encounter
    before s is not a limitation (or who have none since admission), new_limitation_48h (a DNR/DNI or
    other limitation order in [s, s+48h)). NA elsewhere.
    """
    if days.empty:
        return days.assign(eligible=True, hdhd_7d=0, hdhd_30d=0, hdhd_48h=0, hdhd_hosp=np.nan, death_7d=0, death_any_7d=0,
                           icu_discharge_24h=0, unit_exit_24h=0, readmit_72h=np.nan, after_hours_discharge=np.nan,
                           new_limitation_48h=np.nan)
    expired, hospice = _events(hosp, ["expired"]), _events(hosp, ["hospice"])
    s = days["day_start"]
    fu = s + EXPOSURE
    hdhd = _event_time(days, pd.concat([expired, hospice]))
    dead = _event_time(days, expired)
    any_ = _event_time(days, death_times(hosp, patient))  # Expired discharge time, else death_dttm fixed for date-only

    def within(t, horizon):
        return ((t >= fu) & (t < s + pd.Timedelta(days=horizon + 1))).astype(int)

    discharged = days["icu_discharge_dttm"] < fu
    out = days.assign(
        eligible=~(hdhd < fu),  # event-free when follow-up starts: a death inside the exposure window is not its effect
        hdhd_7d=within(hdhd, OUTCOME_DAYS), hdhd_30d=within(hdhd, OUTCOME_DAYS_LONG), hdhd_48h=within(hdhd, 1),
        hdhd_hosp=((hdhd >= fu) & (hdhd <= days["discharge_dttm"])).astype(float).where(days["discharge_dttm"].notna()),  # NA: still open
        death_7d=within(dead, OUTCOME_DAYS), death_any_7d=within(any_, OUTCOME_DAYS),
        icu_discharge_24h=discharged.astype(int),
        unit_exit_24h=(days["stay_out"] < fu).astype(int),
    )
    hour = days["icu_discharge_dttm"].dt.hour
    out["after_hours_discharge"] = ((hour >= AFTER_HOURS[0]) | (hour < AFTER_HOURS[1])).astype(float).where(discharged)
    if blocks is not None:
        b = blocks.sort_values(["hospitalization_id", "icu_admit_dttm"])
        b = b.assign(_next=b.groupby("hospitalization_id")["icu_admit_dttm"].shift(-1))
        nxt = days[["hospitalization_id", "block_index"]].merge(
            b[["hospitalization_id", "block_index", "_next"]], on=["hospitalization_id", "block_index"], how="left")["_next"]
        gap = nxt.to_numpy() - days["icu_discharge_dttm"].to_numpy()
        out["readmit_72h"] = pd.Series(gap <= np.timedelta64(READMIT_HOURS, "h"), index=days.index).astype(float).where(discharged)
    else:
        out["readmit_72h"] = np.nan
    if code_status is not None and len(code_status):
        orders = (code_status[["patient_id", "start_dttm", "code_status_category"]].rename(columns={"start_dttm": "t"})
                  .dropna(subset=["code_status_category"]))  # an uncategorized order is neither full code nor a limitation
        orders = orders.assign(limited=~orders["code_status_category"].str.lower().isin(NON_LIMITATION_CODES))
        orders = orders.groupby(["patient_id", "t"], as_index=False)["limited"].max()  # tied orders: a limitation wins, whatever the row order
        last_t = _event_time(days, orders, "backward")
        at_s = _event_time(days, orders, "backward", value="limited").where(last_t >= days["admission_dttm"])  # resets at admission
        nxt = _event_time(days, orders[orders["limited"]], "forward")
        new = (nxt >= s) & (nxt < s + pd.Timedelta(hours=LIMITATION_HOURS))
        out["new_limitation_48h"] = new.astype(float).where(~at_s.fillna(False).astype(bool))  # NA when limited at s
    else:
        out["new_limitation_48h"] = np.nan
    return out


def covid_hospitalizations(dx: pd.DataFrame):
    """Hospitalizations with ICD-10-CM U07.1 coded anywhere."""
    code = dx["diagnosis_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    return dx.loc[code.str.startswith(COVID_ICD10_PREFIX), "hospitalization_id"].unique()
