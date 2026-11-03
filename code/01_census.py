"""Step 1: ICU stays, rolling census, daily 7am strain and capacity, per unit and per hospital.

Reads hospitalization and adt. Writes
  intermediate_phi/<site>/  icu_blocks, unit_stays, census_by_{unit,hospital}, daily_by_{unit,hospital}
  <site>_upload_to_box/     capacity_by_{unit,hospital}.csv (one row per unit-year), monthly_strain_by_{unit,hospital}.csv,
                            census_qc.csv, monthly_admit_strain_by_unit.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for utils/

import pandas as pd
from clifpy import ClifOrchestrator

from utils import common as c
from utils import figures
from utils.census import (add_normalized, capacities, collapse_icu_blocks, daily_strain, hours_above_capacity,
                          icu_intervals, is_icu, merge_round_trips, rolling_census, unit_stays)

config = c.load_config()
site = config["site_name"]
phi, final = c.phi_dir(site), c.final_dir(site)

co = ClifOrchestrator(data_directory=config["data_directory"], filetype=config["filetype"], timezone=config["timezone"])
co.initialize(tables=["hospitalization", "adt"], columns={
    "hospitalization": ["patient_id", "hospitalization_id", "admission_dttm", "age_at_admission"],
    "adt": ["hospitalization_id", "hospital_id", "in_dttm", "out_dttm", "location_category", "location_type", "location_name"],
})
hosp, adt = co.hospitalization.df, co.adt.df
assert adt["hospital_id"].notna().all(), "adt.hospital_id has nulls: a null key silently empties the census; fill it"
icu_rows_raw = int(is_icu(adt).sum())
adt = merge_round_trips(adt)  # OR / CT / dialysis trips: the bed is held, the return is not an admission (v3, D5)

adult = hosp.loc[hosp["age_at_admission"] >= c.MIN_AGE, "hospitalization_id"]
intervals = icu_intervals(adt, adult)
stays = unit_stays(intervals)
blocks = collapse_icu_blocks(adt)
blocks = blocks[blocks["hospitalization_id"].isin(adult)]
# Extract cutoff: after the last hospital admission no one new arrives and the census only drains. Not the last
# ICU admission: ward patients admitted before the cutoff keep moving to the ICU for weeks after it.
data_end = hosp["admission_dttm"].max()
stays.to_parquet(phi / "unit_stays.parquet", index=False)
blocks.to_parquet(phi / "icu_blocks.parquet", index=False)

unit_types = stays[c.UNIT + ["location_type"]].drop_duplicates(c.UNIT)
levels = {"hospital": (c.HOSPITAL, blocks.rename(columns={"icu_admit_dttm": "in_dttm"})),
          "unit": (c.UNIT, stays)}  # hospital admissions = new ICU stays; unit admissions include ICU-to-ICU transfers
dailies = {}
for level, (by, admits) in levels.items():
    census = rolling_census(intervals, by=by)
    daily = daily_strain(census, admits, by=by)
    daily = daily[daily["day_start"] <= data_end]
    caps = capacities(census, daily, by=by, min_beds=c.MIN_UNIT_BEDS if level == "unit" else 1)  # D6 is a unit rule
    caps = caps.merge(hours_above_capacity(census, caps, by=by), on=by + ["year"], how="left")
    daily = add_normalized(daily, caps, by=by)
    if level == "hospital":  # ED arrivals per 24h: demand that bed-freeing cannot cause (v4 instrument)
        ed = adt.loc[adt["location_category"].str.lower().eq(c.ED_LOCATION), by + ["in_dttm"]].dropna()
        ed_daily = (daily_strain(census, ed, by=by)[by + ["day_start", "admits_24h"]]
                    .rename(columns={"admits_24h": "ed_arrivals_24h"}))
        daily = daily.merge(ed_daily, on=by + ["day_start"], how="left")
        daily["ed_strain"] = daily["ed_arrivals_24h"] / daily["capacity_p99"]
    dailies[level] = daily
    census.to_parquet(phi / f"census_by_{level}.parquet", index=False)
    daily.to_parquet(phi / f"daily_by_{level}.parquet", index=False)

    if level == "unit":
        caps = unit_types.merge(caps, on=by)
        excluded = caps["capacity_p99"].isna()  # unit-years under MIN_UNIT_BEDS (D6), or without a morning census
        n_units = len(caps.drop_duplicates(by))
        if (caps["capacity_p99"] > c.MAX_UNIT_BEDS).any() or n_units > c.MAX_UNITS:
            print(f"WARNING: {n_units} ICU units, some over {c.MAX_UNIT_BEDS} beds. "
                  "Check that location_name names a unit, not a floor or a room.")
    caps.round(2).to_csv(final / f"capacity_by_{level}.csv", index=False)

    monthly = (daily.assign(month=daily["day_start"].dt.tz_localize(None).dt.to_period("M").astype(str))
               .groupby(by + ["month"], as_index=False)
               .agg(n_days=("census", "size"), admissions=("admits_24h", "sum"),
                    mean_admit_strain=("admit_strain", "mean"), mean_census_strain=("census_strain", "mean")))
    monthly = c.suppress_small_rows(monthly, "admissions", ["mean_admit_strain"])
    monthly.round(3).to_csv(final / f"monthly_strain_by_{level}.csv", index=False)
    print(f"\n{level} capacity:\n{caps.round(1).to_string(index=False)}")

figures.monthly_strain(monthly, "mean_admit_strain", "ICU admissions per day / capacity (monthly mean)").savefig(
    final / "monthly_admit_strain_by_unit.png", dpi=150)

icu = is_icu(adt)
qc = pd.DataFrame({"check": ["icu_adt_rows", "icu_rows_folded_into_round_trips", "icu_adt_rows_missing_time_or_unit",
                             "adult_icu_intervals", "adult_icu_stays", "adult_unit_stays", "icu_to_icu_transfers"],
                   "n": [icu_rows_raw, icu_rows_raw - int(icu.sum()),
                         int(adt.loc[icu, ["in_dttm", "out_dttm", "location_name"]].isna().any(axis=1).sum()),
                         len(intervals), len(blocks), len(stays), int(stays["from_icu"].sum())]})
# A census that falls off at the end means the extract misses patients still in hospital at the cutoff.
census = dailies["hospital"].groupby("day_start")["census"].sum()
age = (data_end - census.index).days
last, prior = census[age < 60].mean(), census[(age >= 60) & (age < 120)].mean()
drain_pct = 100 * last / prior if prior > 0 else float("nan")  # nan: empty ICU in the window before (demo)
qc = pd.concat([c.suppress_small_cells(qc, ["n"]), pd.DataFrame({
    "check": ["icu_units", "icu_unit_years_without_capacity_under_4_beds", "census_last_60d_pct_of_prior_60d"],
    "n": pd.Series([n_units, int(excluded.sum()), drain_pct]).round().astype("Int64")})])
qc.to_csv(final / "census_qc.csv", index=False)
print(f"\n{qc.to_string(index=False)}\ndata end (last hospital admission): {data_end:%Y-%m-%d}")
if drain_pct < c.MIN_END_CENSUS_PCT:
    print(f"WARNING: census in the last 60 days is {drain_pct:.0f}% of the 60 days before. "
          "The extract may leave out patients still in hospital at the cutoff.")
