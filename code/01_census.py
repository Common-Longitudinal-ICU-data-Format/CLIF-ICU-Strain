"""ICU census, capacity, and daily 7am strain per hospital. Run from repo root.

Reads config/config.json (or CLIF_CONFIG_PATH); writes
  output/intermediate_phi/<site>/          row-level census and daily strain (never shared)
  output/final_no_phi/<site>_upload_to_box/ capacity, QC counts, monthly strain, figure
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for utils/

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from clifpy import ClifOrchestrator

from utils import strain as s

config = json.loads(Path(os.environ.get("CLIF_CONFIG_PATH", "config/config.json")).read_text())
site = config["site_name"]
phi, final = s.phi_dir(site), s.final_dir(site)
phi.mkdir(parents=True, exist_ok=True)
final.mkdir(parents=True, exist_ok=True)

co = ClifOrchestrator(data_directory=config["data_directory"], filetype=config["filetype"], timezone=config["timezone"])
co.initialize(tables=["hospitalization", "adt"])
hosp, adt = co.hospitalization.df, co.adt.df

adult_ids = hosp.loc[hosp["age_at_admission"] >= s.MIN_AGE, "hospitalization_id"]
intervals = s.icu_intervals(adt, adult_ids)
census = s.rolling_census(intervals, by=["hospital_id"])
blocks = s.collapse_icu_blocks(adt)
admits = blocks[blocks["hospitalization_id"].isin(adult_ids)]
daily = s.daily_strain(census, admits)
# New hospitalizations stop at the extract cutoff; after it the census only drains.
daily = daily[daily["day_start"] <= hosp["admission_dttm"].max()]
caps = s.capacities(census, daily)
daily = s.add_normalized(daily, caps)

census.to_parquet(phi / "census_by_hospital.parquet", index=False)
s.rolling_census(intervals, by=["hospital_id", "location_type", "location_name"]).to_parquet(
    phi / "census_by_unit.parquet", index=False)  # for the unit-level analysis
daily.to_parquet(phi / "daily_strain.parquet", index=False)
caps.to_csv(final / "capacity.csv", index=False)

icu_rows = adt[adt["location_category"].str.lower() == s.ICU_LOCATION]
qc = pd.DataFrame({
    "check": ["icu_adt_rows", "icu_adt_rows_missing_in_or_out", "adult_icu_intervals_kept", "adult_icu_stays"],
    "n": [len(icu_rows), int(icu_rows[["in_dttm", "out_dttm"]].isna().any(axis=1).sum()), len(intervals), len(admits)],
})
s.suppress_small_cells(qc, ["n"]).to_csv(final / "census_qc.csv", index=False)

monthly = (
    daily.assign(month=daily["day_start"].dt.tz_localize(None).dt.to_period("M").astype(str))
    .groupby(["hospital_id", "month"], as_index=False)
    .agg(n_days=("census", "size"), mean_census_norm=(s.PRIMARY_EXPOSURE, "mean"),
         max_census_norm=(s.PRIMARY_EXPOSURE, "max"), mean_admits_norm=("admits_norm", "mean"))
)
monthly.to_csv(final / "monthly_strain.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4))
for h, g in monthly.groupby("hospital_id"):
    ax.plot(pd.PeriodIndex(g["month"], freq="M").to_timestamp(), g["mean_census_norm"], label=h)
ax.axhline(1, color="grey", lw=0.8, ls="--")
ax.set(ylabel="7am ICU census / capacity (monthly mean)", title="ICU strain by month")
ax.legend()
fig.savefig(final / "strain_monthly.png", dpi=150, bbox_inches="tight")

print(caps.to_string(index=False))
print(daily.groupby("hospital_id")["day_start"].agg(["min", "max"]).to_string())  # eyeball for bogus ADT years
