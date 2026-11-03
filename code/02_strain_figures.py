"""Step 2: Will's per-unit figures. Run after 01.

  F1 census_week.png        each unit's rolling census for one week, ^ admissions, v discharges
  F2 daily_admit_strain.png each unit's daily admissions / capacity over the whole period

Both show daily counts of a few patients per unit, so they are written to
intermediate_phi/<site>/figures/ for the local PI only, never to the upload folder.

Usage: uv run python code/02_strain_figures.py [YYYY-MM-DD]   (week start; default: the Monday at mid-period)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils import common as c
from utils import figures

site = c.load_config()["site_name"]
phi, final = c.phi_dir(site), c.final_dir(site)
out = phi / "figures"
out.mkdir(exist_ok=True)

daily = pd.read_parquet(phi / "daily_by_unit.parquet")
census = pd.read_parquet(phi / "census_by_unit.parquet")
stays = pd.read_parquet(phi / "unit_stays.parquet")
caps = pd.read_csv(final / "capacity_by_unit.csv", dtype={"hospital_id": str})  # match the parquet ids

tz = daily["day_start"].dt.tz
if len(sys.argv) > 1:
    week_start = pd.Timestamp(f"{sys.argv[1]} {c.DAY_START_HOUR:02d}:00", tz=tz)
else:
    mid = daily["day_start"].min() + (daily["day_start"].max() - daily["day_start"].min()) / 2
    week_start = (mid + pd.Timedelta(days=(7 - mid.weekday()) % 7)).replace(hour=c.DAY_START_HOUR, minute=0)
caps_year = caps[caps["year"] == week_start.year]  # capacity is per unit-year (D6)
caps = caps_year if len(caps_year) else caps.drop_duplicates(c.UNIT)

figures.census_week(census, stays, caps, week_start).savefig(out / "census_week.png", dpi=150)
figures.daily_admit_strain(daily).savefig(out / "daily_admit_strain.png", dpi=150)
print(f"wrote {out}/census_week.png (week of {week_start:%Y-%m-%d}) and daily_admit_strain.png")
