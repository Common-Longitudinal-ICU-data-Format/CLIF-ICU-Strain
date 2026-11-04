"""Step 4: worst SOFA in the 24 h before each landmark (7am patient-day). Run after 03.

Window per patient-day: [day_start - 24h, day_start). clifpy's compute_sofa_polars joins every event of
a hospitalization to every one of its windows, so each event is first
keyed to its one window (utils.sofa.assign_windows) and cohort-limited copies of the five tables are
written to intermediate_phi/<site>/sofa_input/ with window_id in place of hospitalization_id; clifpy then
scores each window as if it were a hospitalization. The on_imv / on_pressor flags come from the raw
window rows, because clifpy's device_category is only set when the window has an arterial P/F.
Landmarks are keyed by stitched encounter (step 01); each raw row is re-keyed to its encounter first, so a
window that crosses from one linked hospitalization into the next sees both.
Writes intermediate_phi/<site>/sofa_landmark.parquet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import polars as pl

from utils import common as c
from utils.sofa import COMPONENTS, DOSE_COLS, score_windows

config = c.load_config()
phi = c.phi_dir(config["site_name"])
OUT_COLS =["hospitalization_id", "day_start", *COMPONENTS, "sofa_total", "device_category", *DOSE_COLS, "on_imv", "on_pressor"]

windows = (pl.read_parquet(phi / "patient_days.parquet", columns=["hospitalization_id", "day_start"]).unique()
           .with_columns(pl.col("day_start").dt.cast_time_unit("us"))  # events are standardised to microseconds; a site's parquet may be ns
           .sort("hospitalization_id", "day_start")
           .with_columns(window_id=pl.int_range(pl.len()).cast(pl.Utf8), end="day_start",
                         start=pl.col("day_start").dt.offset_by("-1d")))  # previous 7am wall-clock: windows never overlap across DST
encounter_map = (pl.read_parquet(phi / "encounter_map.parquet").with_columns(pl.all().cast(pl.Utf8))
                 .filter(pl.col("encounter_id").is_in(windows["hospitalization_id"].unique().to_list())))
if windows.is_empty():  # clifpy fails on an empty cohort (e.g. the demo, whose units have no capacity)
    pd.DataFrame(columns=OUT_COLS).to_parquet(phi / "sofa_landmark.parquet", index=False)
    print("no patient-days; wrote an empty sofa_landmark.parquet")
    sys.exit(0)

out = score_windows(config, windows, encounter_map, phi / "sofa_input")[OUT_COLS]
out.to_parquet(phi / "sofa_landmark.parquet", index=False)

print(f"SOFA for {out['sofa_total'].notna().sum():,} of {len(out):,} patient-days; "
      f"median {out['sofa_total'].median():.0f} (IQR {out['sofa_total'].quantile(.25):.0f}-{out['sofa_total'].quantile(.75):.0f})")
print("missing by component (%):", (out[COMPONENTS].isna().mean() * 100).round(0).to_dict())
print(f"on_imv {out['on_imv'].mean() * 100:.1f}%  on_pressor {out['on_pressor'].mean() * 100:.1f}%  "
      f"(clifpy device_category == imv {out['device_category'].astype(str).str.lower().eq('imv').mean() * 100:.1f}%)")
