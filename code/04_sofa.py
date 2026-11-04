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
from clifpy.utils.datetime_polars import standardize_datetime_columns
from clifpy.utils.sofa_polars import (REQUIRED_ASSESSMENTS, REQUIRED_LABS, REQUIRED_MEDS, REQUIRED_VITALS,
                                      compute_sofa_polars)

from utils import common as c
from utils.sofa import COMPONENTS, DOSE_COLS, VASOPRESSORS, assign_windows, rescore_sofa

config = c.load_config()
phi = c.phi_dir(config["site_name"])
tz = config["timezone"]
CHUNK_WINDOWS = 50_000               # ~2 GB per clifpy call
OUT_COLS =["hospitalization_id", "day_start", *COMPONENTS, "sofa_total", "device_category", *DOSE_COLS, "on_imv", "on_pressor"]
TABLES = {  # table: (time column, other columns clifpy 0.5.0 reads, row filter)
    "clif_labs": ("lab_result_dttm", ["lab_category", "lab_value", "lab_value_numeric"], pl.col("lab_category").is_in(REQUIRED_LABS)),
    "clif_vitals": ("recorded_dttm", ["vital_category", "vital_value"], pl.col("vital_category").is_in(REQUIRED_VITALS)),
    "clif_patient_assessments": ("recorded_dttm", ["assessment_category", "numerical_value", "categorical_value"],
                                 pl.col("assessment_category").is_in(REQUIRED_ASSESSMENTS)),
    "clif_respiratory_support": ("recorded_dttm", ["device_category", "mode_category", "fio2_set", "lpm_set", "tidal_volume_set", "resp_rate_set"],
                                 pl.lit(True)),
    "clif_medication_admin_continuous": ("admin_dttm", ["med_category", "med_dose", "med_dose_unit"],
                                         pl.col("med_category").is_in(set(REQUIRED_MEDS) | set(VASOPRESSORS))),
}

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


def cohort_rows(table: str, time_col: str, cols: list, keep: pl.Expr) -> pl.DataFrame:
    """One CLIF table restricted to cohort encounters (re-keyed to the encounter id), the columns clifpy reads, and the site timezone."""
    path = Path(config["data_directory"]) / f"{table}.{config['filetype']}"
    lf = (pl.scan_parquet(path) if config["filetype"] == "parquet" else pl.scan_csv(path, try_parse_dates=True))
    lf = (lf.select("hospitalization_id", time_col, *cols).with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
          .filter(keep & pl.col("hospitalization_id").is_in(encounter_map["hospitalization_id"].to_list()))
          .join(encounter_map.lazy(), on="hospitalization_id")  # raw hospitalization -> its stitched encounter
          .drop("hospitalization_id").rename({"encounter_id": "hospitalization_id"}))
    return standardize_datetime_columns(lf, target_timezone=tz, target_time_unit="us", datetime_columns=[time_col]).collect()


sofa_in = phi / "sofa_input"
sofa_in.mkdir(exist_ok=True)
flags = windows.select("window_id")
for table, (time_col, cols, keep) in TABLES.items():
    rows = cohort_rows(table, time_col, cols, keep)
    keyed = assign_windows(rows.lazy(), time_col, windows).collect()
    if table == "clif_respiratory_support":  # clifpy forward-fills FiO2 from the 24 h before each window
        lookback = assign_windows(rows.lazy(), time_col, windows.with_columns(start=pl.col("start") - pl.duration(hours=24), end="start"))
        flags = flags.join(keyed.group_by("hospitalization_id").agg(on_imv=(pl.col("device_category").str.to_lowercase() == "imv").any())
                           .rename({"hospitalization_id": "window_id"}), on="window_id", how="left")
        keyed = pl.concat([keyed, lookback.collect()])
    if table == "clif_vitals":  # clifpy takes the last weight before each dose; carry the pre-window one in at window start
        weights = rows.filter(pl.col("vital_category") == "weight_kg").sort("hospitalization_id", time_col)
        carried = (windows.sort("hospitalization_id", "start")
                   .join_asof(weights, left_on="start", right_on=time_col, by="hospitalization_id", strategy="backward", check_sortedness=False)
                   .filter(pl.col("vital_value").is_not_null())
                   .select(hospitalization_id="window_id", recorded_dttm="start", vital_category="vital_category", vital_value="vital_value"))
        keyed = pl.concat([keyed, carried])
    if table == "clif_medication_admin_continuous":
        flags = flags.join(keyed.group_by("hospitalization_id").agg(on_pressor=(pl.col("med_category").is_in(VASOPRESSORS) & (pl.col("med_dose") > 0)).any())
                           .rename({"hospitalization_id": "window_id"}), on="window_id", how="left")
        keyed = keyed.filter(pl.col("med_category").is_in(REQUIRED_MEDS))  # clifpy scores only its four
    keyed.write_parquet(sofa_in / f"{table}.parquet")
    print(f"{table}: {rows.height:,} cohort rows -> {keyed.height:,} in windows")
    del rows, keyed

# one clifpy call per CHUNK_WINDOWS windows bounds its peak memory (its pandas loaders hold a whole table at once);
# a chunk must be large enough to contain every category (clifpy crashes on e.g. a chunk with no spo2)
chunk_id = (pl.int_range(pl.len()) // CHUNK_WINDOWS).clip(upper_bound=max(windows.height // CHUNK_WINDOWS - 1, 0))  # remainder joins the last chunk
sofa = pl.concat([compute_sofa_polars(str(sofa_in), chunk.select(hospitalization_id="window_id", start_dttm="start", end_dttm="end"),
                                      filetype="parquet", fill_na_scores_with_zero=False, remove_outliers=True, timezone=tz)
                  for _, chunk in windows.group_by(chunk_id)], how="diagonal_relaxed")
out = (windows.select("window_id", "hospitalization_id", "day_start")
       .join(sofa.rename({"hospitalization_id": "window_id"}), on="window_id", how="left")
       .join(flags, on="window_id", how="left").with_columns(pl.col("on_imv", "on_pressor").fill_null(False)))
out = rescore_sofa(out.to_pandas())[OUT_COLS]
out.to_parquet(phi / "sofa_landmark.parquet", index=False)

print(f"SOFA for {out['sofa_total'].notna().sum():,} of {len(out):,} patient-days; "
      f"median {out['sofa_total'].median():.0f} (IQR {out['sofa_total'].quantile(.25):.0f}-{out['sofa_total'].quantile(.75):.0f})")
print("missing by component (%):", (out[COMPONENTS].isna().mean() * 100).round(0).to_dict())
print(f"on_imv {out['on_imv'].mean() * 100:.1f}%  on_pressor {out['on_pressor'].mean() * 100:.1f}%  "
      f"(clifpy device_category == imv {out['device_category'].astype(str).str.lower().eq('imv').mean() * 100:.1f}%)")
