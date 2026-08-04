"""Apply clifpy outlier handling and the respiratory support waterfall.

Must run before criteria evaluation: an SpO2 of 0 or PaO2 of 9999 would
otherwise manufacture false hypoxemia.

Loads labs / vitals / respiratory_support filtered to this project's cohort
hospitalization_ids (the `filters` param pushed into the DuckDB scan) rather
than the full site tables. The source parquets span 2018-2024 site-wide --
roughly 7x more rows than the 2024-only, ~6k-hospitalization cohort needs --
and process_resp_support_waterfall builds an hourly scaffold per encounter,
so running it unfiltered exhausted memory for no analytic benefit.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from clifpy import ClifOrchestrator, apply_outlier_handling, process_resp_support_waterfall

from icu_strain import definitions as d
from icu_strain.pipeline import restore_device_category_case

config = json.loads(Path("config/config.json").read_text())
phi_dir = Path("output/intermediate_phi")
out_dir = Path("output/final_no_phi")
out_dir.mkdir(parents=True, exist_ok=True)

cohort = pd.read_parquet(phi_dir / "cohort.parquet")
cohort_ids = cohort["hospitalization_id"].unique().tolist()

co = ClifOrchestrator(
    data_directory=config["data_directory"],
    filetype=config["filetype"],
    timezone=config["timezone"],
)
co.initialize(
    tables=["labs", "vitals", "respiratory_support"],
    # labs/vitals: trim to the columns outlier handling + downstream analysis
    # actually use. respiratory_support keeps every column -- the waterfall's
    # device/mode heuristics read several optional setter columns.
    columns={
        "labs": ["hospitalization_id", "lab_category", "lab_result_dttm", "lab_value_numeric"],
        "vitals": ["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"],
    },
    filters={
        "labs": {"hospitalization_id": cohort_ids},
        "vitals": {"hospitalization_id": cohort_ids},
        "respiratory_support": {"hospitalization_id": cohort_ids},
    },
)

before = {
    "labs": co.labs.df["lab_value_numeric"].notna().sum(),
    "vitals": co.vitals.df["vital_value"].notna().sum(),
    "respiratory_support": co.respiratory_support.df["fio2_set"].notna().sum(),
}

for tbl in (co.labs, co.vitals, co.respiratory_support):
    apply_outlier_handling(tbl)

after = {
    "labs": co.labs.df["lab_value_numeric"].notna().sum(),
    "vitals": co.vitals.df["vital_value"].notna().sum(),
    "respiratory_support": co.respiratory_support.df["fio2_set"].notna().sum(),
}

# NOTE: these counts are scoped to the cohort's hospitalizations (2024 index
# ICU stays), not the full multi-year site table -- see module docstring.
pd.DataFrame([
    {"table": k, "values_before": before[k], "values_after": after[k],
     "values_removed": before[k] - after[k],
     "scope": "cohort hospitalization_ids only, not the full site table"}
    for k in before
]).to_csv(out_dir / "outlier_summary.csv", index=False)

waterfall = process_resp_support_waterfall(co.respiratory_support.df)
# clifpy's waterfall lower-cases device_category on input and never restores
# it, so downstream exact-match comparisons against d.IMV_DEVICE ("IMV") and
# d.ROOM_AIR_DEVICE ("Room Air") would silently match zero rows otherwise.
waterfall = restore_device_category_case(waterfall)
waterfall.to_parquet(phi_dir / "resp_support_waterfall.parquet", index=False)

# Only the categories the analysis reads: arterial gases for labs, spo2 for
# vitals. Carrying the rest into the intermediate parquets is pure cost.
arterial = {d.ABG_PO2, d.ABG_PCO2, d.ABG_PH}
labs_out = co.labs.df[co.labs.df["lab_category"].isin(arterial)]
vitals_out = co.vitals.df[co.vitals.df["vital_category"] == d.SPO2_VITAL]
labs_out.to_parquet(phi_dir / "labs_clean.parquet", index=False)
vitals_out.to_parquet(phi_dir / "vitals_clean.parquet", index=False)

print(f"waterfall rows: {len(waterfall):,}; "
      f"device NULL rate after: {100.0 * waterfall['device_category'].isna().mean():.1f}%")
