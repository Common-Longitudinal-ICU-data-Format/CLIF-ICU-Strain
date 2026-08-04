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

# Defined here (not down near labs_out/vitals_out) because it's also needed
# to compute the arterial/spo2-scoped before/after stats below.
arterial = {d.ABG_PO2, d.ABG_PCO2, d.ABG_PH}


def _notna_count(df, value_col, category_col=None, category=None):
    if category_col is not None:
        df = df[df[category_col].isin(category) if isinstance(category, set) else df[category_col] == category]
    return df[value_col].notna().sum()


before = {
    "labs": _notna_count(co.labs.df, "lab_value_numeric"),
    "vitals": _notna_count(co.vitals.df, "vital_value"),
    "respiratory_support": _notna_count(co.respiratory_support.df, "fio2_set"),
    "labs_arterial_only": _notna_count(co.labs.df, "lab_value_numeric", "lab_category", arterial),
    "vitals_spo2_only": _notna_count(co.vitals.df, "vital_value", "vital_category", d.SPO2_VITAL),
}

for tbl in (co.labs, co.vitals, co.respiratory_support):
    apply_outlier_handling(tbl)

after = {
    "labs": _notna_count(co.labs.df, "lab_value_numeric"),
    "vitals": _notna_count(co.vitals.df, "vital_value"),
    "respiratory_support": _notna_count(co.respiratory_support.df, "fio2_set"),
    "labs_arterial_only": _notna_count(co.labs.df, "lab_value_numeric", "lab_category", arterial),
    "vitals_spo2_only": _notna_count(co.vitals.df, "vital_value", "vital_category", d.SPO2_VITAL),
}

# Scope is explicit per row: the plain "labs"/"vitals" rows span every
# category in the cohort-filtered table, not just the 3 arterial lab
# categories and 1 spo2 vital category that actually survive into
# labs_clean.parquet/vitals_clean.parquet and reach the ARF criteria -- e.g.
# "338 nullified out of 3.17M labs" alone says nothing about how many
# ARTERIAL values were removed. The *_only rows answer that directly.
_SCOPE = {
    "labs": "cohort hospitalization_ids, ALL lab categories (not just the 3 arterial categories used downstream)",
    "vitals": "cohort hospitalization_ids, ALL vital categories (not just spo2, the only one used downstream)",
    "respiratory_support": "cohort hospitalization_ids, not the full multi-year site table",
    "labs_arterial_only": "cohort hospitalization_ids, restricted to po2_arterial/pco2_arterial/ph_arterial -- exactly what labs_clean.parquet contains",
    "vitals_spo2_only": "cohort hospitalization_ids, restricted to spo2 -- exactly what vitals_clean.parquet contains",
}
pd.DataFrame([
    {"table": k, "values_before": before[k], "values_after": after[k],
     "values_removed": before[k] - after[k], "scope": _SCOPE[k]}
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
labs_out = co.labs.df[co.labs.df["lab_category"].isin(arterial)]
vitals_out = co.vitals.df[co.vitals.df["vital_category"] == d.SPO2_VITAL]
labs_out.to_parquet(phi_dir / "labs_clean.parquet", index=False)
vitals_out.to_parquet(phi_dir / "vitals_clean.parquet", index=False)

print(f"waterfall rows: {len(waterfall):,}; "
      f"device NULL rate after: {100.0 * waterfall['device_category'].isna().mean():.1f}%")
