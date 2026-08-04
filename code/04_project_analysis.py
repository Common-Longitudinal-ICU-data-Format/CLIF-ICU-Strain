"""Classify ARF subtypes and emit the three deliverables. Run from repo root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from icu_strain.aggregate import (
    monthly_imv_counts, monthly_subtype_counts, suppress_small_cells, suppress_small_rows,
    time_to_icu_summary,
)
from icu_strain.paths import final_dir, phi_dir as phi_dir_path
from icu_strain.physiology import classify_arf, evaluate_hypercapnia, evaluate_hypoxemia
from icu_strain.pipeline import attach_abg_availability, build_observation_frame, first_imv_starts

phi_dir = phi_dir_path()
out_dir = final_dir()
out_dir.mkdir(parents=True, exist_ok=True)

cohort = pd.read_parquet(phi_dir / "cohort.parquet")
labs = pd.read_parquet(phi_dir / "labs_clean.parquet")
vitals = pd.read_parquet(phi_dir / "vitals_clean.parquet")
resp = pd.read_parquet(phi_dir / "resp_support_waterfall.parquet")

obs = build_observation_frame(cohort, labs, vitals, resp)
obs = attach_abg_availability(obs, labs, cohort)

hypox = evaluate_hypoxemia(obs)

labs_win = labs.merge(cohort[["hospitalization_id", "window_start", "window_end"]], on="hospitalization_id")
labs_win = labs_win[
    (labs_win["lab_result_dttm"] >= labs_win["window_start"])
    & (labs_win["lab_result_dttm"] <= labs_win["window_end"])
]
hypercap = evaluate_hypercapnia(labs_win)

classified = classify_arf(hypox, hypercap)
classified.to_parquet(phi_dir / "arf_classified.parquet", index=False)

# Deliverable 1.
# Two columns, not one: the waterfall forward-fills device state, so a
# patient arriving already intubated from the ED/OR shows up as IMV at ICU
# admission and would be miscounted as a new initiation by a single column.
# n_imv_any: any IMV during the index ICU stay (includes carried-over
# ventilation). n_imv_new: a genuine in-stay transition into IMV. The gap
# between the two is the already-ventilated-on-arrival population.
imv = first_imv_starts(cohort, resp)

imv_any = imv[["hospitalization_id", "imv_any_start_dttm"]].rename(
    columns={"imv_any_start_dttm": "imv_start_dttm"})
imv_new = imv.dropna(subset=["imv_new_start_dttm"])[["hospitalization_id", "imv_new_start_dttm"]].rename(
    columns={"imv_new_start_dttm": "imv_start_dttm"})

monthly_imv_any = monthly_imv_counts(imv_any).rename(columns={"n_new_imv": "n_imv_any"})
monthly_imv_new = monthly_imv_counts(imv_new).rename(columns={"n_new_imv": "n_imv_new"})
monthly_imv = monthly_imv_any.merge(monthly_imv_new, on="month", how="outer")
monthly_imv[["n_imv_any", "n_imv_new"]] = monthly_imv[["n_imv_any", "n_imv_new"]].fillna(0).astype("int64")
monthly_imv = monthly_imv.sort_values("month").reset_index(drop=True)

suppress_small_cells(monthly_imv, ["n_imv_any", "n_imv_new"]).to_csv(
    out_dir / "monthly_imv.csv", index=False)

# Deliverable 2
subtype_cols = ["hypoxemic", "hypercapnic", "mixed", "total"]
suppress_small_cells(monthly_subtype_counts(cohort, classified), subtype_cols).to_csv(
    out_dir / "monthly_arf_subtypes.csv", index=False)

# Deliverable 3.
# Suppress the whole row, not just n: publishing a median for a subtype with
# n=3 while masking n would leak more than the count it hides.
# suppress_small_rows also keeps n as a real (nullable Int64) integer, unlike
# a plain `.loc[mask, cols] = pd.NA`, which upcasts n to float64 even when
# nothing is suppressed (a published n of 431 would round-trip as "431.0").
t3 = time_to_icu_summary(cohort, classified)
value_cols = ["median_hours", "q25_hours", "q75_hours", "pct_before_icu"]
suppress_small_rows(t3, "n", value_cols).to_csv(out_dir / "time_to_icu_by_subtype.csv", index=False)

# Attrition, for the report. "any" vs "new" IMV split out explicitly so
# neither number can be misread as the other.
pd.DataFrame([
    {"step": "adults with index ICU stay in 2024", "n": len(cohort)},
    {"step": "met any ARF criteria in window", "n": len(classified)},
    {"step": "any IMV during index ICU stay (includes already-intubated-on-arrival)", "n": len(imv_any)},
    {"step": "new IMV initiation during index ICU stay (excludes already-intubated-on-arrival)",
     "n": len(imv_new)},
]).to_csv(out_dir / "attrition.csv", index=False)

print(classified["arf_subtype"].value_counts().to_string())
