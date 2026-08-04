"""Classify ARF subtypes and emit the three deliverables. Run from repo root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from icu_strain import definitions as d
from icu_strain.aggregate import (
    monthly_imv_counts, monthly_subtype_counts, suppress_small_cells, time_to_icu_summary,
)
from icu_strain.physiology import classify_arf, evaluate_hypercapnia, evaluate_hypoxemia
from icu_strain.pipeline import attach_abg_availability, build_observation_frame, first_imv_starts

phi_dir = Path("output/intermediate_phi")
out_dir = Path("output/final_no_phi")
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

# Deliverable 1
imv = first_imv_starts(cohort, resp)
suppress_small_cells(monthly_imv_counts(imv), ["n_new_imv"]).to_csv(
    out_dir / "monthly_imv.csv", index=False)

# Deliverable 2
subtype_cols = ["hypoxemic", "hypercapnic", "mixed", "total"]
suppress_small_cells(monthly_subtype_counts(cohort, classified), subtype_cols).to_csv(
    out_dir / "monthly_arf_subtypes.csv", index=False)

# Deliverable 3.
# Suppress the whole row, not just n: publishing a median for a subtype with
# n=3 while masking n would leak more than the count it hides.
t3 = time_to_icu_summary(cohort, classified)
t3.loc[t3["n"] < d.MIN_CELL_SIZE,
       ["n", "median_hours", "q25_hours", "q75_hours", "pct_before_icu"]] = pd.NA
t3.to_csv(out_dir / "time_to_icu_by_subtype.csv", index=False)

# Attrition, for the report
pd.DataFrame([
    {"step": "adults with index ICU stay in 2024", "n": len(cohort)},
    {"step": "met any ARF criteria in window", "n": len(classified)},
    {"step": "new IMV during index ICU stay", "n": len(imv)},
]).to_csv(out_dir / "attrition.csv", index=False)

print(classified["arf_subtype"].value_counts().to_string())
