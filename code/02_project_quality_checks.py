"""Per-month data availability. Run from repo root."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from clifpy import ClifOrchestrator

from icu_strain import definitions as d

config = json.loads(Path("config/config.json").read_text())
out_dir = Path("output/final_no_phi")
out_dir.mkdir(parents=True, exist_ok=True)

cohort = pd.read_parquet("output/intermediate_phi/cohort.parquet")
cohort_ids = cohort["hospitalization_id"].unique().tolist()

# Push the hospitalization_id filter into the DuckDB scan (clifpy's `filters`
# param) so labs/vitals/respiratory_support are never materialized beyond the
# cohort: the source tables span 2018-2024 site-wide, ~7x more than this
# 2024-only, 6k-hospitalization cohort needs.
co = ClifOrchestrator(
    data_directory=config["data_directory"],
    filetype=config["filetype"],
    timezone=config["timezone"],
)
co.initialize(
    tables=["labs", "vitals", "respiratory_support"],
    columns={
        # lab_value_numeric is needed here (beyond category/dttm) to tell a
        # benign exact-duplicate row apart from a genuine value conflict --
        # see the dup-key quality metrics below.
        "labs": ["hospitalization_id", "lab_category", "lab_result_dttm", "lab_value_numeric"],
        "vitals": ["hospitalization_id", "recorded_dttm", "vital_category"],
        "respiratory_support": ["hospitalization_id", "recorded_dttm", "fio2_set", "device_category"],
    },
    filters={
        "labs": {"hospitalization_id": cohort_ids},
        "vitals": {"hospitalization_id": cohort_ids},
        "respiratory_support": {"hospitalization_id": cohort_ids},
    },
)

arterial = {d.ABG_PO2, d.ABG_PCO2, d.ABG_PH}
labs, vitals, resp = co.labs.df, co.vitals.df, co.respiratory_support.df

cohort["month"] = cohort["icu_admit_dttm"].dt.to_period("M").astype(str)


def _in_window(df, dttm_col, ids_col="hospitalization_id"):
    m = df.merge(cohort[[ids_col, "window_start", "window_end", "month"]], on=ids_col)
    return m[(m[dttm_col] >= m["window_start"]) & (m[dttm_col] <= m["window_end"])]


abg_ids = _in_window(labs[labs["lab_category"].isin(arterial)], "lab_result_dttm")
spo2_ids = _in_window(vitals[vitals["vital_category"] == d.SPO2_VITAL], "recorded_dttm")
fio2_ids = _in_window(resp[resp["fio2_set"].notna()], "recorded_dttm")

rows = []
for month, grp in cohort.groupby("month"):
    n = len(grp)
    ids = set(grp["hospitalization_id"])
    rows.append({
        "month": month,
        "n_hospitalizations": n,
        "pct_with_abg": 100.0 * len(ids & set(abg_ids["hospitalization_id"])) / n,
        "pct_with_spo2": 100.0 * len(ids & set(spo2_ids["hospitalization_id"])) / n,
        "pct_with_fio2": 100.0 * len(ids & set(fio2_ids["hospitalization_id"])) / n,
    })

qc = pd.DataFrame(rows)
# Named "_cohort", not "_raw": this is the null rate over the cohort-filtered
# respiratory_support scan (this driver's `filters` push-down), not the raw
# site-wide table. A consortium consumer pooling this column across sites
# would otherwise compare a cohort-conditional rate here against another
# site's site-wide rate under the same column name.
qc["device_null_rate_cohort"] = 100.0 * resp["device_category"].isna().mean()

# Duplicate-key quality signal, overall (not per-month): what fraction of
# cohort-scoped arterial lab rows (po2_arterial/pco2_arterial/ph_arterial)
# share a (hospitalization_id, lab_result_dttm, lab_category) key with
# another row. Reported as two SEPARATE rates because they are different
# problems: an identical-value duplicate is a benign redundant record, but a
# differing-value duplicate is a genuine conflict that physiology.
# evaluate_hypercapnia must arbitrarily-but-deterministically resolve (see
# its merge_asof tie-break). This rate determines whether this site's
# hypercapnic/mixed counts are comparable to a site without the same
# duplication -- collapsing the two into one number would hide which
# problem (if either) a given site actually has.
dup_key = ["hospitalization_id", "lab_result_dttm", "lab_category"]
arterial_labs = labs[labs["lab_category"].isin(arterial)]
n_arterial = len(arterial_labs)
if n_arterial:
    group_size = arterial_labs.groupby(dup_key)["lab_value_numeric"].transform("size")
    group_nunique = arterial_labs.groupby(dup_key)["lab_value_numeric"].transform("nunique")
    is_conflicting_dup = (group_size > 1) & (group_nunique > 1)
    is_identical_dup = (group_size > 1) & (group_nunique == 1)
    pct_conflicting = 100.0 * is_conflicting_dup.sum() / n_arterial
    pct_identical = 100.0 * is_identical_dup.sum() / n_arterial
else:
    pct_conflicting = pct_identical = 0.0
qc["pct_arterial_dup_key_conflicting_overall"] = pct_conflicting
qc["pct_arterial_dup_key_identical_overall"] = pct_identical

qc.to_csv(out_dir / "quality_checks.csv", index=False)
print(qc.to_string(index=False))
