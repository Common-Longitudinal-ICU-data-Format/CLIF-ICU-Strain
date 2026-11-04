"""Step 3: landmark rows (design v3, E2; v4 adds demand-side exposures and process outcomes). Run after 01.

One row per ICU stay per local 7am s the patient is already in a unit at: that unit's admissions by
other patients over [s, s+24h) as a share of capacity (exposure), in-hospital death or hospice
discharge over [s+24h, s+8d) (outcome), and covariates measured at or before s. v5 adds the acuity-weighted
admissions and census (respiratory_support, medication_admin_continuous and crrt_therapy, read with polars).

Reads patient, hospital_diagnosis, code_status, and step 1's intermediates, including the stitched
encounters (one row per encounter, its final discharge category and time) in place of hospitalization. Writes
  intermediate_phi/<site>/patient_days.parquet
  <site>_upload_to_box/attrition.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import polars as pl
from clifpy import ClifOrchestrator, calculate_cci

from utils import common as c
from utils.patient_days import (add_exposure, add_outcomes, add_present_exposure, add_weighted_exposure, assign_unit, covid_hospitalizations,
                                expand_patient_days, icu_cohort, unit_presence)
from utils.support import KINDS, support_events, support_in, weights

config = c.load_config()
site = config["site_name"]
phi, final = c.phi_dir(site), c.final_dir(site)
start, end = c.study_window()

co = ClifOrchestrator(data_directory=config["data_directory"], filetype=config["filetype"], timezone=config["timezone"])
co.initialize(tables=["patient"], columns={"patient": ["patient_id", "sex_category", "death_dttm"]})
hosp, patient = pd.read_parquet(phi / "encounters.parquet"), co.patient.df  # stitched in step 1
encounter_of = pd.read_parquet(phi / "encounter_map.parquet").set_index("hospitalization_id")["encounter_id"]
stays = pd.read_parquet(phi / "unit_stays.parquet")
blocks = pd.read_parquet(phi / "icu_blocks.parquet")
hosp_daily, unit_daily = pd.read_parquet(phi / "daily_by_hospital.parquet"), pd.read_parquet(phi / "daily_by_unit.parquet")

cohort = icu_cohort(hosp, blocks, start, end)
steps = [("adult ICU stays in study window", cohort)]
days = expand_patient_days(cohort, hosp_daily)
steps.append(("with a 7am after census burn-in", days))
days = assign_unit(days, stays)
days = days[days["incumbent"]]  # in the unit before 7am s; the arrival day (after s) is not a landmark
steps.append(("landmarks: in the unit before 7am s", days))
days = add_exposure(days, stays, unit_daily, hosp_daily).dropna(subset=["admit_strain"])  # no capacity (D6) or no next day
steps.append(("with unit admissions over [s, s+24h) and capacity", days))

# v5: acuity weights (1 + IMV + vasopressor + CRRT) for every adult ICU unit stay, over its first SUPPORT_HOURS (an
# arrival's weight) and over the SUPPORT_HOURS before each 7am it holds (a present patient's weight)
emap = (pl.read_parquet(phi / "encounter_map.parquet").with_columns(pl.all().cast(pl.Utf8))
        .filter(pl.col("encounter_id").is_in(stays["hospitalization_id"].astype(str).unique().tolist())))
events = support_events(config, emap)
span = pd.Timedelta(hours=c.SUPPORT_HOURS)
arrive_end = (stays["in_dttm"] + span).where(stays["in_dttm"] + span < stays["out_dttm"], stays["out_dttm"])
stays = support_in(stays.assign(_end=arrive_end), events, "in_dttm", "_end")
stays["w_arrival"] = weights(stays)
presence = unit_presence(stays, unit_daily)
present_start = (presence["day_start"] - span).where(presence["day_start"] - span > presence["in_dttm"], presence["in_dttm"])
presence = support_in(presence.assign(_start=present_start), events, "_start", "day_start")
presence["w_present"] = weights(presence)
days = add_weighted_exposure(days, stays, presence, unit_daily)
print("support in the first 6 h of a unit stay (% of stays): " + ", ".join(f"{k} {100 * stays[k].mean():.1f}" for k in KINDS)
      + f"; weighted / unweighted admissions in the exposure window {days['wadmits'].sum() / max(days['admits'].sum(), 1):.2f}")
days = add_present_exposure(days, stays, unit_daily)  # v6: admissions while the patient is there, as a rate
left = days["stay_out"] < days["day_end"]
print(f"left the unit inside the exposure window: {100 * left.mean():.0f}% of landmarks, "
      f"median {days.loc[left, 'hours_present'].median():.1f} h present")

try:  # v4 process outcome: new limitation orders; a site without the table gets NA
    co.load_table("code_status", columns=["patient_id", "start_dttm", "code_status_category"],
                  filters={"patient_id": cohort["patient_id"].unique().tolist()})
    code_status = co.code_status.df
except Exception as e:
    print(f"code_status not loaded ({type(e).__name__}): new_limitation_48h will be NA")
    code_status = None
days = add_outcomes(days, hosp, patient, blocks, code_status)
last = hosp_daily["day_start"].max()
# Every incumbent stays in the table (the unit's workload at s counts them all); model rows are flagged, not dropped
days["in_window"] = days["day_start"] + pd.Timedelta(days=c.OUTCOME_DAYS + 1) <= last  # outcome window inside the data
days.loc[days["day_start"] + pd.Timedelta(days=c.OUTCOME_DAYS_LONG + 1) > last, "hdhd_30d"] = np.nan
steps.append(("no death or hospice discharge inside the exposure window", days[days["eligible"]]))
steps.append(("outcome window inside the data", days[days["eligible"] & days["in_window"]]))

dx_cols = ["hospitalization_id", "diagnosis_code", "diagnosis_code_format"]
ids = encounter_of[encounter_of.isin(cohort["hospitalization_id"])].index.tolist()  # every hospitalization of a cohort encounter
try:
    co.load_table("hospital_diagnosis", columns=dx_cols + ["poa_present"], filters={"hospitalization_id": ids})
except Exception:  # site without the column
    co.load_table("hospital_diagnosis", columns=dx_cols, filters={"hospitalization_id": ids})
dx = co.hospital_diagnosis.df
dx = dx.assign(hospitalization_id=dx["hospitalization_id"].map(encounter_of))  # Charlson and COVID per encounter
poa = (dx["poa_present"].astype(str).str.lower().isin(["1", "1.0", "true", "y", "yes"]) if "poa_present" in dx
       else pd.Series(False, index=dx.index))
use_poa = poa.mean() >= c.MIN_POA_SHARE  # Charlson and COVID from codes present on admission (v3); else every code
if use_poa:
    dx = dx[poa]
icd10 = dx["diagnosis_code_format"].astype(str).str.lower().eq("icd10cm").mean()  # clifpy's calculate_cci reads only these
codes_note = f"{'present-on-admission' if use_poa else 'all'} codes ({100 * poa.mean():.0f}% flagged POA, {100 * icd10:.0f}% ICD10CM)"
print(f"Charlson and COVID from {codes_note}")
if icd10 < 0.5:
    print("WARNING: most diagnosis codes are not in ICD10CM format; Charlson will be 0 for most stays.")
cci = calculate_cci(dx)[["hospitalization_id", "cci_score"]]
days = (days.merge(patient[["patient_id", "sex_category"]].drop_duplicates("patient_id"), on="patient_id", how="left")
            .merge(cci, on="hospitalization_id", how="left"))
days["cci_score"] = days["cci_score"].fillna(0)  # no coded comorbidity
days["covid"] = days["hospitalization_id"].isin(covid_hospitalizations(dx)).astype(int)
local = days["day_start"].dt.tz_localize(None)
days["year"], days["month"], days["dow"] = local.dt.year, local.dt.month, local.dt.dayofweek
days["icu_day_cap"] = days["icu_day"].clip(upper=c.ICU_DAY_CAP)
days.to_parquet(phi / "patient_days.parquet", index=False)

model_rows = days[days["eligible"] & days["in_window"]]
death, hdhd, any_ = (model_rows[k].astype(bool) for k in ["death_7d", "hdhd_7d", "death_any_7d"])
attrition = pd.DataFrame([{"step": s, "icu_stays": d["stay_id"].nunique(), "patients": d["patient_id"].nunique(),
                           "landmarks": len(d) if "day_start" in d else pd.NA} for s, d in steps])
attrition = pd.concat([attrition, pd.DataFrame([
    {"step": "landmarks with death or hospice discharge in [s+24h, s+8d)", "landmarks": int(hdhd.sum())},
    {"step": "  in-hospital death", "landmarks": int(death.sum())},
    {"step": "  hospice discharge without in-hospital death", "landmarks": int((hdhd & ~death).sum())},
    {"step": "landmarks with a linked death (patient.death_dttm) but no in-hospital death in the window",
     "landmarks": int((any_ & ~death).sum())},
    {"step": "landmarks with ICU discharge inside the exposure window", "landmarks": int(model_rows["icu_discharge_24h"].sum())},
    {"step": "  readmitted to an ICU within 72h of that discharge", "landmarks": int(model_rows["readmit_72h"].sum())},
    {"step": "  discharged between 18:00 and 06:00", "landmarks": int(model_rows["after_hours_discharge"].sum())},
    {"step": "landmarks full code at s (last order before s is not a limitation)",
     "landmarks": int(model_rows["new_limitation_48h"].notna().sum())},
    {"step": "  with a new limitation order within 48h", "landmarks": int(model_rows["new_limitation_48h"].sum())},
    {"step": f"Charlson and COVID from {codes_note}"},  # so the coordinating site can see each site's covariate source
])]).astype({"icu_stays": "Int64", "patients": "Int64", "landmarks": "Int64"})
c.suppress_small_cells(attrition, ["icu_stays", "patients", "landmarks"]).to_csv(final / "attrition.csv", index=False)
print(attrition.to_string(index=False))
