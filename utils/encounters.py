"""Encounter stitching: a patient's hospitalizations that follow one another within STITCH_HOURS are one
encounter (an ED encounter and the admission after it; a transfer between a system's hospitals).

Step 01 stitches once and writes the encounter table and the hospitalization -> encounter mapping;
every later step keys on the encounter id, which is the encounter's first hospitalization_id.
"""

import pandas as pd
from clifpy.utils.stitching_encounters import stitch_encounters

from .common import STITCH_HOURS

HOSP_COLS = ["patient_id", "hospitalization_id", "admission_dttm", "discharge_dttm", "age_at_admission",
             "admission_type_category", "discharge_category"]
ADT_COLS = ["hospitalization_id", "in_dttm", "out_dttm", "location_category", "hospital_id"]


def stitch(hosp: pd.DataFrame, adt: pd.DataFrame, hours: float = STITCH_HOURS) -> tuple:
    """Link hospitalizations with clifpy, then collapse each linked group into one encounter.

    clifpy 0.5.0 links a hospitalization to the next one when that admission is within `hours` of its
    discharge, but returns only the link (`encounter_block`): the per-encounter discharge it computes is
    dropped. Returns (encounters, adt, mapping):
      encounters: one row per encounter with the first admission, age and admission type, and the last
        discharge and its category (a transfer's 'Acute Care Hospital' becomes the final 'Expired'); NaT
        when any of its hospitalizations is still open;
      adt: hospitalization_id replaced by the encounter's (each row keeps its own hospital_id);
      mapping: hospitalization_id -> encounter_id, for tables keyed by the raw hospitalization.
    """
    h = hosp.reindex(columns=HOSP_COLS)  # clifpy requires every column; a site without one gets NA
    _, _, links = stitch_encounters(h, adt[ADT_COLS], time_interval=hours)
    h = h.merge(links, on="hospitalization_id").sort_values(["encounter_block", "admission_dttm"])
    h["encounter_id"] = h.groupby("encounter_block")["hospitalization_id"].transform("first")
    first = h.drop_duplicates("encounter_id")  # earliest admission
    last = h.sort_values(["encounter_id", "discharge_dttm"], na_position="last").drop_duplicates("encounter_id", keep="last")
    encounters = (first[["encounter_id", "patient_id", "admission_dttm", "age_at_admission", "admission_type_category"]]
                  .merge(last[["encounter_id", "discharge_dttm", "discharge_category"]], on="encounter_id")
                  .merge(h.groupby("encounter_id").size().rename("n_hospitalizations").reset_index(), on="encounter_id")
                  .rename(columns={"encounter_id": "hospitalization_id"}).reset_index(drop=True))
    mapping = h[["hospitalization_id", "encounter_id"]].reset_index(drop=True)
    ids = adt["hospitalization_id"].map(mapping.set_index("hospitalization_id")["encounter_id"])
    return encounters, adt.assign(hospitalization_id=ids.fillna(adt["hospitalization_id"])), mapping


def death_times(encounters: pd.DataFrame, patient: pd.DataFrame) -> pd.DataFrame:
    """One death time per patient who died (columns patient_id, t).

    An encounter discharged Expired gives its discharge time (precise). Otherwise patient.death_dttm, moved
    to a live discharge in the 24 h after it: a date-only death_dttm sits at midnight, before the discharge
    of the encounter the patient left alive that day, and would otherwise date the death inside the stay.
    """
    expired = encounters["discharge_category"].str.lower().eq("expired")
    in_hosp = encounters[expired].groupby("patient_id", as_index=False)["discharge_dttm"].min().rename(columns={"discharge_dttm": "t"})
    d = (patient[["patient_id", "death_dttm"]].dropna().drop_duplicates("patient_id")
         .loc[lambda x: ~x["patient_id"].isin(in_hosp["patient_id"])].sort_values("death_dttm"))
    alive = encounters.loc[~expired, ["patient_id", "discharge_dttm"]].dropna()
    alive = alive.assign(discharge_dttm=alive["discharge_dttm"].astype(d["death_dttm"].dtype),
                         patient_id=alive["patient_id"].astype(d["patient_id"].dtype)).sort_values("discharge_dttm")
    m = pd.merge_asof(d, alive, left_on="death_dttm", right_on="discharge_dttm", by="patient_id",
                      direction="forward", tolerance=pd.Timedelta(hours=24))
    linked = pd.DataFrame({"patient_id": m["patient_id"], "t": m["discharge_dttm"].fillna(m["death_dttm"])})
    return pd.concat([in_hosp, linked], ignore_index=True)
