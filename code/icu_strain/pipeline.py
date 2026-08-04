"""Loading, waterfall processing, and observation assembly."""

import duckdb
import pandas as pd

from . import definitions as d
from .physiology import normalize_fio2

# CLIF 2.1 mCIDE vocabulary for respiratory_support.device_category.
# clifpy.process_resp_support_waterfall lower-cases device_category on input
# and never restores it (it hard-codes "imv", "room air", etc. throughout),
# so exact-match comparisons downstream against d.IMV_DEVICE ("IMV") and
# d.ROOM_AIR_DEVICE ("Room Air") would silently match zero rows unless the
# original casing is restored first. See restore_device_category_case.
_DEVICE_CATEGORY_CASE = {
    v.lower(): v
    for v in (
        "IMV", "NIPPV", "CPAP", "High Flow NC", "Face Mask",
        "Trach Collar", "Nasal Cannula", "Room Air", "Other",
    )
}


def restore_device_category_case(waterfall: pd.DataFrame) -> pd.DataFrame:
    """Undo process_resp_support_waterfall's unconditional device_category lower-casing."""
    out = waterfall.copy()
    out["device_category"] = out["device_category"].str.lower().map(_DEVICE_CATEGORY_CASE)
    return out


def build_observation_frame(cohort, labs, vitals, resp) -> pd.DataFrame:
    """One row per lab/vital observation inside the window, with support state.

    ASOF JOIN attaches the most recent prior waterfall row, which is how the
    forward-filled device and FiO2 reach each observation.
    """
    con = duckdb.connect()
    con.register("cohort", cohort)
    con.register("labs", labs)
    con.register("vitals", vitals)
    con.register("resp", resp)

    obs = con.execute(f"""
        WITH events AS (
            SELECT hospitalization_id, recorded_dttm AS dttm,
                   CASE WHEN vital_category = '{d.SPO2_VITAL}' THEN vital_value END AS spo2,
                   NULL::DOUBLE AS po2
            FROM vitals WHERE vital_category = '{d.SPO2_VITAL}'
            UNION ALL
            SELECT hospitalization_id, lab_result_dttm AS dttm,
                   NULL::DOUBLE AS spo2,
                   CASE WHEN lab_category = '{d.ABG_PO2}' THEN lab_value_numeric END AS po2
            FROM labs WHERE lab_category = '{d.ABG_PO2}'
        ),
        windowed AS (
            SELECT e.* FROM events e
            JOIN cohort c USING (hospitalization_id)
            WHERE e.dttm >= c.window_start AND e.dttm <= c.window_end
        )
        SELECT w.hospitalization_id, w.dttm, w.spo2, w.po2,
               r.device_category, r.fio2_set AS fio2
        FROM windowed w
        ASOF LEFT JOIN resp r
          ON w.hospitalization_id = r.hospitalization_id
         AND w.dttm >= r.recorded_dttm
        ORDER BY w.hospitalization_id, w.dttm
    """).df()
    con.close()

    obs["fio2"] = normalize_fio2(obs["fio2"])

    # DuckDB's .df() round-trip re-localizes tz-aware timestamps to its own
    # IANA alias (e.g. "America/Chicago") regardless of the input columns'
    # tz naming (e.g. pandas/pytz "US/Central") -- same instant, different tz
    # dtype object. classify_arf later takes min(axis=1) across a column
    # built from this dttm and one built straight from cohort/labs/vitals
    # timestamps; mismatched tz dtypes make pandas fall back to an
    # elementwise path that raises comparing a Timestamp to a float.
    # Normalizing back to the cohort's own tz keeps the instant identical
    # and the dtype consistent with the rest of the pipeline.
    cohort_tz = cohort["window_start"].dt.tz if len(cohort) else None
    if cohort_tz is not None and obs["dttm"].dt.tz is not None:
        obs["dttm"] = obs["dttm"].dt.tz_convert(cohort_tz)

    return obs


def attach_abg_availability(obs: pd.DataFrame, labs: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Add hospitalization-level `has_abg`, gating the S/F fallback."""
    arterial = {d.ABG_PO2, d.ABG_PCO2, d.ABG_PH}
    windowed = labs.merge(
        cohort[["hospitalization_id", "window_start", "window_end"]], on="hospitalization_id"
    )
    in_window = (
        (windowed["lab_result_dttm"] >= windowed["window_start"])
        & (windowed["lab_result_dttm"] <= windowed["window_end"])
        & (windowed["lab_category"].isin(arterial))
    )
    ids = set(windowed[in_window]["hospitalization_id"])
    out = obs.copy()
    out["has_abg"] = out["hospitalization_id"].isin(ids)
    return out


def first_imv_starts(cohort: pd.DataFrame, resp: pd.DataFrame) -> pd.DataFrame:
    """Earliest IMV observation inside each hospitalization's index ICU stay."""
    df = resp[resp["device_category"] == d.IMV_DEVICE].merge(
        cohort[["hospitalization_id", "icu_admit_dttm", "icu_discharge_dttm"]],
        on="hospitalization_id", how="inner",
    )
    inside = (df["recorded_dttm"] >= df["icu_admit_dttm"]) & (df["recorded_dttm"] <= df["icu_discharge_dttm"])
    return (
        df[inside].groupby("hospitalization_id", as_index=False)["recorded_dttm"]
        .min().rename(columns={"recorded_dttm": "imv_start_dttm"})
    )
