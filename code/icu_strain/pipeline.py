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
    """Undo process_resp_support_waterfall's unconditional device_category lower-casing.

    Falls back to the original (lower-cased) value for anything outside the
    known CLIF 2.1 vocabulary, rather than dropping it to NaN. `.map()` alone
    returns NaN for an unmapped key, which would silently zero out every
    downstream exact-match comparison for that value -- the exact failure
    this function exists to prevent, just moved one level down (a CLIF 3.0
    site's "hfnc"/"t_piece" vocabulary, unmapped here, would otherwise vanish
    to NaN entirely instead of merely failing to match IMV/Room Air).
    Preserving the original string keeps the signal inspectable for
    debugging. Raising was rejected: aborting a multi-million-row batch
    pipeline over one stray/unrecognized value is worse than continuing with
    an honestly-unmapped-but-visible value; a future CLIF-version-aware
    vocabulary map is a separate concern from this function's one job
    (restore CLIF 2.1 casing for CLIF 2.1 values).
    """
    out = waterfall.copy()
    lowered = out["device_category"].str.lower()
    out["device_category"] = lowered.map(_DEVICE_CATEGORY_CASE).fillna(out["device_category"])
    return out


def _dedupe_resp_timestamps(resp: pd.DataFrame) -> pd.DataFrame:
    """Collapse resp rows that tie on (hospitalization_id, recorded_dttm).

    process_resp_support_waterfall's own final dedup is a full-row
    `drop_duplicates()`, which can still leave rows that agree on
    (hospitalization_id, recorded_dttm) but differ in some other column it
    carries. Both build_observation_frame's ASOF JOIN and
    first_imv_starts' within-stay prev-state shift need exactly one row to
    represent a hospitalization's device state at a given instant; left
    undeduplicated, which literal row wins depends on scan/merge order,
    which is not guaranteed stable across runs -- the same class of bug
    fixed in physiology.evaluate_hypercapnia's merge_asof. Sorting on the
    columns that actually drive downstream logic (device_category,
    fio2_set) before keeping the first row makes the choice reproducible
    regardless of upstream scan order. At every tie observed on one real
    site's data, device_category and fio2_set were already identical
    between the duplicates (so this is a no-op there); this guards the
    general, multi-site case where they might not be.
    """
    if resp.empty:
        return resp
    key = ["hospitalization_id", "recorded_dttm"]
    return (
        resp.sort_values([*key, "device_category", "fio2_set"], na_position="first", kind="stable")
        .drop_duplicates(subset=key, keep="first")
        .reset_index(drop=True)
    )


def build_observation_frame(cohort, labs, vitals, resp) -> pd.DataFrame:
    """One row per lab/vital observation inside the window, with support state.

    ASOF JOIN attaches the most recent prior waterfall row, which is how the
    forward-filled device and FiO2 reach each observation.
    """
    resp = _dedupe_resp_timestamps(resp)

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
    """Earliest within-stay IMV state, split into 'any' vs genuinely 'new'.

    process_resp_support_waterfall forward-fills device_category, so a
    patient already intubated in the ED/OR before ICU admission shows up as
    IMV starting at their very first in-stay respiratory-support row -- that
    is a carried-over state, not a new intubation. Deliverable 1 is read as
    an incidence rate, so the two must not be conflated into one column.

    imv_any_start_dttm: earliest in-stay row with device_category == IMV,
    regardless of what preceded it. Includes patients already ventilated on
    arrival.
    imv_new_start_dttm: earliest in-stay row with device_category == IMV
    whose immediately preceding in-stay row was NOT IMV -- a genuine
    transition. NaT when the patient's very first in-stay row is already
    IMV: there is no preceding in-stay row to compare against, so "new" is
    undetermined and conservatively excluded (they still count in
    imv_any_start_dttm).
    """
    resp = _dedupe_resp_timestamps(resp)
    df = resp.merge(
        cohort[["hospitalization_id", "icu_admit_dttm", "icu_discharge_dttm"]],
        on="hospitalization_id", how="inner",
    )
    inside = (df["recorded_dttm"] >= df["icu_admit_dttm"]) & (df["recorded_dttm"] <= df["icu_discharge_dttm"])
    df = df[inside].sort_values(["hospitalization_id", "recorded_dttm"])
    df["prev_device_category"] = df.groupby("hospitalization_id")["device_category"].shift()

    imv = df[df["device_category"] == d.IMV_DEVICE]
    any_start = (
        imv.groupby("hospitalization_id", as_index=False)["recorded_dttm"]
        .min().rename(columns={"recorded_dttm": "imv_any_start_dttm"})
    )

    is_new_intubation = imv["prev_device_category"].notna() & (imv["prev_device_category"] != d.IMV_DEVICE)
    new_start = (
        imv[is_new_intubation].groupby("hospitalization_id", as_index=False)["recorded_dttm"]
        .min().rename(columns={"recorded_dttm": "imv_new_start_dttm"})
    )

    return any_start.merge(new_start, on="hospitalization_id", how="left")
