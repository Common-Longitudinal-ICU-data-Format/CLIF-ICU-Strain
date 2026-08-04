import pandas as pd
from icu_strain.pipeline import (
    attach_abg_availability,
    build_observation_frame,
    first_imv_starts,
    restore_device_category_case,
)


def test_observation_frame_attaches_most_recent_support_state():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "window_start": pd.to_datetime(["2024-01-01 00:00"]),
        "window_end": pd.to_datetime(["2024-01-02 00:00"]),
    })
    vitals = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 05:00"]),
        "vital_category": ["spo2"],
        "vital_value": [88.0],
    })
    labs = pd.DataFrame(columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"])
    resp = pd.DataFrame({
        "hospitalization_id": ["H1", "H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 04:00"]),
        "device_category": ["Nasal Cannula", "Room Air"],
        "fio2_set": [0.40, 0.21],
    })
    out = build_observation_frame(cohort, labs, vitals, resp)
    row = out[out["spo2"].notna()].iloc[0]
    assert row["device_category"] == "Room Air"   # 04:00 state, not 00:00
    assert row["fio2"] == 0.21


def test_resp_duplicate_timestamp_tie_resolves_deterministically():
    # Two resp rows tied at the exact same (hospitalization_id,
    # recorded_dttm) but disagreeing on device_category -- the same class of
    # raw-data artifact found in real arterial labs (see
    # test_exact_timestamp_tie_resolves_deterministically_by_value in
    # test_physiology_criteria.py), here guarded against in respiratory_support
    # too since the ASOF JOIN below needs exactly one row to represent state
    # at a given instant. Without a deterministic tiebreak, which row DuckDB's
    # ASOF JOIN attaches could depend on scan order. Both input orderings of
    # the tied rows must agree, and on the documented rule: ascending sort by
    # device_category, so "IMV" (< "Room Air" alphabetically) wins.
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "window_start": pd.to_datetime(["2024-01-01 00:00"]),
        "window_end": pd.to_datetime(["2024-01-02 00:00"]),
    })
    vitals = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 05:00"]),
        "vital_category": ["spo2"],
        "vital_value": [88.0],
    })
    labs = pd.DataFrame(columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"])
    resp_a = pd.DataFrame({
        "hospitalization_id": ["H1", "H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 04:00", "2024-01-01 04:00"]),
        "device_category": ["IMV", "Room Air"],
        "fio2_set": [0.5, 0.21],
    })
    resp_b = resp_a.iloc[::-1].reset_index(drop=True)

    row_a = build_observation_frame(cohort, labs, vitals, resp_a)
    row_a = row_a[row_a["spo2"].notna()].iloc[0]
    row_b = build_observation_frame(cohort, labs, vitals, resp_b)
    row_b = row_b[row_b["spo2"].notna()].iloc[0]

    assert row_a["device_category"] == "IMV"
    assert row_b["device_category"] == "IMV"


def test_observation_frame_dttm_tz_matches_cohort_not_duckdbs_own_alias():
    # DuckDB's ASOF JOIN round-trip returns tz-aware timestamps localized to
    # its own IANA alias (e.g. "America/Chicago") regardless of what the
    # input columns were tagged with (e.g. pandas/pytz "US/Central") -- same
    # instant, different tz dtype object. Downstream, classify_arf takes
    # min(axis=1) across a column built from this dttm and one built from a
    # plain-pandas timestamp tagged "US/Central"; mismatched tz dtypes make
    # pandas fall back to an elementwise path that raises comparing a
    # Timestamp to a float. The fix is to normalize dttm back to the
    # cohort's own tz before returning.
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "window_start": pd.to_datetime(["2024-01-01 00:00"]).tz_localize("US/Central"),
        "window_end": pd.to_datetime(["2024-01-02 00:00"]).tz_localize("US/Central"),
    })
    vitals = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 05:00"]).tz_localize("US/Central"),
        "vital_category": ["spo2"],
        "vital_value": [88.0],
    })
    labs = pd.DataFrame(columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"])
    resp = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 00:00"]).tz_localize("US/Central"),
        "device_category": ["Room Air"],
        "fio2_set": [0.21],
    })
    out = build_observation_frame(cohort, labs, vitals, resp)
    assert out["dttm"].dt.tz == cohort["window_start"].dt.tz


def test_observations_outside_window_are_dropped():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "window_start": pd.to_datetime(["2024-01-01 00:00"]),
        "window_end": pd.to_datetime(["2024-01-01 12:00"]),
    })
    vitals = pd.DataFrame({
        "hospitalization_id": ["H1", "H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 06:00", "2024-01-02 06:00"]),
        "vital_category": ["spo2", "spo2"],
        "vital_value": [88.0, 85.0],
    })
    labs = pd.DataFrame(columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"])
    resp = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 00:00"]),
        "device_category": ["Room Air"],
        "fio2_set": [0.21],
    })
    out = build_observation_frame(cohort, labs, vitals, resp)
    assert len(out) == 1


def test_first_imv_start_is_earliest_within_index_icu_stay():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "icu_admit_dttm": pd.to_datetime(["2024-01-01 00:00"]),
        "icu_discharge_dttm": pd.to_datetime(["2024-01-05 00:00"]),
    })
    resp = pd.DataFrame({
        "hospitalization_id": ["H1", "H1", "H1"],
        "recorded_dttm": pd.to_datetime(["2023-12-31 00:00", "2024-01-02 00:00", "2024-01-03 00:00"]),
        "device_category": ["IMV", "IMV", "IMV"],
        "fio2_set": [0.5, 0.5, 0.5],
    })
    out = first_imv_starts(cohort, resp)
    assert len(out) == 1
    # The pre-ICU IMV row is outside the index stay and must not count.
    assert out.iloc[0]["imv_any_start_dttm"] == pd.Timestamp("2024-01-02 00:00")
    # Every in-stay row is IMV, so the very first in-stay row has no
    # preceding in-stay row to compare against -- not a detectable "new"
    # intubation (same pattern as already-ventilated-on-arrival).
    assert pd.isna(out.iloc[0]["imv_new_start_dttm"])


def test_first_imv_start_already_intubated_on_arrival_counts_any_not_new():
    # First in-stay respiratory-support row is already IMV -- the waterfall
    # forward-fills from before ICU admission, so this looks identical to a
    # patient who arrived from the ED/OR already ventilated. That must count
    # as an IMV *stay*, but must NOT be read as a new intubation event.
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "icu_admit_dttm": pd.to_datetime(["2024-01-01 00:00"]),
        "icu_discharge_dttm": pd.to_datetime(["2024-01-05 00:00"]),
    })
    resp = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 00:00"]),
        "device_category": ["IMV"],
        "fio2_set": [0.5],
    })
    out = first_imv_starts(cohort, resp)
    assert len(out) == 1
    assert out.iloc[0]["imv_any_start_dttm"] == pd.Timestamp("2024-01-01 00:00")
    assert pd.isna(out.iloc[0]["imv_new_start_dttm"])


def test_first_imv_start_transition_to_imv_counts_as_both_any_and_new():
    # Patient starts on Room Air, then transitions to IMV within the stay --
    # a genuine new intubation, so it must show up in both columns at the
    # same timestamp.
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "icu_admit_dttm": pd.to_datetime(["2024-01-01 00:00"]),
        "icu_discharge_dttm": pd.to_datetime(["2024-01-05 00:00"]),
    })
    resp = pd.DataFrame({
        "hospitalization_id": ["H1", "H1"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"]),
        "device_category": ["Room Air", "IMV"],
        "fio2_set": [0.21, 0.5],
    })
    out = first_imv_starts(cohort, resp)
    assert len(out) == 1
    assert out.iloc[0]["imv_any_start_dttm"] == pd.Timestamp("2024-01-02 00:00")
    assert out.iloc[0]["imv_new_start_dttm"] == pd.Timestamp("2024-01-02 00:00")


def test_restore_device_category_case_undoes_waterfall_lowercasing():
    # clifpy.process_resp_support_waterfall lower-cases device_category on
    # input and never restores it, so exact-match comparisons against
    # d.IMV_DEVICE ("IMV") / d.ROOM_AIR_DEVICE ("Room Air") would silently
    # match zero rows unless case is restored before those comparisons run.
    waterfall = pd.DataFrame({
        "hospitalization_id": ["H1", "H1", "H1"],
        "device_category": ["imv", "room air", None],
    })
    out = restore_device_category_case(waterfall)
    assert out["device_category"].iloc[0] == "IMV"
    assert out["device_category"].iloc[1] == "Room Air"
    assert pd.isna(out["device_category"].iloc[2])


def test_restore_device_category_case_falls_back_to_original_for_unmapped_value():
    # "hfnc" is not CLIF 2.1 vocabulary (e.g. a CLIF 3.0 site's naming). A
    # bare .map() would turn it into NaN, silently deleting the signal the
    # same way the original bug did -- it must survive unchanged instead.
    waterfall = pd.DataFrame({
        "hospitalization_id": ["H1", "H1"],
        "device_category": ["imv", "hfnc"],
    })
    out = restore_device_category_case(waterfall)
    assert out["device_category"].iloc[0] == "IMV"
    assert out["device_category"].iloc[1] == "hfnc"


def test_attach_abg_availability_true_when_arterial_gas_in_window():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "window_start": pd.to_datetime(["2024-01-01 00:00"]),
        "window_end": pd.to_datetime(["2024-01-02 00:00"]),
    })
    labs = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "lab_result_dttm": pd.to_datetime(["2024-01-01 06:00"]),
        "lab_category": ["po2_arterial"],
        "lab_value_numeric": [80.0],
    })
    obs = pd.DataFrame({"hospitalization_id": ["H1"]})
    out = attach_abg_availability(obs, labs, cohort)
    assert out.iloc[0]["has_abg"] == True


def test_attach_abg_availability_false_when_only_gas_is_outside_window():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "window_start": pd.to_datetime(["2024-01-01 00:00"]),
        "window_end": pd.to_datetime(["2024-01-02 00:00"]),
    })
    labs = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "lab_result_dttm": pd.to_datetime(["2024-01-05 06:00"]),  # after window_end
        "lab_category": ["po2_arterial"],
        "lab_value_numeric": [80.0],
    })
    obs = pd.DataFrame({"hospitalization_id": ["H1"]})
    out = attach_abg_availability(obs, labs, cohort)
    assert out.iloc[0]["has_abg"] == False


def test_attach_abg_availability_false_for_venous_only():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "window_start": pd.to_datetime(["2024-01-01 00:00"]),
        "window_end": pd.to_datetime(["2024-01-02 00:00"]),
    })
    labs = pd.DataFrame({
        "hospitalization_id": ["H1"],
        "lab_result_dttm": pd.to_datetime(["2024-01-01 06:00"]),
        "lab_category": ["pco2_venous"],
        "lab_value_numeric": [45.0],
    })
    obs = pd.DataFrame({"hospitalization_id": ["H1"]})
    out = attach_abg_availability(obs, labs, cohort)
    assert out.iloc[0]["has_abg"] == False
