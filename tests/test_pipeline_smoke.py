import pandas as pd
from icu_strain.pipeline import build_observation_frame, first_imv_starts, restore_device_category_case


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
    assert out.iloc[0]["imv_start_dttm"] == pd.Timestamp("2024-01-02 00:00")


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
