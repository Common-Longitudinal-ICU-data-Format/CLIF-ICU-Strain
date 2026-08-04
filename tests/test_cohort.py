import pandas as pd
from icu_strain.cohort import collapse_icu_blocks, build_cohort


def _adt(rows):
    return pd.DataFrame(rows, columns=["hospitalization_id", "in_dttm", "out_dttm", "location_category"]).assign(
        in_dttm=lambda x: pd.to_datetime(x["in_dttm"]),
        out_dttm=lambda x: pd.to_datetime(x["out_dttm"]),
    )


def test_adjacent_icu_rows_merge_into_one_block():
    adt = _adt([
        ("H1", "2024-03-01 00:00", "2024-03-01 06:00", "icu"),
        ("H1", "2024-03-01 06:00", "2024-03-01 12:00", "icu"),
    ])
    out = collapse_icu_blocks(adt, gap_hours=1)
    assert len(out) == 1
    assert out.iloc[0]["icu_admit_dttm"] == pd.Timestamp("2024-03-01 00:00")
    assert out.iloc[0]["icu_discharge_dttm"] == pd.Timestamp("2024-03-01 12:00")
    assert out.iloc[0]["n_adt_rows_merged"] == 2


def test_gap_beyond_threshold_splits_blocks():
    adt = _adt([
        ("H1", "2024-03-01 00:00", "2024-03-01 06:00", "icu"),
        ("H1", "2024-03-01 10:00", "2024-03-01 12:00", "icu"),
    ])
    out = collapse_icu_blocks(adt, gap_hours=1).sort_values("block_index")
    assert len(out) == 2
    assert list(out["block_index"]) == [0, 1]


def test_non_icu_rows_are_ignored():
    adt = _adt([
        ("H1", "2024-03-01 00:00", "2024-03-01 06:00", "ed"),
        ("H1", "2024-03-01 06:00", "2024-03-01 12:00", "ward"),
    ])
    assert len(collapse_icu_blocks(adt)) == 0


def test_intervening_ward_row_does_not_bridge_icu_blocks():
    # A ward stay between two ICU rows is a real transfer out; the 1h gap rule
    # must not merge across it even though the timestamps are contiguous.
    adt = _adt([
        ("H1", "2024-03-01 00:00", "2024-03-01 06:00", "icu"),
        ("H1", "2024-03-01 06:00", "2024-03-01 06:30", "ward"),
        ("H1", "2024-03-01 06:30", "2024-03-01 12:00", "icu"),
    ])
    out = collapse_icu_blocks(adt, gap_hours=1)
    assert len(out) == 2


def test_build_cohort_keeps_first_block_and_filters():
    hosp = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "hospitalization_id": ["H1", "H2", "H3"],
        "admission_dttm": pd.to_datetime(["2024-03-01 00:00", "2024-03-02 00:00", "2023-06-01 00:00"]),
        "discharge_dttm": pd.to_datetime(["2024-03-10 00:00", "2024-03-12 00:00", "2023-06-10 00:00"]),
        "age_at_admission": [55, 17, 70],
    })
    adt = _adt([
        ("H1", "2024-03-01 04:00", "2024-03-03 00:00", "icu"),
        ("H1", "2024-03-05 00:00", "2024-03-06 00:00", "icu"),
        ("H2", "2024-03-02 04:00", "2024-03-04 00:00", "icu"),   # minor
        ("H3", "2023-06-01 04:00", "2023-06-03 00:00", "icu"),   # wrong year
    ])
    out = build_cohort(hosp, adt, year=2024, min_age=18)
    assert list(out["hospitalization_id"]) == ["H1"]
    assert out.iloc[0]["icu_admit_dttm"] == pd.Timestamp("2024-03-01 04:00")
    assert out.iloc[0]["window_start"] == pd.Timestamp("2024-03-01 00:00")
    assert out.iloc[0]["window_end"] == pd.Timestamp("2024-03-02 04:00")


def test_build_cohort_with_no_icu_rows_returns_empty_frame():
    # Regression: collapse_icu_blocks' empty-return path must carry real dtypes
    # so build_cohort's later `.dt.year` access doesn't raise on an all-object frame.
    hosp = pd.DataFrame({
        "patient_id": ["P1"],
        "hospitalization_id": ["H1"],
        "admission_dttm": pd.to_datetime(["2024-03-01 00:00"]),
        "discharge_dttm": pd.to_datetime(["2024-03-10 00:00"]),
        "age_at_admission": [55],
    })
    adt = _adt([
        ("H1", "2024-03-01 00:00", "2024-03-01 06:00", "ward"),
    ])
    out = build_cohort(hosp, adt, year=2024, min_age=18)
    assert len(out) == 0
    assert list(out.columns) == [
        "patient_id", "hospitalization_id", "admission_dttm", "discharge_dttm",
        "age_at_admission", "icu_admit_dttm", "icu_discharge_dttm",
        "window_start", "window_end",
    ]


def test_build_cohort_filters_on_icu_admission_year_not_hospital_admission_year():
    # Hospital admission in Dec 2023, but the index ICU block starts in Jan 2024:
    # the year filter keys on icu_admit_dttm, so this hospitalization must be kept.
    hosp = pd.DataFrame({
        "patient_id": ["P1"],
        "hospitalization_id": ["H1"],
        "admission_dttm": pd.to_datetime(["2023-12-30 00:00"]),
        "discharge_dttm": pd.to_datetime(["2024-01-10 00:00"]),
        "age_at_admission": [55],
    })
    adt = _adt([
        ("H1", "2024-01-01 04:00", "2024-01-03 00:00", "icu"),
    ])
    out = build_cohort(hosp, adt, year=2024, min_age=18)
    assert list(out["hospitalization_id"]) == ["H1"]
