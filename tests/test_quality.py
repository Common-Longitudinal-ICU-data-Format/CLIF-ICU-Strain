import pandas as pd

from icu_strain.quality import monthly_quality_checks


def _cohort(hosp_ids, admit_dttm):
    df = pd.DataFrame({
        "hospitalization_id": hosp_ids,
        "icu_admit_dttm": pd.to_datetime([admit_dttm] * len(hosp_ids)),
    })
    df["window_start"] = df["icu_admit_dttm"] - pd.Timedelta(hours=1)
    df["window_end"] = df["icu_admit_dttm"] + pd.Timedelta(hours=24)
    return df


def _empty_labs():
    return pd.DataFrame({
        "hospitalization_id": pd.Series(dtype="object"),
        "lab_category": pd.Series(dtype="object"),
        "lab_result_dttm": pd.Series(dtype="datetime64[ns]"),
        "lab_value_numeric": pd.Series(dtype="float64"),
    })


def _empty_vitals():
    return pd.DataFrame({
        "hospitalization_id": pd.Series(dtype="object"),
        "recorded_dttm": pd.Series(dtype="datetime64[ns]"),
        "vital_category": pd.Series(dtype="object"),
    })


def _resp_with_one_non_null_device_row():
    # A single, otherwise-irrelevant row so device_null_rate_cohort's
    # isna().mean() runs over a non-empty column rather than an empty one.
    return pd.DataFrame({
        "hospitalization_id": ["ZZZ"],
        "recorded_dttm": pd.to_datetime(["2024-01-01 00:00"]),
        "fio2_set": [pd.NA],
        "device_category": ["Room Air"],
    })


def test_month_below_min_cell_size_is_suppressed_but_larger_month_is_not():
    """A month with n_hospitalizations < MIN_CELL_SIZE (10) must be suppressed
    in the emitted table -- both the count itself and every per-month
    availability percentage computed from it -- while a month at or above the
    threshold is published unsuppressed.
    """
    big_month = _cohort([f"H{i}" for i in range(12)], "2024-01-05 00:00")   # n=12, >= MIN_CELL_SIZE
    small_month = _cohort([f"S{i}" for i in range(3)], "2024-02-05 00:00")  # n=3, < MIN_CELL_SIZE
    cohort = pd.concat([big_month, small_month], ignore_index=True)

    out = monthly_quality_checks(
        cohort, _empty_labs(), _empty_vitals(), _resp_with_one_non_null_device_row()
    ).set_index("month")

    big = out.loc["2024-01"]
    assert big["n_hospitalizations"] == 12
    assert big["pct_with_abg"] == 0.0  # no labs supplied -- a real, unsuppressed zero
    assert big["pct_with_spo2"] == 0.0
    assert big["pct_with_fio2"] == 0.0

    small = out.loc["2024-02"]
    assert pd.isna(small["n_hospitalizations"])
    assert pd.isna(small["pct_with_abg"])
    assert pd.isna(small["pct_with_spo2"])
    assert pd.isna(small["pct_with_fio2"])


def test_suppression_does_not_touch_overall_non_per_month_columns():
    """device_null_rate_cohort and the dup-key rates are single overall values,
    not conditioned on any one month's small n, so they must survive
    suppression even on a month whose per-month columns get masked.
    """
    small_month = _cohort([f"S{i}" for i in range(3)], "2024-02-05 00:00")

    out = monthly_quality_checks(
        small_month, _empty_labs(), _empty_vitals(), _resp_with_one_non_null_device_row()
    ).set_index("month")

    row = out.loc["2024-02"]
    assert pd.isna(row["n_hospitalizations"])
    assert row["device_null_rate_cohort"] == 0.0
    assert row["pct_arterial_dup_key_conflicting_overall"] == 0.0
    assert row["pct_arterial_dup_key_identical_overall"] == 0.0
