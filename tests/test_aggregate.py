import pandas as pd
from icu_strain.aggregate import (
    suppress_small_cells, monthly_imv_counts, monthly_subtype_counts, time_to_icu_summary,
)


def test_small_counts_are_suppressed_but_zero_is_kept():
    df = pd.DataFrame({"month": ["2024-01", "2024-02", "2024-03"], "n": [0, 5, 40]})
    out = suppress_small_cells(df, ["n"], min_n=10)
    assert out.loc[0, "n"] == 0
    assert pd.isna(out.loc[1, "n"])
    assert out.loc[2, "n"] == 40


def test_monthly_imv_bins_by_imv_start_month():
    imv = pd.DataFrame({
        "hospitalization_id": ["H1", "H2", "H3"],
        "imv_start_dttm": pd.to_datetime(["2024-01-15", "2024-01-20", "2024-02-02"]),
    })
    out = monthly_imv_counts(imv).set_index("month")
    assert out.loc["2024-01", "n_new_imv"] == 2
    assert out.loc["2024-02", "n_new_imv"] == 1


def test_monthly_subtypes_sum_to_total():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1", "H2", "H3"],
        "icu_admit_dttm": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"]),
    })
    classified = pd.DataFrame({
        "hospitalization_id": ["H1", "H2", "H3"],
        "arf_subtype": ["hypoxemic", "mixed", "hypoxemic"],
        "first_arf_dttm": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"]),
    })
    out = monthly_subtype_counts(cohort, classified)
    row = out[out["month"] == "2024-01"].iloc[0]
    assert row["hypoxemic"] + row["hypercapnic"] + row["mixed"] == row["total"] == 3


def test_time_to_icu_reports_negative_deltas_as_pre_icu():
    cohort = pd.DataFrame({
        "hospitalization_id": ["H1", "H2"],
        "icu_admit_dttm": pd.to_datetime(["2024-01-05 12:00", "2024-01-06 12:00"]),
    })
    classified = pd.DataFrame({
        "hospitalization_id": ["H1", "H2"],
        "arf_subtype": ["hypoxemic", "hypoxemic"],
        # H1 met criteria 4h BEFORE ICU; H2 2h after.
        "first_arf_dttm": pd.to_datetime(["2024-01-05 08:00", "2024-01-06 14:00"]),
    })
    out = time_to_icu_summary(cohort, classified).set_index("arf_subtype")
    assert out.loc["hypoxemic", "n"] == 2
    assert out.loc["hypoxemic", "median_hours"] == -1.0   # median of (-4, +2)
    assert out.loc["hypoxemic", "pct_before_icu"] == 50.0


def test_suppression_preserves_integer_dtype_and_float_columns_untouched():
    df = pd.DataFrame({
        "n_int": pd.array([0, 5, 40], dtype="int64"),
        "pct_before_icu": [0.0, 5.0, 99.9],
    })
    out = suppress_small_cells(df, ["n_int", "pct_before_icu"], min_n=10)

    # Integer column keeps integer semantics through suppression and CSV rendering
    # (a naive mask() upcasts int64 -> float64, so 40 would round-trip as "40.0").
    assert out.loc[2, "n_int"] == 40
    assert pd.api.types.is_integer_dtype(out["n_int"])
    csv_text = out[["n_int"]].to_csv(index=False)
    assert "40.0" not in csv_text
    assert "40" in csv_text
    assert pd.isna(out.loc[1, "n_int"])

    # Float columns (e.g. median_hours, pct_before_icu) must NOT be coerced to Int64.
    assert pd.api.types.is_float_dtype(out["pct_before_icu"])
    assert out.loc[0, "pct_before_icu"] == 0.0
    assert pd.isna(out.loc[1, "pct_before_icu"])
    assert out.loc[2, "pct_before_icu"] == 99.9


def test_monthly_imv_counts_on_hand_built_empty_frame_does_not_raise():
    # Untyped empty frame: the exact pattern (pd.DataFrame(columns=[...])) that
    # previously caused a `.dt` accessor crash elsewhere in this codebase.
    imv = pd.DataFrame(columns=["hospitalization_id", "imv_start_dttm"])
    out = monthly_imv_counts(imv)
    assert list(out.columns) == ["month", "n_new_imv"]
    assert len(out) == 0


def test_monthly_subtype_counts_on_hand_built_empty_frame_does_not_raise():
    cohort = pd.DataFrame(columns=["hospitalization_id", "icu_admit_dttm"])
    classified = pd.DataFrame(columns=["hospitalization_id", "arf_subtype", "first_arf_dttm"])
    out = monthly_subtype_counts(cohort, classified)
    assert list(out.columns) == ["month", "hypoxemic", "hypercapnic", "mixed", "total"]
    assert len(out) == 0


def test_time_to_icu_summary_on_hand_built_empty_frame_does_not_raise():
    cohort = pd.DataFrame(columns=["hospitalization_id", "icu_admit_dttm"])
    classified = pd.DataFrame(columns=["hospitalization_id", "arf_subtype", "first_arf_dttm"])
    out = time_to_icu_summary(cohort, classified)
    assert list(out.columns) == [
        "arf_subtype", "n", "median_hours", "q25_hours", "q75_hours", "pct_before_icu",
    ]
    assert len(out) == 0


def test_monthly_imv_counts_deduplicates_hospitalization_across_months():
    # H1 has an IMV-start row in both January and February. It represents one
    # hospitalization's first IMV initiation, so it must be counted exactly
    # once, in the earlier month -- not once per month it happens to appear in.
    imv = pd.DataFrame({
        "hospitalization_id": ["H1", "H1", "H2"],
        "imv_start_dttm": pd.to_datetime(["2024-01-15", "2024-02-01", "2024-02-05"]),
    })
    out = monthly_imv_counts(imv).set_index("month")
    assert out.loc["2024-01", "n_new_imv"] == 1
    assert out.loc["2024-02", "n_new_imv"] == 1
