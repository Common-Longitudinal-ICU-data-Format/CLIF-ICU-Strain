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
