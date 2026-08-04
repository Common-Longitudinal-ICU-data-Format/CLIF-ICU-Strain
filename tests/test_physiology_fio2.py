import pandas as pd
import numpy as np
from icu_strain.physiology import normalize_fio2


def test_fractions_pass_through():
    out = normalize_fio2(pd.Series([0.21, 0.40, 1.0]))
    assert list(out) == [0.21, 0.40, 1.0]


def test_percentages_are_rescaled():
    out = normalize_fio2(pd.Series([21.0, 40.0, 100.0]))
    assert list(out) == [0.21, 0.40, 1.0]


def test_impossible_values_become_nan():
    out = normalize_fio2(pd.Series([-1.0, 0.0, 88880.0, 101.0]))
    assert out.isna().all()


def test_boundary_one_is_treated_as_fraction():
    # 1.0 is 100% oxygen, not 1%. Must not be rescaled to 0.01.
    assert normalize_fio2(pd.Series([1.0])).iloc[0] == 1.0


def test_below_room_air_fraction_is_nan():
    # 0.10 is below the fraction of oxygen in air; not physically deliverable.
    assert np.isnan(normalize_fio2(pd.Series([0.10])).iloc[0])


def test_nan_stays_nan():
    assert normalize_fio2(pd.Series([np.nan])).isna().all()
