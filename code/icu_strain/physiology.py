"""ARF criteria evaluation over the CLIF physiologic timeline."""

import numpy as np
import pandas as pd

from . import definitions as d


def normalize_fio2(s: pd.Series) -> pd.Series:
    """Coerce fio2_set to a fraction in [0.21, 1.0].

    The raw column at ucmc ranges -1.0 to 88880 with a 0.40 median: mostly
    fractions, some percentages, some garbage. FiO2 is the denominator of both
    P/F and S/F, so an unnormalized value silently manufactures a qualifying
    ratio. Anything uninterpretable becomes NaN, which disqualifies the
    observation rather than defaulting it.
    """
    v = pd.to_numeric(s, errors="coerce").astype(float)
    # (1, 100] are percentages. 1.0 itself is 100% oxygen, so it stays a fraction.
    v = v.where(~((v > 1.0) & (v <= 100.0)), v / 100.0)
    # Outside physically deliverable range -> unusable.
    v = v.where((v >= 0.21) & (v <= 1.0), np.nan)
    return v
