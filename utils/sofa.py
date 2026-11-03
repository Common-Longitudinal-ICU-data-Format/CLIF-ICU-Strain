"""Fixes to clifpy 0.5.0's compute_sofa_polars output (scoring rules in clifpy/utils/sofa_polars.py:867-935)."""

import numpy as np
import pandas as pd
import polars as pl

COMPONENTS = ["sofa_cv_97", "sofa_coag", "sofa_liver", "sofa_resp", "sofa_cns", "sofa_renal"]
PRESSORS = ["dopamine", "epinephrine", "norepinephrine", "dobutamine"]
DOSE_COLS = [f"{p}_mcg_kg_min" for p in PRESSORS]
VASOPRESSORS = ["norepinephrine", "epinephrine", "dopamine"]   # dobutamine is an inotrope; clifpy 0.5.0 loads no vasopressin/phenylephrine
ADVANCED_SUPPORT = {"imv", "nippv", "cpap"}


def assign_windows(events: pl.LazyFrame, time_col: str, windows: pl.DataFrame) -> pl.LazyFrame:
    """Re-key each event to the one window [start, end) of its hospitalization that holds it; drop the rest.

    `windows` has hospitalization_id, window_id, start, end; windows of one hospitalization must not overlap.
    This as-of join replaces clifpy's every-event-to-every-window join, which multiplies a hospitalization's
    rows by its number of windows.
    """
    keep = [col for col in events.collect_schema().names() if col != "hospitalization_id"]
    return (events.sort("hospitalization_id", time_col)
            .join_asof(windows.select("hospitalization_id", "window_id", "start", "end").lazy().sort("hospitalization_id", "start"),
                       left_on=time_col, right_on="start", by="hospitalization_id", strategy="backward", check_sortedness=False)
            .filter(pl.col(time_col) < pl.col("end"))
            .select(pl.col("window_id").alias("hospitalization_id"), *keep))


def rescore_sofa(sofa: pd.DataFrame) -> pd.DataFrame:
    """Rescore cardiovascular and respiratory, and total the observed components.

    shortcut: works around clifpy 0.5.0 giving 3 CV points to a dose charted as 0 and leaving
    P/F < 200 off IMV/NIPPV/CPAP unscored; delete once a clifpy release has the fix (branch
    fix/sofa-zero-dose-and-resp; PR #132 overlaps). Devices are matched case-insensitively.
    Like clifpy, the total counts a missing component as 0; unlike it, the total is NA when all are.
    """
    dose = {p: sofa[f"{p}_mcg_kg_min"].fillna(0) for p in PRESSORS}
    cv = np.select(
        [(dose["dopamine"] > 15) | (dose["epinephrine"] > 0.1) | (dose["norepinephrine"] > 0.1),
         (dose["dopamine"] > 5) | (dose["epinephrine"] > 0) | (dose["norepinephrine"] > 0),
         (dose["dopamine"] > 0) | (dose["dobutamine"] > 0),
         sofa["map"] < 70,
         sofa["map"] >= 70],
        [4, 3, 2, 1, 0], default=np.nan)
    if "on_pressor" in sofa:  # a dose clifpy could not weight-normalise (no weight charted) is unknown, not "no pressor"
        unweighable = sofa[DOSE_COLS].isna().all(axis=1) & sofa["on_pressor"].fillna(False).astype(bool)
        cv = np.where(unweighable, np.nan, cv)

    pf = sofa["p_f"]
    advanced = sofa["device_category"].astype(str).str.lower().isin(ADVANCED_SUPPORT)
    resp = np.select(
        [(pf < 100) & advanced, (pf < 200) & advanced, pf < 300, pf < 400, pf >= 400],
        [4, 3, 2, 1, 0], default=np.nan)

    out = sofa.assign(sofa_cv_97=cv, sofa_resp=resp)
    out["sofa_total"] = out[COMPONENTS].sum(axis=1, min_count=1)  # NA only when every component is missing
    return out
