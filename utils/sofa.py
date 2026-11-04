"""Worst SOFA per time window with clifpy 0.5.0, and fixes to its compute_sofa_polars output
(scoring rules in clifpy/utils/sofa_polars.py:867-935)."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from clifpy.utils.datetime_polars import standardize_datetime_columns
from clifpy.utils.sofa_polars import (REQUIRED_ASSESSMENTS, REQUIRED_LABS, REQUIRED_MEDS, REQUIRED_VITALS,
                                      compute_sofa_polars)

COMPONENTS = ["sofa_cv_97", "sofa_coag", "sofa_liver", "sofa_resp", "sofa_cns", "sofa_renal"]
PRESSORS = ["dopamine", "epinephrine", "norepinephrine", "dobutamine"]
DOSE_COLS = [f"{p}_mcg_kg_min" for p in PRESSORS]
VASOPRESSORS = ["norepinephrine", "epinephrine", "dopamine", "vasopressin", "phenylephrine", "angiotensin"]  # on_pressor; dobutamine is an inotrope
ADVANCED_SUPPORT = {"imv", "nippv", "cpap"}
CHUNK_WINDOWS = 50_000               # ~2 GB per clifpy call
TABLES = {  # table: (time column, other columns clifpy 0.5.0 reads, row filter)
    "clif_labs": ("lab_result_dttm", ["lab_category", "lab_value", "lab_value_numeric"], pl.col("lab_category").is_in(REQUIRED_LABS)),
    "clif_vitals": ("recorded_dttm", ["vital_category", "vital_value"], pl.col("vital_category").is_in(REQUIRED_VITALS)),
    "clif_patient_assessments": ("recorded_dttm", ["assessment_category", "numerical_value", "categorical_value"],
                                 pl.col("assessment_category").is_in(REQUIRED_ASSESSMENTS)),
    "clif_respiratory_support": ("recorded_dttm", ["device_category", "mode_category", "fio2_set", "lpm_set", "tidal_volume_set", "resp_rate_set"],
                                 pl.lit(True)),
    "clif_medication_admin_continuous": ("admin_dttm", ["med_category", "med_dose", "med_dose_unit"],
                                         pl.col("med_category").is_in(set(REQUIRED_MEDS) | set(VASOPRESSORS))),
}


def cohort_rows(config: dict, encounter_map: pl.DataFrame, table: str, time_col: str, cols: list, keep: pl.Expr) -> pl.DataFrame:
    """One CLIF table restricted to cohort encounters (re-keyed to the encounter id), the columns asked for, and the site timezone."""
    path = Path(config["data_directory"]) / f"{table}.{config['filetype']}"
    lf = (pl.scan_parquet(path) if config["filetype"] == "parquet" else pl.scan_csv(path, try_parse_dates=True))
    lf = (lf.select("hospitalization_id", time_col, *cols).with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
          .filter(keep & pl.col("hospitalization_id").is_in(encounter_map["hospitalization_id"].to_list()))
          .join(encounter_map.lazy(), on="hospitalization_id")  # raw hospitalization -> its stitched encounter
          .drop("hospitalization_id").rename({"encounter_id": "hospitalization_id"}))
    return standardize_datetime_columns(lf, target_timezone=config["timezone"], target_time_unit="us", datetime_columns=[time_col]).collect()


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
    if "on_pressor" in sofa:  # a pressor clifpy could not weight-normalise (no weight) or does not score (vasopressin, phenylephrine,
        # angiotensin) leaves CV unknown, not scored from a MAP the drug is holding up
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


def score_windows(config: dict, windows: pl.DataFrame, encounter_map: pl.DataFrame, workdir: Path) -> pd.DataFrame:
    """Rescored worst SOFA, on_imv and on_pressor for each window, with the windows' other columns.

    `windows`: hospitalization_id (stitched encounter), window_id (string), start, end; `encounter_map` holds the
    cohort's raw hospitalization_id -> encounter_id. Each event is keyed to its one window (assign_windows) and the
    window-keyed tables are written to `workdir`, where clifpy scores each window as if it were a hospitalization.
    """
    workdir.mkdir(exist_ok=True)
    flags = windows.select("window_id")
    for table, (time_col, cols, keep) in TABLES.items():
        rows = cohort_rows(config, encounter_map, table, time_col, cols, keep)
        keyed = assign_windows(rows.lazy(), time_col, windows).collect()
        if table == "clif_respiratory_support":  # clifpy forward-fills FiO2 from the 24 h before each window
            lookback = assign_windows(rows.lazy(), time_col, windows.with_columns(start=pl.col("start") - pl.duration(hours=24), end="start"))
            flags = flags.join(keyed.group_by("hospitalization_id").agg(on_imv=(pl.col("device_category").str.to_lowercase() == "imv").any())
                               .rename({"hospitalization_id": "window_id"}), on="window_id", how="left")
            keyed = pl.concat([keyed, lookback.collect()])
        if table == "clif_vitals":  # clifpy takes the last weight before each dose; carry the pre-window one in at window start
            weights = rows.filter(pl.col("vital_category") == "weight_kg").sort("hospitalization_id", time_col)
            carried = (windows.sort("hospitalization_id", "start")
                       .join_asof(weights, left_on="start", right_on=time_col, by="hospitalization_id", strategy="backward", check_sortedness=False)
                       .filter(pl.col("vital_value").is_not_null())
                       .select(hospitalization_id="window_id", recorded_dttm="start", vital_category="vital_category", vital_value="vital_value"))
            keyed = pl.concat([keyed, carried])
        if table == "clif_medication_admin_continuous":
            flags = flags.join(keyed.group_by("hospitalization_id").agg(on_pressor=(pl.col("med_category").is_in(VASOPRESSORS) & (pl.col("med_dose") > 0)).any())
                               .rename({"hospitalization_id": "window_id"}), on="window_id", how="left")
            keyed = keyed.filter(pl.col("med_category").is_in(REQUIRED_MEDS))  # clifpy scores only its four
        keyed.write_parquet(workdir / f"{table}.parquet")
        print(f"{table}: {rows.height:,} cohort rows -> {keyed.height:,} in windows")
        del rows, keyed

    # one clifpy call per CHUNK_WINDOWS windows bounds its peak memory (its pandas loaders hold a whole table at once);
    # a chunk must be large enough to contain every category (clifpy crashes on e.g. a chunk with no spo2)
    chunk_id = (pl.int_range(pl.len()) // CHUNK_WINDOWS).clip(upper_bound=max(windows.height // CHUNK_WINDOWS - 1, 0))  # remainder joins the last chunk
    sofa = pl.concat([compute_sofa_polars(str(workdir), chunk.select(hospitalization_id="window_id", start_dttm="start", end_dttm="end"),
                                          filetype="parquet", fill_na_scores_with_zero=False, remove_outliers=True, timezone=config["timezone"])
                      for _, chunk in windows.group_by(chunk_id)], how="diagonal_relaxed")
    out = (windows.drop("start", "end")
           .join(sofa.rename({"hospitalization_id": "window_id"}), on="window_id", how="left")
           .join(flags, on="window_id", how="left").with_columns(pl.col("on_imv", "on_pressor").fill_null(False)))
    return rescore_sofa(out.to_pandas())
