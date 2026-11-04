"""Organ support (invasive ventilation, vasopressors, CRRT) around a unit stay, and the acuity weight it gives a patient."""

from pathlib import Path

import pandas as pd
import polars as pl

from .sofa import VASOPRESSORS, cohort_rows

KINDS = ["imv", "pressor", "crrt"]
TABLES = {  # kind: table, time column, other columns read, row filter
    "imv": ("clif_respiratory_support", "recorded_dttm", ["device_category"], pl.col("device_category").str.to_lowercase() == "imv"),
    "pressor": ("clif_medication_admin_continuous", "admin_dttm", ["med_category", "med_dose"],
                pl.col("med_category").is_in(VASOPRESSORS) & (pl.col("med_dose") > 0)),
    "crrt": ("clif_crrt_therapy", "recorded_dttm", [], pl.lit(True)),
}


def support_events(config: dict, encounter_map: pl.DataFrame) -> pd.DataFrame:
    """One row per (encounter, time, kind) at which that support was charted; a site without crrt_therapy has no crrt rows."""
    parts = []
    for kind, (table, time_col, cols, keep) in TABLES.items():
        if not (Path(config["data_directory"]) / f"{table}.{config['filetype']}").exists():
            print(f"{table} not found: {kind} is not in the acuity weights")
            continue
        rows = cohort_rows(config, encounter_map, table, time_col, cols, keep)
        parts.append(rows.select("hospitalization_id", t=time_col).unique().with_columns(kind=pl.lit(kind)).to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["hospitalization_id", "t", "kind"])


def support_in(windows: pd.DataFrame, events: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """`windows` (hospitalization_id, `start`, `end`) with a bool column per kind: any event in [start, end).

    Each window is looked up on its own (the first event at or after its start), so windows of one
    hospitalization may overlap. Returns a copy with a fresh index.
    """
    out = windows.reset_index(drop=True)
    w = out.reset_index(names="_i").sort_values(start)
    for kind in KINDS:
        e = events.loc[events["kind"] == kind, ["hospitalization_id", "t"]].dropna(subset=["t"])  # a row with no time is no event
        if e.empty:
            out[kind] = False
            continue
        e = e.assign(hospitalization_id=e["hospitalization_id"].astype(w["hospitalization_id"].dtype),
                     t=e["t"].astype(w[start].dtype)).sort_values("t")
        m = pd.merge_asof(w, e, left_on=start, right_on="t", by="hospitalization_id", direction="forward")
        out[kind] = (m["t"] < m[end]).set_axis(m["_i"]).reindex(out.index).to_numpy()
    return out


def weights(flags: pd.DataFrame) -> pd.Series:
    """1 for every patient, plus 1 per kind of support charted in the window."""
    return 1 + flags[KINDS].sum(axis=1)
