"""ARF criteria evaluation over the CLIF physiologic timeline."""

import numpy as np
import pandas as pd

from . import definitions as d


def normalize_fio2(s: pd.Series) -> pd.Series:
    """Coerce fio2_set to a fraction in [0.21, 1.0].

    The raw column holds mostly
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


def evaluate_hypoxemia(obs: pd.DataFrame) -> pd.DataFrame:
    """Return observations meeting any hypoxemic ARF rule.

    `has_abg` is hospitalization-level: True if the hospitalization has any
    arterial blood gas anywhere in its window. The S/F fallback is gated on it
    so that pulse-ox-only patients are not lost, without letting S/F override
    a real gas.
    """
    df = obs.copy()
    df["fio2"] = normalize_fio2(df["fio2"])
    is_room_air = df["device_category"].fillna("") == d.ROOM_AIR_DEVICE
    # Normalize at the boundary: a left-merge miss upstream commonly produces
    # has_abg as float64 with NaN, and `~` on a non-boolean dtype either raises
    # (float) or silently does the wrong thing (object). A NaN means "we don't
    # know of an ABG", which the S/F fallback should treat as False (no ABG).
    has_abg = df["has_abg"].fillna(False).astype(bool)

    rules = {
        "room_air_spo2": is_room_air & (df["spo2"] < d.SPO2_HYPOXEMIA),
        "room_air_pao2": is_room_air & (df["po2"] <= d.PAO2_HYPOXEMIA),
        "pf_ratio": (df["po2"] / df["fio2"]) <= d.PF_RATIO_THRESHOLD,
        "sf_ratio": (~has_abg) & ((df["spo2"] / df["fio2"]) <= d.SF_RATIO_THRESHOLD),
    }

    hits = []
    for name, mask in rules.items():
        m = mask.fillna(False)
        if m.any():
            hits.append(df[m].assign(hypoxemia_rule=name))
    if not hits:
        return df.head(0).assign(hypoxemia_rule=pd.Series(dtype=object))
    return pd.concat(hits, ignore_index=True)


def evaluate_hypercapnia(labs: pd.DataFrame, tolerance_min: int = d.ABG_PAIR_TOLERANCE_MIN) -> pd.DataFrame:
    """Return arterial draws with PaCO2 >= 45 AND pH < 7.35.

    Only exact arterial categories are considered; pco2_venous / ph_venous are
    never matched, since a venous PCO2 >= 45 is physiologically normal.
    """
    pco2 = labs[labs["lab_category"] == d.ABG_PCO2][
        ["hospitalization_id", "lab_result_dttm", "lab_value_numeric"]
    ].rename(columns={"lab_result_dttm": "pco2_dttm", "lab_value_numeric": "pco2"})
    ph = labs[labs["lab_category"] == d.ABG_PH][
        ["hospitalization_id", "lab_result_dttm", "lab_value_numeric"]
    ].rename(columns={"lab_result_dttm": "ph_dttm", "lab_value_numeric": "ph"})

    if pco2.empty or ph.empty:
        # Preserve real dtypes (not all-object) so a downstream min(axis=1)
        # against a real datetime column in classify_arf doesn't raise. This
        # is the common multi-site case where a site's category mapping is
        # missing pco2_arterial or ph_arterial entirely.
        empty = labs.head(0)
        return pd.DataFrame({
            "hospitalization_id": empty["hospitalization_id"],
            "dttm": pd.Series(dtype="datetime64[ns]"),
            "pco2": pd.Series(dtype=float),
            "ph": pd.Series(dtype=float),
        })

    # merge_asof (not an all-pairs merge + tolerance filter) is O(n log n) and
    # never materializes the full N-by-M cross product, which for a
    # high-utilization hospitalization can be hundreds of thousands of rows
    # before a single one is discarded. `direction="nearest"` also collapses
    # to one nearest pH per PaCO2 draw, matching "the same arterial draw"
    # rather than "any pH within the window".
    paired = pd.merge_asof(
        pco2.sort_values("pco2_dttm"),
        ph.sort_values("ph_dttm"),
        left_on="pco2_dttm", right_on="ph_dttm",
        by="hospitalization_id", direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_min),
    )
    # Unmatched rows (no pH within tolerance) come back as NaN/NaT, not
    # dropped -- the comparisons below evaluate to False for NaN, so they
    # cannot spuriously satisfy the conjunction.
    qualifies = (paired["pco2"] >= d.PACO2_HYPERCAPNIA) & (paired["ph"] < d.PH_HYPERCAPNIA)
    out = paired[qualifies.fillna(False)].copy()
    out["dttm"] = out[["pco2_dttm", "ph_dttm"]].min(axis=1)
    return out[["hospitalization_id", "dttm", "pco2", "ph"]].reset_index(drop=True)


def classify_arf(hypox: pd.DataFrame, hypercap: pd.DataFrame) -> pd.DataFrame:
    """Assign one subtype per hospitalization from the union over the window.

    A patient hypoxemic in the ED and hypercapnic in the ICU is 'mixed'.
    first_arf_dttm is the earliest qualifying observation of any type.
    """
    a = hypox.groupby("hospitalization_id")["dttm"].min().rename("first_hypox")
    b = hypercap.groupby("hospitalization_id")["dttm"].min().rename("first_hypercap")
    j = pd.concat([a, b], axis=1)

    has_a, has_b = j["first_hypox"].notna(), j["first_hypercap"].notna()
    j["arf_subtype"] = pd.NA
    j.loc[has_a & ~has_b, "arf_subtype"] = "hypoxemic"
    j.loc[~has_a & has_b, "arf_subtype"] = "hypercapnic"
    j.loc[has_a & has_b, "arf_subtype"] = "mixed"

    j["first_arf_dttm"] = j[["first_hypox", "first_hypercap"]].min(axis=1)
    return j.reset_index()[["hospitalization_id", "arf_subtype", "first_arf_dttm"]]
