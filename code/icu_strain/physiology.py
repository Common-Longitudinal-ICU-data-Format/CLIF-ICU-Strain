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

    rules = {
        "room_air_spo2": is_room_air & (df["spo2"] < d.SPO2_HYPOXEMIA),
        "room_air_pao2": is_room_air & (df["po2"] <= d.PAO2_HYPOXEMIA),
        "pf_ratio": (df["po2"] / df["fio2"]) <= d.PF_RATIO_THRESHOLD,
        "sf_ratio": (~df["has_abg"]) & ((df["spo2"] / df["fio2"]) <= d.SF_RATIO_THRESHOLD),
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
        return pd.DataFrame(columns=["hospitalization_id", "dttm", "pco2", "ph"])

    paired = pco2.merge(ph, on="hospitalization_id", how="inner")
    within = (paired["pco2_dttm"] - paired["ph_dttm"]).abs() <= pd.Timedelta(minutes=tolerance_min)
    paired = paired[within]

    qualifies = (paired["pco2"] >= d.PACO2_HYPERCAPNIA) & (paired["ph"] < d.PH_HYPERCAPNIA)
    out = paired[qualifies].copy()
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
