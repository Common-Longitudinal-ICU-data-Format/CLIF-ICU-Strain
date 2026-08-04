import pandas as pd
from icu_strain.physiology import evaluate_hypoxemia, evaluate_hypercapnia, classify_arf
from icu_strain import definitions as d


def _obs(rows):
    return pd.DataFrame(
        rows,
        columns=["hospitalization_id", "dttm", "spo2", "po2", "fio2", "device_category", "has_abg"],
    ).assign(dttm=lambda x: pd.to_datetime(x["dttm"]))


def test_room_air_spo2_below_90_qualifies():
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", 88, None, 0.21, "Room Air", True)]))
    assert list(out["hypoxemia_rule"]) == ["room_air_spo2"]


def test_room_air_spo2_at_90_does_not_qualify():
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", 90, None, 0.21, "Room Air", True)]))
    assert out.empty


def test_low_spo2_on_supplemental_o2_does_not_trigger_room_air_rule():
    # 88% on 6L nasal cannula must not match the room-air rule.
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", 88, None, 0.40, "Nasal Cannula", True)]))
    assert "room_air_spo2" not in set(out["hypoxemia_rule"])


def test_pf_ratio_at_or_below_300_qualifies():
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", None, 120.0, 0.40, "IMV", True)]))
    assert list(out["hypoxemia_rule"]) == ["pf_ratio"]  # 120/0.40 = 300


def test_sf_fallback_only_applies_without_abg():
    with_abg = _obs([("H1", "2024-03-01 01:00", 90, None, 0.40, "Nasal Cannula", True)])
    without_abg = _obs([("H2", "2024-03-01 01:00", 90, None, 0.40, "Nasal Cannula", False)])
    assert evaluate_hypoxemia(with_abg).empty          # 90/0.40 = 225 but ABG exists
    assert not evaluate_hypoxemia(without_abg).empty   # fallback allowed


def test_null_fio2_disqualifies_ratio_rules():
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", 85, 50.0, None, "Nasal Cannula", False)]))
    assert out.empty


def test_room_air_pao2_at_60_qualifies():
    # PaO2 60 on room air (fio2 0.21) also happens to satisfy pf_ratio
    # (60/0.21 ~ 285.7 <= 300), since pf_ratio is not room-air-gated. Assert
    # room_air_pao2 fired rather than asserting it is the only rule.
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", None, 60.0, 0.21, "Room Air", True)]))
    assert "room_air_pao2" in set(out["hypoxemia_rule"])


def test_room_air_pao2_at_61_does_not_qualify():
    # Isolate the room_air_pao2 boundary from the pf_ratio rule by using a
    # not-room-air device_category, so only room_air_pao2's gate is tested
    # (pf_ratio has no device gate and would otherwise also fire at
    # 61/0.21 ~ 290.5 <= 300).
    out = evaluate_hypoxemia(_obs([("H1", "2024-03-01 01:00", None, 61.0, None, "Room Air", True)]))
    assert "room_air_pao2" not in set(out["hypoxemia_rule"])


def test_has_abg_as_float_nan_treated_as_no_abg():
    # A left-merge miss upstream commonly yields has_abg as float64 with NaN
    # rather than a clean bool. `~` on that dtype must not raise, and NaN
    # must be treated as "no ABG" so the S/F fallback still applies.
    obs = _obs([("H1", "2024-03-01 01:00", 90, None, 0.40, "Nasal Cannula", float("nan"))])
    obs["has_abg"] = obs["has_abg"].astype("float64")
    out = evaluate_hypoxemia(obs)
    assert list(out["hypoxemia_rule"]) == ["sf_ratio"]


def _labs(rows):
    return pd.DataFrame(
        rows, columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"]
    ).assign(lab_result_dttm=lambda x: pd.to_datetime(x["lab_result_dttm"]))


def test_paired_arterial_gas_qualifies():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", d.ABG_PCO2, 50.0),
        ("H1", "2024-03-01 01:00", d.ABG_PH, 7.30),
    ]), tolerance_min=5)
    assert len(out) == 1


def test_high_pco2_with_normal_ph_does_not_qualify():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", d.ABG_PCO2, 50.0),
        ("H1", "2024-03-01 01:00", d.ABG_PH, 7.40),
    ]), tolerance_min=5)
    assert out.empty


def test_venous_gas_never_qualifies():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", "pco2_venous", 55.0),
        ("H1", "2024-03-01 01:00", "ph_venous", 7.30),
    ]), tolerance_min=5)
    assert out.empty


def test_gases_beyond_tolerance_do_not_pair():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", d.ABG_PCO2, 50.0),
        ("H1", "2024-03-01 01:30", d.ABG_PH, 7.30),
    ]), tolerance_min=5)
    assert out.empty


def test_paco2_at_45_with_low_ph_qualifies():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", d.ABG_PCO2, 45.0),
        ("H1", "2024-03-01 01:00", d.ABG_PH, 7.30),
    ]), tolerance_min=5)
    assert len(out) == 1


def test_ph_at_735_does_not_qualify():
    out = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", d.ABG_PCO2, 50.0),
        ("H1", "2024-03-01 01:00", d.ABG_PH, 7.35),
    ]), tolerance_min=5)
    assert out.empty


def test_missing_arterial_category_does_not_crash_classify_arf():
    # A site whose category mapping has no pco2_arterial or ph_arterial rows
    # at all (only venous here) must flow cleanly through evaluate_hypercapnia
    # and classify_arf rather than raising on a dtype mismatch.
    hypercap = evaluate_hypercapnia(_labs([
        ("H1", "2024-03-01 01:00", "pco2_venous", 55.0),
        ("H1", "2024-03-01 01:00", "ph_venous", 7.30),
    ]), tolerance_min=5)
    hypox = pd.DataFrame({"hospitalization_id": pd.Series(dtype=object), "dttm": pd.Series(dtype="datetime64[ns]")})
    out = classify_arf(hypox, hypercap)
    assert out.empty


def test_classify_assigns_mixed_and_earliest_timestamp():
    hypox = pd.DataFrame({
        "hospitalization_id": ["H1", "H2"],
        "dttm": pd.to_datetime(["2024-03-01 05:00", "2024-03-02 01:00"]),
    })
    hypercap = pd.DataFrame({
        "hospitalization_id": ["H1", "H3"],
        "dttm": pd.to_datetime(["2024-03-01 02:00", "2024-03-03 01:00"]),
    })
    out = classify_arf(hypox, hypercap).set_index("hospitalization_id")
    assert out.loc["H1", "arf_subtype"] == "mixed"
    assert out.loc["H1", "first_arf_dttm"] == pd.Timestamp("2024-03-01 02:00")
    assert out.loc["H2", "arf_subtype"] == "hypoxemic"
    assert out.loc["H3", "arf_subtype"] == "hypercapnic"
