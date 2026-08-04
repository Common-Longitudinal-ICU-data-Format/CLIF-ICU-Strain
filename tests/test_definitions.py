from icu_strain import definitions as d


def test_thresholds_match_manuscript():
    assert d.SPO2_HYPOXEMIA == 90
    assert d.PAO2_HYPOXEMIA == 60
    assert d.PF_RATIO_THRESHOLD == 300
    assert d.SF_RATIO_THRESHOLD == 315
    assert d.PACO2_HYPERCAPNIA == 45
    assert d.PH_HYPERCAPNIA == 7.35


def test_abg_categories_are_arterial_only():
    cats = {d.ABG_PO2, d.ABG_PCO2, d.ABG_PH}
    assert cats == {"po2_arterial", "pco2_arterial", "ph_arterial"}
    assert not any("venous" in c for c in cats)


def test_criteria_text_covers_all_subtypes():
    for key in ("hypoxemic", "hypercapnic", "mixed", "not_ards"):
        assert key in d.CRITERIA_TEXT
        assert len(d.CRITERIA_TEXT[key]) > 40


def test_report_never_calls_this_ards():
    for key in ("hypoxemic", "hypercapnic", "mixed"):
        assert "ARDS" not in d.CRITERIA_TEXT[key]
