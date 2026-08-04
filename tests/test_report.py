import pandas as pd
from icu_strain.report import render_report
from icu_strain import definitions as d


def _tables():
    return {
        "attrition": pd.DataFrame({"step": ["adults with ICU stay"], "n": [100]}),
        "monthly_imv": pd.DataFrame({"month": ["2024-01"], "n_new_imv": [42]}),
        "monthly_arf_subtypes": pd.DataFrame({
            "month": ["2024-01"], "hypoxemic": [30], "hypercapnic": [11], "mixed": [15], "total": [56],
        }),
        "time_to_icu_by_subtype": pd.DataFrame({
            "arf_subtype": ["hypoxemic"], "n": [30], "median_hours": [-2.5],
            "q25_hours": [-8.0], "q75_hours": [1.0], "pct_before_icu": [62.0],
        }),
        "quality_checks": pd.DataFrame({
            "month": ["2024-01"], "n_hospitalizations": [100],
            "pct_with_abg": [55.0], "pct_with_spo2": [99.0], "pct_with_fio2": [80.0],
        }),
        "outlier_summary": pd.DataFrame({
            "table": ["vitals"], "values_before": [100], "values_after": [98], "values_removed": [2],
        }),
    }


def test_report_is_a_complete_html_document():
    html = render_report(_tables(), charts={}, meta={"site": "ucmc", "year": 2024})
    assert html.strip().startswith("<!doctype html>")
    assert "</html>" in html


def test_report_states_these_are_not_ards():
    html = render_report(_tables(), charts={}, meta={"site": "ucmc", "year": 2024})
    assert "NOT ARDS" in html
    assert d.CRITERIA_TEXT["not_ards"][:40] in html


def test_report_embeds_the_implemented_thresholds():
    html = render_report(_tables(), charts={}, meta={"site": "ucmc", "year": 2024})
    for token in ("90", "60", "300", "315", "45", "7.35"):
        assert token in html


def test_report_carries_the_sf_fallback_caveat():
    html = render_report(_tables(), charts={}, meta={"site": "ucmc", "year": 2024})
    assert d.SF_FALLBACK_CAVEAT[:40] in html


def test_suppressed_cells_render_as_marker_not_blank():
    tables = _tables()
    tables["monthly_imv"] = pd.DataFrame({"month": ["2024-01"], "n_new_imv": [pd.NA]})
    html = render_report(tables, charts={}, meta={"site": "ucmc", "year": 2024})
    assert "&lt;10" in html or "<10" in html


def test_charts_are_embedded_inline_when_supplied():
    html = render_report(_tables(), charts={"imv": "data:image/png;base64,AAA"},
                         meta={"site": "ucmc", "year": 2024})
    assert "data:image/png;base64,AAA" in html
