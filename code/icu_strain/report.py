"""Render the self-contained ICU strain ARF HTML report.

This module holds only templating/rendering logic: it turns already-computed
tables and chart data URIs into one portable HTML document. It does no I/O
and never touches PHI -- the driver (05_generate_report.py) owns reading the
aggregate CSVs and building the figures; this module just lays them out.

Definitions are rendered directly from `icu_strain.definitions.CRITERIA_TEXT`
and `SF_FALLBACK_CAVEAT` rather than retyped, so the documented criteria
cannot drift from the constants that actually drove the classification.

Note on escaping: table cells and the definitions/caveat prose are emitted
unescaped (`escape=False`), matching the CSV disclosure-control convention
of a literal `<10` for a suppressed cell. A stray `<` not followed by a
letter, `!`, or `?` is not a valid tag opener per the HTML5 tokenizer, so
browsers render it as plain text rather than swallowing content -- `<10` and
`<=` both display correctly without escaping.
"""

from __future__ import annotations

import html as _html

import pandas as pd

from . import definitions as d

# --- palette (dataviz skill default categorical + chrome tokens) ---
# Light-mode values only, reused by the driver's matplotlib figures so chart
# PNGs and the CSS variables below can never drift apart. Charts are baked in
# light-mode colors and shown in a bordered card (see img.chart below), which
# keeps them legible against both a light and a dark page background rather
# than requiring two rendered variants per figure.
PALETTE = {
    "surface": "#fcfcfb",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
    "series_1": "#2a78d6",  # blue
    "series_2": "#eb6834",  # orange
    "series_3": "#1baf7a",  # aqua
    "diverging_neg": "#2a78d6",  # before ICU (blue)
    "diverging_pos": "#e34948",  # after ICU (red)
}

_CSS = """
<style>
:root {
  color-scheme: light;
  --page:            #f9f9f7;
  --surface-1:       #fcfcfb;
  --text-primary:    #0b0b0b;
  --text-secondary:  #52514e;
  --text-muted:      #898781;
  --gridline:        #e1e0d9;
  --baseline:        #c3c2b7;
  --border:          rgba(11,11,11,0.10);
  --series-1:        #2a78d6; /* blue: hypoxemic / IMV any */
  --series-2:        #eb6834; /* orange: hypercapnic / IMV new */
  --series-3:        #1baf7a; /* aqua: mixed */
  --diverging-neg:   #2a78d6; /* before ICU */
  --diverging-pos:   #e34948; /* after ICU */
  --diverging-mid:   #f0efec;
  --status-warning:  #fab219;
  --status-critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page:            #0d0d0d;
    --surface-1:       #1a1a19;
    --text-primary:    #ffffff;
    --text-secondary:  #c3c2b7;
    --text-muted:      #898781;
    --gridline:        #2c2c2a;
    --baseline:        #383835;
    --border:          rgba(255,255,255,0.10);
    --series-1:        #3987e5;
    --series-2:        #d95926;
    --series-3:        #199e70;
    --diverging-neg:   #3987e5;
    --diverging-pos:   #e66767;
    --diverging-mid:   #383835;
    --status-warning:  #fab219;
    --status-critical: #e66767;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5;
}
.page { max-width: 920px; margin: 0 auto; padding: 32px 24px 96px; }
header.report-header { margin-bottom: 32px; }
header.report-header h1 { margin: 0 0 4px; font-size: 28px; }
header.report-header p { margin: 0; color: var(--text-secondary); }
section { margin: 40px 0; }
section > h2 {
  font-size: 20px;
  margin: 0 0 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--gridline);
}
section > h3 { font-size: 15px; margin: 20px 0 8px; color: var(--text-secondary); }
p { margin: 8px 0; }
p.prose { color: var(--text-secondary); }
.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 12px 0;
}
.callout {
  border-radius: 8px;
  padding: 12px 16px;
  margin: 12px 0;
  border-left: 4px solid var(--status-warning);
  background: var(--surface-1);
  color: var(--text-primary);
}
.callout strong.label {
  display: inline-block;
  margin-bottom: 4px;
}
.callout.critical { border-left-color: var(--status-critical); }
dl.criteria dt {
  font-weight: 600;
  margin-top: 12px;
}
dl.criteria dd {
  margin: 4px 0 0;
  color: var(--text-secondary);
}
.table-wrap { overflow-x: auto; margin: 12px 0; }
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 14px;
  font-variant-numeric: tabular-nums;
}
table th, table td {
  padding: 6px 12px;
  text-align: right;
  border-bottom: 1px solid var(--gridline);
  white-space: nowrap;
}
table th:first-child, table td:first-child { text-align: left; }
table thead th {
  color: var(--text-secondary);
  font-weight: 600;
  border-bottom: 1px solid var(--baseline);
}
img.chart {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 16px 0;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: #fcfcfb;
}
.legend-note { font-size: 13px; color: var(--text-muted); }
footer.report-footer {
  margin-top: 48px;
  padding-top: 16px;
  border-top: 1px solid var(--gridline);
  color: var(--text-muted);
  font-size: 13px;
}
</style>
""".strip()


def _format_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-format every cell to its final display string.

    Each column is handled by its own dtype:
    - float64 (hour-scale deltas, percentages): one decimal place. Six
      decimal places on a clinical median duration implies a precision the
      underlying measurement doesn't support, and the chart/prose elsewhere
      in the report already round to roughly this precision, so the tables
      should match rather than contradict them.
    - integer / nullable Int64 (counts): rendered as a plain whole number,
      never "425.0".
    - anything else (strings): passed through unchanged.

    NA is handled per-cell with `pd.isna`, not via `DataFrame.to_html`'s
    `na_rep`: `na_rep` only reliably fires for a real float64 NaN. An Int64
    column's `pd.NA` (exactly what `aggregate.suppress_small_cells` /
    `suppress_small_rows` produce) or an object column's `pd.NA` both fall
    through `na_rep` and print as pandas' own "<NA>" repr instead -- silently
    defeating the suppression marker. Formatting to strings ourselves, cell
    by cell, sidesteps that dtype-dependent behavior entirely.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_float_dtype(s):
            out[col] = s.map(lambda v: "<10" if pd.isna(v) else f"{v:.1f}")
        elif pd.api.types.is_integer_dtype(s):
            out[col] = s.map(lambda v: "<10" if pd.isna(v) else f"{int(v)}")
        else:
            out[col] = s.map(lambda v: "<10" if pd.isna(v) else v)
    return out


def _table_html(df: pd.DataFrame) -> str:
    """Render a DataFrame as HTML with suppressed cells shown as `<10`."""
    inner = _format_display_frame(df).to_html(index=False, escape=False, border=0)
    return f'<div class="table-wrap">{inner}</div>'


def _chart_html(charts: dict, key: str, alt: str) -> str:
    uri = charts.get(key)
    if not uri:
        return ""
    return f'<img class="chart" src="{_html.escape(uri, quote=True)}" alt="{_html.escape(alt)}">'


def _header_section(meta: dict) -> str:
    site = _html.escape(str(meta.get("site", "")))
    year = _html.escape(str(meta.get("year", "")))
    generated = _html.escape(str(meta.get("generated_at", "")))
    subtitle_bits = [b for b in (f"Site: {site}" if site else "", f"Year: {year}" if year else "",
                                  f"Generated: {generated}" if generated else "") if b]
    subtitle = " &middot; ".join(subtitle_bits)
    return f"""
<header class="report-header">
  <h1>ICU Respiratory Strain &mdash; ARF Subtype Report</h1>
  <p>{subtitle}</p>
</header>
"""


def _definitions_section() -> str:
    labels = {
        "hypoxemic": "Hypoxemic ARF subtype",
        "hypercapnic": "Hypercapnic ARF subtype",
        "mixed": "Mixed ARF subtype",
    }
    dt_dd = "".join(
        f"<dt>{_html.escape(label)}</dt><dd>{d.CRITERIA_TEXT[key]}</dd>"
        for key, label in labels.items()
    )
    return f"""
<section id="definitions">
  <h2>Definitions</h2>
  <p class="prose">All thresholds below are the constants actually used to classify every
  hospitalization in this report &mdash; generated directly from
  <code>icu_strain/definitions.py</code>, never retyped, so this section cannot drift from
  what the pipeline computed.</p>
  <dl class="criteria">
    {dt_dd}
  </dl>
  <div class="callout critical">
    <strong class="label">NOT ARDS.</strong>
    {d.CRITERIA_TEXT["not_ards"]}
  </div>
  <div class="callout">
    <strong class="label">SpO2/FiO2 fallback caveat.</strong>
    {d.SF_FALLBACK_CAVEAT}
  </div>
</section>
"""


def _attrition_section(tables: dict) -> str:
    return f"""
<section id="attrition">
  <h2>Cohort attrition</h2>
  <p class="prose">Each step below counts adult hospitalizations remaining after applying the
  named criterion, in the order the pipeline applies it.</p>
  {_table_html(tables["attrition"])}
</section>
"""


def _deliverable1_section(tables: dict, charts: dict) -> str:
    imv = tables["monthly_imv"]
    has_both = "n_imv_any" in imv.columns and "n_imv_new" in imv.columns
    stats_note = ""
    if has_both:
        avg_any = pd.to_numeric(imv["n_imv_any"], errors="coerce").mean()
        avg_new = pd.to_numeric(imv["n_imv_new"], errors="coerce").mean()
        if pd.notna(avg_any) and pd.notna(avg_new) and avg_any > 0:
            pct_arrived = 100.0 * (1 - avg_new / avg_any)
            stats_note = (
                f" In this report, that averages roughly {avg_any:.0f} per month for "
                f"<em>any</em> IMV against {avg_new:.0f} per month for genuinely "
                f"<em>new</em> intubations &mdash; about {pct_arrived:.0f}% of ventilated ICU "
                "patients arrived already intubated."
            )
    return f"""
<section id="deliverable-1">
  <h2>Deliverable 1: Monthly IMV volume</h2>
  <p class="prose"><strong>n_imv_any</strong> counts any invasive mechanical ventilation
  during the index ICU stay, including patients who arrived already intubated (from the ED,
  OR, or an outside transfer). <strong>n_imv_new</strong> counts only genuine new
  intubations that began during the ICU stay. The gap between the two series is itself a
  finding: it is the already-ventilated-on-arrival population, not analysis noise.{stats_note}</p>
  {_chart_html(charts, "imv", "Monthly IMV volume: any IMV vs. new IMV initiations")}
  {_table_html(imv)}
</section>
"""


def _deliverable2_section(tables: dict, charts: dict) -> str:
    return f"""
<section id="deliverable-2">
  <h2>Deliverable 2: Monthly ARF subtype counts</h2>
  <p class="prose">Counts of hospitalizations meeting each ARF subtype's criteria (see
  Definitions above), by month of ICU admission. These are <strong>ARF subtypes, not
  ARDS</strong> &mdash; see the NOT ARDS note above.</p>
  <div class="callout">
    <strong class="label">SpO2/FiO2 fallback caveat.</strong>
    {d.SF_FALLBACK_CAVEAT} Roughly half of this cohort has no arterial blood gas anywhere in
    the window, so a large share of the hypoxemic count below is driven by this uncapped
    fallback rather than an ABG-confirmed ratio.
  </div>
  {_chart_html(charts, "subtypes", "Monthly ARF subtype counts, stacked by subtype")}
  {_table_html(tables["monthly_arf_subtypes"])}
</section>
"""


def _deliverable3_section(tables: dict, charts: dict) -> str:
    return f"""
<section id="deliverable-3">
  <h2>Deliverable 3: Time from ARF onset to ICU admission, by subtype</h2>
  <p class="prose"><code>delta_hours = first_arf_dttm &minus; icu_admit_dttm</code>.
  <strong>Negative values are the headline finding here</strong>: they mean the patient
  already met ARF criteria <em>before</em> ICU transfer, not that criteria were met some
  number of hours after admission. Zero is the moment of ICU admission; the reference line
  in the chart below marks it, with "before ICU" and "after ICU" labelled on either side.</p>
  {_chart_html(charts, "delta", "Signed hours from ARF onset to ICU admission, by subtype, "
                                 "with a zero reference line for ICU admission")}
  {_table_html(tables["time_to_icu_by_subtype"])}
</section>
"""


def _data_quality_section(tables: dict) -> str:
    qc = tables["quality_checks"]
    notes = []
    if "pct_with_abg" in qc.columns:
        abg = pd.to_numeric(qc["pct_with_abg"], errors="coerce")
        if abg.notna().any():
            notes.append(
                f"<li>Monthly arterial blood gas (ABG) availability ranges "
                f"{abg.min():.0f}%&ndash;{abg.max():.0f}% of hospitalizations. Because the "
                "hypercapnic and mixed subtypes require an arterial PaCO2/pH pair, this "
                "availability rate is a ceiling on how many hospitalizations could ever "
                "qualify for those two subtypes &mdash; a low-ABG month understates "
                "hypercapnic/mixed counts, it does not mean those patients were not "
                "hypercapnic.</li>"
            )
    if "pct_arterial_dup_key_conflicting_overall" in qc.columns:
        conflict = pd.to_numeric(qc["pct_arterial_dup_key_conflicting_overall"], errors="coerce").iloc[0]
        identical = pd.to_numeric(qc["pct_arterial_dup_key_identical_overall"], errors="coerce").iloc[0]
        notes.append(
            f"<li>Among duplicate-timestamp arterial lab rows (same hospitalization, same "
            f"draw time), {conflict:.2f}% carry conflicting result values and "
            f"{identical:.2f}% are exact duplicates. The identical share is handled safely "
            "by de-duplication; the conflicting share means two different results were "
            "recorded for what should be one draw, which is a source of classification "
            "noise the pipeline cannot fully resolve on its own.</li>"
        )
    notes_html = f"<ul>{''.join(notes)}</ul>" if notes else ""
    return f"""
<section id="data-quality">
  <h2>Data quality</h2>
  {notes_html}
  <h3>Monthly quality checks</h3>
  {_table_html(qc)}
  <h3>Outlier handling</h3>
  {_table_html(tables["outlier_summary"])}
</section>
"""


def _limitations_section() -> str:
    return f"""
<section id="limitations">
  <h2>Limitations</h2>
  <ul>
    <li>{d.CRITERIA_TEXT["not_ards"]}</li>
    <li>{d.SF_FALLBACK_CAVEAT}</li>
    <li>Single-site, single-year (2024) results. Subtype and IMV rates are not adjusted for
    case mix and should not be extrapolated to other sites or years without re-running the
    pipeline locally.</li>
    <li>Cells representing fewer than {d.MIN_CELL_SIZE} hospitalizations are suppressed and
    shown as <code>&lt;{d.MIN_CELL_SIZE}</code> rather than a blank, so suppression is
    distinguishable from missing data.</li>
    <li>Venous and capillary blood gases are excluded from the hypercapnic criterion by
    design; only arterial draws count.</li>
  </ul>
</section>
"""


def render_report(tables: dict, charts: dict, meta: dict) -> str:
    """Render the complete self-contained HTML report.

    `tables` keys: attrition, monthly_imv, monthly_arf_subtypes,
    time_to_icu_by_subtype, quality_checks, outlier_summary.
    `charts` keys (all optional): imv, subtypes, delta -> base64 PNG data URIs.
    `meta` keys: site, year, and optionally generated_at.
    """
    site = _html.escape(str(meta.get("site", "site")))
    year = _html.escape(str(meta.get("year", "")))
    sections = "\n".join([
        _header_section(meta),
        _definitions_section(),
        _attrition_section(tables),
        _deliverable1_section(tables, charts),
        _deliverable2_section(tables, charts),
        _deliverable3_section(tables, charts),
        _data_quality_section(tables),
        _limitations_section(),
    ])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ICU Strain ARF Subtype Report &mdash; {site} {year}</title>
{_CSS}
</head>
<body>
<div class="page">
{sections}
<footer class="report-footer">
  <p>Generated from aggregate, disclosure-controlled CSVs in
  <code>output/final_no_phi/</code> only. This document contains no PHI and no data
  restricted below the {d.MIN_CELL_SIZE}-hospitalization suppression threshold.</p>
</footer>
</div>
</body>
</html>"""
