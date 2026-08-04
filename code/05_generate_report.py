"""Build the self-contained HTML report. Run from repo root.

Reads only the disclosure-controlled aggregate CSVs already written to
output/final_no_phi/ -- never output/intermediate_phi/ and never the site
data directory. That restriction (not this script's care in using the data)
is what makes the produced report structurally PHI-safe: there is nothing
row-level for it to leak, because it never loads anything row-level.
"""

import base64
import datetime as dt
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from icu_strain.paths import final_dir
from icu_strain.report import PALETTE, render_report

out_dir = final_dir()

SUBTYPE_ORDER = ["hypoxemic", "hypercapnic", "mixed"]
SUBTYPE_LABELS = {"hypoxemic": "Hypoxemic", "hypercapnic": "Hypercapnic", "mixed": "Mixed"}
SUBTYPE_COLORS = {
    "hypoxemic": PALETTE["series_1"],
    "hypercapnic": PALETTE["series_2"],
    "mixed": PALETTE["series_3"],
}


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _style_axes(ax):
    ax.set_facecolor(PALETTE["surface"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["baseline"])
        ax.spines[spine].set_linewidth(1)
    ax.tick_params(colors=PALETTE["text_secondary"], labelsize=9)
    ax.yaxis.grid(True, color=PALETTE["gridline"], linewidth=1)
    ax.set_axisbelow(True)


def make_imv_chart(monthly_imv: pd.DataFrame) -> str:
    """Line chart: n_imv_any vs n_imv_new by month.

    Two series, so a legend is mandatory; direct end-labels supplement it.
    The gap between the two lines is the finding (already-intubated-on-
    arrival), so both are plotted rather than just n_imv_new.
    """
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=PALETTE["surface"])
    _style_axes(ax)

    months = monthly_imv["month"]
    x = np.arange(len(months))
    any_vals = pd.to_numeric(monthly_imv["n_imv_any"], errors="coerce")
    new_vals = pd.to_numeric(monthly_imv["n_imv_new"], errors="coerce")

    ax.plot(x, any_vals, color=SUBTYPE_COLORS["hypoxemic"], linewidth=2, marker="o",
            markersize=6, markeredgecolor=PALETTE["surface"], markeredgewidth=1.5,
            label="Any IMV (incl. already intubated on arrival)")
    ax.plot(x, new_vals, color=SUBTYPE_COLORS["hypercapnic"], linewidth=2, marker="o",
            markersize=6, markeredgecolor=PALETTE["surface"], markeredgewidth=1.5,
            label="New IMV (new in-stay initiation)")

    for vals, color in ((any_vals, SUBTYPE_COLORS["hypoxemic"]), (new_vals, SUBTYPE_COLORS["hypercapnic"])):
        last = vals.dropna()
        if not last.empty:
            xi, yi = last.index[-1], last.iloc[-1]
            ax.annotate(f"{yi:,.0f}", (x[xi], yi), textcoords="offset points", xytext=(6, 0),
                        va="center", fontsize=9, color=PALETTE["text_primary"])

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("Hospitalizations", color=PALETTE["text_secondary"], fontsize=10)
    ax.set_title("Monthly IMV volume: any IMV vs. new IMV initiations",
                 color=PALETTE["text_primary"], fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1,
              fontsize=9, labelcolor=PALETTE["text_secondary"])
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def make_subtype_chart(monthly_subtypes: pd.DataFrame) -> str:
    """Stacked bar: monthly ARF subtype counts (part-to-whole, categorical color)."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=PALETTE["surface"])
    _style_axes(ax)

    months = monthly_subtypes["month"]
    x = np.arange(len(months))
    bottom = np.zeros(len(months))
    for subtype in SUBTYPE_ORDER:
        vals = pd.to_numeric(monthly_subtypes[subtype], errors="coerce").fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottom, width=0.6, color=SUBTYPE_COLORS[subtype],
               edgecolor=PALETTE["surface"], linewidth=2, label=SUBTYPE_LABELS[subtype])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("Hospitalizations", color=PALETTE["text_secondary"], fontsize=10)
    ax.set_title("Monthly ARF subtype counts", color=PALETTE["text_primary"], fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              fontsize=9, labelcolor=PALETTE["text_secondary"])
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def make_delta_chart(time_to_icu: pd.DataFrame) -> str:
    """Diverging horizontal bar: signed median hours to ICU admission by subtype.

    Zero is ICU admission. Negative bars (before ICU) take the diverging
    "cool" pole, positive bars (after ICU) the "warm" pole, with q25-q75
    error whiskers and an explicitly labelled zero reference line.
    """
    df = time_to_icu.copy()
    df["median_hours"] = pd.to_numeric(df["median_hours"], errors="coerce")
    df = df.dropna(subset=["median_hours"])
    df["_order"] = df["arf_subtype"].apply(
        lambda s: SUBTYPE_ORDER.index(s) if s in SUBTYPE_ORDER else len(SUBTYPE_ORDER))
    df = df.sort_values("_order")

    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=PALETTE["surface"])
    _style_axes(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    y = np.arange(len(df))
    medians = df["median_hours"].to_numpy()
    q25 = pd.to_numeric(df["q25_hours"], errors="coerce").to_numpy()
    q75 = pd.to_numeric(df["q75_hours"], errors="coerce").to_numpy()
    colors = [PALETTE["diverging_neg"] if m < 0 else PALETTE["diverging_pos"] for m in medians]

    ax.barh(y, medians, height=0.5, color=colors)
    lower_err = np.clip(medians - q25, 0, None)
    upper_err = np.clip(q75 - medians, 0, None)
    ax.errorbar(medians, y, xerr=[lower_err, upper_err], fmt="none",
                ecolor=PALETTE["text_secondary"], elinewidth=1.5, capsize=4)

    # Value labels sit above each bar (not beside it) so they never cross the
    # errorbar whisker running through the bar's own row.
    for yi, m in zip(y, medians):
        ax.annotate(f"{m:+.1f}h", (m, yi), textcoords="offset points", xytext=(0, 14),
                    va="bottom", ha="center", fontsize=9, color=PALETTE["text_primary"])

    ax.axvline(0, color=PALETTE["text_primary"], linewidth=1.5, zorder=3)
    xmin, xmax = ax.get_xlim()
    ax.annotate("← before ICU", (xmin * 0.5, -0.85), ha="center", va="top", fontsize=9,
                color=PALETTE["text_secondary"])
    ax.annotate("after ICU →", (xmax * 0.5, -0.85), ha="center", va="top", fontsize=9,
                color=PALETTE["text_secondary"])

    ax.set_yticks(y)
    ax.set_yticklabels([SUBTYPE_LABELS.get(s, s) for s in df["arf_subtype"]])
    ax.set_xlabel("Hours relative to ICU admission (median, IQR)",
                  color=PALETTE["text_secondary"], fontsize=10)
    ax.set_title("Time from ARF onset to ICU admission, by subtype",
                 color=PALETTE["text_primary"], fontsize=12, loc="left")
    ax.set_ylim(-1.3, len(df) + 0.5)
    ax.text(0, len(df) + 0.15, "ICU admission (0h)", ha="center", va="bottom", fontsize=9,
            color=PALETTE["text_primary"])
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def main() -> None:
    tables = {
        "attrition": pd.read_csv(out_dir / "attrition.csv"),
        "monthly_imv": pd.read_csv(out_dir / "monthly_imv.csv"),
        "monthly_arf_subtypes": pd.read_csv(out_dir / "monthly_arf_subtypes.csv"),
        "time_to_icu_by_subtype": pd.read_csv(out_dir / "time_to_icu_by_subtype.csv"),
        "quality_checks": pd.read_csv(out_dir / "quality_checks.csv"),
        "outlier_summary": pd.read_csv(out_dir / "outlier_summary.csv"),
    }

    charts = {
        "imv": make_imv_chart(tables["monthly_imv"]),
        "subtypes": make_subtype_chart(tables["monthly_arf_subtypes"]),
        "delta": make_delta_chart(tables["time_to_icu_by_subtype"]),
    }

    meta = {
        "site": "ucmc",
        "year": 2024,
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    html = render_report(tables, charts, meta)
    report_path = out_dir / "icu_strain_report.html"
    report_path.write_text(html)
    print(f"Wrote {report_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
