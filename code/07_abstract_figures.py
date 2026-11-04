"""Step 7 (coordinating site only): the abstract figure from the pooled and per-site aggregates. Run after 06.

  A  each ICU's discharge response to admission load (positive-control OR per +0.1, unit-years pooled per unit),
     ranked, coloured by health system, filled when the gate validates it (lower bound > 1)
  B  the pre-specified estimates: pooled common-effect OR with per-system points

Reads output/final_no_phi/<site>_upload_to_box/{unit_year_gate,model_estimates}.csv and pooled/pooled_estimates.csv.
Systems and units are anonymised (System A, B, C; units ranked). Writes output/final_no_phi/pooled/abstract_figure.{png,pdf}.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import common as c

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False})
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]  # Okabe-Ito, colourblind-safe

src = Path(os.environ.get("CLIF_POOL_DIR", c.output_root() / "final_no_phi"))
out = c.output_root() / "final_no_phi" / "pooled"
sites = sorted(n for n in (p.name.replace("_upload_to_box", "") for p in src.glob("*_upload_to_box")) if n.lower() not in c.DEV_SITES)
system = {s: f"System {chr(65 + i)}" for i, s in enumerate(sites)}
colour = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sites)}

# --- A: per-unit discharge response, unit-years pooled by inverse variance ---
units = []
for s in sites:
    g = pd.read_csv(src / f"{s}_upload_to_box" / "unit_year_gate.csv")
    g = g[g["status"] == "fit"].assign(unit=lambda d: d["unit_year"].str.rsplit("|", n=1).str[0],
                                      b=lambda d: np.log(d["odds_ratio"]),
                                      w=lambda d: 1 / ((np.log(d["ci_high"]) - np.log(d["ci_low"])) / 3.92) ** 2)
    g = g[np.isfinite(g["w"]) & (g["w"] > 0)]  # a unit-year with an unbounded interval carries no weight
    for u, d in g.groupby("unit"):
        b, se = (d["w"] * d["b"]).sum() / d["w"].sum(), np.sqrt(1 / d["w"].sum())
        units.append({"site": s, "unit": u, "or": np.exp(b), "lo": np.exp(b - 1.96 * se), "hi": np.exp(b + 1.96 * se),
                      "n": d["n_landmarks"].fillna(0).sum()})
units = pd.DataFrame(units).sort_values("or").reset_index(drop=True)

# --- B: the pre-specified estimates ---
ROWS = [  # label, model, term
    ("ICU discharge within 24 h (positive control)", "positive_control_icu_discharge_24h", "per10"),
    ("Death or hospice, days 1-8 (primary)", "primary", "per10"),
    ("  instrumented by hospital ED arrivals", "ed_arrivals_iv", "per10"),
    ("  on ventilation or vasopressor", "obligate_icu", "per10"),
    ("  placebo: admissions one week earlier", "placebo_past_strain", "placebo_past10"),
    ("  placebo: ED arrivals after the outcome window", "ed_arrivals_placebo", "placebo_ed10"),
    ("Death or hospice, days 1-31", "hdhd_30d", "per10"),
    ("ICU readmission within 72 h of discharge", "readmit_72h", "per10"),
]
pooled = pd.read_csv(out / "pooled_estimates.csv").set_index(["model", "term"]).sort_index()
per_site = {s: pd.read_csv(src / f"{s}_upload_to_box" / "model_estimates.csv").set_index(["model", "term"]).sort_index()
            for s in sites}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})

x = np.arange(len(units))
for s in sites:
    m = units["site"] == s
    validated = m & (units["lo"] > 1)
    size = 12 + 40 * units["n"] / units["n"].max()
    ax_a.errorbar(x[m], units.loc[m, "or"], yerr=[units.loc[m, "or"] - units.loc[m, "lo"], units.loc[m, "hi"] - units.loc[m, "or"]],
                  fmt="none", ecolor=colour[s], elinewidth=0.6, capsize=0, alpha=0.8)
    ax_a.scatter(x[validated], units.loc[validated, "or"], s=size[validated], color=colour[s], zorder=3,
                 edgecolor="white", linewidth=0.4)
    ax_a.scatter(x[m & ~validated], units.loc[m & ~validated, "or"], s=size[m & ~validated], facecolor="white",
                 edgecolor=colour[s], linewidth=0.8, zorder=3)
ax_a.axhline(1, color="grey", lw=0.6, ls="--")
ax_a.set(yscale="log", xticks=x, xlabel=f"ICUs ranked by estimate (n = {len(units)})",
         ylabel="Odds of ICU discharge within 24 h\nper +0.1 admission load (95% CI)")
ax_a.set_xticklabels([])
ax_a.tick_params(axis="x", length=0)
ax_a.set_yticks([0.7, 1.0, 1.5, 2.0])
ax_a.set_yticklabels(["0.7", "1.0", "1.5", "2.0"])
handles = [plt.Line2D([], [], marker="o", ls="none", color=colour[s], ms=5,
                      label=f"{system[s]} ({int((units['site'] == s).sum())} ICUs)") for s in sites]
ax_a.legend(handles=handles, frameon=False, loc="upper left", fontsize=7, title="filled: lower bound > 1", title_fontsize=7)
ax_a.text(-0.12, 1.04, "A", transform=ax_a.transAxes, fontsize=12, fontweight="bold")

y = np.arange(len(ROWS))[::-1]
for yi, (label, model, term) in zip(y, ROWS):
    if (model, term) in pooled.index:
        p = pooled.loc[(model, term)]
        ax_b.plot([p["ci_low"], p["ci_high"]], [yi, yi], color="black", lw=1.2, solid_capstyle="butt")
        ax_b.plot(p["odds_ratio"], yi, marker="D", color="black", ms=5, zorder=4)
    for j, s in enumerate(sites):
        if (model, term) in per_site[s].index:
            r = per_site[s].loc[(model, term)]
            ax_b.plot(r["odds_ratio"], yi + 0.34 - 0.12 * j, marker="o", ms=3, color=colour[s], alpha=0.9, zorder=3, ls="none")
ax_b.axvline(1, color="grey", lw=0.6, ls="--")
ax_b.set(yticks=y, xscale="log", xlabel="Odds ratio per +0.1 admission load (95% CI)")
ax_b.set_yticklabels([r[0] for r in ROWS], fontsize=7)
ax_b.set_xticks([0.9, 1.0, 1.2, 1.5])
ax_b.set_xticklabels(["0.9", "1.0", "1.2", "1.5"])
ax_b.set_xlim(0.85, 1.6)
ax_b.tick_params(axis="y", length=0)
ax_b.spines["left"].set_visible(False)
ax_b.text(0.99, 0.02, "diamond: pooled, common effect\ndots: health systems", transform=ax_b.transAxes, ha="right", va="bottom",
          fontsize=6.5, color="dimgrey")

fig.tight_layout(w_pad=1.5)
fig.canvas.draw()  # the B label goes at the left edge of the row labels, measured after layout so it cannot push the panel
left = min(t.get_window_extent().x0 for t in ax_b.get_yticklabels()) / fig.bbox.width
fig.text(left, ax_b.get_position().y1 + 0.02, "B", fontsize=12, fontweight="bold", va="bottom")
fig.savefig(out / "abstract_figure.png", dpi=300)
fig.savefig(out / "abstract_figure.pdf")
print(f"wrote {out}/abstract_figure.png and .pdf: {len(units)} units across {len(sites)} systems; "
      f"validated units {int((units['lo'] > 1).sum())}")
