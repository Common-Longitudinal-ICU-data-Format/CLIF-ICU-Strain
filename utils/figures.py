"""Figures: Will's per-unit census week and daily admission strain, and the model's spline curve."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _unit(df: pd.DataFrame) -> pd.Series:
    """Unit label: two hospitals can share a location_name."""
    return df["hospital_id"].astype(str) + " " + df["location_name"].astype(str)


def _unit_axes(n_units: int, height: float = 1.9):
    fig, axes = plt.subplots(n_units, 1, figsize=(11, height * n_units), sharex=True, squeeze=False)
    return fig, axes[:, 0]


def census_week(census: pd.DataFrame, stays: pd.DataFrame, caps: pd.DataFrame, week_start: pd.Timestamp):
    """Will's F1: each unit's rolling census over one week as a step line, ^ at admissions,
    v at discharges, dashed line at capacity (p99)."""
    week_end = week_start + pd.Timedelta(days=7)
    units = caps.assign(unit=_unit(caps)).sort_values("unit")
    census_unit, stays_unit = _unit(census), _unit(stays)
    fig, axes = _unit_axes(len(units))
    for ax, (_, u) in zip(axes, units.iterrows()):
        c = census[census_unit == u["unit"]].sort_values("time")
        before = c[c["time"] < week_start].tail(1).assign(time=week_start)  # census in force at the week start
        c = pd.concat([before, c[(c["time"] >= week_start) & (c["time"] < week_end)]])
        c = pd.concat([c, c.tail(1).assign(time=week_end)])  # carry the last census to the week's end
        ax.step(c["time"], c["census"], where="post", color="k", lw=1)
        s = stays[stays_unit == u["unit"]]
        for col, marker, color in [("in_dttm", "^", "tab:red"), ("out_dttm", "v", "tab:blue")]:
            ev = s[(s[col] >= week_start) & (s[col] < week_end)].sort_values(col)
            y = pd.merge_asof(ev[[col]], c[["time", "census"]], left_on=col, right_on="time")["census"]
            ax.scatter(ev[col], y, marker=marker, color=color, s=22, zorder=3)
        ax.axhline(u["capacity_p99"], color="grey", ls="--", lw=0.8)
        ax.set_ylabel(u["unit"], rotation=0, ha="right", va="center")
        ax.set_ylim(0, np.nanmax([u["capacity_max"], c["census"].max(), 0]) + 1)  # NA capacity: unit under 4 beds
    axes[0].set_title(f"Rolling ICU census, week of {week_start:%Y-%m-%d} (^ admission, v discharge, -- capacity)")
    fig.tight_layout()
    return fig


def daily_admit_strain(daily: pd.DataFrame, window: int = 30):
    """Will's F2: each unit's daily admit_strain (admissions in prior 24h / capacity), with a rolling mean."""
    units = daily.groupby(_unit(daily))
    fig, axes = _unit_axes(units.ngroups)
    for ax, (unit, d) in zip(axes, units):
        d = d.sort_values("day_start")
        t = d["day_start"].dt.tz_localize(None)
        ax.scatter(t, d["admit_strain"], s=2, alpha=0.25, color="grey")
        ax.plot(t, d["admit_strain"].rolling(window, min_periods=window // 2).mean(), color="k", lw=1)
        ax.set_ylabel(unit, rotation=0, ha="right", va="center")
    axes[0].set_title(f"Daily ICU admissions / capacity (w = 1), {window}-day rolling mean")
    fig.tight_layout()
    return fig


def monthly_strain(monthly: pd.DataFrame, col: str, ylabel: str):
    """Monthly mean of a strain column, one line per group (hospital or unit)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    key = _unit(monthly) if "location_name" in monthly else monthly["hospital_id"]
    for name, g in monthly.groupby(key):
        ax.plot(pd.PeriodIndex(g["month"], freq="M").to_timestamp(), g[col], label=name, lw=1)
    ax.set(ylabel=ylabel)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def forest(rows: pd.DataFrame, title: str):
    """Odds ratio and 95% CI per site, pooled last. `rows` has label, odds_ratio, ci_low, ci_high."""
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    ax.errorbar(rows["odds_ratio"], y, xerr=[rows["odds_ratio"] - rows["ci_low"], rows["ci_high"] - rows["odds_ratio"]],
                fmt="o", color="k", capsize=3)
    ax.axvline(1, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y, rows["label"])
    ax.set(xlabel="Odds ratio per +0.1 admission strain (95% CI)", title=title)
    fig.tight_layout()
    return fig


def spline_curve(curve: pd.DataFrame, xlabel: str, ref: float, title: str):
    """Odds ratio (95% CI) vs the reference value across the exposure grid."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["x"], np.exp(curve["estimate"]), color="k")
    ax.fill_between(curve["x"], np.exp(curve["estimate"] - 1.96 * curve["se"]),
                    np.exp(curve["estimate"] + 1.96 * curve["se"]), alpha=0.2, color="k")
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set(xlabel=xlabel, ylabel=f"Odds ratio vs {ref}", title=title)
    fig.tight_layout()
    return fig
