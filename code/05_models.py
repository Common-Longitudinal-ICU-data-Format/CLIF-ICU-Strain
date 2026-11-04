"""Step 5: the pre-specified models (v3 incumbent landmark, E2; v4 exposures and process outcomes), Table 1,
and the estimates this site shares. Run after 04.

Writes to <site>_upload_to_box/: model_estimates.csv, unit_year_gate.csv, table1.csv, spline_primary.png.
Odds ratios are per +0.1 of unit capacity (admissions by others over the next 24h for the primary;
other incumbents on organ support, or hospital ED arrivals, for the v4 exposures). Estimates and SEs are
written unrounded. The exposure checks (positive controls, balance, the per-unit-year gate) are fitted
before any mortality model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import common as c
from utils import figures
from utils.models import drop_constant_outcome_groups, fit_logit, fit_ols, spline, spline_contrasts
from utils.sofa import COMPONENTS

config = c.load_config()
site = config["site_name"]
phi, final = c.phi_dir(site), c.final_dir(site)

days = pd.read_parquet(phi / "patient_days.parquet")
sofa = pd.read_parquet(phi / "sofa_landmark.parquet")  # an empty file (demo) has untyped keys
days = days.merge(sofa.astype({k: days[k].dtype for k in ["hospitalization_id", "day_start"]}),
                  on=["hospitalization_id", "day_start"], how="left")
days["unit"] = days["hospital_id"].astype(str) + "|" + days["location_name"].astype(str)
days["unit_year"] = days["unit"] + "|" + days["year"].astype(str)
iso = days["day_start"].dt.isocalendar()
days["unit_week"] = days["unit"] + "|" + iso["year"].astype(str) + "-" + iso["week"].astype(str)
days["occ_c10"] = (days["census_strain"] - c.OCCUPANCY_CENTER) * 10  # occupancy at s, centred: 0 = 90% full
days["occ_hinge10"] = (days["census_strain"] - c.OCCUPANCY_TIPPING).clip(lower=0) * 10  # slope change above the tipping point
days["placebo10"] = days["placebo_admit_strain"] * 10
days["placebo_past10"] = days["placebo_past_admit_strain"] * 10
for comp in COMPONENTS:
    days[f"{comp}_cat"] = days[comp].fillna(-1).astype(int)  # -1: not measured in [s-24h, s)
days["on_imv"] = days["on_imv"].fillna(False).astype(int)  # any time in [s-24h, s)
days["on_pressor"] = days["on_pressor"].fillna(False).astype(int)
days["obligate"] = days["on_imv"].astype(bool) | days["on_pressor"].astype(bool)  # almost never leaves within 24h
# v4 A2: other incumbents on organ support at s, as a share of capacity (index patient excluded)
support = days["obligate"].astype(int)
days["wocc_strain"] = (support.groupby([days["unit"], days["day_start"]]).transform("sum") - support) / days["capacity_p99"]
days["wocc10"] = days["wocc_strain"] * 10
if "in_window" in days:  # unit workload above counted every incumbent; the models use the eligible rows only
    days = days[days["eligible"] & days["in_window"]].copy()
for col in ["ed_strain", "prior_ed_strain", "placebo_ed_strain"]:  # v4 A3: hospital ED arrivals / hospital ICU capacity
    if col not in days:
        days[col] = np.nan
days["ed10"], days["placebo_ed10"] = days["ed_strain"] * 10, days["placebo_ed_strain"] * 10

SEVERITY = [f"C({comp}_cat)" for comp in COMPONENTS]
FIXED_EFFECTS = ["C(month)", "C(dow)", "C(unit_year)"]
BASE = [spline("age_at_admission"), "C(sex_category)", "cci_score", "covid", *SEVERITY, "on_imv", "on_pressor",
        "prior_admit_strain", "hosp_census_strain", "C(icu_day_cap)", *FIXED_EFFECTS]
COVARIATES = [*BASE, "occ_c10"]  # v3 set; occupancy-exposure models drop occ_c10
CLUSTER = ["patient_id", "unit_week"]  # a patient's landmarks are correlated, and so are a unit's within a week
# name, outcome, exposure (per +0.1), extra terms (already per 0.1), options: rows (a query), cluster, drop (terms),
# spline (shape, contrasts), iv (instrument column: two-stage residual inclusion)
CHECKS = [  # fitted first: does the exposure move what strain should move?
    ("positive_control_icu_discharge_24h", "icu_discharge_24h", "admit_strain", [], {}),
    ("positive_control_unit_exit_24h", "unit_exit_24h", "admit_strain", [], {}),
]
BALANCE = ["sofa_total", "age_at_admission", "cci_score", "on_imv"]  # baseline covariates regressed on the exposure
MODELS = [
    ("primary", "hdhd_7d", "admit_strain", [], {"spline": True}),
    ("placebo_future_strain", "hdhd_7d", "admit_strain", ["placebo10"], {}),
    ("placebo_past_strain", "hdhd_7d", "admit_strain", ["placebo_past10"], {}),
    ("admissions_x_occupancy", "hdhd_7d", "admit_strain", ["per10:occ_c10"], {}),
    ("obligate_icu", "hdhd_7d", "admit_strain", [], {"rows": "obligate"}),
    ("hdhd_30d", "hdhd_30d", "admit_strain", [], {}),
    ("next_day_death_or_hospice", "hdhd_48h", "admit_strain", [], {}),       # Corbett: death in the next 24h
    ("hospital_death_or_hospice", "hdhd_hosp", "admit_strain", [], {}),      # Corbett: whole-hospitalization outcome
    ("in_hospital_death_only", "death_7d", "admit_strain", [], {}),
    ("any_source_death", "death_any_7d", "admit_strain", [], {}),
    ("capacity_max", "hdhd_7d", "admit_strain_max", [], {}),
    ("hospital_wide", "hdhd_7d", "hosp_admit_strain", [], {}),
    ("no_sofa", "hdhd_7d", "admit_strain", [], {"drop": SEVERITY}),
    ("patient_cluster_only", "hdhd_7d", "admit_strain", [], {"cluster": ["patient_id"]}),
    # --- v4 ---
    ("primary_validated_units", "hdhd_7d", "admit_strain", [], {"rows": "validated"}),                         # A1
    ("weighted_occupancy", "hdhd_7d", "wocc_strain", [], {}),                                                  # A2
    ("occupancy_threshold", "hdhd_7d", "census_strain", ["occ_hinge10"], {"drop": ["occ_c10"]}),               # A2
    ("occupancy_spline", "hdhd_7d", "census_strain", [], {"drop": ["occ_c10"], "spline": True}),               # A2
    ("occupancy_x_support", "hdhd_7d", "census_strain", ["wocc10", "per10:wocc10"], {"drop": ["occ_c10"]}),    # A2
    ("ed_arrivals_reduced_form", "hdhd_7d", "ed_strain", [], {}),                                              # A3
    ("ed_arrivals_iv", "hdhd_7d", "admit_strain", [], {"iv": "ed10"}),                                         # A3
    ("ed_arrivals_placebo", "hdhd_7d", "admit_strain", ["placebo_ed10"], {}),                                  # A3
    ("readmit_72h", "readmit_72h", "admit_strain", [], {}),                                                    # B1
    ("after_hours_discharge", "after_hours_discharge", "admit_strain", [], {}),                                # B1
    ("new_limitation_48h", "new_limitation_48h", "admit_strain", [], {}),                                      # B1
    ("obligate_weighted_occupancy", "hdhd_7d", "wocc_strain", [], {"rows": "obligate"}),                       # B3
    ("obligate_ed_arrivals_iv", "hdhd_7d", "admit_strain", [], {"rows": "obligate", "iv": "ed10"}),            # B3
]


def converged(res) -> str:
    return "fit" if res.mle_retvals["converged"] else "not_converged"


def gate_unit_years() -> pd.DataFrame:
    """A1: per unit-year, do admissions displace incumbents? Validated when the positive-control OR's lower bound > 1."""
    rows = []
    for uy, d in days.groupby("unit_year"):
        n_events = int(d["icu_discharge_24h"].sum())
        row = {"site": site, "unit_year": uy, "n_landmarks": len(d), "n_discharges": n_events}
        if n_events < c.MIN_GATE_EVENTS or d["icu_discharge_24h"].nunique() < 2:
            rows.append({**row, "status": "too few discharges"})
            continue
        try:
            r = fit_logit(d.assign(per10=d["admit_strain"] * 10), "icu_discharge_24h", ["per10", "C(month)", "C(dow)"], ["patient_id"])
            ok = bool(r.mle_retvals["converged"] and np.isfinite(r.bse["per10"]))
            lo, hi = r.params["per10"] - 1.96 * r.bse["per10"], r.params["per10"] + 1.96 * r.bse["per10"]
            rows.append({**row, "status": "fit" if ok else "no_se", "odds_ratio": np.exp(r.params["per10"]),
                         "ci_low": np.exp(lo), "ci_high": np.exp(hi), "validated": bool(ok and lo > 0)})
        except Exception as e:
            rows.append({**row, "status": f"failed: {type(e).__name__}"})
    gate = pd.DataFrame(rows, columns=["site", "unit_year", "n_landmarks", "n_discharges", "status", "odds_ratio",
                                       "ci_low", "ci_high", "validated"])
    return gate.assign(validated=gate["validated"].fillna(False).astype(bool))


def fit_model(name, outcome, exposure, extra, opts):
    """Estimate rows for one logistic model, and its spline curve when asked for."""
    base = {"site": site, "model": name, "outcome": outcome, "exposure": exposure, "scale": "log_odds"}
    df = days.query(opts["rows"]) if "rows" in opts else days
    df = df.dropna(subset=[outcome, exposure] + ([opts["iv"]] if "iv" in opts else []))
    kept = drop_constant_outcome_groups(df, outcome, "unit_year")
    if dropped := df["unit_year"].nunique() - kept["unit_year"].nunique():
        print(f"{name}: dropped {dropped} unit-year(s) with no {outcome} variation ({len(df) - len(kept):,} rows)")
    df = kept
    if min(df[outcome].sum(), (1 - df[outcome]).sum()) < c.MIN_CELL_SIZE:
        return [{**base, "status": f"skipped: fewer than {c.MIN_CELL_SIZE} events or non-events"}], None
    df = df.assign(per10=df[exposure] * 10)
    terms = [t for t in COVARIATES if t not in opts.get("drop", [])]
    cluster = opts.get("cluster", CLUSTER)
    out = []
    if "iv" in opts:  # two-stage residual inclusion: the first-stage residual carries the endogenous part of per10
        fs = fit_ols(df, "per10", [opts["iv"], *terms], cluster)
        df = df.assign(_resid=df["per10"] - fs.fittedvalues)
        extra = [*extra, "_resid"]
        out.append({**base, "n_obs": int(fs.nobs), "status": "fit", "term": "first_stage_t2",
                    "estimate": (fs.params[opts["iv"]] / fs.bse[opts["iv"]]) ** 2, "scale": "linear"})
    lin = fit_logit(df, outcome, ["per10", *extra, *terms], cluster)
    meta = {**base, "n_obs": int(lin.nobs), "n_events": int(lin.model.endog.sum()), "status": converged(lin)}
    out += [{**meta, "term": t, "estimate": lin.params[t], "se": lin.bse[t]} for t in ["per10", *extra] if t != "_resid"]
    q = df.loc[lin.model.data.row_labels, exposure].quantile([0.01, 0.25, 0.5, 0.75, 0.99])  # the rows the model kept
    out.append({**meta, "term": "per_iqr", "x": q[0.75], "ref": q[0.25],
                "estimate": lin.params["per10"] * (q[0.75] - q[0.25]) * 10, "se": lin.bse["per10"] * (q[0.75] - q[0.25]) * 10})
    if not opts.get("spline"):
        return out, None
    try:
        spl = fit_logit(df, outcome, [spline(exposure), *terms], cluster)  # shape only
    except Exception as e:  # coincident knots when the exposure takes few distinct values: keep the linear rows
        out.append({**meta, "term": "p75_vs_p25", "status": f"spline_failed: {type(e).__name__}"})
        return out, None
    meta["status"] = converged(spl)
    iqr = spline_contrasts(spl, exposure, [q[0.75]], ref=q[0.25]).iloc[0]
    out.append({**meta, "term": "p75_vs_p25", "x": q[0.75], "ref": q[0.25], "estimate": iqr["estimate"], "se": iqr["se"]})
    curve = spline_contrasts(spl, exposure, np.linspace(q[0.01], q[0.99], 15), ref=q[0.5])
    out += [{**meta, "term": "spline_vs_median", "x": r.x, "ref": q[0.5], "estimate": r.estimate, "se": r.se}
            for r in curve.itertuples()]
    return out, curve


def balance(cov):
    """Does the exposure predict a baseline covariate, within unit-year, month and weekday? It should not."""
    base = {"site": site, "model": f"balance_{cov}", "outcome": cov, "exposure": "admit_strain", "scale": "linear"}
    df = days.dropna(subset=[cov, "admit_strain"])
    df = df.assign(per10=df["admit_strain"] * 10, **{cov: df[cov].astype(float)})
    res = fit_ols(df, cov, ["per10", *FIXED_EFFECTS], CLUSTER)
    return [{**base, "n_obs": int(res.nobs), "status": "fit", "term": "per10", "estimate": res.params["per10"],
             "se": res.bse["per10"]}]


gate = gate_unit_years()
validated = set(gate.loc[gate["validated"], "unit_year"])
days["validated"] = days["unit_year"].isin(validated)
print(f"A1 gate: {len(validated)} of {gate['unit_year'].nunique()} unit-years validated "
      f"({days['validated'].mean() * 100:.0f}% of landmarks)")
c.suppress_small_cells(gate, ["n_landmarks", "n_discharges"]).to_csv(final / "unit_year_gate.csv", index=False)

rows, curves = [], {}
for name, *spec in [*CHECKS, *MODELS]:
    try:
        out, curve = fit_model(name, *spec)
    except Exception as e:  # one failing model must not stop the others from being written
        out, curve = [{"site": site, "model": name, "status": f"failed: {type(e).__name__}: {e}"[:300]}], None
        print(f"{name} failed: {type(e).__name__}: {e}")
    rows += out
    if curve is not None:
        curves[name] = curve
    if name == CHECKS[-1][0]:  # balance tests sit with the checks, before the mortality models
        for cov in BALANCE:
            try:
                rows += balance(cov)
            except Exception as e:
                rows.append({"site": site, "model": f"balance_{cov}", "status": f"failed: {type(e).__name__}: {e}"[:300]})
                print(f"balance_{cov} failed: {type(e).__name__}: {e}")

est = pd.DataFrame(rows).reindex(columns=["site", "model", "outcome", "exposure", "scale", "term", "x", "ref",
                                          "estimate", "se", "n_obs", "n_events", "status"])
est = est.astype({"n_obs": "Int64", "n_events": "Int64"})
log_odds = est["scale"].eq("log_odds")
est = est.assign(odds_ratio=np.exp(est["estimate"]).where(log_odds),
                 ci_low=np.exp(est["estimate"] - 1.96 * est["se"]).where(log_odds),
                 ci_high=np.exp(est["estimate"] + 1.96 * est["se"]).where(log_odds))
est.to_csv(final / "model_estimates.csv", index=False)
if "primary" in curves:
    q = days["admit_strain"].quantile(0.5)
    figures.spline_curve(curves["primary"], "Others' admissions to the unit in the next 24h / capacity", round(q, 2),
                         f"{site}: death or hospice within 7 days").savefig(final / "spline_primary.png", dpi=150)


# --- Table 1: ICU stays, and landmarks ---
def n_pct(mask):
    n = int(mask.sum())
    return f"<{c.MIN_CELL_SIZE}" if 0 < n < c.MIN_CELL_SIZE else f"{n:,} ({100 * mask.mean():.1f}%)"


def med_iqr(x, digits=1):
    return f"{x.median():.{digits}f} ({x.quantile(.25):.{digits}f}-{x.quantile(.75):.{digits}f})"


stays = days.sort_values("icu_day").drop_duplicates("stay_id")
model_n = est.loc[est["model"] == "primary", "n_obs"].max()
table1 = pd.DataFrame([
    ("Landmarks (ICU stay x 7am already in the unit)", f"{len(days):,}"),
    ("ICU stays", f"{len(stays):,}"), ("Patients", f"{days['patient_id'].nunique():,}"),
    ("ICU units", f"{days['unit'].nunique()}"),
    ("Unit-years passing the positive-control gate, n (% of landmarks)", f"{len(validated)} ({days['validated'].mean() * 100:.0f}%)"),
    ("Age, median (IQR)", med_iqr(stays["age_at_admission"], 0)),
    ("Female, n (%)", n_pct(stays["sex_category"].astype(str).str.lower().eq("female"))),
    ("Charlson index, median (IQR)", med_iqr(stays["cci_score"], 0)),
    ("COVID-19 diagnosis, n (%)", n_pct(stays["covid"].eq(1))),
    ("Per landmark: ICU day, median (IQR)", med_iqr(days["icu_day"], 0)),
    ("SOFA in the 24h before the landmark, median (IQR)", med_iqr(days["sofa_total"], 0)),
    ("SOFA with no component measured, n (%)", n_pct(days["sofa_total"].isna())),
    ("Invasive ventilation in the 24h before, n (%)", n_pct(days["on_imv"])),
    ("Vasopressor in the 24h before, n (%)", n_pct(days["on_pressor"])),
    ("Unit occupancy at the landmark (census / capacity), median (IQR)", med_iqr(days["census_strain"], 2)),
    ("Other incumbents on ventilation or vasopressor / capacity, median (IQR)", med_iqr(days["wocc_strain"], 2)),
    ("Exposure: others' admissions in the next 24h / capacity, median (IQR)", med_iqr(days["admit_strain"], 2)),
    ("Hospital ED arrivals in the next 24h / hospital ICU capacity, median (IQR)", med_iqr(days["ed_strain"], 2)),
    ("ICU discharge inside the exposure window, n (%)", n_pct(days["icu_discharge_24h"].eq(1))),
    ("  readmitted to an ICU within 72h, n (% of those discharged)", n_pct(days["readmit_72h"].dropna().eq(1))),
    ("  discharged 18:00-06:00, n (% of those discharged)", n_pct(days["after_hours_discharge"].dropna().eq(1))),
    ("New limitation order within 48h, n (% of full-code landmarks)", n_pct(days["new_limitation_48h"].dropna().eq(1))),
    ("Death or hospice discharge in [s+24h, s+8d), n (%)", n_pct(days["hdhd_7d"].eq(1))),
    ("  in-hospital death, n (%)", n_pct(days["death_7d"].eq(1))),
    ("Landmarks in the primary model (complete cases)", f"{model_n:,.0f}" if pd.notna(model_n) else "0"),
], columns=["characteristic", "value"])
table1.to_csv(final / "table1.csv", index=False)

print(table1.to_string(index=False))
show = est[est["term"].isin(["per10", "placebo10", "placebo_past10", "placebo_ed10", "per10:occ_c10", "occ_hinge10",
                             "wocc10", "per10:wocc10", "first_stage_t2", "per_iqr", "p75_vs_p25"])]
print(show[["model", "term", "estimate", "se", "odds_ratio", "ci_low", "ci_high", "n_obs", "n_events", "status"]]
      .round(4).to_string(index=False))
