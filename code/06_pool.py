"""Step 6 (coordinating site only): pool every site's model_estimates.csv. Run from repo root.

Reads output/final_no_phi/<site>_upload_to_box/model_estimates.csv for every site present
(copy other sites' Box uploads there), or the folder named by CLIF_POOL_DIR. Pools the
per-0.1 log odds ratios (the same scale at every site). Primary: common effect, inverse variance
(D3: with three sites tau^2 cannot be estimated). Beside it: random effects with Paule-Mandel
tau^2 and a t interval. Writes output/final_no_phi/pooled/: pooled_estimates.csv, forest_primary.png
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import common as c
from utils import figures
from utils.models import e_value, random_effects

TERMS = ["per10", "placebo10", "placebo_past10", "per10:occ_c10"]

src = Path(os.environ.get("CLIF_POOL_DIR", c.output_root() / "final_no_phi"))
out = c.output_root() / "final_no_phi" / "pooled"
out.mkdir(parents=True, exist_ok=True)

est = pd.concat([pd.read_csv(f) for f in sorted(src.glob("*_upload_to_box/model_estimates.csv"))])
est = est[(est["status"] == "fit") & est["se"].notna() & est["term"].isin(TERMS)]
rows = []
for (m, t), g in est.groupby(["model", "term"], sort=False):
    r = random_effects(g["estimate"], g["se"])
    rows.append({"model": m, "term": t, "scale": g["scale"].iloc[0], "sites": ", ".join(g["site"]), "n_sites": len(g),
                 "n_obs": g["n_obs"].sum(), "n_events": g["n_events"].sum(),
                 "estimate": r["fixed"], "lo": r["fixed_lo"], "hi": r["fixed_hi"],  # common effect: the primary pool
                 "re_estimate": r["estimate"], "re_lo": r["lo"], "re_hi": r["hi"], "tau2": r["tau2"]})
pooled = pd.DataFrame(rows)
log_odds = pooled["scale"].eq("log_odds")
for src_col, dst in [("estimate", "odds_ratio"), ("lo", "ci_low"), ("hi", "ci_high"),
                     ("re_estimate", "re_odds_ratio"), ("re_lo", "re_ci_low"), ("re_hi", "re_ci_high")]:
    pooled[dst] = np.exp(pooled[src_col]).where(log_odds)
pooled["e_value"] = pooled["odds_ratio"].map(e_value)
pooled["e_value_ci"] = [e_value(lo) if lo > 1 else e_value(hi) if hi < 1 else 1.0
                        for lo, hi in zip(pooled["ci_low"], pooled["ci_high"])]
pooled.loc[~log_odds, ["e_value", "e_value_ci"]] = np.nan
pooled.to_csv(out / "pooled_estimates.csv", index=False)

sites = est[(est["model"] == "primary") & (est["term"] == "per10")].assign(label=lambda d: d["site"])
p = pooled[(pooled["model"] == "primary") & (pooled["term"] == "per10")]
if len(p):  # no site with a fit primary: nothing to draw
    pooled_rows = pd.DataFrame({"label": ["Pooled, common effect", "Pooled, random effects (t)"],
                                "odds_ratio": [p["odds_ratio"].iloc[0], p["re_odds_ratio"].iloc[0]],
                                "ci_low": [p["ci_low"].iloc[0], p["re_ci_low"].iloc[0]],
                                "ci_high": [p["ci_high"].iloc[0], p["re_ci_high"].iloc[0]]})
    figures.forest(pd.concat([sites[["label", "odds_ratio", "ci_low", "ci_high"]], pooled_rows]),
                   "Unit admission strain and death or hospice within 7 days (incumbent landmark)").savefig(
        out / "forest_primary.png", dpi=150)

print(pooled[["model", "term", "n_sites", "n_obs", "n_events", "estimate", "se" if "se" in pooled else "lo", "odds_ratio",
              "ci_low", "ci_high", "re_odds_ratio", "re_ci_low", "re_ci_high", "tau2", "e_value", "e_value_ci"]]
      .round(4).to_string(index=False))
