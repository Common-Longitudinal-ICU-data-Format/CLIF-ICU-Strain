"""Build the ICU cohort for the analysis year. Run from repo root."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clifpy import ClifOrchestrator

from icu_strain import definitions as d
from icu_strain.cohort import build_cohort
from icu_strain.paths import final_dir, phi_dir

config_path = Path(os.environ.get("CLIF_CONFIG_PATH", "config/config.json"))
config = json.loads(config_path.read_text())
out_dir = phi_dir()
out_dir.mkdir(parents=True, exist_ok=True)
final_dir().mkdir(parents=True, exist_ok=True)

# Overridable so the end-to-end demo test can point at whatever year the
# bundled synthetic dataset actually contains, without touching the constant
# that governs the real site analysis.
year = int(os.environ.get("CLIF_ANALYSIS_YEAR", d.ANALYSIS_YEAR))

co = ClifOrchestrator(
    data_directory=config["data_directory"],
    filetype=config["filetype"],
    timezone=config["timezone"],
)
co.initialize(tables=["hospitalization", "adt"])

cohort = build_cohort(co.hospitalization.df, co.adt.df, year=year, min_age=d.MIN_AGE)
cohort.to_parquet(out_dir / "cohort.parquet", index=False)

# Non-PHI (site name + analysis year, no patient data), so it's written
# straight to final_dir(): 05_generate_report.py reads its report metadata
# from here rather than hardcoding it, since 05 reads only from output
# directories, never from config or site data directly.
(final_dir() / "run_meta.json").write_text(
    json.dumps({"site_name": config.get("site_name", ""), "analysis_year": year}, indent=2)
)

print(f"cohort: {len(cohort):,} hospitalizations with an index ICU stay in {year}")
