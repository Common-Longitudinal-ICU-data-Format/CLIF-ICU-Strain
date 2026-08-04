"""Build the ICU cohort for the analysis year. Run from repo root."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clifpy import ClifOrchestrator

from icu_strain import definitions as d
from icu_strain.cohort import build_cohort

config = json.loads(Path("config/config.json").read_text())
out_dir = Path("output/intermediate_phi")
out_dir.mkdir(parents=True, exist_ok=True)

co = ClifOrchestrator(
    data_directory=config["data_directory"],
    filetype=config["filetype"],
    timezone=config["timezone"],
)
co.initialize(tables=["hospitalization", "adt"])

cohort = build_cohort(co.hospitalization.df, co.adt.df, year=d.ANALYSIS_YEAR, min_age=d.MIN_AGE)
cohort.to_parquet(out_dir / "cohort.parquet", index=False)
print(f"cohort: {len(cohort):,} hospitalizations with an index ICU stay in {d.ANALYSIS_YEAR}")
