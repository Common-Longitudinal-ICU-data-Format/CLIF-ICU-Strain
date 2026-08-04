"""Per-month data availability. Run from repo root."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from clifpy import ClifOrchestrator

from icu_strain.paths import final_dir, phi_dir
from icu_strain.quality import monthly_quality_checks

config_path = Path(os.environ.get("CLIF_CONFIG_PATH", "config/config.json"))
config = json.loads(config_path.read_text())
out_dir = final_dir()
out_dir.mkdir(parents=True, exist_ok=True)

cohort = pd.read_parquet(phi_dir() / "cohort.parquet")
cohort_ids = cohort["hospitalization_id"].unique().tolist()

# Push the hospitalization_id filter into the DuckDB scan (clifpy's `filters`
# param) so labs/vitals/respiratory_support are never materialized beyond the
# cohort: the source tables span 2018-2024 site-wide, ~7x more than this
# 2024-only, 6k-hospitalization cohort needs.
co = ClifOrchestrator(
    data_directory=config["data_directory"],
    filetype=config["filetype"],
    timezone=config["timezone"],
)
co.initialize(
    tables=["labs", "vitals", "respiratory_support"],
    columns={
        # lab_value_numeric is needed here (beyond category/dttm) to tell a
        # benign exact-duplicate row apart from a genuine value conflict --
        # see monthly_quality_checks' dup-key metrics.
        "labs": ["hospitalization_id", "lab_category", "lab_result_dttm", "lab_value_numeric"],
        "vitals": ["hospitalization_id", "recorded_dttm", "vital_category"],
        "respiratory_support": ["hospitalization_id", "recorded_dttm", "fio2_set", "device_category"],
    },
    filters={
        "labs": {"hospitalization_id": cohort_ids},
        "vitals": {"hospitalization_id": cohort_ids},
        "respiratory_support": {"hospitalization_id": cohort_ids},
    },
)

qc = monthly_quality_checks(cohort, co.labs.df, co.vitals.df, co.respiratory_support.df)
qc.to_csv(out_dir / "quality_checks.csv", index=False)
print(qc.to_string(index=False))
