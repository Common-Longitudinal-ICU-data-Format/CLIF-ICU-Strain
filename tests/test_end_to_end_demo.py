"""Full pipeline against the bundled synthetic clif_demo dataset.

Guards against import errors, schema drift, and column-name typos that unit
tests with hand-built frames cannot catch.

Runs every driver against a scratch config (CLIF_CONFIG_PATH) and a scratch
output root (CLIF_OUTPUT_ROOT) so this never touches config/config.json or
output/final_no_phi/ -- both belong to the real site analysis.

The demo dataset's dates are shifted decades into the future (a de-identified
synthetic set, not real 2018-2024 site data) and only has a handful of ICU
stays in any given year, so ANALYSIS_YEAR is overridden via CLIF_ANALYSIS_YEAR
to 2201 -- the demo year found to run the full pipeline cleanly. No one in
that year's 3-hospitalization cohort happens to meet ARF criteria, which is
exactly the zero-row path earlier tasks hardened the pipeline against; this
test is what proves that holds end to end, not just in unit tests with
hand-built empty frames.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "clif_demo"
DEMO_ANALYSIS_YEAR = 2201


@pytest.fixture(scope="module")
def demo_output_dir(tmp_path_factory):
    """Run the full 01->05 pipeline once against clif_demo, into a scratch root.

    Module-scoped so the 3 tests below share one pipeline run rather than
    each re-running all 5 drivers.
    """
    tmp_path = tmp_path_factory.mktemp("demo_run")
    config = tmp_path / "config.json"
    config.write_text(
        f'{{"site_name": "demo", "data_directory": "{DEMO}", '
        f'"filetype": "parquet", "timezone": "US/Central", "output_directory": "./output"}}'
    )
    output_root = tmp_path / "output"

    # Merge into os.environ rather than replacing it; a bare env dict drops
    # PATH and the virtualenv, and the subprocess fails for the wrong reason.
    env = {
        **os.environ,
        "CLIF_CONFIG_PATH": str(config),
        "CLIF_OUTPUT_ROOT": str(output_root),
        "CLIF_ANALYSIS_YEAR": str(DEMO_ANALYSIS_YEAR),
    }
    for script in [
        "01_cohort_identification.py",
        "02_project_quality_checks.py",
        "03_outlier_handling.py",
        "04_project_analysis.py",
        "05_generate_report.py",
    ]:
        r = subprocess.run(
            [sys.executable, f"code/{script}"], cwd=REPO, capture_output=True, text=True, env=env
        )
        assert r.returncode == 0, f"{script} failed:\n{r.stderr}"

    return output_root / "final_no_phi"


@pytest.mark.skipif(not DEMO.exists(), reason="demo dataset not present")
def test_pipeline_runs_end_to_end_on_demo(demo_output_dir):
    report = demo_output_dir / "icu_strain_report.html"
    assert report.exists()
    assert "NOT ARDS" in report.read_text()


@pytest.mark.skipif(not DEMO.exists(), reason="demo dataset not present")
def test_subtype_counts_sum_to_total(demo_output_dir):
    df = pd.read_csv(demo_output_dir / "monthly_arf_subtypes.csv")
    unsuppressed = df.dropna(subset=["hypoxemic", "hypercapnic", "mixed", "total"])
    sums = unsuppressed[["hypoxemic", "hypercapnic", "mixed"]].sum(axis=1)
    assert (sums == unsuppressed["total"]).all()


@pytest.mark.skipif(not DEMO.exists(), reason="demo dataset not present")
def test_no_delta_exceeds_the_declared_window(demo_output_dir):
    df = pd.read_csv(demo_output_dir / "time_to_icu_by_subtype.csv")
    assert (df["q75_hours"] <= 24.0).all()
