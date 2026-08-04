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

Because the demo year has 0 ARF hits, the subtype-sum and delta-window
invariants below evaluate over 0-row frames and are true regardless of
whether the pipeline is doing anything correct -- they only catch a schema
typo, not a silently-broken (e.g. accidentally-empty-loaded) run. The known
DEMO_COHORT_N assertion is what actually catches that: it is tied to a fact
about this exact demo dataset/year (verified independently below) that a
silently-empty upstream load would not reproduce.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "clif_demo"
DEMO_SITE_NAME = "demo"
DEMO_ANALYSIS_YEAR = 2201
DEMO_COHORT_N = 3  # adults with an index ICU stay in 2201 in clif_demo, verified by inspection


@pytest.fixture(scope="module")
def demo_output_dir(tmp_path_factory):
    """Run the full 01->05 pipeline once against clif_demo, into a scratch root.

    Module-scoped so the tests below share one pipeline run rather than each
    re-running all 5 drivers.
    """
    tmp_path = tmp_path_factory.mktemp("demo_run")
    config = tmp_path / "config.json"
    config.write_text(
        f'{{"site_name": "{DEMO_SITE_NAME}", "data_directory": "{DEMO}", '
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
    text = report.read_text()

    # Static template boilerplate (present regardless of what data ran) --
    # proves the report template rendered, not that it reflects this run.
    assert "NOT ARDS" in text

    # Data-dependent: run_meta.json is written by 01 from this run's own
    # config/year and read back by 05, rather than 05 hardcoding a literal
    # site/year. This would fail if that plumbing regressed to a hardcoded
    # value that happened to differ from the demo config.
    assert f"Site: {DEMO_SITE_NAME}" in text
    assert f"Year: {DEMO_ANALYSIS_YEAR}" in text


@pytest.mark.skipif(not DEMO.exists(), reason="demo dataset not present")
def test_cohort_size_matches_the_known_demo_count(demo_output_dir):
    """Tied to a known fact about this exact demo dataset/year, unlike the
    two invariant tests below (which hold vacuously on an empty frame). A
    silent empty-load bug anywhere in 01-04 would fail this even though it
    would pass the other two.
    """
    df = pd.read_csv(demo_output_dir / "attrition.csv")
    n_cohort = df.loc[df["step"].str.startswith("adults with index ICU stay"), "n"].iloc[0]
    assert n_cohort == DEMO_COHORT_N


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
