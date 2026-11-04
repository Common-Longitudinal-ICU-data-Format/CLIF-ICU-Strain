"""Run the ICU strain pipeline at one site, in order, stopping at the first step that fails.

    uv run python run_pipeline.py                            # config/config.json, every step
    uv run python run_pipeline.py --from 04_sofa             # start later (earlier outputs already there)

Steps: 01_census -> 02_strain_figures -> 03_patient_days -> 04_sofa -> 05_models.
06_pool.py is for the coordinating site only and is run on its own after every site uploads.
Upload output/final_no_phi/<site>_upload_to_box/ when it finishes.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from utils import common as c

ROOT = Path(__file__).resolve().parent
STEPS = ["01_census", "02_strain_figures", "03_patient_days", "04_sofa", "05_models"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0], help="first step to run")
    args = ap.parse_args()
    cfg = c.load_config()
    if not Path(cfg["data_directory"]).exists():
        sys.exit(f"data_directory not found: {cfg['data_directory']}")
    steps = STEPS[STEPS.index(args.start):]

    print(f"ICU strain at {cfg['site_name']}: {' -> '.join(steps)}")
    for step in steps:
        t = time.time()
        print(f"\n===== {step}  {time.strftime('%H:%M:%S')}", flush=True)
        if subprocess.run([sys.executable, f"code/{step}.py"], cwd=ROOT).returncode:
            sys.exit(f"\n{step} failed: fix it, then rerun with --from {step}")
        print(f"----- {step} done in {time.time() - t:.0f} s")
    upload = ROOT / c.output_root() / "final_no_phi" / f"{cfg['site_name']}_upload_to_box"
    print(f"\nDone. Upload {upload} (aggregates only; never output/intermediate_phi/).")


if __name__ == "__main__":
    main()
