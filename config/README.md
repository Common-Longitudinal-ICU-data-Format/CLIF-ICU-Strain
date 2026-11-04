## Configuration

1. Rename `config_template.json` to `config.json`.
2. Fill in your site-specific settings:
   - `site_name` — your site's identifier; names the output folders
     `output/intermediate_phi/<site_name>/` and `output/final_no_phi/<site_name>_upload_to_box/`.
   - `data_directory` — path to the directory holding your CLIF table files
     (`clif_vitals.parquet`, `clif_labs.parquet`, …). **Defaults to the bundled `clif_demo/`
     dataset so the templates run out of the box; change it to your site's CLIF tables directory
     for a real run.**
   - `filetype` — `"csv"` or `"parquet"`.
   - `timezone` — your data's timezone, e.g. `"US/Eastern"` (required by clifpy).
   - `study_start`, `study_end` (optional) — the ICU admission window, for date-shifted data only; the
     study's is 2019-01-01 to 2025-01-01 (`utils/common.py`).

Outputs always go to `output/` (or `CLIF_OUTPUT_ROOT`).

This file uses the **clifpy config schema**. The scripts in `code/` read it directly; set
`CLIF_CONFIG_PATH` to point them at a different file. `run_pipeline.py` does this for you: keep
several sites as `config_<site>.json` (also git-ignored) and run
`uv run python run_pipeline.py config/config_<site>.json`.

Note: the `.gitignore` in this directory keeps `config.json` out of the remote repository.
