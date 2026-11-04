## Configuration

`config.json` is the only config. Edit it for your site before running:
   - `site_name` — your site's identifier; names the output folders
     `output/intermediate_phi/<site_name>/` and `output/final_no_phi/<site_name>_upload_to_box/`.
   - `data_directory` — path to the directory holding your CLIF table files
     (`clif_vitals.parquet`, `clif_labs.parquet`, …). **Defaults to the bundled `clif_demo/`
     dataset so the templates run out of the box; change it to your site's CLIF tables directory
     for a real run.**
   - `filetype` — `"csv"` or `"parquet"`.
     On Windows write the path with forward slashes (`"C:/CLIF/tables"`) or doubled backslashes: a single
     backslash is an escape character in JSON.
   - `timezone` — your data's timezone, e.g. `"US/Eastern"` (required by clifpy).
   - `study_start`, `study_end` (optional) — the ICU admission window, for date-shifted data only; the
     study's is 2019-01-01 to 2025-01-01 (`utils/common.py`).

Outputs always go to `output/` (or `CLIF_OUTPUT_ROOT`).

This file uses the **clifpy config schema**; every script in `code/` reads it directly. Don't commit your
site's values.
