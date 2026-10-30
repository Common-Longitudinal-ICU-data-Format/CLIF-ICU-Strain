## Code directory

Numbered scripts, run in order from the repo root with `uv run python code/<script>.py`. Each reads
`config/config.json` (or the file named by `CLIF_CONFIG_PATH`) and writes to the site's two output
folders (see [`../output/README.md`](../output/README.md)):

| Script | Reads | Writes |
|---|---|---|
| `01_census.py` | `hospitalization`, `adt` | rolling census and daily 7am strain (`intermediate_phi/<site>/`); capacity, QC counts, monthly strain and figure (`final_no_phi/<site>_upload_to_box/`) |

Later steps (patient-days, SOFA, models, pooling) are added as they are built; see
[`../docs/superpowers/specs/2026-10-19-icu-strain-design-v2.md`](../docs/superpowers/specs/2026-10-19-icu-strain-design-v2.md).

The functions the scripts share (census, capacity, ICU stay blocks, disclosure control, output
paths) live in [`../utils/strain.py`](../utils/strain.py) and are unit-tested in `tests/`.

To try it on the bundled demo data: `cp config/config_template.json config/config.json` then run
`01`. `uv run pytest` runs the unit tests and an end-to-end pass over `clif_demo/`.
