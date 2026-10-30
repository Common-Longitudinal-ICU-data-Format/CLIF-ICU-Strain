## Output directory

Every script writes into two per-site folders, named from `site_name` in `config/config.json`:

* **`final_no_phi/<site_name>_upload_to_box/`** — **aggregate, shareable** results only. This folder,
  and nothing else, is uploaded to Box: no `patient_id` or row-level records, every reported count at
  cell size **n ≥ 11**, no raw `.csv` / `.parquet` data.

* **`intermediate_phi/<site_name>/`** — **patient-level working data (NEVER share).** Rolling census,
  daily strain, patient-day tables. Git-ignored; never uploaded or sent to the PI. See
  [`intermediate_phi/README.md`](intermediate_phi/README.md).

`CLIF_OUTPUT_ROOT` redirects both folders (the demo test uses it to write to a scratch directory).

See [`../guides/primer.md`](../guides/primer.md) for the full data-security rules.
