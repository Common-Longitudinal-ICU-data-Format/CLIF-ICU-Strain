# `intermediate_phi/<site_name>/` — patient-level working data (NEVER share)

Each site's row-level intermediates (rolling census, daily strain, patient-day tables, anything
with `hospitalization_id` or event timestamps) live in a subfolder named from `site_name` in
`config/config.json`. Treat everything here as containing PHI.

- **Never** upload the contents of this folder to Box, share with the PI, or send across the
  consortium.
- Its contents are **git-ignored** so they cannot be committed — only this `README.md` is tracked.
- Share results **only** from `../final_no_phi/<site_name>_upload_to_box/` (aggregate, n ≥ 11, no
  row-level records). See [`../README.md`](../README.md) and
  [`../../guides/primer.md`](../../guides/primer.md).
