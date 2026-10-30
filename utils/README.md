## Utils directory

`strain.py` holds the functions the numbered scripts share: constants, per-site output paths,
small-cell suppression, ICU stay blocks, rolling census, daily 7am strain, capacity. Pure pandas,
unit-tested in `tests/test_strain.py`.

Scripts read `config/config.json` directly (or the file named by `CLIF_CONFIG_PATH`), so there is
no separate config helper.
