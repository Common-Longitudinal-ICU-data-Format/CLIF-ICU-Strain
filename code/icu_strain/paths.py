"""Filesystem roots shared by the numbered drivers.

`CLIF_OUTPUT_ROOT` lets a caller (namely the end-to-end demo test) point every
driver's intermediate_phi/final_no_phi output at a scratch directory instead
of the site's real `output/`, without any driver needing to know it's under
test. Defaults to "output" so normal site runs are unaffected.
"""

import os
from pathlib import Path


def output_root() -> Path:
    return Path(os.environ.get("CLIF_OUTPUT_ROOT", "output"))


def phi_dir() -> Path:
    """Intermediate directory: row-level, PHI-bearing artifacts. Never shared."""
    return output_root() / "intermediate_phi"


def final_dir() -> Path:
    """Final directory: disclosure-controlled aggregates safe to share."""
    return output_root() / "final_no_phi"
