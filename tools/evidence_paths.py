"""Locations of the separately maintained Dinkster evidence checkout."""

import os
from pathlib import Path

EVIDENCE_ROOT = Path(
    os.environ.get(
        "DINKSTER_EVIDENCE_ROOT", Path(__file__).resolve().parents[2] / "dinkster-evidence"
    )
).resolve()
