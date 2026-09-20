"""Locations of the separately maintained Dinkster evidence checkout."""

import os
from pathlib import Path

DINKSTER_ROOT = Path(os.environ.get("DINKSTER_ROOT", Path(__file__).resolve().parents[1])).resolve()

EVIDENCE_ROOT = Path(
    os.environ.get("DINKSTER_EVIDENCE_ROOT", DINKSTER_ROOT.parent / "dinkster-evidence")
).resolve()
