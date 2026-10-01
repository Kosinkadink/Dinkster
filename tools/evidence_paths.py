"""Locations and revision of the validation evidence checkout."""

import os
import subprocess
from pathlib import Path

DINKSTER_ROOT = Path(os.environ.get("DINKSTER_ROOT", Path(__file__).resolve().parents[1])).resolve()
EVIDENCE_REVISION = (DINKSTER_ROOT / "tools/evidence-revision.txt").read_text().strip()

EVIDENCE_ROOT = Path(
    os.environ.get("DINKSTER_EVIDENCE_ROOT", DINKSTER_ROOT.parent / ".dinkster-evidence-source")
).resolve()


def validate_evidence_revision(root: Path = EVIDENCE_ROOT) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    actual = result.stdout.strip() if result.returncode == 0 else "unavailable"
    if actual != EVIDENCE_REVISION:
        raise RuntimeError(
            f"validation requires dinkster-evidence {EVIDENCE_REVISION}, found {actual} at "
            f"{root}; run `uv run --no-sync python scripts/prepare_validation_inputs.py`"
        )
