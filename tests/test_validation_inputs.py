from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import evidence_paths


def test_local_evidence_checkout_is_a_sibling_of_dinkster() -> None:
    assert evidence_paths.EVIDENCE_ROOT.parent == evidence_paths.DINKSTER_ROOT.parent


def test_hosted_checkout_reads_the_local_evidence_revision() -> None:
    action = (
        evidence_paths.DINKSTER_ROOT / ".github/actions/prepare-validation-inputs/action.yml"
    ).read_text()

    assert "revision=$(cat tools/evidence-revision.txt)" in action
    assert "ref: ${{ steps.evidence-revision.outputs.revision }}" in action
    assert evidence_paths.EVIDENCE_REVISION not in action


def test_validation_accepts_the_exact_evidence_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"{evidence_paths.EVIDENCE_REVISION}\n",
        ),
    )

    evidence_paths.validate_evidence_revision(tmp_path)


def test_validation_rejects_a_different_evidence_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=f"{'f' * 40}\n"),
    )

    with pytest.raises(RuntimeError, match="uv run --no-sync python"):
        evidence_paths.validate_evidence_revision(tmp_path)
