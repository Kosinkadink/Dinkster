from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import evidence_paths


def resolved_evidence_root(environment: dict[str, str]) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.evidence_paths import EVIDENCE_ROOT; print(EVIDENCE_ROOT)",
        ],
        cwd=evidence_paths.DINKSTER_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def test_local_evidence_checkout_defaults_to_a_sibling_of_dinkster() -> None:
    environment = os.environ.copy()
    environment.pop("DINKSTER_EVIDENCE_ROOT", None)

    assert resolved_evidence_root(environment) == (
        evidence_paths.DINKSTER_ROOT.parent / ".dinkster-evidence-source"
    )


def test_hosted_evidence_checkout_accepts_an_in_tree_location() -> None:
    environment = os.environ.copy()
    hosted_root = evidence_paths.DINKSTER_ROOT / ".evidence-source"
    environment["DINKSTER_EVIDENCE_ROOT"] = str(hosted_root)

    assert resolved_evidence_root(environment) == hosted_root


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


def test_validation_does_not_require_evidence_for_independent_test_collection(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["DINKSTER_EVIDENCE_ROOT"] = str(tmp_path / "missing")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/test_p2p_artifact_smoke.py",
        ],
        cwd=evidence_paths.DINKSTER_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
