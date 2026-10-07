from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_validation_inputs
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
    assert "python scripts/prepare_validation_inputs.py --evidence-root .evidence-source" in action
    assert evidence_paths.EVIDENCE_REVISION not in action


def test_validation_preparation_materializes_core_owned_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dinkster_root = tmp_path / "Dinkster"
    shim = dinkster_root / "scripts/comfyui_benchmark_nodes/shim.py"
    workflow_tool = dinkster_root / "tools/workflow_benchmark.py"
    shim.parent.mkdir(parents=True)
    workflow_tool.parent.mkdir(parents=True)
    shim.write_text("SHIM = True\n")
    workflow_tool.write_text("TOOL = True\n")

    evidence_root = tmp_path / "evidence"
    evidence_shim = evidence_root / "scripts/comfyui_benchmark_nodes"
    evidence_shim.parent.mkdir(parents=True)
    evidence_shim.write_text("../../Dinkster/scripts/comfyui_benchmark_nodes")

    monkeypatch.setattr(prepare_validation_inputs, "ROOT", dinkster_root)
    monkeypatch.setattr(
        prepare_validation_inputs,
        "MATERIALIZED_FILES",
        (Path("tools/workflow_benchmark.py"),),
    )

    prepare_validation_inputs.materialize_core_sources(evidence_root)

    assert (evidence_shim / "shim.py").read_text() == "SHIM = True\n"
    assert (evidence_root / "tools/workflow_benchmark.py").read_text() == "TOOL = True\n"


def test_validation_preparation_preserves_git_status_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepare_validation_inputs.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" D scripts/comfyui_benchmark_nodes\n"),
    )

    assert prepare_validation_inputs.run("git", "status") == (" D scripts/comfyui_benchmark_nodes")


def test_validation_preparation_rejects_unrelated_dirty_files(tmp_path: Path) -> None:
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me\n")

    with pytest.raises(RuntimeError, match="dirty validation input checkout"):
        prepare_validation_inputs.reset_materialized_sources(tmp_path, "?? notes.txt")

    assert unrelated.read_text() == "keep me\n"


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
            "tests/test_values.py",
        ],
        cwd=evidence_paths.DINKSTER_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
