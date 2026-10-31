from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_validation_inputs
from tools import evidence_paths


def test_public_ci_runs_the_fast_checks_without_private_inputs() -> None:
    import yaml

    text = (evidence_paths.DINKSTER_ROOT / ".github/workflows/ci.yml").read_text()
    workflow = yaml.safe_load(text)
    fast = workflow["jobs"]["fast"]
    assert fast["runs-on"] == "ubuntu-latest"
    assert "secrets." not in text and "vars." not in text
    assert all("if" not in step for step in fast["steps"])
    assert any(step.get("run") == "bash scripts/ci-fast.sh" for step in fast["steps"])
    gate = (evidence_paths.DINKSTER_ROOT / "scripts/ci-fast.sh").read_text()
    assert "export UV_OFFLINE=1" in gate
    assert "-m 'not network'" in gate
    commands = [step.get("run", "") for step in fast["steps"]]
    backend = next(command for command in commands if "uv build --package" in command)
    assert commands.index(backend) < commands.index("bash scripts/ci-fast.sh")
    export = next(
        step for step in fast["steps"] if "DINKSTER_CATALOG_WIRE_OUTPUT" in step.get("env", {})
    )
    assert "test_default_catalog_publishes_only_owned_translation_carriers" in export["run"]
    decode = fast["steps"][-1]
    assert "tools/validate_default_catalog.mjs" in decode["run"]
    assert decode["env"]["CATALOG"] == export["env"]["DINKSTER_CATALOG_WIRE_OUTPUT"]
    assert all("prepare-validation-inputs" not in step.get("uses", "") for step in fast["steps"])


def test_full_validation_collects_coverage_in_the_root_test_shards() -> None:
    import yaml

    workflow = yaml.safe_load(
        (evidence_paths.DINKSTER_ROOT / ".github/workflows/full-validation.yml").read_text()
    )
    jobs = workflow["jobs"]
    assert "coverage" not in jobs
    root = jobs["test"]
    linux = [row for row in root["strategy"]["matrix"]["include"] if row["os"] == "linux"]
    assert [(row["shard"], row["pytest_args"]) for row in linux] == [
        (index, f"-p tools.pytest_file_shard --file-shard {index}/4") for index in range(1, 5)
    ]
    windows = [row for row in root["strategy"]["matrix"]["include"] if row["os"] == "windows"]
    assert [row["pytest_args"] for row in windows] == [
        f"-p tools.pytest_file_shard --file-shard {index}/6"
        + (" --durations=20" if index == 1 else "")
        for index in range(1, 7)
    ]
    assert [row["run_checks"] for row in windows] == [False, False, True, False, False, False]
    assert [row["run_checks"] for row in linux] == [True, False, False, False]
    run = next(step for step in root["steps"] if "--cov-branch" in step.get("run", ""))
    assert run["if"] == "runner.os == 'Linux'"
    assert "${{ matrix.pytest_args }}" in run["run"]
    assert run["env"]["COVERAGE_FILE"] == ".coverage.${{ matrix.shard }}"
    upload = next(
        step for step in root["steps"] if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert upload["if"] == "runner.os == 'Linux'"
    assert upload["with"]["path"] == run["env"]["COVERAGE_FILE"]
    assert upload["with"]["if-no-files-found"] == "error"
    gate = jobs["coverage-gate"]
    assert set(gate["needs"]) == {"validation-plan", "test", "windows-runtime"}
    runtime = jobs["windows-runtime"]
    assert runtime["strategy"]["matrix"]["shard"] == [1, 2, 3]
    runtime_run = next(step for step in runtime["steps"] if "--cov-branch" in step.get("run", ""))
    assert "--file-shard ${{ matrix.shard }}/3" in runtime_run["run"]
    assert "--case-shard-file tests/test_environment_generations.py" in runtime_run["run"]
    runtime_upload = next(
        step for step in runtime["steps"] if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert runtime_upload["if"] == "always()"
    assert runtime_upload["with"]["if-no-files-found"] == "error"
    assert runtime_upload["with"]["include-hidden-files"] is True
    download = next(
        step for step in gate["steps"] if step.get("uses") == "actions/download-artifact@v4"
    )
    combine = gate["steps"][-1]["run"]
    assert "*/packages/" in combine
    assert 'coverage combine --rcfile="$RUNNER_TEMP/coverage-paths.ini" coverage-data' in combine
    assert "for shard in 1 2 3; do" in combine
    assert 'test -f "coverage-data/.coverage.windows-runtime-$shard"' in combine
    for shard in runtime["strategy"]["matrix"]["shard"]:
        coverage_file = runtime_run["env"]["COVERAGE_FILE"].replace(
            "${{ matrix.shard }}", str(shard)
        )
        assert coverage_file == f".coverage.windows-runtime-{shard}"
        assert (
            runtime_upload["with"]["path"].replace("${{ matrix.shard }}", str(shard))
            == coverage_file
        )
        artifact_name = runtime_upload["with"]["name"].replace("${{ matrix.shard }}", str(shard))
        assert artifact_name.startswith(download["with"]["pattern"].removesuffix("*"))
        assert download["with"]["path"] == "coverage-data"
        assert download["with"]["merge-multiple"] is True
    assert "model-tests" not in jobs
    assert "model-tests-gate" not in jobs


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
