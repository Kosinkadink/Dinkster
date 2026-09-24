from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION_PATH = "./.github/actions/torch-cpu-suite"
ACTION = yaml.safe_load((ROOT / ACTION_PATH / "action.yml").read_text(encoding="utf-8"))
PRIVATE_DEPENDENCY_ACTION_PATH = "./.github/actions/check-private-dependencies"
PRIVATE_DEPENDENCY_ACTION = yaml.safe_load(
    (ROOT / PRIVATE_DEPENDENCY_ACTION_PATH / "action.yml").read_text(encoding="utf-8")
)
WORKFLOW = yaml.safe_load(
    (ROOT / ".github/workflows/full-validation.yml").read_text(encoding="utf-8")
)
JOBS = WORKFLOW["jobs"]
PR_WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
PR_JOBS = PR_WORKFLOW["jobs"]
RELEASE_WORKFLOW = yaml.safe_load(
    (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
)
PRIVATE_DEPENDENCIES_AVAILABLE = "steps.private-dependencies.outputs.available == 'true'"
MODEL_CONDITION = "inputs.run-model-tests == 'true'"
MODEL_GROUP_ENV = "env.DINKSTER_MODEL_TEST_GROUP"
RECEIPT_TEST = "tests/test_gen_comfy_source_parity_receipts.py"
ACCEPTANCE_CLOSURE_TEST = (
    ".evidence-source/packages/dinkster-acceptance/tests/test_acceptance_closure.py"
)
ACCEPTANCE_SAMPLING_TEST = (
    ".evidence-source/packages/dinkster-acceptance/tests/test_acceptance_sampling.py"
)
VISION_SUITES = (
    "packages/dinkster-nodes-vision/tests/test_hed.py",
    "packages/dinkster-nodes-vision/tests/test_upscale.py",
    "packages/dinkster-nodes-vision/tests/test_depth_anything_v2.py",
    "packages/dinkster-nodes-vision/tests/test_detr.py",
    "packages/dinkster-nodes-vision/tests/test_rtdetr.py",
    "packages/dinkster-nodes-vision/tests/test_efficient_sam.py",
    "packages/dinkster-nodes-vision/tests/test_birefnet.py",
    "packages/dinkster-nodes-vision/tests/test_depth_anything_v3.py",
    "packages/dinkster-nodes-vision/tests/test_sam31.py",
)


def _bash_executable() -> str:
    if sys.platform != "win32":
        return "bash"
    git = shutil.which("git")
    if git is not None:
        bash = Path(git).parent.parent / "bin" / "bash.exe"
        if bash.is_file():
            return str(bash)
    raise AssertionError("Git for Windows bash is unavailable")


MODEL_GROUPS = (
    {
        "name": "inference and IPAdapter, shard 1 of 8",
        "group": "inference",
        "suites": "inference-torch,model-ipadapter",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 1/8",
    },
    {
        "name": "inference and IPAdapter, shard 2 of 8",
        "group": "inference",
        "suites": "inference-torch,model-ipadapter",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 2/8",
    },
    {
        "name": "inference and IPAdapter, shard 3 of 8",
        "group": "inference",
        "suites": "inference-torch,model-ipadapter",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 3/8",
    },
    {
        "name": "inference and IPAdapter, shard 4 of 8",
        "group": "inference",
        "suites": "inference-torch,model-ipadapter",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 4/8",
    },
    {
        "name": "inference and IPAdapter, shard 5 of 8",
        "group": "inference",
        "suites": "inference-torch,model-ipadapter",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 5/8",
    },
    {
        "name": "inference and IPAdapter, shard 6 of 8",
        "group": "inference",
        "suites": "inference-torch,model-ipadapter",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 6/8",
    },
    {
        "name": "inference and IPAdapter, shard 7 of 8",
        "group": "inference",
        "suites": "inference-torch,model-ipadapter",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 7/8",
    },
    {
        "name": "inference and IPAdapter, shard 8 of 8",
        "group": "inference",
        "suites": "inference-torch,model-ipadapter",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 8/8",
    },
    {
        "name": "acceptance and benchmark",
        "group": "acceptance",
        "suites": "acceptance-sampling,benchmark-loader",
        "pytest-args": "",
    },
    {
        "name": "HED, upscale and EfficientSAM",
        "group": "vision-fast",
        "suites": "hed,upscale,efficient-sam",
        "pytest-args": "",
    },
    {
        "name": "Depth Anything V2, DETR and RT-DETR",
        "group": "vision-detection",
        "suites": "depth-anything-v2,detr,rtdetr",
        "pytest-args": "",
    },
    {
        "name": "BiRefNet and Depth Anything V3",
        "group": "vision-large",
        "suites": "birefnet,depth-anything-v3",
        "pytest-args": "",
    },
    {"name": "SAM 3.1", "group": "vision-sam", "suites": "sam31", "pytest-args": ""},
)
PR_MODEL_GROUPS = (
    {
        "name": "inference and IPAdapter, shard 1 of 8",
        "group": "inference",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 1/8",
    },
    {
        "name": "inference and IPAdapter, shard 2 of 8",
        "group": "inference",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 2/8",
    },
    {
        "name": "inference and IPAdapter, shard 3 of 8",
        "group": "inference",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 3/8",
    },
    {
        "name": "inference and IPAdapter, shard 4 of 8",
        "group": "inference",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 4/8",
    },
    {
        "name": "inference and IPAdapter, shard 5 of 8",
        "group": "inference",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 5/8",
    },
    {
        "name": "inference and IPAdapter, shard 6 of 8",
        "group": "inference",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 6/8",
    },
    {
        "name": "inference and IPAdapter, shard 7 of 8",
        "group": "inference",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 7/8",
    },
    {
        "name": "inference and IPAdapter, shard 8 of 8",
        "group": "inference",
        "pytest-args": "-p tools.pytest_file_shard --file-shard 8/8",
    },
    {"name": "HED, upscale and EfficientSAM", "group": "vision-fast", "pytest-args": ""},
    {
        "name": "Depth Anything V2, DETR and RT-DETR",
        "group": "vision-detection",
        "pytest-args": "",
    },
    {
        "name": "BiRefNet and Depth Anything V3",
        "group": "vision-large",
        "pytest-args": "",
    },
    {"name": "SAM 3.1", "group": "vision-sam", "pytest-args": ""},
)
EXPECTED_MODEL_SUITES = {
    "inference-torch",
    "model-ipadapter",
    "acceptance-sampling",
    "benchmark-loader",
    "hed",
    "upscale",
    "efficient-sam",
    "depth-anything-v2",
    "detr",
    "rtdetr",
    "birefnet",
    "depth-anything-v3",
    "sam31",
}


def _model_group(step: dict[str, object]) -> str:
    command = str(step.get("run", ""))
    name = str(step.get("name", ""))
    if "dinkster-inference-torch/tests" in command:
        return "inference"
    if ACCEPTANCE_SAMPLING_TEST in command or "test_benchmark_inference.py" in command:
        return "acceptance"
    if any(
        token in name or token in command
        for token in ("line and edge", "test_hed.py", "upscale", "efficient_sam")
    ):
        return "vision-fast"
    if any(
        token in name or token in command
        for token in (
            "Depth Anything V2",
            "test_depth_anything_v2.py",
            "DETR",
            "test_detr.py",
            "test_rtdetr.py",
        )
    ):
        return "vision-detection"
    if any(
        token in name or token in command
        for token in ("BiRefNet", "birefnet", "Depth Anything 3", "depth_anything_v3")
    ):
        return "vision-large"
    if "SAM 3.1" in name or "sam31" in command:
        return "vision-sam"
    raise AssertionError(f"unassigned model step: {step}")


def _condition_matches(expression: str, context: dict[str, str]) -> bool:
    """Evaluate the equality/conjunction subset used by these allocation guards."""
    matches = []
    for clause in expression.split("&&"):
        match = re.fullmatch(r"\s*([\w.-]+)\s*==\s*'([^']*)'\s*", clause)
        assert match is not None, clause
        matches.append(context[match[1]].lower() == match[2].lower())
    return all(matches)


def test_every_cpu_composite_caller_declares_its_model_test_allocation() -> None:
    callers = []
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        for name, job in yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"].items():
            for step in job.get("steps", []):
                if step.get("uses") != ACTION_PATH:
                    continue
                callers.append((path.name, name))
                assert step["with"]["run-model-tests"] == (
                    "true" if name in {"engine-tests", "model-tests"} else "false"
                )
                assert job["runs-on"] == "${{ fromJSON(vars.CI_RUNNERS).linux }}"
    assert set(callers) == {
        ("full-validation.yml", "model-tests"),
        ("full-validation.yml", "torch-cpu"),
    }
    assert len(callers) == 2
    assert ACTION["inputs"]["run-model-tests"]["default"] == "false"
    assert ACTION["inputs"]["pytest-args"]["default"] == ""


def test_fork_pull_requests_use_hosted_runners_and_record_private_jobs_not_run() -> None:
    assert set(PR_JOBS) == {"fast"}
    assert PR_JOBS["fast"]["runs-on"] == (
        "${{ fromJSON(vars.CI_RUNNERS)[((github.event_name == 'pull_request' && "
        "github.event.pull_request.head.repo.full_name != github.repository) || "
        "inputs.simulate-fork) && 'forkLinux' || 'linux'] }}"
    )
    for job_name in PR_JOBS:
        steps = PR_JOBS[job_name]["steps"]
        guard = steps[1]
        assert guard == {
            "name": "Check private dependency access",
            "id": "private-dependencies",
            "uses": PRIVATE_DEPENDENCY_ACTION_PATH,
            "with": {
                "secret-name-1": "DINKSTER_EVIDENCE_READ_KEY",
                "secret-value-1": "${{ secrets.DINKSTER_EVIDENCE_READ_KEY }}",
                "force-not-run": "${{ inputs.simulate-fork }}",
            },
        }
        assert all(step["if"] == PRIVATE_DEPENDENCIES_AVAILABLE for step in steps[2:-1])
        assert steps[-1] == {
            "name": "Refuse green validation without private inputs",
            "if": "always() && steps.private-dependencies.outputs.available != 'true'",
            "run": (
                'echo "Required private inputs were unavailable; pull request validation did '
                'not run." >> "$GITHUB_STEP_SUMMARY"\nexit 1\n'
            ),
        }


def test_windows_bash_resolves_from_the_git_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = tmp_path / "Git" / "cmd" / "git.exe"
    bash = tmp_path / "Git" / "bin" / "bash.exe"
    git.parent.mkdir(parents=True)
    bash.parent.mkdir(parents=True)
    git.touch()
    bash.touch()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        shutil, "which", lambda executable: str(git) if executable == "git" else None
    )

    assert _bash_executable() == str(bash)


@pytest.mark.parametrize(
    ("force_not_run", "secret_value_1", "secret_value_2", "expected_available", "summary"),
    [
        ("false", "first", "second", "true", ""),
        (
            "false",
            "",
            "second",
            "false",
            "not run: requires repository secret FIRST\n",
        ),
        (
            "true",
            "first",
            "second",
            "false",
            "not run: requires repository secret FIRST\n"
            "not run: requires repository secret SECOND\n",
        ),
    ],
)
def test_private_dependency_check_reports_each_missing_secret(
    tmp_path: Path,
    force_not_run: str,
    secret_value_1: str,
    secret_value_2: str,
    expected_available: str,
    summary: str,
) -> None:
    output = tmp_path / "output"
    step_summary = tmp_path / "summary"
    result = subprocess.run(
        [
            _bash_executable(),
            "-e",
            "-o",
            "pipefail",
            "-c",
            PRIVATE_DEPENDENCY_ACTION["runs"]["steps"][0]["run"],
        ],
        env={
            **os.environ,
            "FORCE_NOT_RUN": force_not_run,
            "SECRET_NAME_1": "FIRST",
            "SECRET_VALUE_1": secret_value_1,
            "SECRET_NAME_2": "SECOND",
            "SECRET_VALUE_2": secret_value_2,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(step_summary),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == f"available={expected_available}\n"
    actual_summary = step_summary.read_text(encoding="utf-8") if step_summary.exists() else ""
    assert actual_summary == summary


def test_torch_cpu_runs_the_unconditional_suite_on_hosted_linux() -> None:
    assert [name for name in JOBS if name.startswith("torch-cpu")] == ["torch-cpu"]
    job = JOBS["torch-cpu"]
    # timeout-minutes bounds the whole job (comfy-vibe-station#245); the exact
    # value is pinned in test_full_validation_pytest_and_demo_jobs_are_timeout_bounded
    assert set(job) == {"needs", "if", "runs-on", "env", "timeout-minutes", "steps"}
    assert job["needs"] == "validation-plan"
    assert job["if"] == "needs.validation-plan.outputs.run-heavy == 'true'"
    assert job["runs-on"] == "${{ fromJSON(vars.CI_RUNNERS).linux }}"
    assert job["env"] == {"ATEN_CPU_CAPABILITY": "avx2", "ONEDNN_MAX_CPU_ISA": "AVX2"}
    checkout, suite = job["steps"]
    assert checkout == {
        "uses": "actions/checkout@v4",
        "with": {"clean": True, "persist-credentials": False},
    }
    assert suite == {
        "uses": ACTION_PATH,
        "with": {
            "evidence-deploy-key": "${{ secrets.DINKSTER_EVIDENCE_READ_KEY }}",
            "run-model-tests": "false",
        },
    }


def test_artifact_smoke_uses_every_hosted_platform() -> None:
    assert JOBS["p2p-artifact-smoke"]["strategy"]["matrix"] == {
        "os": ["linux", "windows", "macos"],
        "python-version": ["3.12"],
        "include": [
            {"os": "linux", "runner": "linux"},
            {"os": "windows", "runner": "windows"},
            {"os": "macos", "runner": "macos"},
        ],
    }


def test_artifact_smoke_installs_the_locked_default_set_without_private_access() -> None:
    job = JOBS["p2p-artifact-smoke"]
    setup_uv, sync, pytest_run = job["steps"][1:]
    assert setup_uv["uses"] == "astral-sh/setup-uv@v5"
    assert sync == {"run": "uv sync --locked"}
    assert pytest_run == {"run": "uv run --locked pytest -q tests/test_p2p_artifact_smoke.py"}


@pytest.mark.parametrize("enabled", ["", "false", "true"])
def test_all_model_downloads_and_model_pytest_lanes_require_opt_in(enabled: str) -> None:
    downloads = []
    executions = []
    for step in ACTION["runs"]["steps"]:
        command = step.get("run", "")
        is_download = bool(re.search(r"\.(pth|safetensors|onnx)\b", command))
        is_vision_execution = any(suite in command for suite in VISION_SUITES)
        is_execution = (
            "-m pytest" in command
            and RECEIPT_TEST not in command
            and ACCEPTANCE_CLOSURE_TEST not in command
            and (not is_vision_execution or "env" in step)
        )
        if not (is_download or is_execution):
            continue
        group = _model_group(step)
        assert step["if"] == (f"{MODEL_CONDITION} && {MODEL_GROUP_ENV} == '{group}'")
        assert _condition_matches(
            step["if"],
            {"inputs.run-model-tests": enabled, MODEL_GROUP_ENV: group},
        ) == (enabled == "true")
        if is_download:
            downloads.append(step)
        if is_execution:
            assert "run_counted_suite" not in command
            executions.append(step)
    assert len(downloads) == 9
    assert len(executions) == 12
    assert sum(len(step["env"]) for step in executions if "env" in step) == 19


def test_acceptance_sampling_runs_only_in_the_model_lane() -> None:
    executions = [
        step
        for step in ACTION["runs"]["steps"]
        if "-m pytest" in step.get("run", "")
        and "packages/dinkster-acceptance/tests" in step["run"]
    ]
    assert len(executions) == 2
    common, sampling = executions
    assert common["run"].strip() == (
        f".venv-torch/bin/python -m pytest -q {ACCEPTANCE_CLOSURE_TEST}"
    )
    assert common["if"] == f"{MODEL_GROUP_ENV} == ''"
    assert sampling["run"].strip() == (
        f".venv-torch/bin/python -m pytest -q {ACCEPTANCE_SAMPLING_TEST}"
    )
    assert sampling["if"] == (f"{MODEL_CONDITION} && {MODEL_GROUP_ENV} == 'acceptance'")


def test_weight_free_vision_suites_remain_hosted_without_artifacts() -> None:
    for suite in VISION_SUITES:
        executions = [
            step
            for step in ACTION["runs"]["steps"]
            if "-m pytest" in step.get("run", "") and suite in step["run"]
        ]
        assert len(executions) == 2
        hosted, model = executions
        assert hosted["if"] == f"{MODEL_GROUP_ENV} == ''"
        assert "env" not in hosted
        assert model["if"].startswith(f"{MODEL_CONDITION} && {MODEL_GROUP_ENV} == ")
        assert model["env"]


def test_model_lane_commands_artifact_pins_and_environments_match_reviewed_contract() -> None:
    # The digest covers the ordered full lanes, including URLs, SHA-256 checks,
    # selectors and artifact env bindings. Pin changes require deliberate review.
    steps = [
        {key: value for key, value in step.items() if key not in {"if", "name"}}
        for step in ACTION["runs"]["steps"]
        if step.get("if", "").startswith(f"{MODEL_CONDITION} &&")
    ]
    assert hashlib.sha256(json.dumps(steps, sort_keys=True).encode()).hexdigest() == (
        "0484b2100ec9c82b75b71278c52445faa827924eccc4beeb0e5d87f1fdb234b9"
    )


def test_source_receipts_and_torch_typechecks_remain_hosted() -> None:
    retained = []
    for step in ACTION["runs"]["steps"]:
        if (
            step.get("if", "").startswith(f"{MODEL_CONDITION} &&")
            or step.get("name") == "Reclaim runner disk"
        ):
            continue
        if "if" in step:
            assert step["if"] == f"{MODEL_GROUP_ENV} == ''"
        retained.append(step)
    commands = [step.get("run", "").strip() for step in retained]
    projects = {
        path.parent.name
        for path in (ROOT / "packages").glob("*/pyproject.toml")
        if 'venv = ".venv-torch"' in path.read_text(encoding="utf-8")
    }
    projects.add("dinkster-acceptance")
    assert len(projects) == 3
    assert {command for command in commands if "pyright -p" in command} == {
        (
            ".venv/bin/pyright -p .evidence-source/packages/dinkster-acceptance "
            "--pythonpath .venv-torch/bin/python"
            if project == "dinkster-acceptance"
            else f".venv/bin/pyright -p packages/{project}"
        )
        for project in projects
    }
    assert f".venv-torch/bin/python -m pytest -q {RECEIPT_TEST}" in commands
    assert {
        command
        for command in commands
        if "-m pytest" in command and any(suite in command for suite in VISION_SUITES)
    } == {f".venv-torch/bin/python -m pytest -q {suite}" for suite in VISION_SUITES}
    assert f".venv-torch/bin/python -m pytest -q {ACCEPTANCE_CLOSURE_TEST}" in commands
    assert all(ACCEPTANCE_SAMPLING_TEST not in command for command in commands)
    assert ".venv-torch/bin/python tools/gen_comfy_source_parity_receipts.py --check" in commands
    assert 'UV_CONSTRAINT="$RUNNER_TEMP/torch-constraints.txt" ./scripts/setup_envs.sh' in commands
    assert {step["name"] for step in retained if "name" in step} == {
        "Checkout pinned ComfyUI source",
        "Checkout pinned workflow templates",
        "Acquire pinned Impact Pack source",
        "Verify source-generated parity receipts",
        "Test source-parity receipt generation",
        "Assert pinned AVX2 dispatch",
    }


def test_receipts_use_pinned_evidence_with_a_separate_readonly_key() -> None:
    steps = ACTION["runs"]["steps"]
    prepare_path = "./.github/actions/prepare-validation-inputs"
    (prepare,) = [step for step in steps if step.get("uses") == prepare_path]
    assert prepare["with"] == {
        "evidence-deploy-key": "${{ inputs.evidence-deploy-key }}",
        "coverage": "false",
    }
    assert all(
        step.get("with", {}).get("repository") != "Kosinkadink/dinkster-evidence" for step in steps
    )
    helper = yaml.safe_load((ROOT / prepare_path / "action.yml").read_text(encoding="utf-8"))
    assert helper["inputs"]["coverage"]["default"] == "true"
    preparation_steps = helper["runs"]["steps"]
    (access,) = [
        step
        for step in preparation_steps
        if step.get("with", {}).get("repository") == "Kosinkadink/dinkster-evidence"
        and step.get("uses") == "./.github/actions/configure-private-repository"
    ]
    assert access["with"]["deploy-key"] == "${{ inputs.evidence-deploy-key }}"
    (checkout,) = [step for step in preparation_steps if step.get("uses") == "actions/checkout@v4"]
    assert checkout["with"] == {
        "repository": "Kosinkadink/dinkster-evidence",
        "ref": "a5949cd95ab302f73377ad5aa02a6f35565a60b9",
        "path": ".evidence-source",
        "clean": True,
        "persist-credentials": False,
    }
    assert preparation_steps.index(access) < preparation_steps.index(checkout)
    setup = next(step for step in steps if "./scripts/setup_envs.sh" in step.get("run", ""))
    assert steps.index(prepare) < steps.index(setup)
    for name in (
        "Verify source-generated parity receipts",
        "Test source-parity receipt generation",
    ):
        (step,) = [step for step in steps if step.get("name") == name]
        assert steps.index(prepare) < steps.index(step)
        assert step["env"]["DINKSTER_INFERENCE_PARITY_RECORDS"] == (
            "${{ github.workspace }}/.evidence-source/inference-parity/records"
        )
    materialize = next(
        step for step in preparation_steps if "DINKSTER_EVIDENCE_ROOT" in step.get("run", "")
    )
    assert preparation_steps.index(checkout) < preparation_steps.index(materialize)
    for command in (
        "cp -R packages/dinkster-inference-torch .evidence-source/packages/",
        "cp -R scripts/comfyui_benchmark_nodes .evidence-source/scripts/",
        "cp tools/workflow_benchmark*.py .evidence-source/tools/",
        'echo "DINKSTER_EVIDENCE_ROOT=$GITHUB_WORKSPACE/.evidence-source" >> "$GITHUB_ENV"',
        'echo "DINKSTER_ROOT=$GITHUB_WORKSPACE" >> "$GITHUB_ENV"',
    ):
        assert command in materialize["run"]
    for job in JOBS.values():
        for step in job.get("steps", []):
            if step.get("uses") in {ACTION_PATH, prepare_path}:
                assert step["with"]["evidence-deploy-key"] == (
                    "${{ secrets.DINKSTER_EVIDENCE_READ_KEY }}"
                )


def test_public_frontend_checkout_does_not_require_repository_credentials() -> None:
    steps = JOBS["test"]["steps"]
    (checkout,) = [
        step
        for step in steps
        if step.get("with", {}).get("repository") == "Kosinkadink/Dinkster-Frontend"
    ]
    assert checkout["uses"] == "actions/checkout@v4"
    assert "token" not in checkout["with"]
    assert "ssh-key" not in checkout["with"]
    assert all(
        step.get("with", {}).get("repository") != "Kosinkadink/Dinkster-Frontend"
        for step in steps
        if step.get("uses") == "./.github/actions/configure-private-repository"
    )


def test_validation_inputs_expose_existing_git_bash_only_on_windows() -> None:
    helper_path = ROOT / ".github/actions/prepare-validation-inputs/action.yml"
    helper = yaml.safe_load(helper_path.read_text(encoding="utf-8"))
    steps = helper["runs"]["steps"]
    (bootstrap,) = [step for step in steps if "GITHUB_PATH" in step.get("run", "")]
    assert bootstrap["if"] == "runner.os == 'Windows'"
    assert bootstrap["shell"] == "pwsh"
    assert "Join-Path $env:ProgramFiles 'Git/bin'" in bootstrap["run"]
    assert "Test-Path (Join-Path $bashDirectory 'bash.exe') -PathType Leaf" in bootstrap["run"]
    assert "throw 'Git Bash is required for validation input preparation'" in bootstrap["run"]
    assert "$bashDirectory >> $env:GITHUB_PATH" in bootstrap["run"]
    assert all(
        steps.index(bootstrap) < index
        for index, step in enumerate(steps)
        if step.get("shell") == "bash"
    )


def test_validation_history_access_uses_only_step_scoped_credentials() -> None:
    helper_path = ROOT / ".github/actions/prepare-validation-inputs/action.yml"
    helper = yaml.safe_load(helper_path.read_text(encoding="utf-8"))
    (history,) = [
        step
        for step in helper["runs"]["steps"]
        if "git fetch --quiet --depth=1 origin" in step.get("run", "")
    ]
    assert history["if"] == "inputs.coverage == 'true'"
    assert history["env"] == {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": (
            "url.https://x-access-token:${{ github.token }}"
            "@github.com/Kosinkadink/Dinkster.insteadOf"
        ),
        "GIT_CONFIG_VALUE_0": "https://github.com/Kosinkadink/Dinkster",
        "GIT_CONFIG_KEY_1": "credential.helper",
        "GIT_CONFIG_VALUE_1": "",
    }
    assert "git config --global" not in history["run"]
    assert "github.token" not in history["run"]


@pytest.mark.parametrize("event", ["pull_request", "push", "schedule", "workflow_dispatch"])
@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/feature"])
@pytest.mark.parametrize("repository", ["Kosinkadink/Dinkster", "other/Dinkster"])
def test_model_job_runs_only_in_trusted_full_validation(
    event: str, ref: str, repository: str
) -> None:
    job = JOBS["model-tests"]
    # PyYAML reads the YAML 1.1 spelling "on" as True.
    triggers = WORKFLOW[True]
    triggered = event in triggers and (event != "push" or ref == "refs/heads/main")
    allocated = triggered and repository == "Kosinkadink/Dinkster"
    assert allocated == (
        (
            event in {"schedule", "workflow_dispatch"}
            or (event == "push" and ref == "refs/heads/main")
        )
        and repository == "Kosinkadink/Dinkster"
    )
    assert job["runs-on"] == "${{ fromJSON(vars.CI_RUNNERS).linux }}"
    assert job["needs"] == "validation-plan"
    for name, hosted in JOBS.items():
        if name not in {"model-tests", "model-tests-gate", "validation-plan"}:
            assert "model-tests" not in hosted.get("needs", [])


def test_full_model_job_remains_on_main_validation() -> None:
    assert JOBS["model-tests"]["if"] == (
        "needs.validation-plan.outputs.run-heavy == 'true' && "
        "github.repository == 'Kosinkadink/Dinkster'"
    )
    assert "pull_request" not in WORKFLOW[True]
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert workflow[True]["pull_request"] is None


def test_dedicated_job_retains_readonly_credentials_and_cpu_dispatch() -> None:
    job = JOBS["model-tests"]
    assert job["strategy"] == {
        "fail-fast": False,
        "matrix": {"include": list(MODEL_GROUPS)},
    }
    suites = [
        suite
        for group in job["strategy"]["matrix"]["include"]
        for suite in group["suites"].split(",")
    ]
    assert set(suites) == EXPECTED_MODEL_SUITES
    assert all(
        suites.count(suite) == (8 if suite in {"inference-torch", "model-ipadapter"} else 1)
        for suite in EXPECTED_MODEL_SUITES
    )
    assert job["permissions"] == {"contents": "read"}
    assert job["steps"][0] == {
        "uses": "actions/checkout@v4",
        "with": {"clean": True, "persist-credentials": False},
    }
    assert job["steps"][1] == {
        "uses": ACTION_PATH,
        "with": {
            "run-model-tests": "true",
            "evidence-deploy-key": "${{ secrets.DINKSTER_EVIDENCE_READ_KEY }}",
            "pytest-args": "${{ matrix.pytest-args }}",
        },
    }
    assert job["env"] == {
        "ATEN_CPU_CAPABILITY": "avx2",
        "MKL_CBWR": "COMPATIBLE",
        "ONEDNN_MAX_CPU_ISA": "AVX2",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "DINKSTER_MODEL_TEST_GROUP": "${{ matrix.group }}",
    }
    assert JOBS["model-tests-gate"] == {
        "needs": ["validation-plan", "model-tests"],
        "if": (
            "always() && needs.validation-plan.outputs.run-heavy == 'true' && "
            "github.repository == 'Kosinkadink/Dinkster'"
        ),
        "runs-on": "${{ fromJSON(vars.CI_RUNNERS).linux }}",
        "timeout-minutes": 2,
        "steps": [
            {
                "name": "Verify every model-test group passed",
                "run": "test '${{ needs.model-tests.result }}' = success",
            }
        ],
    }
    for name in ("p2p-descriptor-macos", "p2p-artifact-smoke"):
        assert JOBS[name]["if"] == "needs.validation-plan.outputs.run-heavy == 'true'"


def test_pr_workflow_runs_bounded_fast_and_engine_suites() -> None:
    assert set(PR_JOBS) == {"fast"}
    assert set(PR_WORKFLOW[True]) == {"pull_request", "workflow_dispatch"}
    assert PR_WORKFLOW[True]["workflow_dispatch"] == {
        "inputs": {
            "simulate-fork": {
                "description": "Run the fork pull-request policy without repository secrets",
                "required": True,
                "default": False,
                "type": "boolean",
            }
        }
    }
    assert PR_WORKFLOW["concurrency"] == {
        "group": "${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress": True,
    }
    assert WORKFLOW[True] == {
        "push": {"branches": ["main"]},
        "schedule": [
            {"cron": "0 6-22/2 * * *", "timezone": "America/Los_Angeles"},
            {"cron": "23 10 * * *"},
        ],
        "workflow_dispatch": None,
        "workflow_call": None,
    }
    job = PR_JOBS["fast"]
    assert job["timeout-minutes"] == 10
    assert job["steps"][-2] == {
        "if": PRIVATE_DEPENDENCIES_AVAILABLE,
        "run": "bash scripts/ci-fast.sh",
    }
    preparation = [
        step
        for step in job["steps"]
        if step.get("uses") == "./.github/actions/prepare-validation-inputs"
    ]
    assert len(preparation) == 1
    assert preparation[0]["with"]["coverage"] == "false"
    assert "dinkster-training" not in str(PR_WORKFLOW)
    script = (ROOT / "scripts/ci-fast.sh").read_text(encoding="utf-8")
    assert "ruff format --check ." in script
    assert "ruff check ." in script
    assert "uv run --locked pyright" in script
    assert re.findall(r"tests/\S+\.py", script) == [
        "tests/test_extension_contract_pack.py",
        "tests/test_extension_factory_guard.py",
        "tests/test_family_isinstance_guard.py",
        "tests/test_family_registration_gates.py",
        "tests/test_release_install.py",
        "tests/test_schema.py",
        "tests/test_schema_current_contracts.py",
        "tests/test_values.py",
        "tests/test_graph.py",
        "tests/test_graph_wire.py",
    ]
    assert "--cov" not in script
    assert "torch-cpu-suite" not in str(job)


def test_pr_engine_suites_use_cpu_golden_shards_with_a_thirty_minute_bound() -> None:
    assert "engine-tests" not in PR_JOBS
    assert JOBS["model-tests"]["timeout-minutes"] == 30
    assert [row["group"] for row in MODEL_GROUPS].count("inference") == 8
    assert {row["pytest-args"] for row in MODEL_GROUPS if row["group"] == "inference"} == {
        f"-p tools.pytest_file_shard --file-shard {shard}/8" for shard in range(1, 9)
    }


def test_pr_inference_step_applies_only_the_declared_pytest_arguments() -> None:
    (step,) = [
        step
        for step in ACTION["runs"]["steps"]
        if step.get("name") == "Test inference and IPAdapter"
    ]
    assert step["if"] == (f"{MODEL_CONDITION} && {MODEL_GROUP_ENV} == 'inference'")
    assert step["env"] == {"DINKSTER_MODEL_PYTEST_ARGS": "${{ inputs.pytest-args }}"}
    assert "$DINKSTER_MODEL_PYTEST_ARGS" in step["run"]
    assert "packages/dinkster-inference-torch/tests" in step["run"]
    assert "packages/dinkster-model-ipadapter/tests" in step["run"]


def test_windows_file_shards_refresh_tracked_files_after_checkout(tmp_path: Path) -> None:
    job = JOBS["test"]
    windows_rows = [row for row in job["strategy"]["matrix"]["include"] if row["os"] == "windows"]
    assert {row["pytest_args"] for row in windows_rows} == {
        "-p tools.pytest_file_shard --file-shard 1/2",
        "-p tools.pytest_file_shard --file-shard 2/2",
    }

    steps = job["steps"]
    checkout_index = next(
        index for index, step in enumerate(steps) if step.get("uses") == "actions/checkout@v4"
    )
    refresh = steps[checkout_index + 1]
    assert refresh == {
        "name": "Refresh tracked files with current attributes",
        "if": "matrix.os == 'windows'",
        "shell": "pwsh",
        "run": (
            "git rm -r --cached -q .\n"
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
            "git reset --hard -q HEAD\n"
        ),
    }
    repair_commands = "\n".join(
        line for line in refresh["run"].splitlines() if line.startswith("git ")
    )
    assert repair_commands == "git rm -r --cached -q .\ngit reset --hard -q HEAD"

    clone = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(ROOT), str(clone)],
        check=True,
    )
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=clone, check=True)
    relative_license = Path(
        "packages/dinkster-nodes-vision/dinkster_vision_hed_pack/LINEART_LICENSE"
    )
    license_file = clone / relative_license
    attributes = clone / ".gitattributes"
    current_attributes = attributes.read_text(encoding="utf-8")
    assert "**/*_LICENSE text eol=lf\n" in current_attributes
    attributes.write_text(
        current_attributes.replace(
            "**/*_LICENSE text eol=lf\n",
            "**/CLIP_LICENSE text eol=lf\n**/SAM_LICENSE text eol=lf\n",
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".gitattributes"], cwd=clone, check=True)
    license_file.unlink()
    subprocess.run(
        ["git", "checkout", "--", relative_license.as_posix()],
        cwd=clone,
        check=True,
    )
    assert b"\r\n" in license_file.read_bytes()

    subprocess.run(
        ["git", "restore", "--source=HEAD", "--staged", "--worktree", ".gitattributes"],
        cwd=clone,
        check=True,
    )
    lf_bytes = subprocess.run(
        ["git", "show", f"HEAD:{relative_license.as_posix()}"],
        cwd=clone,
        check=True,
        capture_output=True,
    ).stdout
    assert b"\r" not in lf_bytes
    assert b"\r\n" in license_file.read_bytes()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", relative_license.as_posix()],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    subprocess.run(
        [_bash_executable(), "-e", "-o", "pipefail", "-c", repair_commands],
        cwd=clone,
        check=True,
    )
    assert license_file.read_bytes() == lf_bytes
    subprocess.run(
        [_bash_executable(), "-e", "-o", "pipefail", "-c", repair_commands],
        cwd=clone,
        check=True,
    )
    assert license_file.read_bytes() == lf_bytes
    assert (
        subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=clone,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_full_validation_batches_pushes_without_cancelling_active_runs() -> None:
    assert WORKFLOW["permissions"] == {"actions": "read", "contents": "read"}
    assert WORKFLOW["concurrency"] == {
        "group": (
            "${{ github.workflow }}-${{ github.ref }}-"
            "${{ github.event_name == 'push' && 'push' || 'durable' }}"
        ),
        "cancel-in-progress": False,
    }
    plan = JOBS["validation-plan"]
    assert plan["outputs"] == {"run-heavy": "${{ steps.plan.outputs.run-heavy }}"}
    script = plan["steps"][0]["with"]["script"]
    for required in (
        "context.eventName !== 'schedule'",
        "workflow_id: 'full-validation.yml'",
        "branch: 'main'",
        "status: 'success'",
        "per_page: 1",
        "workflow_runs[0]?.head_sha === context.sha",
    ):
        assert required in script
    for name, job in JOBS.items():
        if name in {"validation-plan", "main-status"}:
            continue
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert "validation-plan" in needs, name
        assert "needs.validation-plan.outputs.run-heavy == 'true'" in job["if"], name

    docs = (ROOT / "docs/testing.md").read_text(encoding="utf-8").replace("\n", " ")
    assert "`on.schedule` cron list in that file is the single schedule definition" in docs
    assert "one active push run and only the newest pending push run" in docs
    assert "git merge-base --is-ancestor" in docs


def test_destructive_disk_reclaim_runs_only_on_github_hosted() -> None:
    steps = [step for step in ACTION["runs"]["steps"] if "sudo rm" in step.get("run", "")]
    assert len(steps) == 1
    assert steps[0]["if"] == "runner.environment == 'github-hosted'"
    assert _condition_matches(steps[0]["if"], {"runner.environment": "github-hosted"})


def test_one_required_variable_controls_every_hosted_eligible_job() -> None:
    workflows = (PR_WORKFLOW, WORKFLOW, RELEASE_WORKFLOW)
    for workflow in workflows:
        for name, job in workflow["jobs"].items():
            if "uses" in job:
                continue
            assert "vars.CI_RUNNERS" in str(job["runs-on"]), name
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github/workflows").glob("*.yml"))
    )
    assert "DINKSTER_PR_RUNNER" not in source


def test_full_suites_run_directly_on_hosted_runners() -> None:
    test_commands = "\n".join(str(step.get("run", "")) for step in JOBS["test"]["steps"])
    coverage_commands = "\n".join(str(step.get("run", "")) for step in JOBS["coverage"]["steps"])
    assert "run_counted_suite" not in test_commands
    assert "run_counted_suite" not in coverage_commands


def test_coverage_shards_combine_before_enforcing_the_unchanged_floor() -> None:
    coverage = JOBS["coverage"]
    assert coverage["strategy"] == {
        "fail-fast": False,
        "matrix": {"shard": [1, 2, 3, 4]},
    }
    source = "\n".join(str(step.get("run", "")) for step in coverage["steps"])
    assert "--file-shard=${{ matrix.shard }}/4" in source
    assert "--cov-report=" in source
    assert "--cov-fail-under" not in source
    gate = JOBS["coverage-gate"]
    gate_source = "\n".join(str(step.get("run", "")) for step in gate["steps"])
    assert "test '${{ needs.coverage.result }}' = success" in gate_source
    assert "coverage combine coverage-data" in gate_source
    assert "coverage report --fail-under=80" in gate_source


def test_main_status_records_and_requires_every_complete_lane() -> None:
    status = JOBS["main-status"]
    assert status["if"] == "always()"
    assert status["needs"] == [
        "validation-plan",
        "test",
        "p2p-descriptor-macos",
        "p2p-artifact-smoke",
        "torch-cpu",
        "model-tests-gate",
        "coverage-gate",
        "translation-coverage",
    ]
    source = "\n".join(str(step.get("run", "")) for step in status["steps"])
    assert "main-validation-status.json" in source
    for result in (
        "TEST_RESULT",
        "MACOS_RESULT",
        "ARTIFACT_RESULT",
        "TORCH_RESULT",
        "MODEL_RESULT",
        "COVERAGE_RESULT",
        "TRANSLATION_RESULT",
    ):
        assert f'test "${result}" = success' in source


def test_full_validation_pytest_and_demo_jobs_are_timeout_bounded() -> None:
    pytest_or_demo = {
        name
        for name, job in JOBS.items()
        if any(
            "pytest" in str(step.get("run", ""))
            or "dinkster demo" in str(step.get("run", ""))
            or step.get("uses") == ACTION_PATH
            for step in job["steps"]
        )
    }
    assert pytest_or_demo == {
        "test",
        "p2p-descriptor-macos",
        "p2p-artifact-smoke",
        "torch-cpu",
        "model-tests",
        "coverage",
        "translation-coverage",
    }
    test_job = JOBS["test"]
    assert test_job["timeout-minutes"] == "${{ matrix.timeout-minutes }}"
    assert [row["timeout-minutes"] for row in test_job["strategy"]["matrix"]["include"]] == [
        20,
        20,
        20,
        20,
        30,
        30,
    ]
    assert JOBS["p2p-descriptor-macos"]["timeout-minutes"] == 15
    assert JOBS["p2p-artifact-smoke"]["timeout-minutes"] == 15
    assert JOBS["torch-cpu"]["timeout-minutes"] == 20
    assert JOBS["model-tests"]["timeout-minutes"] == 30
    assert JOBS["coverage"]["timeout-minutes"] == 30
    assert JOBS["coverage-gate"]["timeout-minutes"] == 5
    assert JOBS["translation-coverage"]["timeout-minutes"] == 15
    # A hung pytest run self-identifies the stuck test through pytest's
    # bundled faulthandler_timeout before the job bound releases the runner;
    # it is diagnostic only and never skips or fails a test
    # (comfy-vibe-station#245).
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["pytest"]["ini_options"]["faulthandler_timeout"] == 600
