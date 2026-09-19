from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from tools.evidence_paths import EVIDENCE_ROOT

ROOT = Path(__file__).resolve().parents[1]
ACTION_PATH = "./.github/actions/torch-cpu-suite"
ACTION = yaml.safe_load((ROOT / ACTION_PATH / "action.yml").read_text(encoding="utf-8"))
WORKFLOW = yaml.safe_load(
    (ROOT / ".github/workflows/full-validation.yml").read_text(encoding="utf-8")
)
JOBS = WORKFLOW["jobs"]
MODEL_CONDITION = "inputs.run-model-tests == 'true'"
RECEIPT_TEST = "tests/test_gen_comfy_source_parity_receipts.py"
ACCEPTANCE_CLOSURE_TEST = (
    ".evidence-source/packages/dinkster-acceptance/tests/test_acceptance_closure.py"
)
ACCEPTANCE_SAMPLING_TEST = (
    ".evidence-source/packages/dinkster-acceptance/tests/test_acceptance_sampling.py"
)
TRAINING_SUITE = "packages/dinkster-training-torch/tests"
VISION_SUITES = (
    "packages/dinkster-vision-hed/tests",
    "packages/dinkster-vision-upscale/tests",
    "packages/dinkster-vision-depth-anything-v2/tests",
    "packages/dinkster-vision-detr/tests",
    "packages/dinkster-vision-rtdetr/tests",
    "packages/dinkster-vision-efficient-sam/tests",
    "packages/dinkster-vision-birefnet/tests",
    "packages/dinkster-vision-depth-anything-v3/tests",
    "packages/dinkster-vision-sam31/tests",
)


def _condition_matches(expression: str, context: dict[str, str]) -> bool:
    """Evaluate the equality/conjunction subset used by these allocation guards."""
    matches = []
    for clause in expression.split("&&"):
        match = re.fullmatch(r"\s*([\w.-]+)\s*==\s*'([^']*)'\s*", clause)
        assert match is not None, clause
        matches.append(context[match[1]].lower() == match[2].lower())
    return all(matches)


def test_every_hosted_composite_caller_explicitly_excludes_model_tests() -> None:
    callers = []
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        for name, job in yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"].items():
            for step in job.get("steps", []):
                if step.get("uses") != ACTION_PATH:
                    continue
                callers.append((path.name, name))
                assert step["with"]["run-model-tests"] == (
                    "true" if name == "model-tests" else "false"
                )
                if name != "model-tests":
                    assert job["runs-on"] == "ubuntu-latest"
    assert set(callers) == {
        ("full-validation.yml", "model-tests"),
        *(("full-validation.yml", f"torch-cpu-try{number}") for number in range(1, 6)),
    }
    assert len(callers) == 6
    assert ACTION["inputs"]["run-model-tests"]["default"] == "false"


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
            and TRAINING_SUITE not in command
            and (not is_vision_execution or "env" in step)
        )
        if not (is_download or is_execution):
            continue
        assert step["if"] == MODEL_CONDITION
        assert _condition_matches(step["if"], {"inputs.run-model-tests": enabled}) == (
            enabled == "true"
        )
        if is_download:
            downloads.append(step)
        if is_execution:
            executions.append(step)
    assert len(downloads) == 9
    assert len(executions) == 12
    assert sum(len(step["env"]) for step in executions if "env" in step) == 18


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
    assert "if" not in common
    assert sampling["run"].strip() == (
        f".venv-torch/bin/python -m pytest -q {ACCEPTANCE_SAMPLING_TEST}"
    )
    assert sampling["if"] == MODEL_CONDITION


def test_weight_free_vision_suites_remain_hosted_without_artifacts() -> None:
    for suite in VISION_SUITES:
        executions = [
            step
            for step in ACTION["runs"]["steps"]
            if "-m pytest" in step.get("run", "") and suite in step["run"]
        ]
        assert len(executions) == 2
        hosted, model = executions
        assert "if" not in hosted
        assert "env" not in hosted
        assert model["if"] == MODEL_CONDITION
        assert model["env"]


def test_model_lane_commands_artifact_pins_and_environments_match_reviewed_contract() -> None:
    # The digest covers the ordered full lanes, including URLs, SHA-256 checks,
    # selectors and artifact env bindings. Pin changes require deliberate review.
    steps = [
        {key: value for key, value in step.items() if key not in {"if", "name"}}
        for step in ACTION["runs"]["steps"]
        if step.get("if") == MODEL_CONDITION
    ]
    assert hashlib.sha256(json.dumps(steps, sort_keys=True).encode()).hexdigest() == (
        "0fb8c48ce81fddb3057ce834a0486c5cf6f12a1cf872215eb697a3dfc88a82b0"
    )


def test_source_receipts_typechecks_and_training_suite_remain_hosted() -> None:
    retained = []
    for step in ACTION["runs"]["steps"]:
        if step.get("if") == MODEL_CONDITION or step.get("name") == "Reclaim runner disk":
            continue
        assert "if" not in step
        retained.append(step)
    commands = [step.get("run", "").strip() for step in retained]
    projects = {
        path.parent.name
        for path in (
            *(ROOT / "packages").glob("*/pyproject.toml"),
            EVIDENCE_ROOT / "packages/dinkster-acceptance/pyproject.toml",
        )
        if 'venv = ".venv-torch"' in path.read_text(encoding="utf-8")
    }
    assert len(projects) == 12
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
    training_steps = [step for step in retained if TRAINING_SUITE in step.get("run", "")]
    assert training_steps == [
        {
            "name": "Test training runtime",
            "shell": "bash",
            "env": {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
            "run": f".venv-torch/bin/python -m pytest -q {TRAINING_SUITE}",
        }
    ]
    assert {
        command
        for command in commands
        if "-m pytest" in command and any(suite in command for suite in VISION_SUITES)
    } == {f".venv-torch/bin/python -m pytest -q {suite}" for suite in VISION_SUITES}
    assert f".venv-torch/bin/python -m pytest -q {ACCEPTANCE_CLOSURE_TEST}" in commands
    assert all(ACCEPTANCE_SAMPLING_TEST not in command for command in commands)
    assert ".venv-torch/bin/python tools/gen_comfy_source_parity_receipts.py --check" in commands
    assert "UV_CONSTRAINT=/tmp/torch-constraints.txt ./scripts/setup_envs.sh" in commands
    assert {step["name"] for step in retained if "name" in step} == {
        "Checkout pinned ComfyUI source",
        "Checkout pinned workflow templates",
        "Acquire pinned Impact Pack source",
        "Verify source-generated parity receipts",
        "Test source-parity receipt generation",
        "Assert pinned CPU dispatch",
        "Test training runtime",
    }


@pytest.mark.parametrize("enabled", ["", "false", "true"])
@pytest.mark.parametrize("event", ["pull_request", "push", "schedule", "workflow_dispatch"])
@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/feature"])
@pytest.mark.parametrize("repository", ["Kosinkadink/Dinkster", "other/Dinkster"])
def test_dedicated_job_allocates_only_for_enabled_trusted_full_validation(
    enabled: str, event: str, ref: str, repository: str
) -> None:
    job = JOBS["model-tests"]
    # PyYAML reads the YAML 1.1 spelling "on" as True.
    triggers = WORKFLOW[True]
    triggered = event in triggers and (event != "push" or ref == "refs/heads/main")
    allocated = triggered and _condition_matches(
        job["if"],
        {
            "vars.DINKSTER_MODEL_TESTS_ENABLED": enabled,
            "github.event_name": event,
            "github.ref": ref,
            "github.repository": repository,
        },
    )
    assert allocated == (
        enabled == "true"
        and (
            event in {"schedule", "workflow_dispatch"}
            or (event == "push" and ref == "refs/heads/main")
        )
        and repository == "Kosinkadink/Dinkster"
    )
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "dinkster-model-tests"]
    assert "needs" not in job
    for name, hosted in JOBS.items():
        if name != "model-tests":
            assert "model-tests" not in hosted.get("needs", [])


def test_dedicated_job_retains_readonly_credentials_and_cpu_dispatch() -> None:
    job = JOBS["model-tests"]
    assert job["permissions"] == {"contents": "read"}
    assert job["steps"][0] == {
        "uses": "actions/checkout@v4",
        "with": {"persist-credentials": False},
    }
    assert job["steps"][1] == {
        "uses": ACTION_PATH,
        "with": {
            "run-model-tests": "true",
            "identity-deploy-key": "${{ secrets.DINKSTER_IDENTITY_DEPLOY_KEY }}",
            "evidence-deploy-key": "${{ secrets.DINKSTER_EVIDENCE_READ_KEY }}",
        },
    }
    assert job["env"] == {
        "ATEN_CPU_CAPABILITY": "avx2",
        "ONEDNN_MAX_CPU_ISA": "AVX2",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    for name in ("p2p-descriptor-macos", "p2p-artifact-smoke"):
        assert "if" not in JOBS[name]


def test_pr_workflow_has_only_the_bounded_weight_free_subset() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert set(workflow["jobs"]) == {"fast"}
    assert set(workflow[True]) == {"pull_request", "workflow_dispatch"}
    assert WORKFLOW[True] == {
        "push": {"branches": ["main"]},
        "schedule": [{"cron": "23 10 * * *"}],
        "workflow_dispatch": None,
    }
    job = workflow["jobs"]["fast"]
    assert job["timeout-minutes"] == 5
    assert job["steps"][-1] == {"run": "bash scripts/ci-fast.sh"}
    preparation = [
        step
        for step in job["steps"]
        if step.get("uses") == "./.github/actions/prepare-validation-inputs"
    ]
    assert len(preparation) == 1
    assert preparation[0]["with"]["coverage"] == "false"
    script = (ROOT / "scripts/ci-fast.sh").read_text(encoding="utf-8")
    assert "ruff format --check ." in script
    assert "ruff check ." in script
    assert "uv run --locked pyright" in script
    assert re.findall(r"tests/\S+\.py", script) == [
        "tests/test_schema.py",
        "tests/test_values.py",
        "tests/test_graph.py",
        "tests/test_graph_wire.py",
    ]
    assert "--cov" not in script
    assert "torch-cpu-suite" not in str(job)


@pytest.mark.parametrize("environment", ["github-hosted", "self-hosted"])
def test_destructive_disk_reclaim_never_runs_on_self_hosted(environment: str) -> None:
    steps = [step for step in ACTION["runs"]["steps"] if "sudo rm" in step.get("run", "")]
    assert len(steps) == 1
    assert steps[0]["if"] == "runner.environment == 'github-hosted'"
    assert _condition_matches(steps[0]["if"], {"runner.environment": environment}) == (
        environment == "github-hosted"
    )
