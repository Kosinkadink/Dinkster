from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
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
MODEL_GROUP_ENV = "env.DINKSTER_MODEL_TEST_GROUP"
RECEIPT_TEST = "tests/test_gen_comfy_source_parity_receipts.py"
ACCEPTANCE_CLOSURE_TEST = (
    ".evidence-source/packages/dinkster-acceptance/tests/test_acceptance_closure.py"
)
ACCEPTANCE_SAMPLING_TEST = (
    ".evidence-source/packages/dinkster-acceptance/tests/test_acceptance_sampling.py"
)
TRAINING_SUITE = "packages/dinkster-training-torch/tests"
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
MODEL_GROUPS = (
    {
        "name": "inference and IPAdapter",
        "group": "inference",
        "suites": "inference-torch,model-ipadapter",
    },
    {
        "name": "acceptance and benchmark",
        "group": "acceptance",
        "suites": "acceptance-sampling,benchmark-loader",
    },
    {
        "name": "HED, upscale and EfficientSAM",
        "group": "vision-fast",
        "suites": "hed,upscale,efficient-sam",
    },
    {
        "name": "Depth Anything V2, DETR and RT-DETR",
        "group": "vision-detection",
        "suites": "depth-anything-v2,detr,rtdetr",
    },
    {
        "name": "BiRefNet and Depth Anything V3",
        "group": "vision-large",
        "suites": "birefnet,depth-anything-v3",
    },
    {"name": "SAM 3.1", "group": "vision-sam", "suites": "sam31"},
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


def test_every_cpu_composite_caller_explicitly_excludes_model_tests() -> None:
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
                assert job["runs-on"] == [
                    "self-hosted",
                    "Linux",
                    "X64",
                    "cpu-golden-avx2",
                ]
    assert set(callers) == {
        ("full-validation.yml", "model-tests"),
        ("full-validation.yml", "torch-cpu"),
    }
    assert len(callers) == 2
    assert ACTION["inputs"]["run-model-tests"]["default"] == "false"


def test_torch_cpu_has_one_contract_guard_and_an_unconditional_suite() -> None:
    assert [name for name in JOBS if name.startswith("torch-cpu")] == ["torch-cpu"]
    job = JOBS["torch-cpu"]
    assert set(job) == {"needs", "if", "runs-on", "env", "steps"}
    assert job["needs"] == "validation-plan"
    assert job["if"] == "needs.validation-plan.outputs.run-heavy == 'true'"
    assert job["runs-on"] == ["self-hosted", "Linux", "X64", "cpu-golden-avx2"]
    assert job["env"] == {"ATEN_CPU_CAPABILITY": "avx2", "ONEDNN_MAX_CPU_ISA": "AVX2"}
    guard, checkout, suite = job["steps"]
    assert set(guard) == {"name", "run"}
    assert checkout == {
        "uses": "actions/checkout@v4",
        "with": {"clean": True, "persist-credentials": False},
    }
    assert suite == {
        "uses": ACTION_PATH,
        "with": {
            "identity-deploy-key": "${{ secrets.DINKSTER_IDENTITY_DEPLOY_KEY }}",
            "evidence-deploy-key": "${{ secrets.DINKSTER_EVIDENCE_READ_KEY }}",
            "run-model-tests": "false",
        },
    }


@pytest.mark.skipif(os.name != "posix", reason="Linux CPU guard runs in Bash")
@pytest.mark.parametrize(
    ("vendor", "flags", "accepted"),
    [
        ("AuthenticAMD", "sse2 avx avx2", True),
        ("GenuineIntel", "sse2 avx avx2", False),
        ("AuthenticAMD", "sse2 avx", False),
        ("AuthenticAMD", "sse2 avx avx2 avx512f", False),
        ("AuthenticAMD", "sse2 avx avx20", False),
    ],
)
def test_torch_cpu_guard_executes_the_golden_contract(
    tmp_path: Path, vendor: str, flags: str, accepted: bool
) -> None:
    cpuinfo = tmp_path / "cpu info"
    cpuinfo.write_text(
        f"model name : Test CPU\nvendor_id : {vendor}\nflags : {flags}\n", encoding="utf-8"
    )
    script = JOBS["torch-cpu"]["steps"][0]["run"]
    assert "/proc/cpuinfo" in script
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-o",
            "pipefail",
            "-c",
            script.replace("/proc/cpuinfo", shlex.quote(str(cpuinfo))),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if accepted else 1), result.stderr
    if not accepted:
        assert (
            "CPU golden contract requires AuthenticAMD with AVX2 and without AVX-512"
            in result.stderr
        )


def test_artifact_smoke_uses_only_available_self_hosted_platforms() -> None:
    assert JOBS["p2p-artifact-smoke"]["strategy"]["matrix"] == {
        "os": ["linux", "windows", "macos"],
        "python-version": ["3.12"],
        "include": [
            {"os": "linux", "labels": ["self-hosted", "linux", "x64"]},
            {"os": "windows", "labels": ["self-hosted", "windows", "x64"]},
            {"os": "macos", "labels": ["self-hosted", "macos", "arm64"]},
        ],
    }


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
        group = _model_group(step)
        assert step["if"] == (f"{MODEL_CONDITION} && {MODEL_GROUP_ENV} == '{group}'")
        assert _condition_matches(
            step["if"],
            {"inputs.run-model-tests": enabled, MODEL_GROUP_ENV: group},
        ) == (enabled == "true")
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
        "5583341b0b7a3877b63970114606a11f5c6f3dd426cea7c592eee8f4a285dc1d"
    )


def test_source_receipts_typechecks_and_training_suite_remain_hosted() -> None:
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
        for path in (
            *(ROOT / "packages").glob("*/pyproject.toml"),
            EVIDENCE_ROOT / "packages/dinkster-acceptance/pyproject.toml",
        )
        if 'venv = ".venv-torch"' in path.read_text(encoding="utf-8")
    }
    assert len(projects) == 4
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
            "if": f"{MODEL_GROUP_ENV} == ''",
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
    assert 'UV_CONSTRAINT="$RUNNER_TEMP/torch-constraints.txt" ./scripts/setup_envs.sh' in commands
    assert {step["name"] for step in retained if "name" in step} == {
        "Checkout pinned ComfyUI source",
        "Checkout pinned workflow templates",
        "Acquire pinned Impact Pack source",
        "Verify source-generated parity receipts",
        "Test source-parity receipt generation",
        "Assert pinned CPU dispatch",
        "Test training runtime",
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
        and step.get("uses") == "./.github/actions/configure-dinkster-identity"
    ]
    assert access["with"]["deploy-key"] == "${{ inputs.evidence-deploy-key }}"
    (checkout,) = [step for step in preparation_steps if step.get("uses") == "actions/checkout@v4"]
    assert checkout["with"] == {
        "repository": "Kosinkadink/dinkster-evidence",
        "ref": "16d3d1dae062266232758b07cda181ca3ad881e3",
        "path": ".evidence-source",
        "clean": True,
        "persist-credentials": False,
    }
    assert preparation_steps.index(access) < preparation_steps.index(checkout)
    (identity,) = [
        step
        for step in steps
        if step.get("uses") == "./.github/actions/configure-dinkster-identity"
    ]
    assert identity["with"]["deploy-key"] == "${{ inputs.identity-deploy-key }}"
    setup = next(step for step in steps if "./scripts/setup_envs.sh" in step.get("run", ""))
    assert steps.index(prepare) < steps.index(identity) < steps.index(setup)
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
    assert job["runs-on"] == ["self-hosted", "Linux", "X64", "cpu-golden-avx2"]
    assert job["needs"] == "validation-plan"
    for name, hosted in JOBS.items():
        if name not in {"model-tests", "model-tests-gate", "validation-plan"}:
            assert "model-tests" not in hosted.get("needs", [])


def test_model_job_has_no_pull_request_label_path() -> None:
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
    assert len(suites) == len(set(suites))
    assert set(suites) == EXPECTED_MODEL_SUITES
    assert job["permissions"] == {"contents": "read"}
    assert job["steps"][0] == {
        "uses": "actions/checkout@v4",
        "with": {"clean": True, "persist-credentials": False},
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
        "DINKSTER_MODEL_TEST_GROUP": "${{ matrix.group }}",
    }
    assert JOBS["model-tests-gate"] == {
        "needs": ["validation-plan", "model-tests"],
        "if": (
            "always() && needs.validation-plan.outputs.run-heavy == 'true' && "
            "github.repository == 'Kosinkadink/Dinkster'"
        ),
        "runs-on": ["self-hosted", "linux", "x64"],
        "steps": [
            {
                "name": "Verify every model-test group passed",
                "run": "test '${{ needs.model-tests.result }}' = success",
            }
        ],
    }
    for name in ("p2p-descriptor-macos", "p2p-artifact-smoke"):
        assert JOBS[name]["if"] == "needs.validation-plan.outputs.run-heavy == 'true'"


def test_pr_workflow_has_only_the_bounded_weight_free_subset() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert set(workflow["jobs"]) == {"fast"}
    assert set(workflow[True]) == {"pull_request", "workflow_dispatch"}
    assert WORKFLOW[True] == {
        "push": {"branches": ["main"]},
        "schedule": [
            {"cron": "0 6-22/2 * * *", "timezone": "America/Los_Angeles"},
            {"cron": "23 10 * * *"},
        ],
        "workflow_dispatch": None,
        "workflow_call": None,
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
        "tests/test_extension_contract_pack.py",
        "tests/test_extension_factory_guard.py",
        "tests/test_family_registration_gates.py",
        "tests/test_schema.py",
        "tests/test_values.py",
        "tests/test_graph.py",
        "tests/test_graph_wire.py",
    ]
    assert "--cov" not in script
    assert "torch-cpu-suite" not in str(job)


def test_full_validation_schedule_and_concurrency_keep_durable_runs_alive() -> None:
    assert WORKFLOW["permissions"] == {"actions": "read", "contents": "read"}
    assert WORKFLOW["concurrency"] == {
        "group": (
            "${{ github.workflow }}-${{ github.ref }}-"
            "${{ github.event_name == 'push' && 'push' || 'durable' }}"
        ),
        "cancel-in-progress": "${{ github.event_name == 'push' }}",
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
        if name == "validation-plan":
            continue
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert "validation-plan" in needs, name
        assert "needs.validation-plan.outputs.run-heavy == 'true'" in job["if"], name

    docs = (ROOT / "docs/testing.md").read_text(encoding="utf-8").replace("\n", " ")
    assert "`on.schedule` cron list in that file is the single schedule definition" in docs


@pytest.mark.parametrize("environment", ["github-hosted", "self-hosted"])
def test_destructive_disk_reclaim_never_runs_on_self_hosted(environment: str) -> None:
    steps = [step for step in ACTION["runs"]["steps"] if "sudo rm" in step.get("run", "")]
    assert len(steps) == 1
    assert steps[0]["if"] == "runner.environment == 'github-hosted'"
    assert _condition_matches(steps[0]["if"], {"runner.environment": environment}) == (
        environment == "github-hosted"
    )
