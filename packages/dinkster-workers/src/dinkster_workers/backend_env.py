"""ROCm and XPU backend environments and their smoke reports.

Each supported backend cell (accelerator family plus operating system) has
one frozen recipe: an isolated venv name, a pinned torch build, the vendor
wheel index that serves it, and the workspace packages the backend smoke
and test lanes import. The torch build is the pinned fact; support
packages float, like the CUDA validation lane's. The recipes are the
single source of truth for the executable setup scripts
(scripts/setup_env_rocm.* and scripts/setup_env_xpu.*); a test compares
the scripts' executable commands against :func:`setup_commands` so they
cannot drift silently.

The ROCm and XPU torch packages are different builds of the same ``torch``
distribution and cannot coexist in one environment, so every cell gets its
own venv and its own worker process - even on a host with both GPUs.

A backend claims nothing until its smoke report passes on real hardware.
:func:`validate_smoke_report` checks that a report produced by
scripts/rocm_smoke.py or scripts/xpu_smoke.py records the identity evidence
a support cell requires: host OS build or kernel, driver, torch and backend
runtime versions, device name and architecture, and the baseline capability
probes. Validation proves the report is complete, not that the backend
works; only the recorded probe results say that.

Model-family validation runs one level above the smoke lane:
scripts/family_validation.py executes one real inference workload per cell
(family, execution mode) and emits a report that
:func:`validate_family_report` checks for the same identity evidence plus
the workload, input artifact digests, memory telemetry, and per-check
outcomes. Compile-mode reports must reference a passing eager report for
the cell: eager execution is the support contract, and torch.compile
claims exist only relative to proven eager behavior.

This module stays torch-free: the torch-free root environment inspects
recipes and parses reports, while torch itself exists only inside the
backend venvs the recipes describe.
"""

from __future__ import annotations
from dinkster_values import MEBIBYTE

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

_ROCM_INDEX_URL = "https://repo.amd.com/rocm/whl-multi-arch/"
_ROCM_TORCH_REQUIREMENT = "torch[device-all]==2.12.0+rocm7.14.0"
_XPU_INDEX_URL = "https://download.pytorch.org/whl/xpu"
_XPU_TORCH_REQUIREMENT = "torch==2.13.0+xpu"

# dinkster_inference_torch imports dinkster_kitchen unconditionally, so every
# backend cell needs it. The pin must be the pure-Python wheel: PyPI's
# platform wheels for win_amd64 and linux x86_64 carry CUDA-only compiled
# kernels, while the pure wheel provides the device-agnostic eager backend
# that ROCm and XPU cells run on. uv enforces the sha256 fragment.
_KITCHEN_REQUIREMENT = (
    "dinkster-kitchen@https://files.pythonhosted.org/packages/2e/20/"
    "84e29ca1dedcd51eb5edd297d3c2f6c665cf2e30bb9237892f0f8d108d0d/"
    "dinkster_kitchen-0.2.35.post1-py3-none-any.whl"
    "#sha256=31458547cdcf9ff26974a4955cf79e83ebdf50077666720d3bb3255786c5fc4f"
)

_SUPPORT_PACKAGES = (
    "pytest",
    "numpy",
    "scipy",
    "torchsde",
    "tqdm",
    "pillow",
    "packaging",
    "tokenizers==0.23.1",
    _KITCHEN_REQUIREMENT,
)

_EDITABLE_PACKAGES = (
    "packages/dinkster-schema",
    "packages/dinkster-graph",
    "packages/dinkster-values",
    "packages/dinkster-protocol",
    "packages/dinkster-caches",
    "packages/dinkster-assets",
    "packages/dinkster-memory",
    "packages/dinkster-workers",
    "packages/dinkster-inference",
    "packages/dinkster-inference-torch",
    "packages/dinkster-image-document",
    "packages/dinkster-video",
    "packages/dinkster-api",
    "packages/dinkster-nodes-generation",
    "packages/dinkster-native",
    "packages/dinkster-compat-comfy",
)


@dataclass(frozen=True)
class BackendEnvRecipe:
    """One backend support cell's isolated environment definition."""

    cell: str
    accelerator: str
    os_family: str
    venv: str
    python_version: str
    index_url: str
    torch_requirement: str
    support_packages: tuple[str, ...] = _SUPPORT_PACKAGES
    editable_packages: tuple[str, ...] = _EDITABLE_PACKAGES


def _recipe(
    accelerator: str, os_family: str, index_url: str, torch_requirement: str
) -> BackendEnvRecipe:
    return BackendEnvRecipe(
        cell=f"{os_family}-{accelerator}",
        accelerator=accelerator,
        os_family=os_family,
        venv=f".venv-{accelerator}",
        python_version="3.12",
        index_url=index_url,
        torch_requirement=torch_requirement,
    )


BACKEND_ENV_RECIPES: Mapping[str, BackendEnvRecipe] = MappingProxyType(
    {
        recipe.cell: recipe
        for recipe in (
            _recipe("rocm", "windows", _ROCM_INDEX_URL, _ROCM_TORCH_REQUIREMENT),
            _recipe("rocm", "linux", _ROCM_INDEX_URL, _ROCM_TORCH_REQUIREMENT),
            _recipe("xpu", "windows", _XPU_INDEX_URL, _XPU_TORCH_REQUIREMENT),
            _recipe("xpu", "linux", _XPU_INDEX_URL, _XPU_TORCH_REQUIREMENT),
        )
    }
)


def venv_python(recipe: BackendEnvRecipe) -> str:
    """The venv interpreter path in the cell's native path style."""
    if recipe.os_family == "windows":
        return f"{recipe.venv}\\Scripts\\python.exe"
    return f"{recipe.venv}/bin/python"


def setup_commands(recipe: BackendEnvRecipe) -> tuple[tuple[str, ...], ...]:
    """The exact commands that build the cell's isolated environment.

    All cells install through uv: the editable workspace packages depend on
    each other by distribution name, which plain pip would try to resolve
    from PyPI (where they do not exist) instead of the workspace.
    """
    python = venv_python(recipe)
    editable_args: list[str] = []
    for package in recipe.editable_packages:
        editable_args.extend(("-e", package))
    return (
        # --clear makes reruns deterministic: uv refuses to reuse an
        # existing venv without it, and a fresh environment is the point
        # of the setup script.
        ("uv", "venv", recipe.venv, "--clear", "--python", recipe.python_version),
        (
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--index-url",
            recipe.index_url,
            recipe.torch_requirement,
        ),
        ("uv", "pip", "install", "--python", python, *recipe.support_packages, *editable_args),
        (python, "-c", "import dinkster_compat_comfy.native_arm"),
    )


SMOKE_REPORT_VERSION = 3

_REQUIRED_HOST_FIELDS = ("platform", "os_version", "machine", "python")

_REQUIRED_PROBES = (
    "storage_float32",
    "storage_float16",
    "storage_bfloat16",
    "matmul_float32",
    "matmul_float16",
    "matmul_bfloat16",
    "storage_int8",
    "cast_int8",
    "storage_float8_e4m3fn",
    "cast_float8_e4m3fn",
    "storage_float8_e5m2",
    "cast_float8_e5m2",
    "sdpa_float16",
    "mem_get_info",
    "empty_cache",
    "synchronize",
    "attention_route",
    "dtype_policy",
    "quant_dequant",
)

_SMOKE_GATE_PROBES = (
    "storage_float32",
    "storage_float16",
    "storage_bfloat16",
    "matmul_float32",
    "matmul_float16",
    "matmul_bfloat16",
    "sdpa_float16",
    "mem_get_info",
    "empty_cache",
    "synchronize",
    "attention_route",
    "dtype_policy",
    "quant_dequant",
)


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


def _sha256_hex(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identity_problems(fields: Mapping[str, object], accelerator: str) -> list[str]:
    """Backend identity evidence shared by smoke and family reports."""
    problems: list[str] = []
    if fields.get("accelerator") != accelerator:
        problems.append(f"accelerator is not {accelerator!r}")

    host = _as_mapping(fields.get("host"))
    if host is None:
        problems.append("host section missing")
    else:
        for field in _REQUIRED_HOST_FIELDS:
            if not _nonempty_str(host.get(field)):
                problems.append(f"host.{field} missing or empty")

    driver = fields.get("driver")
    if not _nonempty_str(driver):
        problems.append("driver identity missing or empty")
    elif cast("str", driver).strip().lower().startswith("unknown"):
        problems.append("driver identity is an unknown placeholder")

    torch_section = _as_mapping(fields.get("torch"))
    if torch_section is None:
        problems.append("torch section missing")
    else:
        if not _nonempty_str(torch_section.get("version")):
            problems.append("torch.version missing or empty")
        if not _nonempty_str(torch_section.get("backend_runtime")):
            problems.append("torch.backend_runtime missing or empty")

    devices = fields.get("devices")
    if not isinstance(devices, list) or not devices:
        problems.append("devices list missing or empty")
    else:
        for position, device_entry in enumerate(cast("list[object]", devices)):
            device = _as_mapping(device_entry)
            if device is None:
                problems.append(f"devices[{position}] is not a mapping")
                continue
            if not isinstance(device.get("index"), int):
                problems.append(f"devices[{position}].index missing")
            if not _nonempty_str(device.get("name")):
                problems.append(f"devices[{position}].name missing or empty")
            if not _nonempty_str(device.get("architecture")):
                problems.append(f"devices[{position}].architecture missing or empty")
            total = device.get("total_memory")
            if not isinstance(total, int) or total <= 0:
                problems.append(f"devices[{position}].total_memory missing or not positive")
    return problems


def _artifact_entry_problems(artifacts: list[object], problems: list[str]) -> list[str]:
    """Validate each artifact entry's shape; return the recorded roles."""
    roles: list[str] = []
    for position, artifact_entry in enumerate(artifacts):
        artifact = _as_mapping(artifact_entry)
        if artifact is None:
            problems.append(f"artifacts[{position}] is not a mapping")
            continue
        for name in ("role", "path"):
            if not _nonempty_str(artifact.get(name)):
                problems.append(f"artifacts[{position}].{name} missing or empty")
        role = artifact.get("role")
        if isinstance(role, str) and role:
            roles.append(role)
        if not _sha256_hex(artifact.get("sha256")):
            problems.append(f"artifacts[{position}].sha256 is not a sha256 hex digest")
        size = artifact.get("bytes")
        if not isinstance(size, int) or size <= 0:
            problems.append(f"artifacts[{position}].bytes missing or not positive")
    return roles


def _unload_ceiling_problems(
    checks: Mapping[str, object] | None,
    memory: Mapping[str, object] | None,
    problems: list[str],
) -> None:
    """A passing unload check cannot coexist with residual memory above
    the ceiling."""
    if checks is None or memory is None:
        return
    unload = _as_mapping(checks.get("unload"))
    residual = memory.get("residual_allocated_bytes")
    if (
        unload is not None
        and unload.get("ok") is True
        and isinstance(residual, int)
        and residual > FAMILY_RESIDUAL_CEILING_BYTES
    ):
        problems.append(
            "memory.residual_allocated_bytes exceeds the unload ceiling"
            " despite a passing unload check"
        )


def _all_ok_problems(
    fields: Mapping[str, object],
    checks: Mapping[str, object] | None,
    problems: list[str],
) -> None:
    """``all_ok`` must equal the conjunction of the recorded check
    outcomes - a report claiming success over failed checks is corrupt."""
    all_ok = fields.get("all_ok")
    if not isinstance(all_ok, bool):
        problems.append("all_ok missing")
        return
    if checks is None:
        return
    outcomes = [
        entry.get("ok")
        for entry in (_as_mapping(value) for value in checks.values())
        if entry is not None
    ]
    if outcomes and all_ok != all(outcome is True for outcome in outcomes):
        problems.append("all_ok is inconsistent with the recorded checks")


def _check_entry_problems(
    section: Mapping[str, object] | None,
    section_name: str,
    required: tuple[str, ...],
) -> list[str]:
    """Every required {ok, detail} entry present and well-formed."""
    if section is None:
        return [f"{section_name} section missing"]
    problems: list[str] = []
    for name in required:
        entry = _as_mapping(section.get(name))
        if entry is None:
            problems.append(f"{section_name}.{name} missing")
            continue
        if not isinstance(entry.get("ok"), bool):
            problems.append(f"{section_name}.{name}.ok missing")
        if not _nonempty_str(entry.get("detail")):
            problems.append(f"{section_name}.{name}.detail missing or empty")
    return problems


def validate_smoke_report(report: object, *, accelerator: str) -> tuple[str, ...]:
    """Problems that make a backend smoke report unusable as evidence.

    Returns an empty tuple when the report records every required identity
    field and probe result. A complete report is necessary but not
    sufficient for a support claim: the probe outcomes and a real-hardware
    run decide that.
    """
    if accelerator not in ("rocm", "xpu"):
        raise ValueError(f"unknown backend accelerator {accelerator!r}")
    fields = _as_mapping(report)
    if fields is None:
        return ("report is not a mapping",)
    problems: list[str] = []
    if fields.get("report_version") != SMOKE_REPORT_VERSION:
        problems.append(f"report_version is not {SMOKE_REPORT_VERSION}")
    problems.extend(_identity_problems(fields, accelerator))
    problems.extend(
        _check_entry_problems(_as_mapping(fields.get("probes")), "probes", _REQUIRED_PROBES)
    )
    if not isinstance(fields.get("baseline_ok"), bool):
        problems.append("baseline_ok missing")
    return tuple(problems)


def smoke_gate_problems(report: object, *, accelerator: str) -> tuple[str, ...]:
    """Problems that prevent a smoke report from proving backend support."""
    problems = list(validate_smoke_report(report, accelerator=accelerator))
    fields = _as_mapping(report)
    if fields is None:
        return tuple(problems)
    if fields.get("baseline_ok") is not True:
        problems.append("baseline_ok is not true")
    probes = _as_mapping(fields.get("probes"))
    if probes is not None:
        for name in _SMOKE_GATE_PROBES:
            entry = _as_mapping(probes.get(name))
            if entry is not None and entry.get("ok") is not True:
                problems.append(f"probes.{name} did not pass")
    return tuple(problems)


def format_smoke_receipt(report: object, *, accelerator: str) -> str:
    """A short ASCII receipt for one ROCm or XPU smoke run."""
    problems = smoke_gate_problems(report, accelerator=accelerator)
    fields = _as_mapping(report) or {}
    devices = fields.get("devices")
    device_fields: Mapping[str, object] = {}
    if isinstance(devices, list) and devices:
        device_fields = _as_mapping(cast("list[object]", devices)[0]) or {}
    torch_fields = _as_mapping(fields.get("torch")) or {}
    probes = _as_mapping(fields.get("probes")) or {}

    def passed(name: str) -> bool:
        entry = _as_mapping(probes.get(name))
        return entry is not None and entry.get("ok") is True

    required_passed = sum(passed(name) for name in _SMOKE_GATE_PROBES)
    optional = tuple(name for name in _REQUIRED_PROBES if name not in _SMOKE_GATE_PROBES)
    optional_passed = sum(passed(name) for name in optional)
    lines = (
        "DINKSTER ACCELERATOR SMOKE",
        f"status: {'PASS' if not problems else 'FAIL'}",
        f"backend: {accelerator}",
        f"device: {device_fields.get('name', 'unknown')}",
        f"architecture: {device_fields.get('architecture', 'unknown')}",
        f"driver: {fields.get('driver', 'unknown')}",
        f"torch: {torch_fields.get('version', 'unknown')}",
        f"runtime: {torch_fields.get('backend_runtime', 'unknown')}",
        f"required probes: {required_passed}/{len(_SMOKE_GATE_PROBES)}",
        f"optional probes: {optional_passed}/{len(optional)}",
    )
    return "\n".join(lines)


FAMILY_REPORT_VERSION = 1

FAMILY_VALIDATION_FAMILIES = ("sd15", "sdxl", "flux", "gguf", "lora")
FAMILY_VALIDATION_MODES = ("eager", "compile")

#: Detected runtime family identities each validation family may report.
FAMILY_VALIDATION_FAMILY_IDS = MappingProxyType(
    {
        "sd15": ("dinkster.sd15",),
        "sdxl": ("dinkster.sdxl", "dinkster.sdxl_refiner"),
        "flux": ("dinkster.flux_dev", "dinkster.flux_schnell"),
        "gguf": ("dinkster.sd15", "dinkster.sdxl"),
        "lora": ("dinkster.sd15", "dinkster.sdxl"),
        "zimage": ("dinkster.z_image",),
        "wan21": ("dinkster.wan21",),
        "wan21_infinitetalk": ("dinkster.wan21",),
        "wan21_humo": ("dinkster.wan21",),
        "anima": ("dinkster.anima",),
        "minimax_h3": ("dinkster.minimax_h3",),
        "chroma": ("dinkster.chroma",),
    }
)

#: A passing unload check certifies at most this much allocator residue.
FAMILY_RESIDUAL_CEILING_BYTES = 1_048_576

#: (required, optional) artifact roles each validation family records.
_FAMILY_ARTIFACT_ROLES: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = MappingProxyType(
    {
        "sd15": (("checkpoint",), ()),
        "sdxl": (("checkpoint",), ()),
        "flux": (("checkpoint",), ()),
        "gguf": (("diffusion_gguf", "clip_l", "vae"), ("clip_g",)),
        "lora": (("checkpoint", "lora"), ()),
    }
)

_REQUIRED_FAMILY_CHECKS = (
    "load",
    "encode_text",
    "sample",
    "decode",
    "finite_output",
    "second_run_reuse",
    "unload",
)
_LORA_FAMILY_CHECKS = ("lora_apply", "lora_effect", "lora_restore")
_COMPILE_FAMILY_CHECKS = ("compile_parity",)

_REQUIRED_WORKLOAD_INTS = ("seed", "steps", "width", "height")


def required_family_checks(family: str, mode: str) -> tuple[str, ...]:
    """The check names one family-validation cell must record."""
    if family not in FAMILY_VALIDATION_FAMILIES:
        raise ValueError(f"unknown validation family {family!r}")
    if mode not in FAMILY_VALIDATION_MODES:
        raise ValueError(f"unknown validation mode {mode!r}")
    if family == "lora" and mode == "compile":
        raise ValueError("the lora family has no compile mode")
    checks = _REQUIRED_FAMILY_CHECKS
    if family == "lora":
        checks += _LORA_FAMILY_CHECKS
    if mode == "compile":
        checks += _COMPILE_FAMILY_CHECKS
    return checks


def validate_family_report(report: object, *, accelerator: str) -> tuple[str, ...]:
    """Problems that make a family-validation report unusable as evidence.

    A complete report records the backend identity a smoke report carries,
    plus the executed cell (family and execution mode), the exact workload,
    the input artifacts by digest, memory telemetry, and every required
    check outcome. The evidence must be internally consistent with the
    declared cell: the detected family_id must belong to the family, the
    artifact roles must be exactly the family's inputs, workload fields
    that only one family or mode records must not appear elsewhere, and a
    passing unload check cannot coexist with residual memory above the
    ceiling. ``all_ok`` must equal the conjunction of the recorded check
    outcomes - a report claiming success over failed checks is corrupt,
    not merely incomplete. A compile-mode report must reference a passing
    eager report for the same family and accelerator (digest plus its
    all_ok), and an eager-mode report must not carry such a reference.
    """
    if accelerator not in ("rocm", "xpu"):
        raise ValueError(f"unknown backend accelerator {accelerator!r}")
    fields = _as_mapping(report)
    if fields is None:
        return ("report is not a mapping",)
    problems: list[str] = []
    if fields.get("report_version") != FAMILY_REPORT_VERSION:
        problems.append(f"report_version is not {FAMILY_REPORT_VERSION}")
    problems.extend(_identity_problems(fields, accelerator))

    family = fields.get("family")
    if family not in FAMILY_VALIDATION_FAMILIES:
        problems.append(f"family is not one of {FAMILY_VALIDATION_FAMILIES}")
        family = None
    mode = fields.get("mode")
    if mode not in FAMILY_VALIDATION_MODES:
        problems.append(f"mode is not one of {FAMILY_VALIDATION_MODES}")
        mode = None
    if family == "lora" and mode == "compile":
        problems.append("the lora family has no compile mode")
        mode = None
    if family is None:
        if not _nonempty_str(fields.get("family_id")):
            problems.append("family_id missing or empty")
    elif fields.get("family_id") not in FAMILY_VALIDATION_FAMILY_IDS[cast("str", family)]:
        problems.append(
            f"family_id is not one of {FAMILY_VALIDATION_FAMILY_IDS[cast('str', family)]}"
        )

    workload = _as_mapping(fields.get("workload"))
    if workload is None:
        problems.append("workload section missing")
    else:
        if not _nonempty_str(workload.get("prompt")):
            problems.append("workload.prompt missing or empty")
        if not isinstance(workload.get("negative_prompt"), str):
            problems.append("workload.negative_prompt missing")
        for name in ("sampler_id", "scheduler_id"):
            if not _nonempty_str(workload.get(name)):
                problems.append(f"workload.{name} missing or empty")
        for name in _REQUIRED_WORKLOAD_INTS:
            value = workload.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f"workload.{name} missing or not an integer")
            elif name != "seed" and value <= 0:
                problems.append(f"workload.{name} not positive")
        cfg = workload.get("cfg")
        if not _finite_number(cfg):
            problems.append("workload.cfg missing or not a finite number")
        if family is not None:
            if family == "flux":
                if not _finite_number(workload.get("guidance")):
                    problems.append("workload.guidance missing or not a finite number")
            elif workload.get("guidance") is not None:
                problems.append("workload.guidance is only recorded for the flux family")
            for name in ("lora_strength_model", "lora_strength_clip"):
                if family == "lora":
                    if not _finite_number(workload.get(name)):
                        problems.append(f"workload.{name} missing or not a finite number")
                elif workload.get(name) is not None:
                    problems.append(f"workload.{name} is only recorded for the lora family")
        if mode is not None:
            tolerance = workload.get("compile_tolerance")
            if mode == "compile":
                if not _finite_number(tolerance) or cast("float", tolerance) <= 0:
                    problems.append(
                        "workload.compile_tolerance missing or not a finite positive number"
                    )
            elif tolerance is not None:
                problems.append("workload.compile_tolerance is only recorded in compile mode")

    artifacts = fields.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        problems.append("artifacts list missing or empty")
    else:
        roles = _artifact_entry_problems(cast("list[object]", artifacts), problems)
        if family is not None:
            required_roles, optional_roles = _FAMILY_ARTIFACT_ROLES[cast("str", family)]
            for role in required_roles:
                if roles.count(role) != 1:
                    problems.append(f"artifacts must record role {role!r} exactly once")
            for role in roles:
                if role not in required_roles and role not in optional_roles:
                    problems.append(f"artifacts role {role!r} is not part of a {family} cell")
                elif role in optional_roles and roles.count(role) > 1:
                    problems.append(f"artifacts role {role!r} recorded more than once")

    memory = _as_mapping(fields.get("memory"))
    if memory is None:
        problems.append("memory section missing")
    else:
        for name in ("peak_allocated_bytes", "residual_allocated_bytes"):
            value = memory.get(name)
            if not isinstance(value, int) or value < 0:
                problems.append(f"memory.{name} missing or negative")

    checks = _as_mapping(fields.get("checks"))
    if family is not None and mode is not None:
        required = required_family_checks(cast("str", family), cast("str", mode))
        problems.extend(_check_entry_problems(checks, "checks", required))
    _unload_ceiling_problems(checks, memory, problems)
    _all_ok_problems(fields, checks, problems)

    eager_reference = fields.get("eager_report")
    if mode == "compile":
        reference = _as_mapping(eager_reference)
        if reference is None:
            problems.append("compile mode requires an eager_report reference")
        else:
            if not _sha256_hex(reference.get("digest")):
                problems.append("eager_report.digest is not a sha256 hex digest")
            if reference.get("all_ok") is not True:
                problems.append("eager_report.all_ok is not true")
            if family is not None and reference.get("family") != family:
                problems.append("eager_report.family does not match this report")
            if reference.get("accelerator") != accelerator:
                problems.append("eager_report.accelerator does not match this report")
    elif eager_reference is not None:
        problems.append("eager mode must not carry an eager_report reference")
    return tuple(problems)


BENCHMARK_REPORT_VERSION = 1

#: Systems a head-to-head benchmark report can come from.
BENCHMARK_SYSTEMS = ("dinkster", "comfyui")

#: ComfyUI revision used by pinned head-to-head workloads.
BENCHMARK_COMFYUI_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

#: Placement provenance recorded by each benchmark runner.
BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT = "production_residency"
BENCHMARK_DINKSTER_DIAGNOSTIC_PLACEMENT = "direct_placement_diagnostic"
BENCHMARK_COMFYUI_PLACEMENT = "comfyui_model_management"
BENCHMARK_MINIMAX_H3_DINKSTER_EXECUTION_PATH = "generation_ksampler_multistream"
BENCHMARK_MINIMAX_H3_COMFYUI_EXECUTION_PATH = "sampler_custom_advanced"
_BENCHMARK_DINKSTER_PLACEMENTS = (
    BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT,
    BENCHMARK_DINKSTER_DIAGNOSTIC_PLACEMENT,
)

#: Pinned Anima workload values shared by both runners and the report gate.
BENCHMARK_ANIMA_PROMPT = "a photograph of an astronaut riding a horse"
BENCHMARK_PRIMARY_VARIANT = "primary"
BENCHMARK_ANIMA_FALLBACK_VARIANT = "fallback_768"
_BENCHMARK_ANIMA_VARIANTS = (
    BENCHMARK_PRIMARY_VARIANT,
    BENCHMARK_ANIMA_FALLBACK_VARIANT,
)

#: Model families the inference benchmark measures (issue #667 scope).
BENCHMARK_FAMILIES = (
    "sd15",
    "sdxl",
    "lora",
    "zimage",
    "wan21",
    "wan21_infinitetalk",
    "wan21_humo",
    "anima",
    "minimax_h3",
    "flux",
    "chroma",
)

#: Artifact roles each benchmark family records, all required exactly once.
_BENCHMARK_ARTIFACT_ROLES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "sd15": ("checkpoint",),
        "sdxl": ("checkpoint",),
        "lora": ("checkpoint", "lora"),
        "zimage": ("diffusion", "text_encoder", "vae"),
        "wan21": ("diffusion", "text_encoder", "vae"),
        "wan21_infinitetalk": (
            "diffusion",
            "text_encoder",
            "vae",
            "lora",
            "model_patch",
            "audio_encoder",
            "clip_vision",
            "input_image",
            "input_audio_1",
            "input_audio_2",
        ),
        "wan21_humo": (
            "diffusion",
            "text_encoder",
            "vae",
            "lora",
            "audio_encoder",
            "input_image",
            "input_audio",
        ),
        "anima": ("diffusion", "text_encoder", "vae"),
        "minimax_h3": ("diffusion", "text_encoder", "video_vae", "audio_vae"),
        "flux": ("diffusion", "clip_l", "text_encoder", "vae"),
        "chroma": ("diffusion", "text_encoder", "vae"),
    }
)

#: Pinned artifacts whose content is part of the workload definition.
_BENCHMARK_PINNED_DIGESTS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "wan21_infinitetalk": MappingProxyType(
            {
                "input_image": "88a9d7bd3832304a5b66626c442886f0b82ddbce176089e504b8aeaf4cc3333e",
                "input_audio_1": "d008494976e34b05108f181942a6d4363e2bf1176ebabc10ecb69d2e61245afb",
                "input_audio_2": "632aecb453a9a58d37f9f9e70d07f6748ab604af59a564b84eb76031440d3545",
            }
        ),
        "wan21_humo": MappingProxyType(
            {
                "diffusion": "222ddeac4dea6b78363cb5be78c47660c92963a69386026cd6dc0de4d3094f66",
                "text_encoder": "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
                "vae": "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
                "lora": "85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3",
                "audio_encoder": "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
                "input_image": "3a6662eba09c10b72d763cb947ca38e717998bafc55d7d0c14f72a1410ee1eb0",
                "input_audio": "4e920892d3d33ebb8a04d772960a027f185fa55213ce0c60cfd0ec3faf191e8f",
            }
        ),
        "anima": MappingProxyType(
            {
                "diffusion": "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e",
                "text_encoder": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
                "vae": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
            }
        ),
        "minimax_h3": MappingProxyType(
            {
                "diffusion": "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5",
                "text_encoder": "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
                "video_vae": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
                "audio_vae": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
            }
        ),
        "flux": MappingProxyType(
            {
                "diffusion": "4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7",
                "clip_l": "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
                "text_encoder": "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635",
                "vae": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
            }
        ),
        "chroma": MappingProxyType(
            {
                "diffusion": "a2928ca6075f308f4d5e2182e2b96120fa8ad270ec6ea9b1b5c724c85c49a575",
                "text_encoder": "a498f0485dc9536735258018417c3fd7758dc3bccc0a645feaa472b34955557a",
                "vae": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
            }
        ),
    }
)

#: Video families whose workload carries a frame count.
_BENCHMARK_VIDEO_FAMILIES = frozenset({"wan21", "wan21_infinitetalk", "wan21_humo", "minimax_h3"})

#: Video families that prepare audio conditioning before sampling.
_BENCHMARK_AUDIO_FAMILIES = frozenset({"wan21_infinitetalk", "wan21_humo"})

#: Families measured eager-only; the compile column covers checkpoint
#: families.
_BENCHMARK_EAGER_ONLY_FAMILIES = frozenset(
    {"zimage", "wan21", "wan21_infinitetalk", "wan21_humo", "anima", "minimax_h3", "flux", "chroma"}
)

#: Backends a benchmark may run on; cuda exists for development hosts,
#: the head-to-head lanes are rocm and xpu.
BENCHMARK_ACCELERATORS = ("rocm", "xpu", "cuda")

#: Offload mechanisms a Dinkster benchmark process can run under
#: (the DINKSTER_AIMDO_ARM assembly modes).
BENCHMARK_RESIDENCY_MECHANISMS = ("off", "auto", "on")

#: VRAM regimes a mechanism-comparison cell runs in: the device as-is,
#: or constrained behind a ballast allocation.
BENCHMARK_RESIDENCY_REGIMES = ("open", "constrained")

#: Resolved GPU shared-usage counter scopes a report can record
#: (process on Windows, machine on WSL, off elsewhere).
BENCHMARK_RESIDENCY_SPILL_SCOPES = ("process", "machine", "off")

_MIB = MEBIBYTE

_REQUIRED_BENCHMARK_CHECKS = (
    "load",
    "encode_text",
    "cold_run",
    "finite_output",
    "warm_runs",
    "unload",
)
_BENCHMARK_COLD_PHASES = ("load_s", "encode_s", "sample_s", "decode_s", "total_s")
_BENCHMARK_WARM_FIELDS = ("sample_s", "decode_s", "total_s")
_BENCHMARK_WARM_MEDIANS = ("median_sample_s", "median_decode_s", "median_total_s")


def required_benchmark_checks(family: str) -> tuple[str, ...]:
    """The check names one benchmark cell must record."""
    if family not in BENCHMARK_FAMILIES:
        raise ValueError(f"unknown benchmark family {family!r}")
    checks = _REQUIRED_BENCHMARK_CHECKS
    if family == "lora":
        checks = checks[:1] + ("lora_apply",) + checks[1:]
    if family in _BENCHMARK_AUDIO_FAMILIES:
        checks = checks[:2] + ("encode_audio",) + checks[2:]
    return checks


def _step_wall_problems(value: object, where: str, steps: object, problems: list[str]) -> None:
    """step_wall_ms is optional (dispatch-side boundaries are informational),
    but when present it must carry one finite positive entry per step."""
    if value is None:
        return
    if not isinstance(value, list):
        problems.append(f"{where}.step_wall_ms is not a list")
        return
    entries = cast("list[object]", value)
    if isinstance(steps, int) and not isinstance(steps, bool) and len(entries) != steps:
        problems.append(f"{where}.step_wall_ms does not record one entry per step")
    for position, entry in enumerate(entries):
        if not _finite_number(entry) or cast("float", entry) <= 0:
            problems.append(f"{where}.step_wall_ms[{position}] is not a finite positive number")


def _phase_problems(
    section: Mapping[str, object], names: tuple[str, ...], where: str, problems: list[str]
) -> None:
    for name in names:
        value = section.get(name)
        if not _finite_number(value) or cast("float", value) <= 0:
            problems.append(f"{where}.{name} missing or not a finite positive number")


def _byte_count(value: object) -> int | None:
    """The value as an integer byte count, or None when it is not one."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _residency_problems(
    fields: Mapping[str, object],
    system: object,
    accelerator: str,
    family: object,
    problems: list[str],
    *,
    canonical_evidence: bool,
) -> None:
    """The optional residency section is dinkster-only mechanism-comparison
    evidence: the offload mechanism the process ran under, the VRAM
    regime, and best-effort GPU shared-usage observations. Shared usage
    alone cannot distinguish staging from driver spill, including when
    it stays constant. Reports without the section remain readable, but
    cannot serve as canonical generic or H3 production evidence."""
    expected_routes: set[str] = set()
    if system == "dinkster" and fields.get("placement") == BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT:
        if family == "minimax_h3":
            expected_routes = {"diffusion", "conditioner", "video_vae", "audio_vae"}
        elif canonical_evidence and family in ("sd15", "sdxl", "lora", "zimage", "wan21", "flux"):
            expected_routes = {"runtime"}
    if "residency" not in fields:
        if canonical_evidence and expected_routes:
            problems.append("canonical production evidence requires residency route facts")
        return
    section = _as_mapping(fields.get("residency"))
    if section is None:
        problems.append("residency is not a mapping")
        return
    if system != "dinkster":
        problems.append("residency is only recorded by a dinkster report")
    requested = section.get("mechanism")
    if requested not in BENCHMARK_RESIDENCY_MECHANISMS:
        problems.append(f"residency.mechanism is not one of {BENCHMARK_RESIDENCY_MECHANISMS}")
        requested = None
    bootstrap = section.get("aimdo_bootstrap_succeeded")
    if bootstrap is not None and not isinstance(bootstrap, bool):
        problems.append("residency.aimdo_bootstrap_succeeded is not a boolean or null")
    route_fields = _as_mapping(section.get("routes", {}))
    if route_fields is None:
        problems.append("residency.routes is not a mapping")
        routes: Mapping[str, object] = {}
    else:
        routes = route_fields
    if expected_routes and set(routes) != expected_routes:
        problems.append(
            "residency.routes must record exactly " + ", ".join(sorted(expected_routes))
        )
    required_actual = (
        "aimdo" if requested == "on" or (requested == "auto" and accelerator == "cuda") else None
    )
    for role, value in routes.items():
        route = _as_mapping(value)
        where = f"residency.routes.{role}"
        if route is None:
            problems.append(f"{where} is not a mapping")
            continue
        if route.get("requested") != requested:
            problems.append(f"{where}.requested does not match residency.mechanism")
        actual = route.get("mechanism")
        if actual not in ("aimdo", "eager"):
            problems.append(f"{where}.mechanism is not aimdo or eager")
        fallback_reason = route.get("fallback_reason")
        if "fallback_reason" not in route or (
            fallback_reason is not None and not _nonempty_str(fallback_reason)
        ):
            problems.append(f"{where}.fallback_reason is not a non-empty string or null")
        component_fields: dict[str, tuple[str, ...]] = {}
        for name in ("dynamic_components", "resident_components", "fallback_components"):
            components = route.get(name)
            if not isinstance(components, list) or any(
                not _nonempty_str(item) for item in cast("list[object]", components)
            ):
                problems.append(f"{where}.{name} is not a list of non-empty strings")
                component_fields[name] = ()
            else:
                component_fields[name] = tuple(cast("list[str]", components))
        if canonical_evidence:
            if requested == "off" and (actual != "eager" or component_fields["dynamic_components"]):
                problems.append(f"{where} records dynamic residency with the off selector")
            if (
                requested in ("auto", "on")
                and actual == "eager"
                and not _nonempty_str(fallback_reason)
            ):
                problems.append(f"{where} has no reason for eager fallback")
        if (family == "minimax_h3" or canonical_evidence) and required_actual is not None:
            if actual != required_actual:
                problems.append(f"{where}.mechanism is not required {required_actual}")
            if fallback_reason is not None or component_fields["fallback_components"]:
                problems.append(f"{where} fell back from required {required_actual} residency")
            if not component_fields["dynamic_components"]:
                problems.append(f"{where} did not enroll a dynamic component")
    if (
        (family == "minimax_h3" or (canonical_evidence and expected_routes))
        and required_actual == "aimdo"
        and bootstrap is not True
    ):
        problems.append(
            "residency.aimdo_bootstrap_succeeded is not true for required Aimdo residency"
        )
    regime = section.get("regime")
    if regime not in BENCHMARK_RESIDENCY_REGIMES:
        problems.append(f"residency.regime is not one of {BENCHMARK_RESIDENCY_REGIMES}")
        regime = None
    leave_free = section.get("leave_free_mib")
    ballast = section.get("ballast_bytes")
    if regime == "constrained":
        leave_free_value = _byte_count(leave_free)
        if leave_free_value is None or leave_free_value <= 0:
            problems.append("residency.leave_free_mib missing or not positive when constrained")
        ballast_value = _byte_count(ballast)
        if ballast_value is None or ballast_value < 0:
            problems.append("residency.ballast_bytes missing or negative when constrained")
    elif regime == "open":
        if leave_free is not None:
            problems.append("residency.leave_free_mib is only recorded when constrained")
        if ballast is not None:
            problems.append("residency.ballast_bytes is only recorded when constrained")
    scope = section.get("spill_scope")
    if scope not in BENCHMARK_RESIDENCY_SPILL_SCOPES:
        problems.append(f"residency.spill_scope is not one of {BENCHMARK_RESIDENCY_SPILL_SCOPES}")
        scope = None
    samples: dict[str, int | None] = {}
    for name in ("shared_before_bytes", "shared_warm_bytes", "shared_after_bytes"):
        value = section.get(name)
        sample = _byte_count(value)
        if value is not None and (sample is None or sample < 0):
            problems.append(f"residency.{name} is not a non-negative integer byte count")
            sample = None
        samples[name] = sample
        if scope == "off" and value is not None:
            problems.append(f"residency.{name} is recorded with spill_scope off")
    growth_value = section.get("shared_growth_bytes")
    growth = _byte_count(growth_value)
    if growth_value is not None and growth is None:
        problems.append("residency.shared_growth_bytes is not an integer byte count")
    warm = samples["shared_warm_bytes"]
    after = samples["shared_after_bytes"]
    if warm is not None and after is not None:
        if growth != after - warm:
            problems.append(
                "residency.shared_growth_bytes is not shared_after_bytes - shared_warm_bytes"
            )
    elif growth_value is not None:
        problems.append("residency.shared_growth_bytes requires warm and after samples")
    detected = section.get("shared_spill_detected")
    if detected is not None and not isinstance(detected, bool):
        problems.append("residency.shared_spill_detected is not a boolean or null")
    if canonical_evidence and ("shared_spill_detected" not in section or detected is not None):
        problems.append(
            "residency.shared_spill_detected must be null: shared-usage samples cannot assess spill"
        )


def validate_benchmark_report(
    report: object,
    *,
    accelerator: str,
    canonical_evidence: bool = False,
    expected_comfyui_commit: str | None = None,
) -> tuple[str, ...]:
    """Problems that make an inference benchmark report unusable as evidence.

    A complete report records the backend identity the smoke and family
    reports carry, the measuring system (dinkster or comfyui), the exact
    workload, the input artifacts by digest, synchronized cold and warm
    phase timings, memory telemetry, and every required check outcome.
    Cold covers the first image in a fresh process including model load;
    warm covers repeated sample+decode in the same process, so the warm
    run count must match the workload's declared count. Compile mode is
    a dinkster-only informational column - the head-to-head comparison is
    eager against eager - so a comfyui report must be eager. ``all_ok``
    must equal the conjunction of the recorded check outcomes. Canonical
    evidence excludes Dinkster's explicit direct-placement diagnostic and
    requires route facts for generic and H3 production handles. CUDA auto
    and explicit on must record successful bootstrap, Aimdo enrollment,
    and no component fallback, not merely a production placement label.
    """
    if accelerator not in BENCHMARK_ACCELERATORS:
        raise ValueError(f"unknown benchmark accelerator {accelerator!r}")
    fields = _as_mapping(report)
    if fields is None:
        return ("report is not a mapping",)
    problems: list[str] = []
    if fields.get("report_version") != BENCHMARK_REPORT_VERSION:
        problems.append(f"report_version is not {BENCHMARK_REPORT_VERSION}")
    problems.extend(_identity_problems(fields, accelerator))

    system = fields.get("system")
    if system not in BENCHMARK_SYSTEMS:
        problems.append(f"system is not one of {BENCHMARK_SYSTEMS}")
        system = None
    family = fields.get("family")
    if family not in BENCHMARK_FAMILIES:
        problems.append(f"family is not one of {BENCHMARK_FAMILIES}")
        family = None
    mode = fields.get("mode")
    if mode not in FAMILY_VALIDATION_MODES:
        problems.append(f"mode is not one of {FAMILY_VALIDATION_MODES}")
        mode = None
    if system == "comfyui" and mode == "compile":
        problems.append("a comfyui report must be eager; compile is dinkster-only")
    if mode == "compile" and family in _BENCHMARK_EAGER_ONLY_FAMILIES:
        problems.append(f"{family} reports must be eager")
    if family == "anima" and fields.get("variant") not in _BENCHMARK_ANIMA_VARIANTS:
        problems.append(f"variant is not one of {_BENCHMARK_ANIMA_VARIANTS} for an Anima report")
    if system == "dinkster":
        placement = fields.get("placement")
        if placement not in _BENCHMARK_DINKSTER_PLACEMENTS:
            problems.append(
                f"placement is not one of {_BENCHMARK_DINKSTER_PLACEMENTS} for a Dinkster report"
            )
        elif canonical_evidence and placement != BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT:
            problems.append(
                "canonical evidence requires Dinkster placement "
                f"{BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT!r}"
            )
        elif family in (
            "wan21_infinitetalk",
            "wan21_humo",
            "anima",
            "minimax_h3",
            "flux",
            "chroma",
        ) and (placement != BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT):
            problems.append(
                f"placement is not {BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT!r} "
                f"for a Dinkster {family} report"
            )
        if family is None:
            if not _nonempty_str(fields.get("family_id")):
                problems.append("family_id missing or empty")
        elif fields.get("family_id") not in FAMILY_VALIDATION_FAMILY_IDS[cast("str", family)]:
            problems.append(
                f"family_id is not one of {FAMILY_VALIDATION_FAMILY_IDS[cast('str', family)]}"
            )
    elif system == "comfyui":
        if fields.get("placement") != BENCHMARK_COMFYUI_PLACEMENT:
            problems.append(
                f"placement is not {BENCHMARK_COMFYUI_PLACEMENT!r} for a ComfyUI report"
            )
        if family in ("wan21_humo", "minimax_h3", "flux", "chroma"):
            family_label = {
                "wan21_humo": "HuMo",
                "minimax_h3": "MiniMax H3",
                "flux": "Flux",
                "chroma": "Chroma",
            }[cast("str", family)]
            comfyui = _as_mapping(fields.get("comfyui"))
            if comfyui is None:
                problems.append(f"comfyui section missing for a ComfyUI {family_label} report")
            expected_commit = expected_comfyui_commit or BENCHMARK_COMFYUI_COMMIT
            if comfyui is not None and comfyui.get("commit") != expected_commit:
                requirement = "required" if expected_comfyui_commit is not None else "pinned"
                problems.append(
                    f"comfyui.commit is not the {requirement} {family_label} commit "
                    f"{expected_commit}"
                )
        if family == "anima":
            comfyui = _as_mapping(fields.get("comfyui"))
            if comfyui is None:
                problems.append("comfyui section missing for a ComfyUI anima report")
            elif comfyui.get("commit") != BENCHMARK_COMFYUI_COMMIT:
                problems.append(
                    f"comfyui.commit is not the pinned anima commit {BENCHMARK_COMFYUI_COMMIT}"
                )
        if not _nonempty_str(fields.get("family_id")):
            problems.append("family_id missing or empty")
    if family == "minimax_h3" and system in BENCHMARK_SYSTEMS:
        expected_execution_path = (
            BENCHMARK_MINIMAX_H3_DINKSTER_EXECUTION_PATH
            if system == "dinkster"
            else BENCHMARK_MINIMAX_H3_COMFYUI_EXECUTION_PATH
        )
        if fields.get("execution_path") != expected_execution_path:
            problems.append(
                f"execution_path is not the pinned MiniMax H3 {system} path "
                f"{expected_execution_path!r}"
            )

    steps: object = None
    warm_count: object = None
    workload = _as_mapping(fields.get("workload"))
    if workload is None:
        problems.append("workload section missing")
    else:
        if not _nonempty_str(workload.get("prompt")):
            problems.append("workload.prompt missing or empty")
        if not isinstance(workload.get("negative_prompt"), str):
            problems.append("workload.negative_prompt missing")
        for name in ("sampler_id", "scheduler_id"):
            if not _nonempty_str(workload.get(name)):
                problems.append(f"workload.{name} missing or empty")
        for name in (*_REQUIRED_WORKLOAD_INTS, "warm_runs"):
            value = workload.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f"workload.{name} missing or not an integer")
            elif name != "seed" and value <= 0:
                problems.append(f"workload.{name} not positive")
        steps = workload.get("steps")
        warm_count = workload.get("warm_runs")
        if not _finite_number(workload.get("cfg")):
            problems.append("workload.cfg missing or not a finite number")
        for name in ("lora_strength_model", "lora_strength_clip"):
            if family == "lora":
                if not _finite_number(workload.get(name)):
                    problems.append(f"workload.{name} missing or not a finite number")
            elif family == "wan21_infinitetalk" and name == "lora_strength_model":
                if not _finite_number(workload.get(name)):
                    problems.append(f"workload.{name} missing or not a finite number")
            elif family == "wan21_humo" and name == "lora_strength_model":
                if not _finite_number(workload.get(name)):
                    problems.append(f"workload.{name} missing or not a finite number")
            elif family is not None and workload.get(name) is not None:
                problems.append(f"workload.{name} is only recorded for the lora family")
        if family == "flux":
            if not _finite_number(workload.get("guidance")):
                problems.append("workload.guidance missing or not a finite number")
        elif family is not None and workload.get("guidance") is not None:
            problems.append("workload.guidance is only recorded for the flux family")
        if family == "wan21_infinitetalk":
            motion_frame_count = workload.get("motion_frame_count")
            if (
                not isinstance(motion_frame_count, int)
                or isinstance(motion_frame_count, bool)
                or motion_frame_count <= 0
            ):
                problems.append("workload.motion_frame_count missing or not a positive integer")
            audio_scale = workload.get("audio_scale")
            if not _finite_number(audio_scale) or cast("float", audio_scale) <= 0:
                problems.append("workload.audio_scale missing or not a finite positive number")
            if workload.get("speaker_mask_layout") != "left_right_half":
                problems.append("workload.speaker_mask_layout is not 'left_right_half'")
            expected_workload: dict[str, object] = {
                "prompt": "The camera zooms in. Two characters are talking.",
                "negative_prompt": "",
                "sampler_id": "dinkster.euler" if system == "dinkster" else "euler",
                "scheduler_id": "dinkster.normal" if system == "dinkster" else "normal",
                "seed": 0,
                "steps": 6,
                "width": 832,
                "height": 480,
                "length": 81,
                "cfg": 1.0,
                "warm_runs": 3,
                "lora_strength_model": 1.0,
                "motion_frame_count": 9,
                "audio_scale": 1.0,
                "speaker_mask_layout": "left_right_half",
            }
            for name, expected in expected_workload.items():
                if workload.get(name) != expected:
                    problems.append(f"workload.{name} is not the pinned InfiniteTalk value")
        if family == "wan21_humo":
            expected_workload = {
                "prompt": (
                    "A young boy in sci-fi style clothing is talking to the camera"
                    " in an alien desert."
                ),
                "negative_prompt": (
                    "\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce\u683c\uff0c\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c\u9759\u6b62\uff0c\u6574\u4f53\u53d1\u7070\uff0c\u6700\u5dee\u8d28\u91cf\uff0c\u4f4e\u8d28\u91cf\uff0c"
                    "JPEG\u538b\u7f29\u6b8b\u7559\uff0c\u4e11\u964b\u7684\uff0c\u6b8b\u7f3a\u7684\uff0c\u591a\u4f59\u7684\u624b\u6307\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8\uff0c\u7578\u5f62\u7684\uff0c\u6bc1\u5bb9\u7684\uff0c"
                    "\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53\uff0c\u624b\u6307\u878d\u5408\uff0c\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762\uff0c\u6742\u4e71\u7684\u80cc\u666f\uff0c\u4e09\u6761\u817f\uff0c\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012\u7740\u8d70"
                ),
                "sampler_id": "dinkster.uni_pc" if system == "dinkster" else "uni_pc",
                "scheduler_id": "dinkster.simple" if system == "dinkster" else "simple",
                "seed": 0,
                "steps": 6,
                "width": 640,
                "height": 640,
                "length": 97,
                "cfg": 1.0,
                "warm_runs": 3,
                "lora_strength_model": 1.0,
            }
            for name, expected in expected_workload.items():
                if workload.get(name) != expected:
                    problems.append(f"workload.{name} is not the pinned HuMo value")
        if family == "anima":
            expected_workload = {
                "prompt": BENCHMARK_ANIMA_PROMPT,
                "negative_prompt": "",
                "sampler_id": "dinkster.er_sde" if system == "dinkster" else "er_sde",
                "scheduler_id": "dinkster.simple" if system == "dinkster" else "simple",
                "seed": 875817230929465,
                "steps": 30,
                "cfg": 4.0,
                "warm_runs": 5,
            }
            for name, expected in expected_workload.items():
                if workload.get(name) != expected:
                    problems.append(f"workload.{name} is not the pinned Anima value")
            geometry = (workload.get("width"), workload.get("height"))
            variant = fields.get("variant")
            if variant == BENCHMARK_PRIMARY_VARIANT:
                expected_geometry = (1024, 1024)
            elif variant == BENCHMARK_ANIMA_FALLBACK_VARIANT:
                expected_geometry = (768, 768)
            else:
                expected_geometry = None
            if expected_geometry is not None and geometry != expected_geometry:
                problems.append(
                    f"workload Anima geometry {geometry!r} does not match"
                    f" variant {fields.get('variant')!r}"
                )
        if family == "minimax_h3":
            expected_workload = {
                "prompt": "A red square centered on a black background.",
                "negative_prompt": "",
                "sampler_id": "dinkster.res_multistep" if system == "dinkster" else "res_multistep",
                "scheduler_id": "dinkster.simple" if system == "dinkster" else "simple",
                "seed": 20260813,
                "steps": 20,
                "width": 1344,
                "height": 768,
                "length": 124,
                "cfg": 1.0,
                "warm_runs": 3,
            }
            for name, expected in expected_workload.items():
                if workload.get(name) != expected:
                    problems.append(f"workload.{name} is not the pinned MiniMax H3 value")
        if family == "flux":
            expected_workload = {
                "prompt": "a photograph of an astronaut riding a horse",
                "negative_prompt": "",
                "sampler_id": "dinkster.euler" if system == "dinkster" else "euler",
                "scheduler_id": "dinkster.simple" if system == "dinkster" else "simple",
                "seed": 667,
                "steps": 20,
                "width": 1024,
                "height": 1024,
                "cfg": 1.0,
                "warm_runs": 5,
                "guidance": 3.5,
            }
            for name, expected in expected_workload.items():
                if workload.get(name) != expected:
                    problems.append(f"workload.{name} is not the pinned Flux value")
        if family == "chroma":
            expected_workload = {
                "prompt": "a photograph of an astronaut riding a horse",
                "negative_prompt": "",
                "sampler_id": "dinkster.euler" if system == "dinkster" else "euler",
                "scheduler_id": "dinkster.beta" if system == "dinkster" else "beta",
                "seed": 667,
                "steps": 26,
                "width": 1024,
                "height": 1024,
                "cfg": 3.5,
                "warm_runs": 5,
            }
            for name, expected in expected_workload.items():
                if workload.get(name) != expected:
                    problems.append(f"workload.{name} is not the pinned Chroma value")
        length = workload.get("length")
        if family in _BENCHMARK_VIDEO_FAMILIES:
            if not isinstance(length, int) or isinstance(length, bool):
                problems.append("workload.length missing or not an integer")
            elif length <= 0:
                problems.append("workload.length not positive")
            elif family == "minimax_h3" and (length < 5 or (length - 5) % 17 != 0):
                problems.append("workload.length not a 17k+5 frame count")
            elif family != "minimax_h3" and (length - 1) % 4 != 0:
                # The causal video VAE decodes 4*T - 3 frames from T
                # latent frames; any other length silently decodes to a
                # different frame count than the workload records.
                problems.append("workload.length not a 4k+1 frame count")
        elif family is not None and length is not None:
            problems.append("workload.length is only recorded for video families")

    artifacts = fields.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        problems.append("artifacts list missing or empty")
    else:
        roles = _artifact_entry_problems(cast("list[object]", artifacts), problems)
        if family is not None:
            required_roles = _BENCHMARK_ARTIFACT_ROLES[cast("str", family)]
            for role in required_roles:
                if roles.count(role) != 1:
                    problems.append(f"artifacts must record role {role!r} exactly once")
            for role in roles:
                if role not in required_roles:
                    problems.append(f"artifacts role {role!r} is not part of a {family} cell")
            for role, expected_digest in _BENCHMARK_PINNED_DIGESTS.get(
                cast("str", family), {}
            ).items():
                matching = [
                    artifact
                    for entry in cast("list[object]", artifacts)
                    if (artifact := _as_mapping(entry)) is not None and artifact.get("role") == role
                ]
                if len(matching) == 1 and matching[0].get("sha256") != expected_digest:
                    problems.append(f"artifacts role {role!r} does not match its pinned digest")

    timings = _as_mapping(fields.get("timings"))
    if timings is None:
        problems.append("timings section missing")
    else:
        import_s = timings.get("import_s")
        if import_s is not None and (not _finite_number(import_s) or cast("float", import_s) <= 0):
            problems.append("timings.import_s is not a finite positive number")
        cold = _as_mapping(timings.get("cold"))
        if cold is None:
            problems.append("timings.cold section missing")
        else:
            cold_phases = _BENCHMARK_COLD_PHASES
            if family in _BENCHMARK_AUDIO_FAMILIES:
                cold_phases += ("audio_encode_s",)
            _phase_problems(cold, cold_phases, "timings.cold", problems)
            lora_s = cold.get("lora_s")
            if family == "lora":
                if not _finite_number(lora_s) or cast("float", lora_s) <= 0:
                    problems.append("timings.cold.lora_s missing or not a finite positive number")
            elif family is not None and lora_s is not None:
                problems.append("timings.cold.lora_s is only recorded for the lora family")
            _step_wall_problems(cold.get("step_wall_ms"), "timings.cold", steps, problems)
        warm = _as_mapping(timings.get("warm"))
        if warm is None:
            problems.append("timings.warm section missing")
        else:
            runs = warm.get("runs")
            if not isinstance(runs, list) or not runs:
                problems.append("timings.warm.runs missing or empty")
            else:
                entries = cast("list[object]", runs)
                if (
                    isinstance(warm_count, int)
                    and not isinstance(warm_count, bool)
                    and len(entries) != warm_count
                ):
                    problems.append(
                        "timings.warm.runs does not match the workload's warm_runs count"
                    )
                for position, run_entry in enumerate(entries):
                    run = _as_mapping(run_entry)
                    where = f"timings.warm.runs[{position}]"
                    if run is None:
                        problems.append(f"{where} is not a mapping")
                        continue
                    _phase_problems(run, _BENCHMARK_WARM_FIELDS, where, problems)
                    _step_wall_problems(run.get("step_wall_ms"), where, steps, problems)
            _phase_problems(warm, _BENCHMARK_WARM_MEDIANS, "timings.warm", problems)

    memory = _as_mapping(fields.get("memory"))
    if memory is None:
        problems.append("memory section missing")
    else:
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            value = memory.get(name)
            if not isinstance(value, int) or value <= 0:
                problems.append(f"memory.{name} missing or not positive")
        residual = memory.get("residual_allocated_bytes")
        if not isinstance(residual, int) or residual < 0:
            problems.append("memory.residual_allocated_bytes missing or negative")
        rss = memory.get("peak_rss_bytes")
        if not isinstance(rss, int) or rss <= 0:
            problems.append("memory.peak_rss_bytes is not a positive integer")

    _residency_problems(
        fields, system, accelerator, family, problems, canonical_evidence=canonical_evidence
    )

    checks = _as_mapping(fields.get("checks"))
    if family is not None:
        required = required_benchmark_checks(cast("str", family))
        problems.extend(_check_entry_problems(checks, "checks", required))
    _unload_ceiling_problems(checks, memory, problems)
    _all_ok_problems(fields, checks, problems)
    return tuple(problems)


__all__ = [
    "BACKEND_ENV_RECIPES",
    "BENCHMARK_ACCELERATORS",
    "BENCHMARK_ANIMA_FALLBACK_VARIANT",
    "BENCHMARK_ANIMA_PROMPT",
    "BENCHMARK_COMFYUI_COMMIT",
    "BENCHMARK_COMFYUI_PLACEMENT",
    "BENCHMARK_DINKSTER_DIAGNOSTIC_PLACEMENT",
    "BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT",
    "BENCHMARK_FAMILIES",
    "BENCHMARK_PRIMARY_VARIANT",
    "BENCHMARK_REPORT_VERSION",
    "BENCHMARK_RESIDENCY_MECHANISMS",
    "BENCHMARK_RESIDENCY_REGIMES",
    "BENCHMARK_RESIDENCY_SPILL_SCOPES",
    "BENCHMARK_SYSTEMS",
    "FAMILY_REPORT_VERSION",
    "FAMILY_RESIDUAL_CEILING_BYTES",
    "FAMILY_VALIDATION_FAMILIES",
    "FAMILY_VALIDATION_FAMILY_IDS",
    "FAMILY_VALIDATION_MODES",
    "SMOKE_REPORT_VERSION",
    "BackendEnvRecipe",
    "required_benchmark_checks",
    "required_family_checks",
    "format_smoke_receipt",
    "setup_commands",
    "smoke_gate_problems",
    "validate_benchmark_report",
    "validate_family_report",
    "validate_smoke_report",
    "venv_python",
]
