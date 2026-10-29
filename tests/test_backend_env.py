"""Backend environment recipes and smoke-report validation.

CPU-side proof for the ROCm/XPU environment groundwork: the recipes pin the
ruled backend floors, the setup scripts embed the same pins, command
construction is exact, and smoke-report validation demands every identity
field a support cell requires. None of this claims hardware support.
"""

from __future__ import annotations

import json
import shlex
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_inference import MINIMAX_H3
from dinkster_workers.backend_env import (
    BACKEND_ENV_RECIPES,
    BENCHMARK_ANIMA_FALLBACK_VARIANT,
    BENCHMARK_ANIMA_PROMPT,
    BENCHMARK_FAMILIES,
    BENCHMARK_PRIMARY_VARIANT,
    BENCHMARK_REPORT_VERSION,
    BENCHMARK_SYSTEMS,
    FAMILY_REPORT_VERSION,
    FAMILY_VALIDATION_FAMILIES,
    FAMILY_VALIDATION_MODES,
    SMOKE_REPORT_VERSION,
    format_smoke_receipt,
    required_benchmark_checks,
    required_family_checks,
    setup_commands,
    smoke_gate_problems,
    validate_benchmark_report,
    validate_family_report,
    validate_smoke_report,
    venv_python,
)
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).parents[1]

SETUP_SCRIPTS = {
    "windows-rocm": "scripts/setup_env_rocm.ps1",
    "linux-rocm": "scripts/setup_env_rocm.sh",
    "windows-xpu": "scripts/setup_env_xpu.ps1",
    "linux-xpu": "scripts/setup_env_xpu.sh",
}

HUMO_COMFYUI_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
HUMO_ARTIFACT_DIGESTS = {
    "diffusion": "222ddeac4dea6b78363cb5be78c47660c92963a69386026cd6dc0de4d3094f66",
    "text_encoder": "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
    "vae": "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
    "lora": "85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3",
    "audio_encoder": "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
    "input_image": "3a6662eba09c10b72d763cb947ca38e717998bafc55d7d0c14f72a1410ee1eb0",
    "input_audio": "4e920892d3d33ebb8a04d772960a027f185fa55213ce0c60cfd0ec3faf191e8f",
}

H3_RESIDENCY_REQUIREMENTS = {
    "residency_route_roles": MINIMAX_H3.engine.residency_route_roles,
    "requires_accelerator_residency": MINIMAX_H3.engine.requires_accelerator_residency,
}
ANIMA_ARTIFACT_DIGESTS = {
    "diffusion": "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e",
    "text_encoder": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
    "vae": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
}

MINIMAX_H3_ARTIFACT_DIGESTS = {
    "diffusion": "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5",
    "text_encoder": "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
    "video_vae": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
    "audio_vae": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
}
FLUX_ARTIFACT_DIGESTS = {
    "diffusion": "4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7",
    "clip_l": "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
    "text_encoder": "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635",
    "vae": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
}

CHROMA_ARTIFACT_DIGESTS = {
    "diffusion": "a2928ca6075f308f4d5e2182e2b96120fa8ad270ec6ea9b1b5c724c85c49a575",
    "text_encoder": "a498f0485dc9536735258018417c3fd7758dc3bccc0a645feaa472b34955557a",
    "vae": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
}


def complete_report(accelerator: str) -> dict[str, Any]:
    """A structurally complete smoke report, as the smoke scripts emit."""
    runtime = "hip 7.14.0" if accelerator == "rocm" else "xpu 2.13"
    probes = {
        name: {"ok": True, "detail": "ok"}
        for name in (
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
    }
    if accelerator == "rocm":
        torch_version = "2.12.0+rocm7.14.0"
        device_name = "AMD Radeon PRO W7800"
        architecture = "gfx1100"
        driver = "AMD Radeon PRO W7800 driver 32.0.31021.5001"
        attention_route: dict[str, Any] = {
            "device_kind": "rocm",
            "device_sm": 110,
            "providers": [["hip", "7.14.0"], ["torch", torch_version]],
        }
        dtype_policy: dict[str, Any] = {
            "fp16": {"storage": True, "compute": True},
            "bf16": {"storage": True, "compute": True},
            "fp8_native_matmul": False,
            "lora_patch_dtype": "float16",
        }
    else:
        torch_version = "2.13.0+xpu"
        device_name = "Intel(R) Arc(TM) B570 Graphics"
        architecture = "Xe2"
        driver = "Intel Arc driver 32.0.101.8860"
        attention_route = {
            "device_kind": "xpu",
            "device_sm": None,
            "providers": [["torch", torch_version], ["xpu", "20260500"]],
        }
        dtype_policy = {
            "fp16": {"storage": True, "compute": True},
            "bf16": {"storage": True, "compute": True},
            "fp8_native_matmul": False,
            "lora_patch_dtype": "float32",
        }
    return {
        "report_version": SMOKE_REPORT_VERSION,
        "accelerator": accelerator,
        "host": {
            "platform": "Windows-11-10.0.26200-SP0",
            "os_version": "10.0.26200",
            "machine": "AMD64",
            "python": "3.12.8",
        },
        "driver": driver,
        "torch": {"version": torch_version, "backend_runtime": runtime},
        "devices": [
            {
                "index": 0,
                "name": device_name,
                "architecture": architecture,
                "total_memory": 34342961152,
            }
        ],
        "memory": {"mem_get_info_free": 3, "mem_get_info_total": 4},
        "attention_route": attention_route,
        "dtype_policy": dtype_policy,
        "quant_dequant": {
            "gguf": {"Q8_0": {"bit_exact": True, "max_abs_diff": 0.0}},
            "int8": {"bit_exact": True, "max_abs_diff": 0.0},
            "fp8_e4m3fn": {"bit_exact": True, "max_abs_diff": 0.0},
            "kitchen": None,
        },
        "probes": probes,
        "baseline_ok": True,
    }


class TestRecipes:
    def test_all_four_support_cells_exist(self) -> None:
        assert set(BACKEND_ENV_RECIPES) == {
            "windows-rocm",
            "linux-rocm",
            "windows-xpu",
            "linux-xpu",
        }
        for cell, recipe in BACKEND_ENV_RECIPES.items():
            assert recipe.cell == cell
            assert cell == f"{recipe.os_family}-{recipe.accelerator}"

    def test_rocm_cells_pin_the_ruled_floor(self) -> None:
        for cell in ("windows-rocm", "linux-rocm"):
            recipe = BACKEND_ENV_RECIPES[cell]
            assert recipe.torch_requirement == "torch[device-all]==2.12.0+rocm7.14.0"
            assert recipe.index_url == "https://repo.amd.com/rocm/whl-multi-arch/"
            assert recipe.venv == ".venv-rocm"

    def test_xpu_cells_pin_the_ruled_floor(self) -> None:
        for cell in ("windows-xpu", "linux-xpu"):
            recipe = BACKEND_ENV_RECIPES[cell]
            assert recipe.torch_requirement == "torch==2.13.0+xpu"
            assert recipe.index_url == "https://download.pytorch.org/whl/xpu"
            assert recipe.venv == ".venv-xpu"

    def test_cells_pin_the_pure_python_kitchen_wheel(self) -> None:
        # PyPI's platform wheels for win_amd64 and linux x86_64 are CUDA
        # builds; ROCm and XPU cells need the pure-Python eager backend,
        # hash-pinned so uv verifies the download.
        for recipe in BACKEND_ENV_RECIPES.values():
            kitchen = [
                package
                for package in recipe.support_packages
                if package.startswith("dinkster-kitchen@")
            ]
            assert kitchen == [
                "dinkster-kitchen@https://files.pythonhosted.org/packages/2e/20/"
                "84e29ca1dedcd51eb5edd297d3c2f6c665cf2e30bb9237892f0f8d108d0d/"
                "dinkster_kitchen-0.2.35.post1-py3-none-any.whl"
                "#sha256=31458547cdcf9ff26974a4955cf79e83ebdf50077666720d3bb3255786c5fc4f"
            ]

    def test_backends_never_share_a_venv(self) -> None:
        venvs = {recipe.accelerator: recipe.venv for recipe in BACKEND_ENV_RECIPES.values()}
        assert venvs["rocm"] != venvs["xpu"]

    def test_venv_python_uses_the_cell_path_style(self) -> None:
        assert venv_python(BACKEND_ENV_RECIPES["windows-rocm"]) == ".venv-rocm\\Scripts\\python.exe"
        assert venv_python(BACKEND_ENV_RECIPES["linux-xpu"]) == ".venv-xpu/bin/python"


class TestSetupCommands:
    def test_torch_installs_alone_from_the_backend_index(self) -> None:
        for recipe in BACKEND_ENV_RECIPES.values():
            create, torch_install, packages, benchmark_import = setup_commands(recipe)
            assert create == (
                "uv",
                "venv",
                recipe.venv,
                "--clear",
                "--python",
                recipe.python_version,
            )
            assert torch_install == (
                "uv",
                "pip",
                "install",
                "--python",
                venv_python(recipe),
                "--index-url",
                recipe.index_url,
                recipe.torch_requirement,
            )
            assert recipe.index_url not in packages
            assert benchmark_import == (
                venv_python(recipe),
                "-c",
                "import dinkster_compat_comfy.native_arm",
            )

    def test_workspace_packages_install_editable(self) -> None:
        recipe = BACKEND_ENV_RECIPES["windows-xpu"]
        packages = setup_commands(recipe)[2]
        for support in recipe.support_packages:
            assert support in packages
        for editable in recipe.editable_packages:
            position = packages.index(editable)
            assert packages[position - 1] == "-e"

    def test_benchmark_import_dependencies_are_installed(self) -> None:
        for recipe in BACKEND_ENV_RECIPES.values():
            assert "packages/dinkster-graph" in recipe.editable_packages
            assert "packages/dinkster-compat-comfy" in recipe.editable_packages

    def test_workspace_dependency_closure_is_installed(self) -> None:
        projects: dict[str, tuple[str, tuple[str, ...]]] = {}
        for pyproject_path in (REPO_ROOT / "packages").glob("*/pyproject.toml"):
            project = cast("dict[str, Any]", tomllib.loads(pyproject_path.read_text())["project"])
            projects[cast("str", project["name"])] = (
                pyproject_path.parent.relative_to(REPO_ROOT).as_posix(),
                tuple(cast("list[str]", project.get("dependencies", []))),
            )

        for recipe in BACKEND_ENV_RECIPES.values():
            installed = set(recipe.editable_packages)
            for package_path in recipe.editable_packages:
                project_name = next(
                    name for name, (path, _dependencies) in projects.items() if path == package_path
                )
                for requirement in projects[project_name][1]:
                    dependency = projects.get(Requirement(requirement).name)
                    if dependency is not None:
                        assert dependency[0] in installed, (
                            f"{package_path} requires workspace package {dependency[0]}"
                        )


def executable_commands(script: Path) -> list[list[str]]:
    """Every command a setup script executes, as unquoted token lists."""
    text = script.read_text()
    # Join PowerShell backtick and POSIX backslash line continuations.
    text = text.replace("`\n", " ").replace("\\\n", " ")
    commands: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line in ('$ErrorActionPreference = "Stop"', "set -euo pipefail"):
            continue
        if line.startswith("if ($LASTEXITCODE"):
            continue
        commands.append([token.strip('"') for token in shlex.split(line, posix=False)])
    return commands


class TestSetupScriptSync:
    @pytest.mark.parametrize("cell", sorted(SETUP_SCRIPTS))
    def test_script_commands_match_the_recipe_exactly(self, cell: str) -> None:
        # Full-command equality: the scripts run the recipe's setup commands
        # and then the backend smoke, nothing more, nothing reordered.
        recipe = BACKEND_ENV_RECIPES[cell]
        separator = "\\" if recipe.os_family == "windows" else "/"
        smoke = [
            venv_python(recipe),
            f"scripts{separator}{recipe.accelerator}_smoke.py",
            "--json",
            f"{recipe.accelerator}-report.json",
        ]
        expected = [list(command) for command in setup_commands(recipe)] + [smoke]
        assert executable_commands(REPO_ROOT / SETUP_SCRIPTS[cell]) == expected


class TestSmokeReportValidation:
    @pytest.mark.parametrize("accelerator", ["rocm", "xpu"])
    def test_complete_report_is_accepted(self, accelerator: str) -> None:
        report = complete_report(accelerator)
        assert validate_smoke_report(report, accelerator=accelerator) == ()
        # The JSON roundtrip the smoke scripts perform must not change that.
        roundtrip = json.loads(json.dumps(report))
        assert validate_smoke_report(roundtrip, accelerator=accelerator) == ()

    def test_unknown_accelerator_is_a_caller_error(self) -> None:
        with pytest.raises(ValueError, match="cuda"):
            validate_smoke_report(complete_report("rocm"), accelerator="cuda")

    def test_non_mapping_report_is_rejected(self) -> None:
        assert validate_smoke_report("not a report", accelerator="rocm") == (
            "report is not a mapping",
        )

    def test_backend_crossover_is_rejected(self) -> None:
        problems = validate_smoke_report(complete_report("rocm"), accelerator="xpu")
        assert any("accelerator" in problem for problem in problems)

    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (lambda r: r.pop("report_version"), "report_version"),
            (lambda r: r.pop("host"), "host section"),
            (lambda r: r["host"].pop("os_version"), "host.os_version"),
            (lambda r: r["host"].update(python=""), "host.python"),
            (lambda r: r.update(driver="  "), "driver identity"),
            (
                lambda r: r.update(driver="unknown (Win32_VideoController query failed)"),
                "driver identity is an unknown placeholder",
            ),
            (lambda r: r.pop("torch"), "torch section"),
            (lambda r: r["torch"].pop("backend_runtime"), "torch.backend_runtime"),
            (lambda r: r.update(devices=[]), "devices list"),
            (lambda r: r["devices"][0].pop("architecture"), "devices[0].architecture"),
            (lambda r: r["devices"][0].update(total_memory=0), "devices[0].total_memory"),
            (lambda r: r["probes"].pop("sdpa_float16"), "probes.sdpa_float16"),
            (lambda r: r["probes"].pop("attention_route"), "probes.attention_route"),
            (lambda r: r["probes"]["mem_get_info"].pop("ok"), "probes.mem_get_info.ok"),
            (
                lambda r: r["probes"]["matmul_float32"].update(detail=""),
                "probes.matmul_float32.detail",
            ),
            (lambda r: r.pop("baseline_ok"), "baseline_ok"),
        ],
    )
    def test_missing_evidence_is_named(self, mutate: Any, expected: str) -> None:
        report = complete_report("rocm")
        mutate(report)
        problems = validate_smoke_report(report, accelerator="rocm")
        assert any(expected in problem for problem in problems), problems

    def test_failed_probes_do_not_invalidate_the_report(self) -> None:
        # A probe failure is evidence, not a malformed report: refusals must
        # be recordable so unsupported cells stay explicit.
        report = complete_report("xpu")
        report["probes"]["sdpa_float16"] = {"ok": False, "detail": "unsupported"}
        report["baseline_ok"] = False
        assert validate_smoke_report(report, accelerator="xpu") == ()

    @pytest.mark.parametrize("accelerator", ["rocm", "xpu"])
    def test_complete_passing_report_produces_pass_receipt(self, accelerator: str) -> None:
        report = complete_report(accelerator)
        assert smoke_gate_problems(report, accelerator=accelerator) == ()
        receipt = format_smoke_receipt(report, accelerator=accelerator)
        assert receipt.startswith("DINKSTER ACCELERATOR SMOKE\nstatus: PASS\n")
        assert f"backend: {accelerator}" in receipt
        assert "required probes: 13/13" in receipt
        assert "optional probes: 6/6" in receipt

    @pytest.mark.parametrize("failed_probe", ["mem_get_info", "attention_route", "quant_dequant"])
    def test_shipped_capability_failure_produces_fail_receipt(self, failed_probe: str) -> None:
        report = complete_report("rocm")
        report["probes"][failed_probe] = {"ok": False, "detail": "not available"}
        assert smoke_gate_problems(report, accelerator="rocm") == (
            f"probes.{failed_probe} did not pass",
        )
        assert "status: FAIL" in format_smoke_receipt(report, accelerator="rocm")

    def test_optional_quant_storage_failure_remains_evidence_only(self) -> None:
        report = complete_report("xpu")
        report["probes"]["storage_float8_e5m2"] = {"ok": False, "detail": "not available"}
        assert smoke_gate_problems(report, accelerator="xpu") == ()
        receipt = format_smoke_receipt(report, accelerator="xpu")
        assert "status: PASS" in receipt
        assert "optional probes: 5/6" in receipt

    def test_false_baseline_claim_produces_fail_receipt(self) -> None:
        report = complete_report("xpu")
        report["baseline_ok"] = False
        assert smoke_gate_problems(report, accelerator="xpu") == ("baseline_ok is not true",)
        assert "status: FAIL" in format_smoke_receipt(report, accelerator="xpu")


EAGER_DIGEST = "a" * 64


def complete_family_report(
    accelerator: str, family: str = "sd15", mode: str = "eager"
) -> dict[str, Any]:
    """A structurally complete family report, as family_validation.py emits."""
    smoke = complete_report(accelerator)
    family_ids = {
        "sd15": "dinkster.sd15",
        "sdxl": "dinkster.sdxl",
        "flux": "dinkster.flux_dev",
        "gguf": "dinkster.sd15",
        "lora": "dinkster.sd15",
    }
    report: dict[str, Any] = {
        "report_version": FAMILY_REPORT_VERSION,
        "accelerator": accelerator,
        "host": smoke["host"],
        "driver": smoke["driver"],
        "torch": smoke["torch"],
        "devices": smoke["devices"],
        "family": family,
        "mode": mode,
        "family_id": family_ids[family],
        "workload": {
            "prompt": "a photograph of an astronaut riding a horse",
            "negative_prompt": "",
            "sampler_id": "dinkster.euler",
            "scheduler_id": "dinkster.simple",
            "seed": 591,
            "steps": 8,
            "width": 512,
            "height": 512,
            "cfg": 7.0,
            "guidance": 3.5 if family == "flux" else None,
            "lora_strength_model": 1.0 if family == "lora" else None,
            "lora_strength_clip": 1.0 if family == "lora" else None,
            "compile_tolerance": 5e-3 if mode == "compile" else None,
        },
        "artifacts": [
            {
                "role": role,
                "path": f"/models/{role}.safetensors",
                "sha256": "b" * 64,
                "bytes": 4_265_146_304,
            }
            for role in {
                "gguf": ("diffusion_gguf", "clip_l", "clip_g", "vae"),
                "lora": ("checkpoint", "lora"),
            }.get(family, ("checkpoint",))
        ],
        "memory": {"peak_allocated_bytes": 6_442_450_944, "residual_allocated_bytes": 0},
        "checks": {
            name: {"ok": True, "detail": "ok"} for name in required_family_checks(family, mode)
        },
        "all_ok": True,
    }
    if mode == "compile":
        report["eager_report"] = {
            "digest": EAGER_DIGEST,
            "all_ok": True,
            "family": family,
            "accelerator": accelerator,
        }
    return report


class TestRequiredFamilyChecks:
    def test_every_cell_requires_the_base_checks(self) -> None:
        base = {
            "load",
            "encode_text",
            "sample",
            "decode",
            "finite_output",
            "second_run_reuse",
            "unload",
        }
        for family in FAMILY_VALIDATION_FAMILIES:
            for mode in FAMILY_VALIDATION_MODES:
                if family == "lora" and mode == "compile":
                    continue
                assert base <= set(required_family_checks(family, mode))

    def test_lora_adds_patch_lifecycle_checks(self) -> None:
        checks = set(required_family_checks("lora", "eager"))
        assert {"lora_apply", "lora_effect", "lora_restore"} <= checks
        assert "lora_apply" not in required_family_checks("sd15", "eager")

    def test_compile_adds_parity(self) -> None:
        assert "compile_parity" in required_family_checks("flux", "compile")
        assert "compile_parity" not in required_family_checks("flux", "eager")

    def test_unknown_cell_is_a_caller_error(self) -> None:
        with pytest.raises(ValueError, match="wan"):
            required_family_checks("wan", "eager")
        with pytest.raises(ValueError, match="jit"):
            required_family_checks("sd15", "jit")

    def test_lora_compile_is_not_a_cell(self) -> None:
        with pytest.raises(ValueError, match="no compile mode"):
            required_family_checks("lora", "compile")


class TestFamilyReportValidation:
    @pytest.mark.parametrize("accelerator", ["rocm", "xpu"])
    @pytest.mark.parametrize(
        ("family", "mode"),
        [
            (family, mode)
            for family in FAMILY_VALIDATION_FAMILIES
            for mode in FAMILY_VALIDATION_MODES
            if not (family == "lora" and mode == "compile")
        ],
    )
    def test_complete_report_is_accepted(self, accelerator: str, family: str, mode: str) -> None:
        report = complete_family_report(accelerator, family, mode)
        assert validate_family_report(report, accelerator=accelerator) == ()
        roundtrip = json.loads(json.dumps(report))
        assert validate_family_report(roundtrip, accelerator=accelerator) == ()

    def test_unknown_accelerator_is_a_caller_error(self) -> None:
        with pytest.raises(ValueError, match="cuda"):
            validate_family_report(complete_family_report("rocm"), accelerator="cuda")

    def test_non_mapping_report_is_rejected(self) -> None:
        assert validate_family_report(None, accelerator="xpu") == ("report is not a mapping",)

    def test_backend_crossover_is_rejected(self) -> None:
        problems = validate_family_report(complete_family_report("rocm"), accelerator="xpu")
        assert any("accelerator" in problem for problem in problems)

    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (lambda r: r.pop("report_version"), "report_version"),
            (lambda r: r["host"].pop("os_version"), "host.os_version"),
            (lambda r: r.update(driver="unknown"), "driver identity"),
            (lambda r: r["torch"].pop("backend_runtime"), "torch.backend_runtime"),
            (lambda r: r.update(devices=[]), "devices list"),
            (lambda r: r.update(family="wan"), "family is not one of"),
            (lambda r: r.update(mode="jit"), "mode is not one of"),
            (lambda r: r.pop("family_id"), "family_id"),
            (lambda r: r.update(family_id="dinkster.sdxl"), "family_id is not one of"),
            (lambda r: r.pop("workload"), "workload section"),
            (lambda r: r["workload"].update(prompt=""), "workload.prompt"),
            (lambda r: r["workload"].pop("negative_prompt"), "workload.negative_prompt"),
            (lambda r: r["workload"].update(sampler_id=" "), "workload.sampler_id"),
            (lambda r: r["workload"].update(seed="591"), "workload.seed"),
            (lambda r: r["workload"].update(steps=0), "workload.steps not positive"),
            (lambda r: r["workload"].update(width=True), "workload.width"),
            (lambda r: r["workload"].update(cfg="7"), "workload.cfg"),
            (lambda r: r["workload"].update(cfg=float("inf")), "workload.cfg"),
            (lambda r: r["workload"].update(guidance=3.5), "workload.guidance"),
            (
                lambda r: r["workload"].update(lora_strength_model=1.0),
                "workload.lora_strength_model",
            ),
            (
                lambda r: r["workload"].update(compile_tolerance=5e-3),
                "workload.compile_tolerance",
            ),
            (lambda r: r.update(artifacts=[]), "artifacts list"),
            (lambda r: r["artifacts"][0].pop("role"), "artifacts[0].role"),
            (lambda r: r["artifacts"][0].update(sha256="zz"), "artifacts[0].sha256"),
            (lambda r: r["artifacts"][0].update(bytes=0), "artifacts[0].bytes"),
            (lambda r: r["artifacts"][0].update(role="lora"), "not part of a sd15 cell"),
            (
                lambda r: r["artifacts"].append(dict(r["artifacts"][0])),
                "role 'checkpoint' exactly once",
            ),
            (lambda r: r.pop("memory"), "memory section"),
            (
                lambda r: r["memory"].update(peak_allocated_bytes=-1),
                "memory.peak_allocated_bytes",
            ),
            (
                lambda r: r["memory"].pop("residual_allocated_bytes"),
                "memory.residual_allocated_bytes",
            ),
            (lambda r: r.pop("checks"), "checks section"),
            (lambda r: r["checks"].pop("second_run_reuse"), "checks.second_run_reuse"),
            (lambda r: r["checks"]["unload"].pop("ok"), "checks.unload.ok"),
            (lambda r: r["checks"]["sample"].update(detail=""), "checks.sample.detail"),
            (lambda r: r.pop("all_ok"), "all_ok"),
        ],
    )
    def test_missing_evidence_is_named(self, mutate: Any, expected: str) -> None:
        report = complete_family_report("rocm")
        mutate(report)
        problems = validate_family_report(report, accelerator="rocm")
        assert any(expected in problem for problem in problems), problems

    def test_lora_report_requires_the_patch_checks(self) -> None:
        report = complete_family_report("xpu", family="lora")
        del report["checks"]["lora_restore"]
        problems = validate_family_report(report, accelerator="xpu")
        assert any("checks.lora_restore" in problem for problem in problems)

    def test_failed_checks_are_recordable_but_must_match_all_ok(self) -> None:
        report = complete_family_report("rocm")
        report["checks"]["finite_output"] = {"ok": False, "detail": "nan in output"}
        report["all_ok"] = False
        assert validate_family_report(report, accelerator="rocm") == ()

    def test_all_ok_over_failed_checks_is_corrupt(self) -> None:
        report = complete_family_report("rocm")
        report["checks"]["finite_output"] = {"ok": False, "detail": "nan in output"}
        problems = validate_family_report(report, accelerator="rocm")
        assert any("all_ok is inconsistent" in problem for problem in problems)

    def test_false_all_ok_over_passing_checks_is_corrupt(self) -> None:
        report = complete_family_report("xpu")
        report["all_ok"] = False
        problems = validate_family_report(report, accelerator="xpu")
        assert any("all_ok is inconsistent" in problem for problem in problems)

    def test_compile_requires_an_eager_reference(self) -> None:
        report = complete_family_report("rocm", mode="compile")
        del report["eager_report"]
        problems = validate_family_report(report, accelerator="rocm")
        assert any("compile mode requires an eager_report" in problem for problem in problems)

    def test_compile_reference_must_be_a_passing_digest(self) -> None:
        report = complete_family_report("rocm", mode="compile")
        report["eager_report"] = {"digest": "not-hex", "all_ok": True}
        problems = validate_family_report(report, accelerator="rocm")
        assert any("eager_report.digest" in problem for problem in problems)
        report["eager_report"] = {"digest": EAGER_DIGEST, "all_ok": False}
        problems = validate_family_report(report, accelerator="rocm")
        assert any("eager_report.all_ok" in problem for problem in problems)

    def test_eager_mode_refuses_an_eager_reference(self) -> None:
        report = complete_family_report("xpu")
        report["eager_report"] = {"digest": EAGER_DIGEST, "all_ok": True}
        problems = validate_family_report(report, accelerator="xpu")
        assert any("eager mode must not carry" in problem for problem in problems)

    def test_lora_compile_report_is_rejected(self) -> None:
        report = complete_family_report("rocm", family="lora")
        report["mode"] = "compile"
        problems = validate_family_report(report, accelerator="rocm")
        assert any("no compile mode" in problem for problem in problems)

    def test_flux_requires_guidance(self) -> None:
        report = complete_family_report("rocm", family="flux")
        report["workload"]["guidance"] = None
        problems = validate_family_report(report, accelerator="rocm")
        assert any("workload.guidance" in problem for problem in problems)

    def test_lora_requires_strengths(self) -> None:
        report = complete_family_report("xpu", family="lora")
        report["workload"]["lora_strength_clip"] = None
        problems = validate_family_report(report, accelerator="xpu")
        assert any("workload.lora_strength_clip" in problem for problem in problems)

    @pytest.mark.parametrize("tolerance", [None, 0.0, -1e-3, float("nan"), float("inf")])
    def test_compile_tolerance_must_be_finite_positive(self, tolerance: Any) -> None:
        report = complete_family_report("rocm", mode="compile")
        report["workload"]["compile_tolerance"] = tolerance
        problems = validate_family_report(report, accelerator="rocm")
        assert any("workload.compile_tolerance" in problem for problem in problems)

    def test_gguf_artifact_roles_are_bound(self) -> None:
        report = complete_family_report("xpu", family="gguf")
        report["artifacts"] = [entry for entry in report["artifacts"] if entry["role"] != "vae"]
        problems = validate_family_report(report, accelerator="xpu")
        assert any("role 'vae' exactly once" in problem for problem in problems)

    def test_optional_artifact_role_recorded_once(self) -> None:
        report = complete_family_report("xpu", family="gguf")
        clip_g = next(entry for entry in report["artifacts"] if entry["role"] == "clip_g")
        report["artifacts"].append(dict(clip_g))
        problems = validate_family_report(report, accelerator="xpu")
        assert any("'clip_g' recorded more than once" in problem for problem in problems)

    def test_passing_unload_cannot_leave_residual_above_ceiling(self) -> None:
        report = complete_family_report("rocm")
        report["memory"]["residual_allocated_bytes"] = 2_097_152
        problems = validate_family_report(report, accelerator="rocm")
        assert any("exceeds the unload ceiling" in problem for problem in problems)

    def test_failed_unload_may_record_residual_above_ceiling(self) -> None:
        report = complete_family_report("rocm")
        report["memory"]["residual_allocated_bytes"] = 2_097_152
        report["checks"]["unload"] = {"ok": False, "detail": "allocator still holds 2097152 B"}
        report["all_ok"] = False
        assert validate_family_report(report, accelerator="rocm") == ()

    def test_compile_reference_must_name_the_same_cell(self) -> None:
        report = complete_family_report("rocm", mode="compile")
        report["eager_report"]["family"] = "sdxl"
        problems = validate_family_report(report, accelerator="rocm")
        assert any("eager_report.family" in problem for problem in problems)
        report = complete_family_report("rocm", mode="compile")
        report["eager_report"]["accelerator"] = "xpu"
        problems = validate_family_report(report, accelerator="rocm")
        assert any("eager_report.accelerator" in problem for problem in problems)


def complete_benchmark_report(
    accelerator: str, family: str = "sd15", system: str = "dinkster", mode: str = "eager"
) -> dict[str, Any]:
    """A structurally complete benchmark report, as benchmark_inference.py
    emits."""
    smoke = complete_report(accelerator)
    family_ids = {
        "sd15": "dinkster.sd15",
        "sdxl": "dinkster.sdxl",
        "lora": "dinkster.sdxl",
        "zimage": "dinkster.z_image",
        "wan21": "dinkster.wan21",
        "wan21_infinitetalk": "dinkster.wan21",
        "wan21_humo": "dinkster.wan21",
        "anima": "dinkster.anima",
        "minimax_h3": "dinkster.minimax_h3",
        "flux": "dinkster.flux_dev",
        "chroma": "dinkster.chroma",
    }
    artifact_roles = {
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
    infinitetalk = family == "wan21_infinitetalk"
    humo = family == "wan21_humo"
    anima = family == "anima"
    minimax_h3 = family == "minimax_h3"
    flux = family == "flux"
    chroma = family == "chroma"
    audio_family = infinitetalk or humo
    steps = 6 if audio_family else (30 if anima else (26 if chroma else 20))
    warm_runs = 3 if audio_family or minimax_h3 else 5
    input_digests = {
        "input_image": (
            "3a6662eba09c10b72d763cb947ca38e717998bafc55d7d0c14f72a1410ee1eb0"
            if humo
            else "88a9d7bd3832304a5b66626c442886f0b82ddbce176089e504b8aeaf4cc3333e"
        ),
        "input_audio": "4e920892d3d33ebb8a04d772960a027f185fa55213ce0c60cfd0ec3faf191e8f",
        "input_audio_1": "d008494976e34b05108f181942a6d4363e2bf1176ebabc10ecb69d2e61245afb",
        "input_audio_2": "632aecb453a9a58d37f9f9e70d07f6748ab604af59a564b84eb76031440d3545",
    }
    if humo:
        input_digests.update(HUMO_ARTIFACT_DIGESTS)
    if anima:
        input_digests.update(ANIMA_ARTIFACT_DIGESTS)
    if minimax_h3:
        input_digests.update(MINIMAX_H3_ARTIFACT_DIGESTS)
    if flux:
        input_digests.update(FLUX_ARTIFACT_DIGESTS)
    if chroma:
        input_digests.update(CHROMA_ARTIFACT_DIGESTS)
    warm_entry = {
        "sample_s": 4.1,
        "decode_s": 0.6,
        "total_s": 4.7,
        "step_wall_ms": [205.0] * steps,
    }
    report: dict[str, Any] = {
        "report_version": BENCHMARK_REPORT_VERSION,
        "system": system,
        "accelerator": accelerator,
        "host": smoke["host"],
        "driver": smoke["driver"],
        "torch": smoke["torch"],
        "devices": smoke["devices"],
        "family": family,
        "mode": mode,
        "placement": (
            "production_residency" if system == "dinkster" else "comfyui_model_management"
        ),
        **({"variant": BENCHMARK_PRIMARY_VARIANT} if anima else {}),
        **(
            {
                "execution_path": (
                    "generation_ksampler_multistream"
                    if system == "dinkster"
                    else "sampler_custom_advanced"
                )
            }
            if minimax_h3
            else {}
        ),
        **(
            {"comfyui": {"version": "0.3.75", "commit": HUMO_COMFYUI_COMMIT}}
            if system == "comfyui"
            and family in ("wan21_humo", "anima", "minimax_h3", "flux", "chroma")
            else {}
        ),
        "family_id": family_ids[family] if system == "dinkster" else "sd_xl_base_1.0",
        "workload": {
            "prompt": (
                "A red square centered on a black background."
                if minimax_h3
                else (
                    "A young boy in sci-fi style clothing is talking to the camera "
                    "in an alien desert."
                    if humo
                    else (
                        "The camera zooms in. Two characters are talking."
                        if infinitetalk
                        else (
                            BENCHMARK_ANIMA_PROMPT
                            if anima
                            else "a photograph of an astronaut riding a horse"
                        )
                    )
                )
            ),
            "negative_prompt": (
                "\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce\u683c\uff0c\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c\u9759\u6b62\uff0c\u6574\u4f53\u53d1\u7070\uff0c\u6700\u5dee\u8d28\u91cf\uff0c\u4f4e\u8d28\u91cf\uff0c"
                "JPEG\u538b\u7f29\u6b8b\u7559\uff0c\u4e11\u964b\u7684\uff0c\u6b8b\u7f3a\u7684\uff0c\u591a\u4f59\u7684\u624b\u6307\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8\uff0c\u7578\u5f62\u7684\uff0c\u6bc1\u5bb9\u7684\uff0c"
                "\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53\uff0c\u624b\u6307\u878d\u5408\uff0c\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762\uff0c\u6742\u4e71\u7684\u80cc\u666f\uff0c\u4e09\u6761\u817f\uff0c\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012\u7740\u8d70"
                if humo
                else ""
            ),
            "sampler_id": (
                ("res_multistep" if system == "comfyui" else "dinkster.res_multistep")
                if minimax_h3
                else (
                    ("uni_pc" if system == "comfyui" else "dinkster.uni_pc")
                    if humo
                    else (
                        ("er_sde" if system == "comfyui" else "dinkster.er_sde")
                        if anima
                        else (
                            "euler"
                            if (infinitetalk or flux or chroma) and system == "comfyui"
                            else "dinkster.euler"
                        )
                    )
                )
            ),
            "scheduler_id": (
                ("beta" if system == "comfyui" else "dinkster.beta")
                if chroma
                else (
                    ("simple" if system == "comfyui" else "dinkster.simple")
                    if humo or anima or minimax_h3 or flux
                    else (
                        "normal"
                        if infinitetalk and system == "comfyui"
                        else ("dinkster.normal" if infinitetalk else "dinkster.simple")
                    )
                )
            ),
            "seed": (
                20260813
                if minimax_h3
                else (0 if audio_family else (875817230929465 if anima else 667))
            ),
            "steps": steps,
            "width": 1344 if minimax_h3 else (640 if humo else (832 if infinitetalk else 1024)),
            "height": 768 if minimax_h3 else (640 if humo else (480 if infinitetalk else 1024)),
            "length": (
                124
                if minimax_h3
                else (97 if humo else (81 if infinitetalk else (33 if family == "wan21" else None)))
            ),
            "cfg": (
                1.0
                if audio_family or minimax_h3 or flux
                else (4.0 if anima else (3.5 if chroma else 7.0))
            ),
            "warm_runs": warm_runs,
            "guidance": 3.5 if flux else None,
            "lora_strength_model": 1.0
            if family in ("lora", "wan21_infinitetalk", "wan21_humo")
            else None,
            "lora_strength_clip": 1.0 if family == "lora" else None,
            "motion_frame_count": 9 if infinitetalk else None,
            "audio_scale": 1.0 if infinitetalk else None,
            "speaker_mask_layout": "left_right_half" if infinitetalk else None,
        },
        "artifacts": [
            {
                "role": role,
                "path": f"/models/{role}.safetensors",
                "sha256": input_digests.get(role, "b" * 64),
                "bytes": 6_938_078_334,
            }
            for role in artifact_roles[family]
        ],
        "timings": {
            "import_s": 1.8,
            "cold": {
                "load_s": 12.5,
                "encode_s": 0.4,
                "sample_s": 5.2,
                "decode_s": 0.7,
                "total_s": 18.8,
                "step_wall_ms": [260.0] * steps,
                **({"lora_s": 2.1} if family == "lora" else {}),
                **({"audio_encode_s": 1.7} if audio_family else {}),
            },
            "warm": {
                "runs": [dict(warm_entry) for _ in range(warm_runs)],
                "median_sample_s": 4.1,
                "median_decode_s": 0.6,
                "median_total_s": 4.7,
            },
        },
        "memory": {
            "peak_allocated_bytes": 7_812_667_392,
            "peak_reserved_bytes": 8_589_934_592,
            "residual_allocated_bytes": 0,
            "peak_rss_bytes": 17_179_869_184,
        },
        "checks": {
            name: {"ok": True, "detail": "ok"} for name in required_benchmark_checks(family)
        },
        "all_ok": True,
    }
    if system == "dinkster" and family in (
        "sd15",
        "sdxl",
        "lora",
        "zimage",
        "wan21",
        "flux",
        "minimax_h3",
    ):
        report["residency"] = open_residency_section(
            ("diffusion", "conditioner", "video_vae", "audio_vae") if minimax_h3 else ("runtime",)
        )
    return report


class TestRequiredBenchmarkChecks:
    def test_every_cell_requires_the_base_checks(self) -> None:
        base = {"load", "encode_text", "cold_run", "finite_output", "warm_runs", "unload"}
        for family in BENCHMARK_FAMILIES:
            assert base <= set(required_benchmark_checks(family))

    def test_lora_adds_the_patch_check(self) -> None:
        assert "lora_apply" in required_benchmark_checks("lora")
        assert "lora_apply" not in required_benchmark_checks("sdxl")

    def test_humo_adds_the_audio_encode_check(self) -> None:
        assert "encode_audio" in required_benchmark_checks("wan21_humo")
        assert "encode_audio" not in required_benchmark_checks("wan21")

    def test_unknown_family_is_a_caller_error(self) -> None:
        with pytest.raises(ValueError, match="sd3"):
            required_benchmark_checks("sd3")


class TestBenchmarkReportValidation:
    @pytest.mark.parametrize("accelerator", ["rocm", "xpu", "cuda"])
    @pytest.mark.parametrize("family", BENCHMARK_FAMILIES)
    @pytest.mark.parametrize("system", BENCHMARK_SYSTEMS)
    def test_complete_report_is_accepted(self, accelerator: str, family: str, system: str) -> None:
        report = complete_benchmark_report(accelerator, family, system)
        requirements = H3_RESIDENCY_REQUIREMENTS if family == "minimax_h3" else {}
        assert validate_benchmark_report(report, accelerator=accelerator, **requirements) == ()
        roundtrip = json.loads(json.dumps(report))
        assert validate_benchmark_report(roundtrip, accelerator=accelerator, **requirements) == ()

    def test_dinkster_compile_report_is_accepted(self) -> None:
        report = complete_benchmark_report("cuda", mode="compile")
        assert validate_benchmark_report(report, accelerator="cuda") == ()

    def test_unknown_accelerator_is_a_caller_error(self) -> None:
        with pytest.raises(ValueError, match="mps"):
            validate_benchmark_report(complete_benchmark_report("rocm"), accelerator="mps")

    def test_non_mapping_report_is_rejected(self) -> None:
        assert validate_benchmark_report(None, accelerator="xpu") == ("report is not a mapping",)

    def test_backend_crossover_is_rejected(self) -> None:
        problems = validate_benchmark_report(complete_benchmark_report("rocm"), accelerator="xpu")
        assert any("accelerator" in problem for problem in problems)

    def test_comfyui_compile_is_rejected(self) -> None:
        report = complete_benchmark_report("rocm", system="comfyui", mode="compile")
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any("compile is dinkster-only" in problem for problem in problems)

    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (lambda r: r.pop("report_version"), "report_version"),
            (lambda r: r.update(system="onnx"), "system is not one of"),
            (lambda r: r.update(driver="unknown"), "driver identity"),
            (lambda r: r.update(family="sd3"), "family is not one of"),
            (lambda r: r.update(mode="jit"), "mode is not one of"),
            (lambda r: r.pop("placement"), "placement is not one of"),
            (lambda r: r.update(family_id="dinkster.sdxl"), "family_id is not one of"),
            (lambda r: r.pop("workload"), "workload section"),
            (lambda r: r["workload"].pop("warm_runs"), "workload.warm_runs"),
            (lambda r: r["workload"].update(warm_runs=0), "workload.warm_runs not positive"),
            (lambda r: r["workload"].update(cfg=float("nan")), "workload.cfg"),
            (
                lambda r: r["workload"].update(lora_strength_model=1.0),
                "workload.lora_strength_model",
            ),
            (lambda r: r.update(artifacts=[]), "artifacts list"),
            (lambda r: r["artifacts"][0].update(sha256="zz"), "artifacts[0].sha256"),
            (lambda r: r["artifacts"][0].update(role="lora"), "not part of a sd15 cell"),
            (lambda r: r.pop("timings"), "timings section"),
            (lambda r: r["timings"].update(import_s=0.0), "timings.import_s"),
            (lambda r: r["timings"].pop("cold"), "timings.cold section"),
            (lambda r: r["timings"]["cold"].pop("load_s"), "timings.cold.load_s"),
            (lambda r: r["timings"]["cold"].update(sample_s=0.0), "timings.cold.sample_s"),
            (lambda r: r["timings"]["cold"].update(lora_s=2.1), "timings.cold.lora_s"),
            (
                lambda r: r["timings"]["cold"].update(step_wall_ms=[260.0]),
                "one entry per step",
            ),
            (
                lambda r: r["timings"]["cold"]["step_wall_ms"].__setitem__(0, 0.0),
                "step_wall_ms[0]",
            ),
            (lambda r: r["timings"].pop("warm"), "timings.warm section"),
            (lambda r: r["timings"]["warm"].update(runs=[]), "timings.warm.runs missing"),
            (
                lambda r: r["timings"]["warm"]["runs"].pop(),
                "does not match the workload's warm_runs count",
            ),
            (
                lambda r: r["timings"]["warm"]["runs"][0].pop("sample_s"),
                "timings.warm.runs[0].sample_s",
            ),
            (
                lambda r: r["timings"]["warm"].pop("median_total_s"),
                "timings.warm.median_total_s",
            ),
            (lambda r: r.pop("memory"), "memory section"),
            (
                lambda r: r["memory"].update(peak_reserved_bytes=0),
                "memory.peak_reserved_bytes",
            ),
            (
                lambda r: r["memory"].update(peak_rss_bytes=-1),
                "memory.peak_rss_bytes",
            ),
            (lambda r: r["checks"].pop("warm_runs"), "checks.warm_runs"),
            (lambda r: r.pop("all_ok"), "all_ok"),
        ],
    )
    def test_missing_evidence_is_named(self, mutate: Any, expected: str) -> None:
        report = complete_benchmark_report("rocm")
        mutate(report)
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any(expected in problem for problem in problems), problems

    def test_step_wall_is_optional(self) -> None:
        report = complete_benchmark_report("xpu")
        report["timings"]["cold"]["step_wall_ms"] = None
        for entry in report["timings"]["warm"]["runs"]:
            del entry["step_wall_ms"]
        assert validate_benchmark_report(report, accelerator="xpu") == ()

    def test_rss_is_required(self) -> None:
        report = complete_benchmark_report("xpu")
        report["memory"]["peak_rss_bytes"] = None
        problems = validate_benchmark_report(report, accelerator="xpu")
        assert "memory.peak_rss_bytes is not a positive integer" in problems

    def test_dinkster_placement_is_required_and_closed(self) -> None:
        report = complete_benchmark_report("cuda")
        report["placement"] = "driver-paging"
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("placement is not one of" in problem for problem in problems)

    def test_direct_diagnostic_is_standalone_only(self) -> None:
        report = complete_benchmark_report("cuda")
        report["placement"] = "direct_placement_diagnostic"
        assert validate_benchmark_report(report, accelerator="cuda") == ()
        problems = validate_benchmark_report(
            report,
            accelerator="cuda",
            canonical_evidence=True,
        )
        assert "canonical evidence requires Dinkster placement 'production_residency'" in problems

    @pytest.mark.parametrize("family", ["wan21_infinitetalk", "wan21_humo", "minimax_h3"])
    def test_audio_video_workloads_require_production_residency(self, family: str) -> None:
        report = complete_benchmark_report("cuda", family=family)
        report["placement"] = "direct_placement_diagnostic"
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"placement is not 'production_residency' for a Dinkster {family} report" in problem
            for problem in problems
        )

    def test_comfyui_placement_is_required_and_closed(self) -> None:
        report = complete_benchmark_report("cuda", system="comfyui")
        del report["placement"]
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("placement is not 'comfyui_model_management'" in problem for problem in problems)

        report = complete_benchmark_report("cuda", system="comfyui")
        report["placement"] = "production_residency"
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("placement is not 'comfyui_model_management'" in problem for problem in problems)

    def test_lora_report_requires_lora_evidence(self) -> None:
        report = complete_benchmark_report("rocm", family="lora")
        del report["checks"]["lora_apply"]
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any("checks.lora_apply" in problem for problem in problems)
        report = complete_benchmark_report("rocm", family="lora")
        del report["timings"]["cold"]["lora_s"]
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any("timings.cold.lora_s" in problem for problem in problems)
        report = complete_benchmark_report("rocm", family="lora")
        report["workload"]["lora_strength_clip"] = None
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any("workload.lora_strength_clip" in problem for problem in problems)

    def test_lora_artifact_roles_are_bound(self) -> None:
        report = complete_benchmark_report("xpu", family="lora")
        report["artifacts"] = [entry for entry in report["artifacts"] if entry["role"] != "lora"]
        problems = validate_benchmark_report(report, accelerator="xpu")
        assert any("role 'lora' exactly once" in problem for problem in problems)

    def test_infinitetalk_audio_evidence_and_artifact_roles_are_bound(self) -> None:
        report = complete_benchmark_report("cuda", family="wan21_infinitetalk")
        del report["checks"]["encode_audio"]
        del report["timings"]["cold"]["audio_encode_s"]
        report["artifacts"] = [
            entry for entry in report["artifacts"] if entry["role"] != "model_patch"
        ]
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("checks.encode_audio" in problem for problem in problems)
        assert any("timings.cold.audio_encode_s" in problem for problem in problems)
        assert any("role 'model_patch' exactly once" in problem for problem in problems)

    @pytest.mark.parametrize(
        "role", ["audio_encoder", "clip_vision", "input_image", "input_audio_1", "input_audio_2"]
    )
    def test_infinitetalk_requires_each_conditioning_artifact(self, role: str) -> None:
        report = complete_benchmark_report("cuda", family="wan21_infinitetalk")
        report["artifacts"] = [entry for entry in report["artifacts"] if entry["role"] != role]
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(f"role {role!r} exactly once" in problem for problem in problems)

    def test_infinitetalk_workload_settings_are_bound(self) -> None:
        report = complete_benchmark_report("cuda", family="wan21_infinitetalk")
        report["workload"]["motion_frame_count"] = 0
        report["workload"]["audio_scale"] = None
        report["workload"]["speaker_mask_layout"] = "overlap"
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("workload.motion_frame_count" in problem for problem in problems)
        assert any("workload.audio_scale" in problem for problem in problems)
        assert any("workload.speaker_mask_layout" in problem for problem in problems)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("sampler_id", "dinkster.uni_pc"), ("scheduler_id", "dinkster.simple")],
    )
    def test_infinitetalk_rejects_the_wrong_sampling_settings(
        self, field: str, value: object
    ) -> None:
        report = complete_benchmark_report("cuda", family="wan21_infinitetalk")
        report["workload"][field] = value
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"workload.{field} is not the pinned InfiniteTalk value" in problem
            for problem in problems
        )

    def test_infinitetalk_rejects_the_wrong_lora_strength(self) -> None:
        report = complete_benchmark_report("cuda", family="wan21_infinitetalk")
        report["workload"]["lora_strength_model"] = 0.5
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "workload.lora_strength_model is not the pinned InfiniteTalk value" in problem
            for problem in problems
        )

    @pytest.mark.parametrize("role", ["input_image", "input_audio_1", "input_audio_2"])
    def test_infinitetalk_rejects_an_unpinned_input_digest(self, role: str) -> None:
        report = complete_benchmark_report("cuda", family="wan21_infinitetalk")
        entry = next(entry for entry in report["artifacts"] if entry["role"] == role)
        entry["sha256"] = "f" * 64
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"role {role!r} does not match its pinned digest" in problem for problem in problems
        )

    def test_humo_pinned_inputs_and_fixed_workload_are_bound(self) -> None:
        report = complete_benchmark_report("cuda", family="wan21_humo")
        assert validate_benchmark_report(report, accelerator="cuda") == ()
        input_audio = next(
            artifact for artifact in report["artifacts"] if artifact["role"] == "input_audio"
        )
        input_audio["sha256"] = "f" * 64
        report["workload"]["sampler_id"] = "dinkster.euler"
        del report["timings"]["cold"]["audio_encode_s"]
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("input_audio" in problem and "pinned digest" in problem for problem in problems)
        assert any(
            "workload.sampler_id" in problem and "pinned HuMo" in problem for problem in problems
        )
        assert any("timings.cold.audio_encode_s" in problem for problem in problems)

    @pytest.mark.parametrize("role", HUMO_ARTIFACT_DIGESTS)
    def test_humo_rejects_each_unpinned_artifact_digest(self, role: str) -> None:
        report = complete_benchmark_report("cuda", family="wan21_humo")
        artifact = next(entry for entry in report["artifacts"] if entry["role"] == role)
        artifact["sha256"] = "f" * 64
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"role {role!r} does not match its pinned digest" in problem for problem in problems
        )

    @pytest.mark.parametrize("system", BENCHMARK_SYSTEMS)
    def test_humo_requires_the_system_placement(self, system: str) -> None:
        report = complete_benchmark_report("cuda", family="wan21_humo", system=system)
        report.pop("placement")
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "placement is not" in problem and system in problem.lower() for problem in problems
        )

    def test_humo_comfyui_requires_the_exact_commit(self) -> None:
        report = complete_benchmark_report("cuda", family="wan21_humo", system="comfyui")
        report["comfyui"]["commit"] = HUMO_COMFYUI_COMMIT[:12]
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "comfyui.commit is not the pinned HuMo commit" in problem for problem in problems
        )

        report = complete_benchmark_report("cuda", family="wan21_humo", system="comfyui")
        report.pop("comfyui")
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("comfyui section missing" in problem for problem in problems)

    @pytest.mark.parametrize("role", ANIMA_ARTIFACT_DIGESTS)
    def test_anima_rejects_each_unpinned_artifact_digest(self, role: str) -> None:
        report = complete_benchmark_report("cuda", family="anima")

        artifact = next(entry for entry in report["artifacts"] if entry["role"] == role)
        artifact["sha256"] = "f" * 64
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"role {role!r} does not match its pinned digest" in problem for problem in problems
        )

    @pytest.mark.parametrize("role", MINIMAX_H3_ARTIFACT_DIGESTS)
    def test_minimax_h3_rejects_each_unpinned_artifact_digest(self, role: str) -> None:
        report = complete_benchmark_report("cuda", family="minimax_h3")
        artifact = next(entry for entry in report["artifacts"] if entry["role"] == role)
        artifact["sha256"] = "f" * 64
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"role {role!r} does not match its pinned digest" in problem for problem in problems
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("prompt", "different"),
            ("negative_prompt", "different"),
            ("sampler_id", "dinkster.euler"),
            ("scheduler_id", "dinkster.normal"),
            ("seed", 0),
            ("steps", 29),
            ("cfg", 3.0),
            ("warm_runs", 4),
        ],
    )
    def test_anima_rejects_each_unpinned_workload_field(self, field: str, value: object) -> None:
        report = complete_benchmark_report("cuda", family="anima")
        report["workload"][field] = value
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"workload.{field} is not the pinned Anima value" in problem for problem in problems
        )

    def test_anima_variant_is_required_and_bound_to_its_geometry(self) -> None:
        primary = complete_benchmark_report("cuda", family="anima")
        primary.pop("variant")
        problems = validate_benchmark_report(primary, accelerator="cuda")
        assert any("variant is not one of" in problem for problem in problems)

        fallback = complete_benchmark_report("cuda", family="anima")
        fallback["variant"] = BENCHMARK_ANIMA_FALLBACK_VARIANT
        fallback["workload"]["width"] = 768
        fallback["workload"]["height"] = 768
        assert validate_benchmark_report(fallback, accelerator="cuda") == ()

        fallback["workload"]["width"] = 1024
        problems = validate_benchmark_report(fallback, accelerator="cuda")
        assert any("does not match variant" in problem for problem in problems)

        primary = complete_benchmark_report("cuda", family="anima")
        primary["workload"]["width"] = 768
        primary["workload"]["height"] = 768
        problems = validate_benchmark_report(primary, accelerator="cuda")
        assert any("does not match variant" in problem for problem in problems)

    @pytest.mark.parametrize("variant", [[], {}, 1, None])
    def test_anima_malformed_variant_fails_validation(self, variant: object) -> None:
        report = complete_benchmark_report("cuda", family="anima")
        report["variant"] = variant

        problems = validate_benchmark_report(report, accelerator="cuda")

        assert any("variant is not one of" in problem for problem in problems)

    def test_anima_requires_production_placement_and_the_exact_comfyui_commit(self) -> None:
        dinkster = complete_benchmark_report("cuda", family="anima")
        dinkster["placement"] = "direct_placement_diagnostic"
        problems = validate_benchmark_report(dinkster, accelerator="cuda")
        assert any("for a Dinkster anima report" in problem for problem in problems)

        comfyui = complete_benchmark_report("cuda", family="anima", system="comfyui")
        comfyui["comfyui"]["commit"] = HUMO_COMFYUI_COMMIT[:12]
        problems = validate_benchmark_report(comfyui, accelerator="cuda")
        assert any("pinned anima commit" in problem for problem in problems)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("prompt", "different"),
            ("negative_prompt", "negative"),
            ("sampler_id", "dinkster.euler"),
            ("scheduler_id", "dinkster.normal"),
            ("seed", 0),
            ("steps", 8),
            ("width", 768),
            ("height", 1344),
            ("length", 107),
            ("cfg", 2.0),
            ("warm_runs", 2),
        ],
    )
    def test_minimax_h3_rejects_each_unpinned_workload_field(
        self, field: str, value: object
    ) -> None:
        report = complete_benchmark_report("cuda", family="minimax_h3")
        report["workload"][field] = value
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"workload.{field} is not the pinned MiniMax H3 value" in problem
            for problem in problems
        )

    def test_minimax_h3_comfyui_requires_the_exact_commit(self) -> None:
        report = complete_benchmark_report("cuda", family="minimax_h3", system="comfyui")
        report["comfyui"]["commit"] = HUMO_COMFYUI_COMMIT[:12]
        problems = validate_benchmark_report(
            report, accelerator="cuda", expected_comfyui_commit=HUMO_COMFYUI_COMMIT
        )
        assert any(
            "comfyui.commit is not the required MiniMax H3 commit" in problem
            for problem in problems
        )

    def test_minimax_h3_comfyui_accepts_the_run_specific_commit(self) -> None:
        report = complete_benchmark_report("cuda", family="minimax_h3", system="comfyui")
        expected = "f" * 40
        report["comfyui"]["commit"] = expected

        problems = validate_benchmark_report(
            report, accelerator="cuda", expected_comfyui_commit=expected
        )

        assert not any("comfyui.commit" in problem for problem in problems)

    @pytest.mark.parametrize("system", ["dinkster", "comfyui"])
    def test_minimax_h3_requires_the_system_execution_path(self, system: str) -> None:
        report = complete_benchmark_report("cuda", family="minimax_h3", system=system)
        report["execution_path"] = "wrong"
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"execution_path is not the pinned MiniMax H3 {system} path" in problem
            for problem in problems
        )

    @pytest.mark.parametrize("system", ["dinkster", "comfyui"])
    def test_minimax_h3_rejects_a_missing_execution_path(self, system: str) -> None:
        report = complete_benchmark_report("cuda", family="minimax_h3", system=system)
        del report["execution_path"]
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"execution_path is not the pinned MiniMax H3 {system} path" in problem
            for problem in problems
        )

    def test_minimax_h3_length_must_be_17k_plus_5(self) -> None:
        report = complete_benchmark_report("cuda", family="minimax_h3")
        report["workload"]["length"] = 125
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("not a 17k+5 frame count" in problem for problem in problems)

    @pytest.mark.parametrize("role", FLUX_ARTIFACT_DIGESTS)
    def test_flux_rejects_each_unpinned_artifact_digest(self, role: str) -> None:
        report = complete_benchmark_report("cuda", family="flux")
        artifact = next(entry for entry in report["artifacts"] if entry["role"] == role)
        artifact["sha256"] = "f" * 64
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"role {role!r} does not match its pinned digest" in problem for problem in problems
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("prompt", "different"),
            ("negative_prompt", "different"),
            ("sampler_id", "dinkster.uni_pc"),
            ("scheduler_id", "dinkster.normal"),
            ("seed", 0),
            ("steps", 19),
            ("width", 768),
            ("height", 768),
            ("cfg", 3.5),
            ("warm_runs", 4),
            ("guidance", 4.0),
        ],
    )
    def test_flux_rejects_each_unpinned_workload_field(self, field: str, value: object) -> None:
        report = complete_benchmark_report("cuda", family="flux")
        report["workload"][field] = value
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"workload.{field} is not the pinned Flux value" in problem for problem in problems
        )

    def test_flux_requires_a_finite_guidance(self) -> None:
        report = complete_benchmark_report("cuda", family="flux")
        report["workload"]["guidance"] = None
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "workload.guidance missing or not a finite number" in problem for problem in problems
        )

    def test_guidance_is_banned_outside_the_flux_family(self) -> None:
        report = complete_benchmark_report("rocm", family="sd15")
        report["workload"]["guidance"] = 3.5
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any(
            "workload.guidance is only recorded for the flux family" in problem
            for problem in problems
        )

    def test_flux_requires_production_placement(self) -> None:
        report = complete_benchmark_report("cuda", family="flux")
        report["placement"] = "direct_placement_diagnostic"
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("for a Dinkster flux report" in problem for problem in problems)

    def test_flux_comfyui_requires_the_exact_commit(self) -> None:
        report = complete_benchmark_report("cuda", family="flux", system="comfyui")
        report["comfyui"]["commit"] = HUMO_COMFYUI_COMMIT[:12]
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "comfyui.commit is not the pinned Flux commit" in problem for problem in problems
        )

        report = complete_benchmark_report("cuda", family="flux", system="comfyui")
        report.pop("comfyui")
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("comfyui section missing" in problem for problem in problems)

    @pytest.mark.parametrize("role", CHROMA_ARTIFACT_DIGESTS)
    def test_chroma_rejects_each_unpinned_artifact_digest(self, role: str) -> None:
        report = complete_benchmark_report("cuda", family="chroma")
        artifact = next(entry for entry in report["artifacts"] if entry["role"] == role)
        artifact["sha256"] = "f" * 64
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"role {role!r} does not match its pinned digest" in problem for problem in problems
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("prompt", "different"),
            ("negative_prompt", "different"),
            ("sampler_id", "dinkster.uni_pc"),
            ("scheduler_id", "dinkster.simple"),
            ("seed", 0),
            ("steps", 20),
            ("width", 768),
            ("height", 768),
            ("cfg", 1.0),
            ("warm_runs", 4),
        ],
    )
    def test_chroma_rejects_each_unpinned_workload_field(self, field: str, value: object) -> None:
        report = complete_benchmark_report("cuda", family="chroma")
        report["workload"][field] = value
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            f"workload.{field} is not the pinned Chroma value" in problem for problem in problems
        )

    def test_guidance_is_banned_for_chroma(self) -> None:
        report = complete_benchmark_report("cuda", family="chroma")
        report["workload"]["guidance"] = 3.5
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "workload.guidance is only recorded for the flux family" in problem
            for problem in problems
        )

    def test_chroma_requires_production_placement(self) -> None:
        report = complete_benchmark_report("cuda", family="chroma")
        report["placement"] = "direct_placement_diagnostic"
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("for a Dinkster chroma report" in problem for problem in problems)

    def test_chroma_comfyui_requires_the_exact_commit(self) -> None:
        report = complete_benchmark_report("cuda", family="chroma", system="comfyui")
        report["comfyui"]["commit"] = HUMO_COMFYUI_COMMIT[:12]
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "comfyui.commit is not the pinned Chroma commit" in problem for problem in problems
        )

        report = complete_benchmark_report("cuda", family="chroma", system="comfyui")
        report.pop("comfyui")
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any("comfyui section missing" in problem for problem in problems)

    def test_wan21_length_is_required_and_positive(self) -> None:
        report = complete_benchmark_report("xpu", family="wan21")
        del report["workload"]["length"]
        problems = validate_benchmark_report(report, accelerator="xpu")
        assert any("workload.length missing or not an integer" in problem for problem in problems)
        report = complete_benchmark_report("xpu", family="wan21")
        report["workload"]["length"] = None
        problems = validate_benchmark_report(report, accelerator="xpu")
        assert any("workload.length missing or not an integer" in problem for problem in problems)
        report = complete_benchmark_report("xpu", family="wan21")
        report["workload"]["length"] = 0
        problems = validate_benchmark_report(report, accelerator="xpu")
        assert any("workload.length not positive" in problem for problem in problems)
        report = complete_benchmark_report("xpu", family="wan21")
        report["workload"]["length"] = 34
        problems = validate_benchmark_report(report, accelerator="xpu")
        assert any("not a 4k+1 frame count" in problem for problem in problems)

    def test_length_is_banned_outside_video_families(self) -> None:
        report = complete_benchmark_report("rocm", family="sd15")
        report["workload"]["length"] = 33
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any("only recorded for video families" in problem for problem in problems)

    def test_split_artifact_roles_are_bound(self) -> None:
        report = complete_benchmark_report("xpu", family="zimage")
        report["artifacts"] = [
            entry for entry in report["artifacts"] if entry["role"] != "text_encoder"
        ]
        problems = validate_benchmark_report(report, accelerator="xpu")
        assert any("role 'text_encoder' exactly once" in problem for problem in problems)

    def test_eager_only_families_reject_compile(self) -> None:
        for family in (
            "zimage",
            "wan21",
            "wan21_infinitetalk",
            "wan21_humo",
            "minimax_h3",
            "flux",
            "chroma",
        ):
            report = complete_benchmark_report("cuda", family=family, mode="compile")
            problems = validate_benchmark_report(report, accelerator="cuda")
            assert any(f"{family} reports must be eager" in problem for problem in problems)

    def test_comfyui_family_id_is_free_form_but_required(self) -> None:
        report = complete_benchmark_report("rocm", system="comfyui")
        assert validate_benchmark_report(report, accelerator="rocm") == ()
        report["family_id"] = " "
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any("family_id missing or empty" in problem for problem in problems)

    def test_failed_checks_are_recordable_but_must_match_all_ok(self) -> None:
        report = complete_benchmark_report("rocm")
        report["checks"]["warm_runs"] = {"ok": False, "detail": "nan in warm run 2"}
        report["all_ok"] = False
        assert validate_benchmark_report(report, accelerator="rocm") == ()

    def test_all_ok_over_failed_checks_is_corrupt(self) -> None:
        report = complete_benchmark_report("rocm")
        report["checks"]["warm_runs"] = {"ok": False, "detail": "nan in warm run 2"}
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any("all_ok is inconsistent" in problem for problem in problems)

    def test_passing_unload_cannot_leave_residual_above_ceiling(self) -> None:
        report = complete_benchmark_report("rocm")
        report["memory"]["residual_allocated_bytes"] = 2_097_152
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any("exceeds the unload ceiling" in problem for problem in problems)


_MIB = 1024 * 1024


def open_residency_section(roles: tuple[str, ...] = ("runtime",)) -> dict[str, Any]:
    """A valid open-regime residency section, as benchmark_inference.py
    emits it."""
    return {
        "mechanism": "auto",
        "aimdo_bootstrap_succeeded": True,
        "routes": {
            role: {
                "requested": "auto",
                "mechanism": "aimdo",
                "fallback_reason": None,
                "dynamic_components": ["component"],
                "resident_components": [],
                "fallback_components": [],
            }
            for role in roles
        },
        "regime": "open",
        "leave_free_mib": None,
        "ballast_bytes": None,
        "spill_scope": "process",
        "shared_before_bytes": 100,
        "shared_warm_bytes": 1_000_000_000,
        "shared_after_bytes": 1_000_000_000,
        "shared_growth_bytes": 0,
        "shared_spill_detected": None,
    }


def constrained_residency_section() -> dict[str, Any]:
    """A valid constrained-regime section with unattributed shared-memory growth."""
    section = open_residency_section()
    section.update(
        regime="constrained",
        leave_free_mib=768,
        ballast_bytes=2048 * _MIB,
        shared_after_bytes=1_000_000_000 + 64 * _MIB + 1,
        shared_growth_bytes=64 * _MIB + 1,
    )
    return section


class TestBenchmarkResidencySection:
    @pytest.mark.parametrize("detected", [True, False])
    def test_historical_spill_flags_are_readable_but_not_canonical(self, detected: bool) -> None:
        report = complete_benchmark_report("cuda")
        report["residency"].update(shared_spill_detected=detected, spill_threshold_mib=64)
        assert validate_benchmark_report(report, accelerator="cuda") == ()
        assert any(
            "shared-usage samples cannot assess spill" in problem
            for problem in validate_benchmark_report(
                report, accelerator="cuda", canonical_evidence=True
            )
        )

    def test_canonical_spill_assessment_must_be_explicitly_null(self) -> None:
        report = complete_benchmark_report("cuda")
        del report["residency"]["shared_spill_detected"]
        assert any(
            "shared_spill_detected must be null" in problem
            for problem in validate_benchmark_report(
                report, accelerator="cuda", canonical_evidence=True
            )
        )

    @pytest.mark.parametrize("scope", ["process", "machine"])
    @pytest.mark.parametrize("growth", [-100, 0, 64 * _MIB + 1])
    def test_shared_growth_is_observation_not_spill_assessment(
        self, scope: str, growth: int
    ) -> None:
        report = complete_benchmark_report("cuda")
        report["residency"].update(
            spill_scope=scope,
            shared_after_bytes=1_000_000_000 + growth,
            shared_growth_bytes=growth,
        )
        assert validate_benchmark_report(report, accelerator="cuda", canonical_evidence=True) == ()

    def test_reports_without_the_section_stay_valid(self) -> None:
        report = complete_benchmark_report("cuda")
        report.pop("residency")
        assert "residency" not in report
        assert validate_benchmark_report(report, accelerator="cuda") == ()

    @pytest.mark.parametrize(
        "family", ["sd15", "sdxl", "lora", "zimage", "wan21", "flux", "minimax_h3"]
    )
    def test_canonical_production_requires_residency_and_exact_route_keys(
        self, family: str
    ) -> None:
        report = complete_benchmark_report("cuda", family=family)
        requirements = H3_RESIDENCY_REQUIREMENTS if family == "minimax_h3" else {}
        assert (
            validate_benchmark_report(
                report, accelerator="cuda", canonical_evidence=True, **requirements
            )
            == ()
        )
        section = report.pop("residency")
        assert (
            "canonical production evidence requires residency route facts"
            in validate_benchmark_report(
                report, accelerator="cuda", canonical_evidence=True, **requirements
            )
        )
        report["residency"] = section
        for routes in ({}, {"unexpected": next(iter(section["routes"].values()))}):
            report["residency"] = {**section, "routes": routes}
            assert any(
                "residency.routes must record exactly" in problem
                for problem in validate_benchmark_report(
                    report,
                    accelerator="cuda",
                    canonical_evidence=True,
                    **requirements,
                )
            )

    @pytest.mark.parametrize(
        ("accelerator", "selector"),
        [("cuda", "auto"), ("cuda", "on"), ("rocm", "on"), ("xpu", "on")],
    )
    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (
                lambda s: s.update(aimdo_bootstrap_succeeded=False),
                "bootstrap_succeeded is not true",
            ),
            (lambda s: s.pop("aimdo_bootstrap_succeeded"), "bootstrap_succeeded is not true"),
            (lambda s: s.update(aimdo_bootstrap_succeeded=1), "not a boolean or null"),
            (
                lambda s: s["routes"]["runtime"].update(mechanism="eager"),
                "mechanism is not required aimdo",
            ),
            (
                lambda s: s["routes"]["runtime"].update(fallback_reason="admission failed"),
                "fell back",
            ),
            (lambda s: s["routes"]["runtime"].pop("fallback_reason"), "fallback_reason"),
            (lambda s: s["routes"]["runtime"].update(fallback_components=["text"]), "fell back"),
            (lambda s: s["routes"]["runtime"].update(dynamic_components=[]), "did not enroll"),
            (lambda s: s["routes"]["runtime"].update(requested="off"), "requested does not match"),
        ],
    )
    def test_canonical_generic_aimdo_evidence_fails_closed(
        self, accelerator: str, selector: str, mutate: Any, expected: str
    ) -> None:
        report = complete_benchmark_report(accelerator)
        section = report["residency"]
        section["mechanism"] = selector
        section["routes"]["runtime"]["requested"] = selector
        assert (
            validate_benchmark_report(report, accelerator=accelerator, canonical_evidence=True)
            == ()
        )
        mutate(section)
        assert any(
            expected in problem
            for problem in validate_benchmark_report(
                report, accelerator=accelerator, canonical_evidence=True
            )
        )

    @pytest.mark.parametrize(
        ("accelerator", "selector"), [("cuda", "off"), ("rocm", "auto"), ("xpu", "auto")]
    )
    def test_canonical_generic_records_explicit_eager_or_capability_fallback(
        self, accelerator: str, selector: str
    ) -> None:
        report = complete_benchmark_report(accelerator)
        section = report["residency"]
        section.update(mechanism=selector, aimdo_bootstrap_succeeded=None)
        section["routes"]["runtime"].update(
            requested=selector,
            mechanism="eager",
            dynamic_components=[],
            resident_components=["component"],
            fallback_reason="auto requires NVIDIA CUDA" if selector == "auto" else None,
        )
        assert (
            validate_benchmark_report(report, accelerator=accelerator, canonical_evidence=True)
            == ()
        )
        if selector == "off":
            section["routes"]["runtime"].update(mechanism="aimdo", dynamic_components=["component"])
            expected = "records dynamic residency with the off selector"
        else:
            section["routes"]["runtime"]["fallback_reason"] = None
            expected = "has no reason for eager fallback"
        assert any(
            expected in problem
            for problem in validate_benchmark_report(
                report, accelerator=accelerator, canonical_evidence=True
            )
        )

    def test_open_and_constrained_sections_are_accepted(self) -> None:
        report = complete_benchmark_report("cuda")
        report["residency"] = open_residency_section()
        assert validate_benchmark_report(report, accelerator="cuda") == ()
        report["residency"] = constrained_residency_section()
        assert validate_benchmark_report(report, accelerator="cuda") == ()

    def test_cuda_minimax_h3_requires_actual_aimdo_routes(self) -> None:
        report = complete_benchmark_report("cuda", family="minimax_h3")
        section = open_residency_section()
        section.update(mechanism="auto", aimdo_bootstrap_succeeded=True)
        route = {
            "requested": "auto",
            "mechanism": "aimdo",
            "fallback_reason": None,
            "dynamic_components": ["component"],
            "resident_components": [],
            "fallback_components": [],
        }
        routes = {
            role: dict(route) for role in ("diffusion", "conditioner", "video_vae", "audio_vae")
        }
        section["routes"] = routes
        report["residency"] = section
        assert "catalog residency requirements missing for family-specific routes" in (
            validate_benchmark_report(report, accelerator="cuda")
        )
        assert (
            validate_benchmark_report(report, accelerator="cuda", **H3_RESIDENCY_REQUIREMENTS) == ()
        )

        routes["conditioner"].update(
            mechanism="eager",
            fallback_reason="aimdo activation failed",
            dynamic_components=[],
        )
        problems = validate_benchmark_report(
            report, accelerator="cuda", **H3_RESIDENCY_REQUIREMENTS
        )
        assert "residency.routes.conditioner.mechanism is not required aimdo" in problems
        assert "residency.routes.conditioner fell back from required aimdo residency" in problems

    def test_off_scope_null_samples_are_accepted(self) -> None:
        report = complete_benchmark_report("cuda")
        section = open_residency_section()
        section.update(
            mechanism="off",
            routes={},
            spill_scope="off",
            shared_before_bytes=None,
            shared_warm_bytes=None,
            shared_after_bytes=None,
            shared_growth_bytes=None,
            shared_spill_detected=None,
        )
        report["residency"] = section
        assert validate_benchmark_report(report, accelerator="cuda") == ()

    def test_comfyui_reports_do_not_carry_residency(self) -> None:
        report = complete_benchmark_report("rocm", system="comfyui")
        report["residency"] = open_residency_section()
        problems = validate_benchmark_report(report, accelerator="rocm")
        assert any("only recorded by a dinkster report" in problem for problem in problems)

    def test_non_mapping_section_is_rejected(self) -> None:
        report = complete_benchmark_report("cuda")
        report["residency"] = "auto"
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert "residency is not a mapping" in problems

    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (lambda s: s.update(mechanism="eager"), "residency.mechanism is not one of"),
            (lambda s: s.update(mechanism="sticky"), "residency.mechanism is not one of"),
            (lambda s: s.update(regime="cramped"), "residency.regime is not one of"),
            (lambda s: s.update(spill_scope="auto"), "residency.spill_scope is not one of"),
            (
                lambda s: s.update(leave_free_mib=768),
                "residency.leave_free_mib is only recorded when constrained",
            ),
            (
                lambda s: s.update(ballast_bytes=1),
                "residency.ballast_bytes is only recorded when constrained",
            ),
            (
                lambda s: s.update(shared_before_bytes=-1),
                "residency.shared_before_bytes is not a non-negative integer",
            ),
            (
                lambda s: s.update(shared_growth_bytes=5),
                "shared_after_bytes - shared_warm_bytes",
            ),
            (
                lambda s: s.update(shared_warm_bytes=None),
                "residency.shared_growth_bytes requires warm and after samples",
            ),
            (
                lambda s: s.update(shared_spill_detected="safe"),
                "residency.shared_spill_detected is not a boolean or null",
            ),
        ],
    )
    def test_open_section_defects_are_named(self, mutate: Any, expected: str) -> None:
        report = complete_benchmark_report("cuda")
        section = open_residency_section()
        mutate(section)
        report["residency"] = section
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(expected in problem for problem in problems), problems

    @pytest.mark.parametrize("value", [None, 0, True])
    def test_constrained_section_requires_the_free_target(self, value: Any) -> None:
        report = complete_benchmark_report("cuda")
        section = constrained_residency_section()
        section["leave_free_mib"] = value
        report["residency"] = section
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "residency.leave_free_mib missing or not positive" in problem for problem in problems
        ), problems

    @pytest.mark.parametrize("value", [None, -1, True])
    def test_constrained_section_requires_the_ballast_size(self, value: Any) -> None:
        report = complete_benchmark_report("cuda")
        section = constrained_residency_section()
        section["ballast_bytes"] = value
        report["residency"] = section
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "residency.ballast_bytes missing or negative" in problem for problem in problems
        ), problems

    def test_zero_ballast_is_a_valid_constrained_run(self) -> None:
        report = complete_benchmark_report("cuda")
        section = constrained_residency_section()
        section["ballast_bytes"] = 0
        report["residency"] = section
        assert validate_benchmark_report(report, accelerator="cuda") == ()

    def test_off_scope_rejects_recorded_samples(self) -> None:
        report = complete_benchmark_report("cuda")
        section = open_residency_section()
        section["spill_scope"] = "off"
        report["residency"] = section
        problems = validate_benchmark_report(report, accelerator="cuda")
        assert any(
            "residency.shared_warm_bytes is recorded with spill_scope off" in problem
            for problem in problems
        ), problems
