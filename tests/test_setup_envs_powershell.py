from __future__ import annotations

import ast
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from dinkster_workers.backend_env import BACKEND_ENV_RECIPES
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
POWERSHELL_SETUP = REPO_ROOT / "scripts" / "setup_envs.ps1"
POSIX_SETUP = REPO_ROOT / "scripts" / "setup_envs.sh"
GPU_SETUP_DOC = REPO_ROOT / "packages" / "dinkster-inference-torch" / "README.md"
GPU_TEST = REPO_ROOT / "packages" / "dinkster-inference-torch" / "tests" / "test_gpu.py"
INFERENCE_TORCH_PROJECT = REPO_ROOT / "packages" / "dinkster-inference-torch"
MINIMAX_MUSIC3_COMPONENT = "minimax_music3_component"
GPU_MODEL_PACKS = {
    "packages/dinkster-model-triposplat": "dinkster_model_triposplat",
    "packages/dinkster-model-wan": "dinkster_model_wan",
}


def _powershell_package_array(source: str, name: str) -> list[str]:
    match = re.search(rf"\${name} = @\((.*?)\n\)", source, re.DOTALL)
    assert match is not None
    return re.findall(r'^\s+"(packages/[^\"]+)"', match.group(1), re.MULTILINE)


def _powershell_dependency_array(source: str, name: str) -> list[str]:
    match = re.search(rf"\${name} = @\((.*?)\n\s*\) \+", source, re.DOTALL)
    assert match is not None
    dependencies = re.findall(r'"([^\"]+)"', match.group(1))
    if "$KitchenCpuWheel" in match.group(1):
        dependencies.append("$KitchenCpuWheel")
    return dependencies


def _powershell_requirement_names(source: str, name: str) -> set[str]:
    requirements = _powershell_dependency_array(source, name)
    resolved: list[str] = []
    for requirement in requirements:
        if not requirement.startswith("$"):
            resolved.append(requirement)
            continue
        variable = requirement.removeprefix("$")
        match = re.search(rf'^\${variable} = "([^\"]+)"$', source, re.MULTILINE)
        assert match is not None
        resolved.append(match.group(1))
    return _requirement_names(resolved)


def _posix_editables(source: str, start: str, end: str) -> list[str]:
    section = source.split(start, 1)[1].split(end, 1)[0]
    return re.findall(r"-e ['\"]?(packages/[^\s'\"\\]+)", section)


def _requirement_names(requirements: list[str] | tuple[str, ...]) -> set[str]:
    return {
        Requirement(requirement).name.lower()
        for requirement in requirements
        if not requirement.startswith("$")
    }


def _posix_requirement_names(source: str, start: str, end: str) -> set[str]:
    section = start + source.rsplit(start, 1)[1].split(end, 1)[0]
    command_lines: list[str] = []
    for line in section.splitlines():
        command_lines.append(line)
        if not line.rstrip().endswith("\\"):
            break
    command = "\n".join(command_lines)
    tokens = shlex.split(command.replace("\\\n", " "))
    requirements = tokens[tokens.index("--python") + 2 : tokens.index("-e")]
    return _requirement_names(requirements)


def _posix_variable_install_requirement_names(source: str, python: str, variable: str) -> set[str]:
    match = re.search(rf'^{variable}="([^\"]+)"$', source, re.MULTILINE)
    assert match is not None
    assert f'uv pip install --python {python} "${variable}"' in source
    return _requirement_names((match.group(1),))


def _module_level_statements(statements: list[ast.stmt]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for statement in statements:
        result.append(statement)
        if isinstance(statement, ast.If | ast.Try):
            result.extend(_module_level_statements(statement.body))
            result.extend(_module_level_statements(statement.orelse))
            if isinstance(statement, ast.Try):
                result.extend(_module_level_statements(statement.finalbody))
                for handler in statement.handlers:
                    result.extend(_module_level_statements(handler.body))
        elif isinstance(statement, ast.With):
            result.extend(_module_level_statements(statement.body))
    return result


def _recursive_module_imports(module: str) -> set[str]:
    source_root = INFERENCE_TORCH_PROJECT / "src" / "dinkster_inference_torch"
    pending = [module]
    visited: set[str] = set()
    imports: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        path = source_root / f"{current.replace('.', '/')}.py"
        if not path.is_file():
            continue
        statements = _module_level_statements(ast.parse(path.read_text()).body)
        for statement in statements:
            if isinstance(statement, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom):
                if statement.level == 0:
                    if statement.module is not None:
                        imports.add(statement.module.split(".", 1)[0])
                    continue
                parent = current.split(".")[: -statement.level]
                relative = parent + ([statement.module] if statement.module else [])
                if statement.module is not None:
                    pending.append(".".join(relative))
                for alias in statement.names:
                    candidate = ".".join((*relative, alias.name))
                    if (source_root / f"{candidate.replace('.', '/')}.py").is_file():
                        pending.append(candidate)
    return {name.replace("_", "-").lower() for name in imports - sys.stdlib_module_names}


def test_powershell_setup_matches_posix_editable_package_closure() -> None:
    powershell = POWERSHELL_SETUP.read_text()
    posix = POSIX_SETUP.read_text()

    assert _powershell_package_array(powershell, "CpuEditablePackages") == _posix_editables(
        posix,
        "uv pip install --python .venv-torch/bin/python pytest packaging",
        "# The direct PyPI URL forces",
    )
    assert _powershell_package_array(powershell, "GpuEditablePackages") == _posix_editables(
        posix,
        "uv pip install --python .venv-gpu/bin/python \\",
        "\n\nelse",
    )


def test_supported_execution_environments_cover_minimax_music3_runtime_imports() -> None:
    powershell = POWERSHELL_SETUP.read_text()
    posix = POSIX_SETUP.read_text()
    project = tomllib.loads((INFERENCE_TORCH_PROJECT / "pyproject.toml").read_text())["project"]
    base_requirements = _requirement_names(project["dependencies"])
    imported_distributions = _recursive_module_imports(MINIMAX_MUSIC3_COMPONENT)

    contracts = {
        "package torch extra": base_requirements
        | _requirement_names(project["optional-dependencies"]["torch"]),
        "PowerShell CPU": base_requirements
        | _powershell_requirement_names(powershell, "CpuDependencies")
        | {"torch"},
        "PowerShell CUDA": base_requirements
        | _powershell_requirement_names(powershell, "GpuDependencies")
        | {"torch"},
        "POSIX CPU": base_requirements
        | _posix_requirement_names(
            posix,
            "uv pip install --python .venv-torch/bin/python pytest packaging",
            "# The direct PyPI URL forces",
        )
        | _posix_variable_install_requirement_names(
            posix, ".venv-torch/bin/python", "kitchen_cpu_wheel"
        )
        | {"torch"},
        "POSIX CUDA": base_requirements
        | _posix_requirement_names(
            posix,
            "uv pip install --python .venv-gpu/bin/python \\",
            "\n\nelse",
        )
        | {"torch"},
    }
    contracts.update(
        {
            f"{cell} backend": base_requirements
            | _requirement_names((*recipe.support_packages, recipe.torch_requirement))
            for cell, recipe in BACKEND_ENV_RECIPES.items()
        }
    )

    missing = {
        name: sorted(imported_distributions - installed)
        for name, installed in contracts.items()
        if imported_distributions - installed
    }
    assert not missing

    root_project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    assert "torch" not in _requirement_names(root_project["dependencies"])


def test_gpu_setup_installs_model_packs_imported_by_gpu_tests() -> None:
    powershell = POWERSHELL_SETUP.read_text()
    posix = POSIX_SETUP.read_text()
    documentation = GPU_SETUP_DOC.read_text()
    gpu_test = GPU_TEST.read_text()
    expected = set(GPU_MODEL_PACKS)

    assert all(f"from {module} import" in gpu_test for module in GPU_MODEL_PACKS.values())
    assert expected <= set(_powershell_package_array(powershell, "GpuEditablePackages"))
    assert expected <= set(
        _posix_editables(
            posix,
            "uv pip install --python .venv-gpu/bin/python \\",
            "\n\nelse",
        )
    )
    assert expected <= set(
        _posix_editables(
            documentation,
            "uv pip install --python .venv-gpu/bin/python \\",
            "DINKSTER_ENABLE_GPU_TESTS=1",
        )
    )


def test_powershell_setup_pins_native_windows_test_environments() -> None:
    setup = POWERSHELL_SETUP.read_text()

    assert _powershell_dependency_array(setup, "CpuDependencies") == [
        "pytest",
        "packaging",
        "numpy>=1.26",
        "scipy>=1.11",
        "simpleeval==1.0.3",
        "onnxruntime==1.29.0",
        "opencv-python-headless==5.0.0.93",
        "pillow==12.0.0",
        "safetensors==0.8.0",
        "sentencepiece==0.2.1",
        "tokenizers==0.23.1",
        "transformers==5.16.1",
        "dinkster-aimdo==0.5.5.post2",
        "$KitchenCpuWheel",
    ]
    assert _powershell_dependency_array(setup, "GpuDependencies") == [
        "pytest",
        "numpy",
        "scipy",
        "torchsde",
        "tqdm",
        "pillow",
        "packaging",
        "safetensors==0.8.0",
        "sentencepiece==0.2.1",
        "tokenizers==0.23.1",
        "dinkster-kitchen==0.2.35.post1",
        "dinkster-aimdo==0.5.5.post2",
        "triton-windows==3.7.1.post27",
    ]
    assert '"3.12"' in setup
    assert '"torch==2.13.0+cpu", "torchvision==0.28.0+cpu"' in setup
    assert '"torch==2.13.0+cu130"' in setup
    assert '"triton-windows==3.7.1.post27"' in setup
    assert (
        "dinkster_kitchen-0.2.35.post1-py3-none-any.whl#sha256="
        "31458547cdcf9ff26974a4955cf79e83ebdf50077666720d3bb3255786c5fc4f"
    ) in setup
    assert "Test-Path (Join-Path" in setup
    assert '"Python.h"' in setup
    assert "Get-Command nvidia-smi -ErrorAction SilentlyContinue" in setup
    assert "no NVIDIA GPU detected - skipping .venv-gpu" in setup


def test_powershell_setup_limits_force_and_has_no_private_installer() -> None:
    setup = POWERSHELL_SETUP.read_text()
    force_body = setup.split("if ($Force) {", 1)[1].split("Write-Host", 1)[0]

    for variable, path in (
        ("RootEnvironment", ".venv"),
        ("TorchEnvironment", ".venv-torch"),
        ("GpuEnvironment", ".venv-gpu"),
        ("SetupExtras", ".venv-gpu-extras"),
    ):
        assert f'${variable} = Join-Path $RepoRoot "{path}"' in setup
    assert force_body.count("Remove-Item -Recurse") == 1
    assert "$RootEnvironment, $TorchEnvironment, $GpuEnvironment, $SetupExtras" in force_body
    assert "DINKSTER_AIMDO_TOKEN" not in setup
    assert "install_dinkster_aimdo.py" not in setup


def test_setup_scripts_skip_optional_acceptance_package_when_absent() -> None:
    powershell = POWERSHELL_SETUP.read_text()
    posix = POSIX_SETUP.read_text()
    notice = "dinkster-evidence not found - skipping optional dinkster-acceptance"

    assert 'if [ -d "$acceptance_package" ]; then' in posix
    assert posix.count('if [ "$install_acceptance" = 1 ]; then') == 2
    assert notice in posix
    assert "$InstallAcceptance = Test-Path $AcceptancePackage -PathType Container" in powershell
    assert powershell.count("if ($InstallAcceptance) {") == 2
    assert notice in powershell


def test_powershell_setup_isolates_root_sync_and_prints_runnable_gates() -> None:
    setup = POWERSHELL_SETUP.read_text()
    sync = setup.split('Write-Host "==> .venv (torch-free root env)"', 1)[1].split(
        'Write-Host "==> .venv-torch', 1
    )[0]

    assert '[Environment]::SetEnvironmentVariable("UV_PROJECT", $null, "Process")' in sync
    assert '"UV_PROJECT_ENVIRONMENT", $RootEnvironment, "Process"' in sync
    assert '"sync", "--project", $RepoRoot, "--python", "3.12", "--all-packages"' in sync
    assert "$PreviousProject" not in sync
    cleanup = setup.rsplit("finally {", 1)[1]
    assert '"UV_PROJECT", $PreviousProject, "Process"' in cleanup
    assert '"UV_PROJECT_ENVIRONMENT", $PreviousProjectEnvironment, "Process"' in cleanup
    for command in (
        ".venv\\Scripts\\ruff.exe check .",
        ".venv\\Scripts\\pyright.exe",
        ".venv\\Scripts\\python.exe -m pytest -q",
        ".venv-torch\\Scripts\\python.exe -m pytest -q packages\\dinkster-inference-torch\\tests",
        ".venv-gpu\\Scripts\\python.exe -m pytest -q packages\\dinkster-inference-torch\\tests",
    ):
        assert command in setup


def test_setup_scripts_print_cuda_reference_validation_gate() -> None:
    powershell = POWERSHELL_SETUP.read_text()
    posix = POSIX_SETUP.read_text()

    assert '$env:DINKSTER_VALIDATE_REFERENCE_GOLDENS = "1"' in powershell
    assert "Remove-Item Env:\\DINKSTER_VALIDATE_REFERENCE_GOLDENS" in powershell
    assert "DINKSTER_ENABLE_GPU_TESTS=1 DINKSTER_VALIDATE_REFERENCE_GOLDENS=1" in posix
    assert ".venv-gpu/bin/python -m pytest -q packages/dinkster-inference-torch/tests" in posix


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell execution requires Windows")
def test_powershell_setup_root_sync_ignores_cwd_and_ambient_uv_target(tmp_path: Path) -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    assert powershell is not None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv.cmd"
    uv.write_text(
        "@echo off\r\n"
        '> "%SYNC_PROBE%" echo project=%UV_PROJECT%\r\n'
        '>> "%SYNC_PROBE%" echo environment=%UV_PROJECT_ENVIRONMENT%\r\n'
        '>> "%SYNC_PROBE%" echo args=%*\r\n'
        "exit /b 73\r\n",
        newline="",
    )
    probe = tmp_path / "sync.txt"
    restore_probe = tmp_path / "restored.txt"
    environment = os.environ | {
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "UV_PROJECT": str(tmp_path / "wrong-project"),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "wrong-environment"),
        "SYNC_PROBE": str(probe),
        "RESTORE_PROBE": str(restore_probe),
        "SETUP_SCRIPT": str(POWERSHELL_SETUP),
    }
    result = subprocess.run(
        (
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "try { & $env:SETUP_SCRIPT } catch {} ; "
            "[IO.File]::WriteAllLines($env:RESTORE_PROBE, @($env:UV_PROJECT, "
            "$env:UV_PROJECT_ENVIRONMENT)); exit 1",
        ),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    lines = probe.read_text().splitlines()
    assert lines[:2] == [
        "project=",
        f"environment={REPO_ROOT / '.venv'}",
    ]
    assert lines[2].startswith(f"args=sync --project {REPO_ROOT} --python 3.12 --all-packages")
    assert restore_probe.read_text().splitlines() == [
        str(tmp_path / "wrong-project"),
        str(tmp_path / "wrong-environment"),
    ]
