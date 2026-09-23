from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POWERSHELL_SETUP = REPO_ROOT / "scripts" / "setup_envs.ps1"
POSIX_SETUP = REPO_ROOT / "scripts" / "setup_envs.sh"
GPU_SETUP_DOC = REPO_ROOT / "packages" / "dinkster-inference-torch" / "README.md"
GPU_TEST = REPO_ROOT / "packages" / "dinkster-inference-torch" / "tests" / "test_gpu.py"
GPU_MODEL_PACKS = {
    "packages/dinkster-model-triposplat": "dinkster_model_triposplat",
    "packages/dinkster-model-wan": "dinkster_model_wan",
}


def _powershell_package_array(source: str, name: str) -> list[str]:
    match = re.search(rf"\${name} = @\((.*?)\n\)", source, re.DOTALL)
    assert match is not None
    return re.findall(r'^\s+"(packages/[^\"]+)"', match.group(1), re.MULTILINE)


def _powershell_dependency_array(source: str, name: str) -> list[str]:
    match = re.search(rf"\${name} = @\((.*?)\n[ \t]*\) \+", source, re.DOTALL)
    assert match is not None
    dependencies = re.findall(r'"([^\"]+)"', match.group(1))
    if "$KitchenCpuWheel" in match.group(1):
        dependencies.append("$KitchenCpuWheel")
    return dependencies


def _posix_editables(source: str, start: str, end: str) -> list[str]:
    section = source.split(start, 1)[1].split(end, 1)[0]
    return re.findall(r"-e ['\"]?(packages/[^\s'\"\\]+)", section)


def test_powershell_setup_matches_posix_editable_package_closure() -> None:
    powershell = POWERSHELL_SETUP.read_text()
    posix = POSIX_SETUP.read_text()

    assert _powershell_package_array(powershell, "CpuEditablePackages") == _posix_editables(
        posix,
        "uv pip install --python .venv-torch/bin/python --reinstall-package dinkster-nodes-std",
        "# The direct PyPI URL forces",
    )
    assert _powershell_package_array(powershell, "GpuEditablePackages") == _posix_editables(
        posix,
        "uv pip install --python .venv-gpu/bin/python \\",
        "\n\nelse",
    )


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


def test_torch_environments_include_server_without_host_package_leakage() -> None:
    powershell = POWERSHELL_SETUP.read_text()
    posix = POSIX_SETUP.read_text()

    assert posix.count("-e .") == 2
    assert powershell.count('@("-e", $RepoRoot)') == 2
    for setup in (powershell, posix):
        assert (
            setup.count(
                "import site; import av, dinkster.serve, dinkster_model_triposplat.provider, torch"
            )
            == 2
        )
        assert setup.count("assert site.ENABLE_USER_SITE is False") == 2
        assert setup.count("assert version('av') == '18.1.0'") == 2
        assert "--system-site-packages" not in setup
        assert "PYTHONPATH" not in setup
    assert posix.count("/python -I -c") == 2
    assert powershell.count('"-I"') == 2


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
    assert '"torch==2.13.0+cu130", "torchvision==0.28.0+cu130"' in setup
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
    assert '"sync", "--project", $RepoRoot, "--python", "3.12", "--all-packages",' in sync
    assert setup.count('"--reinstall-package", "dinkster-nodes-std"') == 3
    assert POSIX_SETUP.read_text().count("--reinstall-package dinkster-nodes-std") == 3
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
