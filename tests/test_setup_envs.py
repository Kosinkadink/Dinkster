"""Contributor selection uses producer-owned hardware bounds, never local torch pins."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import setup_envs


@pytest.fixture
def bases(monkeypatch):
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_envs.platform, "machine", lambda: "AMD64")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return [
        {"variant": "cpu", "platform": {"os": "windows", "arch": "x86_64"}, "nvidia": None},
        {
            "variant": "newer",
            "platform": {"os": "windows", "arch": "x86_64"},
            "nvidia": {
                "minimumDriver": "580",
                "minimumComputeCapability": "7.5",
                "maximumExclusiveComputeCapability": None,
            },
        },
        {
            "variant": "older",
            "platform": {"os": "windows", "arch": "x86_64"},
            "nvidia": {
                "minimumDriver": "525",
                "minimumComputeCapability": "5.0",
                "maximumExclusiveComputeCapability": "10.0",
            },
        },
    ]


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [
        ("580.1, 8.9", "newer"),
        ("579.9, 8.9", "older"),
        ("580.1, 6.1", "older"),
        ("524.9, 8.9", "cpu"),
        ("579.9, 10.0", "cpu"),
        ("580.1, 10.0", "newer"),
        ("580.1, 8.9\n530.1, 6.1", "older"),
    ],
)
def test_selects_compatible_bundle_from_manifest_bounds(bases, monkeypatch, hardware, expected):
    monkeypatch.setattr(setup_envs.subprocess, "check_output", lambda *args, **kwargs: hardware)
    assert setup_envs.select_base(bases)["variant"] == expected


@pytest.mark.parametrize("mask", ["", "-1"])
def test_disabled_gpu_selects_cpu_bundle(bases, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert setup_envs.select_base(bases)["variant"] == "cpu"
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases, "newer")


def test_unavailable_driver_selects_cpu_and_explicit_variant_cannot_bypass_bounds(
    bases, monkeypatch
):
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(setup_envs.subprocess, "check_output", unavailable)
    assert setup_envs.select_base(bases)["variant"] == "cpu"
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases, "older")


def test_no_bundle_for_another_platform_or_architecture(bases, monkeypatch):
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases)
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="Intel macOS is unsupported"):
        setup_envs.select_base(bases)
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_envs.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases)


@pytest.mark.parametrize("arch", ["arm64", "aarch64"])
def test_apple_silicon_selects_mac_bundle_without_cuda_probe(bases, monkeypatch, arch):
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup_envs.platform, "machine", lambda: arch)
    mac = {"variant": "mac-arm64", "platform": {"os": "macos", "arch": "arm64"}, "nvidia": None}
    bases.append(mac)
    assert setup_envs.select_base(bases) == mac
    assert setup_envs.select_base(bases, "mac-arm64") == mac
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases, "cpu")
    monkeypatch.setattr(setup_envs.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="Intel macOS is unsupported"):
        setup_envs.select_base(bases, "mac-arm64")


def test_setup_ignores_cwd_and_inherited_uv_project_target(tmp_path: Path):
    tools = tmp_path / "tools"
    tools.mkdir()
    probe = tmp_path / "probe.json"
    script = tools / "probe.py"
    script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SETUP_PROBE']).write_text(json.dumps({\n"
        "'project': os.environ.get('UV_PROJECT'),\n"
        "'environment': os.environ.get('UV_PROJECT_ENVIRONMENT'),\n"
        "'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
        "sys.exit(73)\n"
    )
    uv = tools / ("uv.cmd" if os.name == "nt" else "uv")
    uv.write_text(
        f'@"{sys.executable}" "{script}" %*\n'
        if os.name == "nt"
        else f'#!/bin/sh\n{shlex.quote(sys.executable)} {shlex.quote(str(script))} "$@"\n'
    )
    uv.chmod(0o755)
    root = Path(setup_envs.__file__).resolve().parents[1]
    environment = os.environ | {
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "UV_PROJECT": str(tmp_path / "wrong-project"),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "wrong-environment"),
        "SETUP_PROBE": str(probe),
    }
    environment.pop("DINKSTER_SETUP_BOOTSTRAPPED", None)
    result = subprocess.run(
        [sys.executable, str(root / "scripts/setup_envs.py")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert probe.is_file(), f"stdout={result.stdout}\nstderr={result.stderr}"
    observed = json.loads(probe.read_text())
    assert Path(observed["project"]) == root
    assert Path(observed["environment"]) == root / ".run/bootstrap"
    assert Path(observed["cwd"]) == root
    assert observed["args"] == ["sync", "--locked", "--package", "dinkster-supervisor", "--no-dev"]
    assert not (tmp_path / "wrong-environment").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows runtime admission")
@pytest.mark.parametrize("matching_base", [False, True])
def test_setup_checks_reused_windows_environment_base(tmp_path, monkeypatch, matching_base):
    from dinkster_supervisor import environment_artifacts, environment_downloads

    from scripts import build_release

    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/base_bundles.json").write_text('{"win-cpu": "identity"}')
    gate = "packages/dinkster-supervisor/src/dinkster_supervisor/sqlite_runtime.py"
    (root / gate).parent.mkdir(parents=True)
    shutil.copyfile(Path(setup_envs.__file__).resolve().parents[1] / gate, root / gate)
    base_root = Path(sys.base_prefix).parent if matching_base else tmp_path / "other-base"
    base = {
        "id": "sha256:identity",
        "variant": "win-cpu",
        "python": sys.version.split()[0],
        "interpreter": {"executable": Path(sys.base_prefix).name + "/python.exe"},
        "wheels": [
            {
                "name": "fixture",
                "version": "1.0",
                "filename": "fixture-1.0-py3-none-any.whl",
                "sha256": "a" * 64,
            }
        ],
        "replaces": ["fixture"],
    }
    environment = root / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    (root / ".run").mkdir()
    receipt = root / ".run/environment-base.json"
    receipt.write_text(json.dumps(base))
    monkeypatch.setattr(environment_downloads, "read_metadata", lambda *args: b"")
    monkeypatch.setattr(environment_artifacts, "validate_manifest", lambda *args: base)
    monkeypatch.setattr(environment_artifacts, "acquire_base", lambda *args: base_root)
    monkeypatch.setattr(setup_envs, "select_base", lambda *args: base)
    monkeypatch.setattr(build_release, "workspace_projects", lambda *args: {})
    run = setup_envs.command

    def fixture_command(*args, **kwargs):
        if args[:2] == ("uv", "export"):
            (root / ".run/workspace-requirements.txt").write_text("")
        elif args[0] != "uv":
            run(*args, **kwargs)

    monkeypatch.setattr(setup_envs, "command", fixture_command)
    if matching_base:
        assert setup_envs.setup(root) == environment / "Scripts/python.exe"
    else:
        before = receipt.read_bytes()
        with pytest.raises(subprocess.CalledProcessError):
            setup_envs.setup(root)
        assert receipt.read_bytes() == before
