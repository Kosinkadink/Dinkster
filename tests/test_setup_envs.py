"""Contributor selection uses producer-owned hardware bounds, never local torch pins."""

from __future__ import annotations

import json
import os
import shlex
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
    with pytest.raises(RuntimeError, match="issues/575"):
        setup_envs.select_base(bases)
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_envs.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases)


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
    observed = json.loads(probe.read_text())
    assert Path(observed["project"]) == root
    assert Path(observed["environment"]) == root / ".run/bootstrap"
    assert Path(observed["cwd"]) == root
    assert observed["args"] == ["sync", "--locked", "--package", "dinkster-supervisor", "--no-dev"]
    assert not (tmp_path / "wrong-environment").exists()
