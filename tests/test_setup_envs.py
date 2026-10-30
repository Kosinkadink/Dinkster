"""Contributor selection uses producer-owned hardware bounds, never local torch pins."""

from __future__ import annotations

import subprocess

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
