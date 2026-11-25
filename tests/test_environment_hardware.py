from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from dinkster_supervisor import environment_hardware as hardware


@pytest.mark.parametrize(
    ("system", "arch", "driver", "capability", "expected"),
    [
        ("win32", "AMD64", "580.82", "12.0", ["win-nvidia"]),
        ("win32", "AMD64", "560.76", "7.5", ["win-nvidia-cu126"]),
        ("win32", "AMD64", "560.75", "7.5", ["win-cpu"]),
        ("win32", "AMD64", "580.82", "6.1", ["win-cpu"]),
        ("linux", "x86_64", "580.82", "12.0", ["linux-nvidia"]),
        ("linux", "x86_64", "580.81", "12.0", ["linux-cpu"]),
        ("linux", "x86_64", None, None, ["linux-cpu"]),
        ("darwin", "arm64", None, None, ["mac-arm64"]),
        ("darwin", "x86_64", None, None, []),
    ],
)
def test_build_selection_uses_published_limits_and_never_offers_cpu_with_supported_card(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    arch: str,
    driver: str | None,
    capability: str | None,
    expected: list[str],
) -> None:
    monkeypatch.setattr(hardware.sys, "platform", system)
    monkeypatch.setattr(hardware.platform, "machine", lambda: arch)
    manifests: dict[str, dict[str, object]] = {}
    for name, os_name, architecture, limits in [
        ("win-cpu", "windows", "x86_64", None),
        ("linux-cpu", "linux", "x86_64", None),
        ("mac-arm64", "macos", "arm64", None),
        ("win-nvidia-cu126", "windows", "x86_64", ("560.76", "7.5", "12.0")),
        ("win-nvidia", "windows", "x86_64", ("580.82", "8.0", None)),
        ("linux-nvidia", "linux", "x86_64", ("580.82", "8.0", None)),
    ]:
        manifests[name] = {
            "platform": {"os": os_name, "arch": architecture},
            "nvidia": None
            if limits is None
            else dict(
                zip(
                    (
                        "minimumDriver",
                        "minimumComputeCapability",
                        "maximumExclusiveComputeCapability",
                    ),
                    limits,
                    strict=True,
                )
            ),
        }
    cards: list[hardware.GraphicsCard] = [
        {
            "id": "0",
            "name": "NVIDIA card",
            "driver": driver,
            "computeCapability": capability,
            "memoryMiB": 8192,
        }
    ]
    assert hardware.supported_builds(manifests, cards) == expected


def test_card_detection_reads_nvidia_smi_without_loading_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware.sys, "platform", "linux")

    def command(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert args == [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
        ]
        return subprocess.CompletedProcess(args, 0, '0, "NVIDIA, one", 580.82, 12.0, 24576 MiB\n')

    monkeypatch.setattr(hardware.subprocess, "run", command)
    assert hardware.detect_graphics_cards() == [
        {
            "id": "0",
            "name": "NVIDIA, one",
            "driver": "580.82",
            "computeCapability": "12.0",
            "memoryMiB": 24576,
        }
    ]


@pytest.mark.parametrize("single", [False, True])
def test_windows_fallback_retains_card_identity_without_inventing_driver_facts(
    monkeypatch: pytest.MonkeyPatch, single: bool
) -> None:
    import json

    monkeypatch.setattr(hardware.sys, "platform", "win32")
    card = {"Name": "NVIDIA card", "PNPDeviceID": "PCI\\VEN_10DE", "AdapterCompatibility": "NVIDIA"}
    rows = card if single else [None, {"Name": 3}, {"Name": "Intel", "PNPDeviceID": "other"}, card]

    def command(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        assert args[0] == "powershell.exe"
        return subprocess.CompletedProcess(args, 0, json.dumps(rows))

    monkeypatch.setattr(hardware.subprocess, "run", command)
    assert hardware.detect_graphics_cards() == [
        {
            "id": "PCI\\VEN_10DE",
            "name": "NVIDIA card",
            "driver": None,
            "computeCapability": None,
            "memoryMiB": None,
        }
    ]


@pytest.mark.parametrize("nvidia", [False, True])
def test_missing_driver_tool_does_not_hide_an_nvidia_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvidia: bool
) -> None:
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    device = tmp_path / "0000:01:00.0"
    device.mkdir()
    (device / "vendor").write_text("0x10de" if nvidia else "0x8086")
    (device / "class").write_text("0x030000\n")

    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(hardware.subprocess, "run", missing)
    monkeypatch.setattr(hardware, "Path", lambda _: tmp_path)
    assert hardware.detect_graphics_cards() == (None if nvidia else [])
