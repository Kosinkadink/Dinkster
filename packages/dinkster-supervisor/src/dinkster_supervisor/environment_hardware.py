"""Read host graphics-card facts and select from published base manifests."""

from __future__ import annotations

import csv
import io
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast


class GraphicsCard(TypedDict):
    id: str
    name: str
    driver: str | None
    computeCapability: str | None
    memoryMiB: int | None


def detect_graphics_cards() -> list[GraphicsCard] | None:
    """Inspect hardware without importing CUDA or acquiring software."""
    if sys.platform == "darwin":
        return []
    nvidia_smi = "nvidia-smi"
    if (
        sys.platform == "linux"
        and shutil.which(nvidia_smi) is None
        and Path("/usr/lib/wsl/lib/nvidia-smi").is_file()
    ):
        nvidia_smi = "/usr/lib/wsl/lib/nvidia-smi"
    try:
        output = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
    except FileNotFoundError:
        if sys.platform != "win32":
            try:
                devices = Path("/sys/bus/pci/devices")
                if not devices.is_dir():
                    return None
                for device in devices.iterdir():
                    if (device / "vendor").read_text().strip() == "0x10de" and (
                        device / "class"
                    ).read_text().startswith("0x03"):
                        return None
                return []
            except OSError:
                return None
    except (OSError, subprocess.SubprocessError):
        return None
    else:
        cards: list[GraphicsCard] = []
        for row in csv.reader(io.StringIO(output), skipinitialspace=True):
            if len(row) != 5:
                return None
            index, name, driver, capability, memory = (value.strip() for value in row)
            if (
                not index.isdecimal()
                or not name
                or not re.fullmatch(r"\d+(?:\.\d+)+", driver)
                or not re.fullmatch(r"\d+\.\d+", capability)
                or version(capability) == (0,)
                or not re.fullmatch(r"[1-9]\d* MiB", memory)
                or any(card["id"] == index for card in cards)
            ):
                return None
            cards.append(
                {
                    "id": index,
                    "name": name,
                    "driver": driver,
                    "computeCapability": capability,
                    "memoryMiB": int(memory.removesuffix(" MiB")),
                }
            )
        return cards
    try:
        output = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object PNPDeviceID,Name,AdapterCompatibility | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        parsed: object = json.loads(output or "[]")
        rows = cast("list[object]", parsed) if isinstance(parsed, list) else [parsed]
        cards = []
        for value in rows:
            if not isinstance(value, dict):
                continue
            row = cast("dict[str, object]", value)
            name, identifier = row.get("Name"), row.get("PNPDeviceID")
            if (
                isinstance(name, str)
                and isinstance(identifier, str)
                and "nvidia" in f"{row.get('AdapterCompatibility')} {name}".lower()
            ):
                cards.append(
                    {
                        "id": identifier,
                        "name": name,
                        "driver": None,
                        "computeCapability": None,
                        "memoryMiB": None,
                    }
                )
        return cards
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def version(value: str) -> tuple[int, ...]:
    parts = [int(part) for part in value.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def supported_cards(base: dict[str, object], cards: list[GraphicsCard]) -> list[GraphicsCard]:
    limits = cast("dict[str, str | None] | None", base["nvidia"])
    if limits is None:
        return []
    maximum = limits["maximumExclusiveComputeCapability"]
    return [
        card
        for card in cards
        if (driver := card["driver"]) is not None
        and (capability := card["computeCapability"]) is not None
        and version(driver) >= version(cast("str", limits["minimumDriver"]))
        and version(capability) >= version(cast("str", limits["minimumComputeCapability"]))
        and (maximum is None or version(capability) < version(maximum))
    ]


def supported_builds(
    manifests: dict[str, dict[str, object]], cards: list[GraphicsCard]
) -> list[str]:
    """Prefer the newest compatible GPU base, with native CPU alternatives last."""
    os_name = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform)
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64"}.get(platform.machine().lower())
    native = {
        name: base
        for name, base in manifests.items()
        if base["platform"] == {"os": os_name, "arch": arch}
    }
    gpu = [name for name, base in native.items() if supported_cards(base, cards)]
    return sorted(
        gpu,
        key=lambda name: version(
            cast("dict[str, str]", native[name]["nvidia"])["minimumComputeCapability"]
        ),
        reverse=True,
    ) + sorted(name for name, base in native.items() if base["nvidia"] is None)


def build_reason(base: dict[str, object]) -> str:
    limits = cast("dict[str, str | None] | None", base["nvidia"])
    if limits is None:
        if cast("dict[str, str]", base["platform"])["os"] == "macos":
            return "Uses the graphics built into this Mac."
        return "Does not use the graphics card and is much slower."
    minimum = ".".join(str(part) for part in version(cast("str", limits["minimumDriver"])))
    return f"Needs NVIDIA driver {minimum} or newer."


def build_label(variant: str) -> str:
    return {
        "win-nvidia": "NVIDIA graphics",
        "win-nvidia-cu126": "NVIDIA graphics, older drivers",
        "linux-nvidia": "NVIDIA graphics",
        "mac-arm64": "Mac with Apple Silicon",
        "win-cpu": "CPU only",
        "linux-cpu": "CPU only",
    }.get(variant, variant)


def hardware_selection(
    manifests: dict[str, dict[str, object]],
    cards: list[GraphicsCard] | None,
    requested: str | None = None,
    assigned: tuple[str, ...] = (),
) -> dict[str, object]:
    """Shared read-only hardware response for setup planning and CLI consumers."""
    if cards is None:
        return {
            "detectedCards": None,
            "choices": None,
            "variant": None,
            "buildLabel": None,
            "buildReason": None,
            "gpus": [],
            "refusal": {
                "code": "hardware-unavailable",
                "message": "Graphics-card detection failed. "
                "Check the graphics driver and try again.",
            },
        }
    builds = supported_builds(manifests, cards)
    if assigned:
        gpu_default = bool(builds and manifests[builds[0]]["nvidia"] is not None)
        builds = [
            name
            for name in builds
            if manifests[name]["nvidia"] is None
            or set(assigned).issubset(
                card["id"] for card in supported_cards(manifests[name], cards)
            )
        ]
        if gpu_default and not any(manifests[name]["nvidia"] is not None for name in builds):
            builds = []
    if requested is not None and requested not in builds:
        raise ValueError("That build is not supported on this computer.")
    variant = requested or (builds[0] if builds else None)
    return {
        "detectedCards": cards,
        "choices": [
            {
                "variant": name,
                "label": build_label(name),
                "reason": build_reason(manifests[name]),
            }
            for name in builds
        ],
        "variant": variant,
        "buildLabel": build_label(variant) if variant is not None else None,
        "buildReason": build_reason(manifests[variant]) if variant is not None else None,
        "gpus": [
            card["id"]
            for card in supported_cards(manifests[variant], cards)
            if not assigned or card["id"] in assigned
        ]
        if variant is not None
        else [],
        "refusal": None
        if builds
        else {
            "code": "unsupported-hardware",
            "message": "No published build supports this computer.",
        },
    }
