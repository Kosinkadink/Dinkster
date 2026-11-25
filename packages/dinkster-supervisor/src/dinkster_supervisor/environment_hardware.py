"""Read host graphics-card facts and select from published base manifests."""

from __future__ import annotations

import csv
import io
import json
import platform
import re
import subprocess
import sys
from typing import TypedDict, cast


class GraphicsCard(TypedDict):
    id: str
    name: str
    driver: str | None
    computeCapability: str | None
    memoryMiB: int | None


def detect_graphics_cards() -> list[GraphicsCard]:
    """Inspect hardware without importing CUDA or acquiring software."""
    if sys.platform == "darwin":
        return []
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
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
            return []
    except (OSError, subprocess.SubprocessError):
        return []
    else:
        cards: list[GraphicsCard] = []
        for row in csv.reader(io.StringIO(output), skipinitialspace=True):
            if len(row) != 5:
                return []
            index, name, driver, capability, memory = (value.strip() for value in row)
            if (
                not index.isdecimal()
                or not name
                or not re.fullmatch(r"\d+(?:\.\d+)+", driver)
                or not re.fullmatch(r"\d+\.\d+", capability)
                or version(capability) == (0, 0)
                or not re.fullmatch(r"[1-9]\d* MiB", memory)
                or any(card["id"] == index for card in cards)
            ):
                return []
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
        rows = json.loads(output or "[]")
        rows = rows if isinstance(rows, list) else [rows]
        return [
            {
                "id": row["PNPDeviceID"],
                "name": row["Name"],
                "driver": None,
                "computeCapability": None,
                "memoryMiB": None,
            }
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("Name"), str)
            and isinstance(row.get("PNPDeviceID"), str)
            and "nvidia" in f"{row.get('AdapterCompatibility')} {row['Name']}".lower()
        ]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


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
    """Prefer the newest compatible GPU base; offer CPU only without a supported card."""
    os_name = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform)
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64"}.get(platform.machine().lower())
    native = {
        name: base
        for name, base in manifests.items()
        if base["platform"] == {"os": os_name, "arch": arch}
    }
    gpu = [name for name, base in native.items() if supported_cards(base, cards)]
    if gpu:
        return sorted(
            gpu,
            key=lambda name: version(
                cast("dict[str, str]", native[name]["nvidia"])["minimumComputeCapability"]
            ),
            reverse=True,
        )
    return sorted(name for name, base in native.items() if base["nvidia"] is None)


def build_label(variant: str) -> str:
    return {
        "win-nvidia": "Windows with NVIDIA graphics",
        "win-nvidia-cu126": "Windows with NVIDIA graphics (CUDA 12.6)",
        "linux-nvidia": "Linux with NVIDIA graphics",
        "mac-arm64": "Mac with Apple Silicon",
        "win-cpu": "Windows with CPU only",
        "linux-cpu": "Linux with CPU only",
    }.get(variant, variant)
