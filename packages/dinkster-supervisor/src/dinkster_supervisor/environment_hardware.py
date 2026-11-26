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
from typing import Any, NotRequired, TypedDict, cast


class GraphicsCard(TypedDict):
    id: str
    name: str
    driver: str | None
    computeCapability: str | None
    memoryMiB: int | None
    vendor: NotRequired[str]
    detectionError: NotRequired[str]
    detectionTool: NotRequired[str]
    pciAddress: NotRequired[str]


def hardware_command(arguments: list[str]) -> str:
    return subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=5).stdout


def pci_address(value: object) -> str:
    match = re.fullmatch(r"([0-9a-f]{4,8}):([0-9a-f]{2}):([0-9a-f]{2}\.[0-7])", str(value).lower())
    if match is None:
        raise ValueError("Invalid PCI address")
    return f"{int(match[1], 16):04x}:{match[2]}:{match[3]}"


def card_inventory() -> list[GraphicsCard] | None:
    vendors = {"10de": "nvidia", "1002": "amd", "8086": "intel"}
    cards: list[GraphicsCard] = []
    try:
        if sys.platform == "linux":
            devices = Path("/sys/bus/pci/devices")
            if not devices.is_dir():
                return None
            for device in sorted(devices.iterdir()):
                vendor = vendors.get((device / "vendor").read_text().strip().removeprefix("0x"))
                if vendor and (device / "class").read_text().startswith(("0x03", "0x12")):
                    cards.append(
                        {
                            "id": device.name,
                            "name": "",
                            "vendor": vendor,
                            "pciAddress": pci_address(device.name),
                            "driver": None,
                            "computeCapability": None,
                            "memoryMiB": None,
                        }
                    )
            return cards
        output = hardware_command(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object PNPDeviceID,Name | ConvertTo-Json -Compress",
            ]
        )
        parsed: object = json.loads(output or "[]")
        rows = cast("list[object]", parsed) if isinstance(parsed, list) else [parsed]
        for value in rows:
            if not isinstance(value, dict):
                continue
            row = cast("dict[str, object]", value)
            name, identifier = row.get("Name"), row.get("PNPDeviceID")
            if not isinstance(identifier, str):
                continue
            match = re.search(r"VEN_([0-9A-F]{4})", identifier.upper())
            if match and (vendor := vendors.get(match[1].lower())):
                cards.append(
                    {
                        "id": identifier,
                        "name": name if isinstance(name, str) else "",
                        "vendor": vendor,
                        "driver": None,
                        "computeCapability": None,
                        "memoryMiB": None,
                    }
                )
        return cards
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def vendor_cards(vendor: str, tool: str) -> list[GraphicsCard]:
    cards: list[GraphicsCard] = []
    if vendor == "nvidia":
        output = hardware_command(
            [
                tool,
                "--query-gpu=index,name,driver_version,compute_cap,memory.total,pci.bus_id",
                "--format=csv,noheader",
            ]
        )
        for row in csv.reader(io.StringIO(output), skipinitialspace=True):
            if len(row) != 6:
                raise ValueError("Incomplete NVIDIA card facts")
            index, name, driver, capability, memory, address = (value.strip() for value in row)
            if (
                not re.fullmatch(r"\d+\.\d+", capability)
                or version(capability) == (0,)
                or not re.fullmatch(r"[1-9]\d* MiB", memory)
            ):
                raise ValueError("Invalid NVIDIA card facts")
            cards.append(
                {
                    "id": index,
                    "name": name,
                    "vendor": vendor,
                    "driver": driver,
                    "computeCapability": capability,
                    "memoryMiB": int(memory.removesuffix(" MiB")),
                    "pciAddress": pci_address(address),
                }
            )
    else:
        arguments = [tool, "static", "--asic", "--bus", "--driver", "--vram", "--json"]
        if vendor == "intel":
            arguments = [tool, "discovery", "-j"]
        parsed: Any = json.loads(hardware_command(arguments))
        rows: Any
        if vendor == "intel":
            rows = parsed["device_list"]
        else:
            rows = (
                cast("dict[str, Any]", parsed).get("gpu_data", parsed)
                if isinstance(parsed, dict)
                else parsed
            )
            if isinstance(rows, dict):
                rows = [rows]
        if not isinstance(rows, list):
            raise ValueError("Invalid graphics-card facts")
        for row in cast("list[dict[str, Any]]", rows):
            if vendor == "intel":
                index = str(row["device_id"])
                row = json.loads(hardware_command([tool, "discovery", "-d", index, "-j"]))
                name, driver = row["device_name"], row["driver_version"]
                address = row["pci_bdf_address"]
                memory = row.get("memory_physical_size_byte")
                memory = int(str(memory)) // (1024 * 1024) if str(memory).isdecimal() else None
            else:
                index = str(row["gpu"])
                name, driver = row["asic"]["market_name"], row["driver"]["version"]
                address = row["bus"]["bdf"]
                memory = row.get("vram", {}).get("size")
                if isinstance(memory, dict):
                    size = cast("dict[str, Any]", memory)
                    memory = size["value"] if size.get("unit") == "MB" else None
                memory = int(str(memory)) if str(memory).isdecimal() else None
            if not isinstance(name, str) or not isinstance(driver, str):
                raise ValueError("Invalid graphics-card facts")
            cards.append(
                {
                    "id": index,
                    "name": name,
                    "vendor": vendor,
                    "driver": driver,
                    "computeCapability": None,
                    "memoryMiB": memory,
                    "pciAddress": pci_address(address),
                }
            )
    for card in cards:
        if (
            not card["id"].isdecimal()
            or not card["name"]
            or not isinstance(card["driver"], str)
            or not re.fullmatch(r"\d+(?:\.\d+)*", card["driver"])
            or sum(other["id"] == card["id"] for other in cards) != 1
        ):
            raise ValueError("Invalid graphics-card facts")
    return cards


def detect_graphics_cards() -> list[GraphicsCard] | None:
    """Inspect hardware without importing GPU runtimes or acquiring software."""
    if sys.platform == "darwin":
        return []
    inventory = card_inventory()
    cards: list[GraphicsCard] = []
    for vendor, filename in [("nvidia", "nvidia-smi"), ("amd", "amd-smi"), ("intel", "xpu-smi")]:
        found = [card for card in inventory or [] if card.get("vendor") == vendor]
        # WSL may expose NVIDIA through its tool but not through PCI sysfs.
        if inventory is not None and not found and vendor != "nvidia":
            continue
        tool = filename
        if (
            vendor == "nvidia"
            and sys.platform == "linux"
            and shutil.which(tool) is None
            and Path("/usr/lib/wsl/lib/nvidia-smi").is_file()
        ):
            tool = "/usr/lib/wsl/lib/nvidia-smi"
        error = "detection-tool-failed"
        try:
            reported = vendor_cards(vendor, tool)
        except FileNotFoundError:
            reported, error = [], "detection-tool-missing"
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            reported = []
        cards.extend(reported)
        remaining = list(reported)
        for card in found:
            match = next(
                (
                    other
                    for other in remaining
                    if (
                        card.get("pciAddress") == other.get("pciAddress")
                        if "pciAddress" in card
                        else card["name"] == other["name"]
                    )
                ),
                remaining[0] if "pciAddress" not in card and remaining else None,
            )
            if match is not None:
                remaining.remove(match)
            else:
                cards.append({**card, "detectionError": error, "detectionTool": filename})
    return cards if cards or inventory is not None else None


def version(value: str) -> tuple[int, ...]:
    parts = [int(part) for part in value.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def graphics_vendor(base: dict[str, object]) -> str | None:
    return next(
        (vendor for vendor in ("nvidia", "amd", "intel") if base.get(vendor) is not None), None
    )


def supported_cards(base: dict[str, object], cards: list[GraphicsCard]) -> list[GraphicsCard]:
    vendor = graphics_vendor(base)
    if vendor is None:
        return []
    limits = cast("dict[str, str | None]", base[vendor])
    maximum = limits.get("maximumExclusiveComputeCapability")
    return [
        card
        for card in cards
        if card.get("vendor", "nvidia") == vendor
        and not card.get("detectionError")
        and (driver := card["driver"]) is not None
        and version(driver) >= version(cast("str", limits["minimumDriver"]))
        and (
            vendor != "nvidia"
            or (
                (capability := card["computeCapability"]) is not None
                and version(capability) >= version(cast("str", limits["minimumComputeCapability"]))
                and (maximum is None or version(capability) < version(maximum))
            )
        )
    ]


def native_bases(manifests: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    os_name = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform)
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64"}.get(platform.machine().lower())
    return {
        name: base
        for name, base in manifests.items()
        if base["platform"] == {"os": os_name, "arch": arch}
    }


def supported_builds(
    manifests: dict[str, dict[str, object]], cards: list[GraphicsCard]
) -> list[str]:
    """Prefer the newest compatible GPU base, with native CPU alternatives last."""
    native = native_bases(manifests)
    gpu = [name for name, base in native.items() if supported_cards(base, cards)]
    gpu.sort(
        key=lambda name: tuple(
            version(
                cast(
                    "dict[str, str]", native[name][cast("str", graphics_vendor(native[name]))]
                ).get(field, "0")
            )
            for field in [
                "minimumComputeCapability",
                "minimumDriver",
            ]
        ),
        reverse=True,
    )
    gpu.sort(
        key=lambda name: next(
            index
            for index, card in enumerate(cards)
            if card.get("vendor", "nvidia") == graphics_vendor(native[name])
        )
    )
    return gpu + [name for name, base in native.items() if graphics_vendor(base) is None]


def unusable_cards(
    manifests: dict[str, dict[str, object]], cards: list[GraphicsCard]
) -> list[dict[str, object]]:
    native = native_bases(manifests)
    result: list[dict[str, object]] = []
    for card in cards:
        vendor = card.get("vendor", "nvidia")
        bases = [base for base in native.values() if graphics_vendor(base) == vendor]
        if any(supported_cards(base, [card]) for base in bases):
            continue
        facts: dict[str, object] = {
            "id": card["id"],
            "name": card["name"] or None,
            "vendor": vendor,
        }
        if not bases:
            facts["reason"] = "no-published-build"
        elif error := card.get("detectionError"):
            facts.update(
                reason=error,
                tool=card.get(
                    "detectionTool",
                    {"nvidia": "nvidia-smi", "amd": "amd-smi", "intel": "xpu-smi"}[vendor],
                ),
            )
        elif card["driver"] is None or (vendor == "nvidia" and card["computeCapability"] is None):
            facts.update(
                reason="detection-tool-failed",
                tool={"nvidia": "nvidia-smi", "amd": "amd-smi", "intel": "xpu-smi"}[vendor],
            )
        else:
            limits = [cast("dict[str, str | None]", base[vendor]) for base in bases]
            if vendor == "nvidia":
                capability = version(cast("str", card["computeCapability"]))
                if all(
                    capability < version(cast("str", limit["minimumComputeCapability"]))
                    for limit in limits
                ):
                    facts["reason"] = "card-too-old"
                limits = [
                    limit
                    for limit in limits
                    if capability >= version(cast("str", limit["minimumComputeCapability"]))
                    and (
                        limit.get("maximumExclusiveComputeCapability") is None
                        or capability
                        < version(cast("str", limit["maximumExclusiveComputeCapability"]))
                    )
                ]
            if limits:
                minimum = min(
                    (cast("str", limit["minimumDriver"]) for limit in limits), key=version
                )
                facts.update(reason="driver-too-old", minimumDriver=minimum, driver=card["driver"])
            elif "reason" not in facts:
                facts["reason"] = "no-published-build"
        result.append(facts)
    return result


def build_reason(base: dict[str, object]) -> str:
    vendor = graphics_vendor(base)
    if vendor is None:
        if cast("dict[str, str]", base["platform"])["os"] == "macos":
            return "Uses the graphics built into this Mac."
        return "Does not use the graphics card and is much slower."
    limits = cast("dict[str, str | None]", base[vendor])
    minimum = ".".join(str(part) for part in version(cast("str", limits["minimumDriver"])))
    brand = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}[vendor]
    return f"Needs {brand} driver {minimum} or newer."


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
        gpu_default = bool(builds and graphics_vendor(manifests[builds[0]]) is not None)
        builds = [
            name
            for name in builds
            if graphics_vendor(manifests[name]) is None
            or set(assigned).issubset(
                card["id"] for card in supported_cards(manifests[name], cards)
            )
        ]
        if gpu_default and not any(graphics_vendor(manifests[name]) is not None for name in builds):
            builds = []
    if requested is not None and requested not in builds:
        raise ValueError("That build is not supported on this computer.")
    unusable = unusable_cards(manifests, cards)
    repairable = any(
        card["reason"] in {"driver-too-old", "detection-tool-missing", "detection-tool-failed"}
        for card in unusable
    )
    variant = requested or (
        builds[0]
        if builds and (graphics_vendor(manifests[builds[0]]) is not None or not repairable)
        else None
    )
    assignments = {
        name: [
            card["id"]
            for card in supported_cards(manifests[name], cards)
            if not assigned or card["id"] in assigned
        ]
        for name in builds
    }
    return {
        "detectedCards": cards,
        "unusableCards": unusable,
        "system": {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform),
        "choices": [
            {
                "variant": name,
                "label": manifests[name].get("displayName", build_label(name)),
                "reason": build_reason(manifests[name]),
                "gpus": assignments[name],
            }
            for name in builds
        ],
        "variant": variant,
        "buildLabel": manifests[variant].get("displayName", build_label(variant))
        if variant is not None
        else None,
        "buildReason": build_reason(manifests[variant]) if variant is not None else None,
        "gpus": assignments[variant] if variant is not None else [],
        "refusal": None
        if builds
        else {
            "code": "unsupported-hardware",
            "message": "No published build supports this computer.",
        },
    }
