from __future__ import annotations

import platform
import sys
from pathlib import Path


def platform_golden_path(path: Path, torch_version: str) -> Path:
    if sys.platform.startswith("linux"):
        return path
    key = f"{sys.platform}-py{platform.python_version()}-torch{torch_version}"
    return path.with_name(f"{path.stem}.{key}{path.suffix}")


def cpu_identity() -> str:
    """CPU model of the host executing the generator.

    Executed-reference goldens that are asserted bit-exactly are pinned to
    the mint host's CPU class: conv, FFT, and vectorized libm kernels
    dispatch on microarchitecture and drift by ULPs across CPUs (same class
    as the GPU host drift in Dinkster issue #636). The test loader skips
    exact-equality suites when the live CPU differs from the recorded one.
    """
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown-cpu"


def tuple_provenance(torch_version: str, *, pin_cpu: bool = False) -> dict[str, str]:
    """Provenance keys the platform-tuple loader validates. The Linux base
    fixture stays free of provenance unless the generator opts into CPU
    pinning (pin_cpu=True, for fixtures whose executed kernels drift across
    CPU microarchitectures); other platforms record the full platform
    tuple."""
    if sys.platform.startswith("linux"):
        return {"cpu": cpu_identity()} if pin_cpu else {}
    return platform_provenance(torch_version, pin_cpu=pin_cpu)


def platform_provenance(torch_version: str, *, pin_cpu: bool = False) -> dict[str, str]:
    provenance: dict[str, str] = {}
    if pin_cpu:
        provenance["cpu"] = cpu_identity()
    provenance.update(
        {
            "python": f"Python {platform.python_version()}",
            "torch": torch_version,
        }
    )
    if not sys.platform.startswith("linux"):
        provenance.update({"os": platform.platform(), "platform": sys.platform})
    return provenance
