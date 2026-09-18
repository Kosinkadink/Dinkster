"""Host accelerator detection: which ``[pack.extra-requires]`` key applies.

The vocabulary is :data:`~dinkster_workers.manifest.KNOWN_ACCELERATORS`
(``cpu``/``cuda``/``mps``/``rocm``/``xpu``). Detection is a cheap,
side-effect-free host inspection - driver files and vendor tools, never a
torch import and never a CUDA context (an idle GPU-touched process holds
~300-450 MiB VRAM; detection must cost zero). It answers "what wheels
would work here", not "what is torch using right now".

Explicit selection always wins: detection is only the ``auto`` fallback.
A server/CLI/desktop layer passes ``--accelerator`` (or
``$DINKSTER_ACCELERATOR``) through :func:`resolve_accelerator`; provisioning
then installs the matching ``[pack.extra-requires]`` list next to a
pack's base ``requires``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .manifest import KNOWN_ACCELERATORS

ACCELERATOR_ENV = "DINKSTER_ACCELERATOR"
"""Environment override read by :func:`resolve_accelerator` when the
selection is ``auto`` - how a supervisor or desktop shell pins the
accelerator for every child without threading a flag everywhere."""


class AcceleratorError(Exception):
    """An accelerator selection that is not in the known vocabulary."""


def detect_accelerator(
    *,
    sys_platform: str = sys.platform,
    machine: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.exists,
) -> str:
    """Best-effort host accelerator, never raising and never touching a
    GPU. The injectable seams (``which``/``exists``/platform facts) are
    for tests - CI has no GPUs and detection must not depend on them.

    Order matters only where families could coexist: vendor driver
    evidence (NVIDIA, then AMD, then Intel) beats the CPU fallback, and
    Apple Silicon is MPS by construction. Wrong detection is never
    destructive - the user overrides with an explicit selection and every
    plan prints which accelerator it resolved."""
    if sys_platform == "darwin":
        arch = machine if machine is not None else _host_machine()
        return "mps" if arch == "arm64" else "cpu"
    if (
        exists("/proc/driver/nvidia/version")
        or which("nvidia-smi") is not None
        or exists(r"C:\Windows\System32\nvml.dll")
    ):
        return "cuda"
    if (
        exists("/sys/module/amdgpu")
        or exists("/opt/rocm")
        or which("rocm-smi") is not None
        or which("rocminfo") is not None
    ):
        return "rocm"
    if which("xpu-smi") is not None or exists("/sys/module/intel_vsec"):
        return "xpu"
    return "cpu"


def detect_runtime(
    accelerator: str,
    *,
    run: Callable[[list[str]], str | None] | None = None,
    read_text: Callable[[str], str | None] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Advisory runtime/toolchain facts for the selected accelerator, as
    sorted (key, value) pairs - what a snapshot records as additional
    scope. Best-effort and never raising: unknown/undetectable = ().

    Same zero-VRAM stance as :func:`detect_accelerator`: never a torch
    import, never a CUDA/ROCm context in THIS process. The one subprocess
    it may run is the vendor's own status tool (``nvidia-smi``), which
    reports via the driver without creating a compute context here.
    Facts are advisory scope only - restore may narrate a difference,
    it never changes pin reuse."""
    runner = run if run is not None else _run_tool
    reader = read_text if read_text is not None else _read_file
    facts: dict[str, str] = {}
    if accelerator == "cuda":
        output = runner(["nvidia-smi"])
        if output:
            driver = re.search(r"Driver Version:\s*([\w.]+)", output)
            if driver:
                facts["driver"] = driver.group(1)
            cuda = re.search(r"CUDA Version:\s*([\w.]+)", output)
            if cuda:
                facts["cuda"] = cuda.group(1)
    elif accelerator == "rocm":
        version = reader("/opt/rocm/.info/version")
        if version:
            facts["rocm"] = version.strip()
    return tuple(sorted(facts.items()))


def _run_tool(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _read_file(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def resolve_accelerator(selection: str = "auto", *, environ: dict[str, str] | None = None) -> str:
    """Turn a user selection into a concrete accelerator: an explicit
    known key is authoritative, ``auto`` consults ``$DINKSTER_ACCELERATOR``
    then detection. Unknown keys fail loudly - a typo'd accelerator would
    otherwise silently select the wrong dependency set."""
    env = environ if environ is not None else dict(os.environ)
    if selection == "auto":
        selection = env.get(ACCELERATOR_ENV, "").strip() or "auto"
    if selection == "auto":
        return detect_accelerator()
    if selection not in KNOWN_ACCELERATORS:
        raise AcceleratorError(
            f"unknown accelerator {selection!r}; expected auto or one of "
            f"{', '.join(KNOWN_ACCELERATORS)}"
        )
    return selection


def _host_machine() -> str:
    import platform

    return platform.machine()


__all__ = [
    "ACCELERATOR_ENV",
    "AcceleratorError",
    "detect_accelerator",
    "detect_runtime",
    "resolve_accelerator",
]
