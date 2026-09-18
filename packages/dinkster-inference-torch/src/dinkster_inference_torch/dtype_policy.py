"""Per-backend float dtype capability facts.

should_use_fp16 and should_use_bf16 in the reference ComfyUI
(comfy/model_management.py @ b78cec87) fold three questions into one
boolean: whether kernels run natively in the dtype, whether weights may
rest in it and be cast at use (the manual_cast regime), and whether
live memory pressure justifies that cast. Dinkster records the first two
as observed facts (:class:`DtypeSupport`) and leaves memory pressure to
the residency governor. The reference's CLI-flag and exotic-backend
branches (force_fp16/fp32, directml, npu, mlu, ixuca) have no Dinkster
equivalent; unknown device types raise loudly instead of guessing.

Native fp8 matrix multiplication is a separate fact answered by
:func:`~.quant_linear.supports_fp8_matmul`, which consumes the AMD
architecture gate defined here.
"""

from __future__ import annotations

import os
import platform
import re

import torch
from dinkster_inference.devices import (
    NVIDIA_10_SERIES,
    NVIDIA_16_SERIES,
    DtypeSupport,
    nvidia_bf16_compute,
    nvidia_fp16_support,
)

# AMD_RDNA2_AND_OLDER_ARCH @ b78cec87: no native bf16 compute, but
# weights may rest in bf16 and cast at use.
AMD_RDNA2_AND_OLDER_ARCH = (
    "gfx1030",
    "gfx1031",
    "gfx1035",
    "gfx1010",
    "gfx1011",
    "gfx1012",
    "gfx906",
    "gfx900",
    "gfx803",
)
AMD_ENABLE_MIOPEN_ENV = "COMFYUI_ENABLE_MIOPEN"

# SUPPORT_FP8_OPS AMD auto-enable @ b78cec87: the only architectures
# with native fp8 matmul, and only on torch >= 2.7 with ROCm >= 6.4.
AMD_FP8_MATMUL_ARCH = ("gfx1200", "gfx1201", "gfx950")


def torch_version_numeric() -> tuple[int, int]:
    """The (major, minor) pair used by upstream's version gates."""
    values: list[int] = []
    for part in str(torch.__version__).split("+", 1)[0].split(".")[:2]:
        match = re.match(r"\d+", part)
        values.append(0 if match is None else int(match.group()))
    while len(values) < 2:
        values.append(0)
    return values[0], values[1]


def rocm_version_numeric() -> tuple[int, int]:
    """The (major, minor) pair of the HIP runtime, mirroring upstream's
    parse; (6, -1) on non-ROCm builds or unparseable versions."""
    parts = str(getattr(torch.version, "hip", None)).split(".")[:2]
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (6, -1)


def amd_gfx_arch(device: torch.device) -> str | None:
    """The gfx architecture of an AMD device on a ROCm torch build
    (``gfx1100``), None for every other build or device."""
    if device.type != "cuda" or not getattr(torch.version, "hip", None):
        return None
    name = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", "") or "")
    arch = name.split(":")[0].strip()
    return arch or None


def configure_amd_miopen() -> bool:
    """Disable MIOpen on newer AMD architectures unless explicitly enabled."""
    arch = amd_gfx_arch(torch.device("cuda"))
    if (
        arch is None
        or re.fullmatch(r"gfx[0-9a-f]+", arch) is None
        or any(a in arch for a in AMD_RDNA2_AND_OLDER_ARCH)
        or os.getenv(AMD_ENABLE_MIOPEN_ENV) == "1"
    ):
        return False
    torch.backends.cudnn.enabled = False
    return True


def amd_fp8_matmul_supported(device: torch.device) -> bool:
    """SUPPORT_FP8_OPS AMD auto-enable @ b78cec87 for one device."""
    arch = amd_gfx_arch(device)
    if arch is None or not any(a in arch for a in AMD_FP8_MATMUL_ARCH):
        return False
    return torch_version_numeric() >= (2, 7) and rocm_version_numeric() >= (6, 4)


def fp16_support(device: torch.device) -> DtypeSupport:
    """should_use_fp16 @ b78cec87 as capability facts.

    The reference refuses fp16 outright on CPU and on pre-Pascal
    NVIDIA cards, cast included; XPU answers with the device's fp16
    capability flag for both regimes; 10-series NVIDIA cards run fp16
    kernels profitably only on Windows; the 16-series cards have
    broken fp16 kernels but still take fp16 storage with cast-at-use.
    """
    if device.type == "cpu":
        return DtypeSupport(storage=False, compute=False)
    if device.type == "mps":
        return DtypeSupport(storage=True, compute=True)
    if device.type == "xpu":
        native = bool(torch.xpu.get_device_properties(device).has_fp16)
        return DtypeSupport(storage=native, compute=native)
    if device.type != "cuda":
        raise ValueError(f"no fp16 dtype policy for device type {device.type!r}")
    if getattr(torch.version, "hip", None):
        return DtypeSupport(storage=True, compute=True)
    props = torch.cuda.get_device_properties(device)
    return nvidia_fp16_support(props.major, props.name, windows=any(platform.win32_ver()))


def _mac_release_major() -> int:
    release = platform.mac_ver()[0]
    match = re.match(r"\d+", release)
    return 0 if match is None else int(match.group())


def _mac_version() -> tuple[int, ...] | None:
    """mac_version @ b78cec87: the macOS release as an int tuple, None elsewhere."""
    release = platform.mac_ver()[0]
    if not release:
        return None
    try:
        return tuple(int(part) for part in release.split("."))
    except ValueError:
        return None


def force_fp16_attention_upcast() -> bool:
    """force_upcast_attention_dtype @ b78cec87 as a capability fact.

    FP16 attention math produces black images on macOS 14.5 and later,
    so the reference forces it to float32 there. The reference's CLI
    force flag has no Dinkster equivalent.
    """
    version = _mac_version()
    return version is not None and version >= (14, 5)


def bf16_support(device: torch.device) -> DtypeSupport:
    """should_use_bf16 @ b78cec87 as capability facts.

    The reference refuses bf16 outright on CPU (kernels exist but are
    far too slow), and MPS has no bf16 dtype at all before macOS 14.
    XPU answers with torch.xpu.is_bf16_supported() for both regimes.
    RDNA2-and-older AMD architectures take bf16 storage with
    cast-at-use but have no native bf16 compute; newer AMD
    architectures answer through the same capability-major check as
    NVIDIA (gfx1100 reports major 11). Pre-Ampere NVIDIA cards never
    compute in bf16, and take bf16 storage only when the build can
    cast it (torch.cuda.is_bf16_supported).
    """
    if device.type == "cpu":
        return DtypeSupport(storage=False, compute=False)
    if device.type == "mps":
        supported = _mac_release_major() >= 14
        return DtypeSupport(storage=supported, compute=supported)
    if device.type == "xpu":
        supported = bool(torch.xpu.is_bf16_supported())
        return DtypeSupport(storage=supported, compute=supported)
    if device.type != "cuda":
        raise ValueError(f"no bf16 dtype policy for device type {device.type!r}")
    arch = amd_gfx_arch(device)
    if arch is not None and any(a in arch for a in AMD_RDNA2_AND_OLDER_ARCH):
        return DtypeSupport(storage=True, compute=False)
    props = torch.cuda.get_device_properties(device)
    if nvidia_bf16_compute(props.major):
        return DtypeSupport(storage=True, compute=True)
    return DtypeSupport(storage=bool(torch.cuda.is_bf16_supported()), compute=False)


__all__ = [
    "AMD_ENABLE_MIOPEN_ENV",
    "AMD_FP8_MATMUL_ARCH",
    "AMD_RDNA2_AND_OLDER_ARCH",
    "NVIDIA_10_SERIES",
    "NVIDIA_16_SERIES",
    "DtypeSupport",
    "amd_fp8_matmul_supported",
    "amd_gfx_arch",
    "bf16_support",
    "configure_amd_miopen",
    "force_fp16_attention_upcast",
    "fp16_support",
    "rocm_version_numeric",
    "torch_version_numeric",
]
