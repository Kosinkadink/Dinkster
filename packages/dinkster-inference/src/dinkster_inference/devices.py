"""Devices and dtypes as data; precision as an explicit plan.

ComfyUI's model_management answers "what device, what dtype" through
module-global probing and ~40 policy functions consulted ad hoc
(reference: comfy/model_management.py @ b78cec87). Dinkster inverts that:
capabilities are observed once into a value, policy is a protocol that
turns capabilities into a PrecisionPlan, and everything downstream
consumes the plan. Nothing here imports torch - a dtype is a name plus
geometry, and mapping to a framework dtype is the executing backend's
job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DTypeKind = Literal["float", "int", "bool", "complex"]


@dataclass(frozen=True)
class DType:
    """A dtype as pure geometry: name, bit width, and kind.

    ``bits`` may be sub-byte (4-bit quant formats); byte sizing is done
    at the buffer level (TensorGeometry.nbytes), never per element.
    """

    name: str
    bits: int
    kind: DTypeKind = "float"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dtype name must not be empty")
        if self.bits <= 0:
            raise ValueError(f"dtype bits must be positive, got {self.bits}")


FLOAT64 = DType("float64", 64)
FLOAT32 = DType("float32", 32)
FLOAT16 = DType("float16", 16)
BFLOAT16 = DType("bfloat16", 16)
FLOAT8_E4M3 = DType("float8_e4m3fn", 8)
FLOAT8_E5M2 = DType("float8_e5m2", 8)
FLOAT8_E4M3FNUZ = DType("float8_e4m3fnuz", 8)
FLOAT8_E5M2FNUZ = DType("float8_e5m2fnuz", 8)
FLOAT8_E8M0 = DType("float8_e8m0", 8)
FLOAT6_E2M3 = DType("float6_e2m3", 6)
FLOAT6_E3M2 = DType("float6_e3m2", 6)
FLOAT4 = DType("float4_e2m1", 4)
NVFP4 = DType("nvfp4", 4)
COMPLEX64 = DType("complex64", 64, "complex")
INT8 = DType("int8", 8, "int")
INT16 = DType("int16", 16, "int")
INT32 = DType("int32", 32, "int")
INT64 = DType("int64", 64, "int")
UINT8 = DType("uint8", 8, "int")
UINT16 = DType("uint16", 16, "int")
UINT32 = DType("uint32", 32, "int")
UINT64 = DType("uint64", 64, "int")
BOOL = DType("bool", 8, "bool")


@dataclass(frozen=True)
class DeviceRef:
    """A compute device: kind plus optional index (``cuda:0``, ``cpu``)."""

    kind: str
    index: int | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("device kind must not be empty")
        if self.index is not None and self.index < 0:
            raise ValueError(f"device index must be >= 0, got {self.index}")

    def __str__(self) -> str:
        return self.kind if self.index is None else f"{self.kind}:{self.index}"

    def residency_key(self) -> str:
        """This device's residency class in dinkster-memory's vocabulary:
        ``ram`` for the CPU, ``vram:<device>`` for accelerators."""
        return "ram" if self.kind == "cpu" else f"vram:{self}"


CPU = DeviceRef("cpu")


@dataclass(frozen=True)
class DtypeSupport:
    """Whether weights may rest in a dtype and whether kernels run in it.

    ``storage`` without ``compute`` is exactly the reference's
    manual_cast regime: weights rest in the dtype and are cast to a
    computable dtype at use.
    """

    storage: bool
    compute: bool

    @property
    def manual_cast(self) -> bool:
        return self.storage and not self.compute


# should_use_fp16 @ b78cec87: the 10-series list matches lowercased
# device names; the 16-series list matches case-sensitively.
NVIDIA_10_SERIES = (
    "1080",
    "1070",
    "titan x",
    "p3000",
    "p3200",
    "p4000",
    "p4200",
    "p5000",
    "p5200",
    "p6000",
    "1060",
    "1050",
    "p40",
    "p100",
    "p6",
    "p4",
)
NVIDIA_16_SERIES = (
    "1660",
    "1650",
    "1630",
    "T500",
    "T550",
    "T600",
    "MX550",
    "MX450",
    "CMP 30HX",
    "T2000",
    "T1000",
    "T1200",
)


def nvidia_fp16_support(major: int, name: str, *, windows: bool) -> DtypeSupport:
    """should_use_fp16 @ b78cec87 for one NVIDIA CUDA device, from
    driver-reported facts (compute capability major, device name).

    The reference refuses fp16 outright on pre-Pascal cards, cast
    included; 10-series cards run fp16 kernels profitably only on
    Windows; the 16-series cards have broken fp16 kernels but still
    take fp16 storage with cast-at-use.
    """
    if major >= 8:
        return DtypeSupport(storage=True, compute=True)
    if major < 6:
        return DtypeSupport(storage=False, compute=False)
    lowered = name.lower()
    if any(x in lowered for x in NVIDIA_10_SERIES):
        return DtypeSupport(storage=True, compute=windows)
    if major < 7:
        return DtypeSupport(storage=True, compute=False)
    if any(x in name for x in NVIDIA_16_SERIES):
        return DtypeSupport(storage=True, compute=False)
    return DtypeSupport(storage=True, compute=True)


def nvidia_bf16_compute(major: int) -> bool:
    """should_use_bf16 @ b78cec87 native-compute answer for one NVIDIA
    CUDA device: only Ampere and newer compute in bf16. Storage on
    older cards depends on the torch build's cast support, which only
    the executing backend can observe."""
    return major >= 8


def nvidia_compute_dtypes(major: int, name: str, *, windows: bool) -> frozenset[str]:
    """Native compute dtype names for one NVIDIA CUDA device, per the
    reference gates. float32 is always computable."""
    names = {"float32"}
    if nvidia_fp16_support(major, name, windows=windows).compute:
        names.add("float16")
    if nvidia_bf16_compute(major):
        names.add("bfloat16")
    return frozenset(names)


@dataclass(frozen=True)
class DeviceCapabilities:
    """What one device can actually do - observed once, passed as a value.

    ``compute_dtypes`` are dtypes the device can run kernels in;
    ``storage_dtypes`` are dtypes weights may rest in (a superset when
    cast-at-use is available). ``total_memory`` is bytes, None when the
    backend cannot report it.
    """

    device: DeviceRef
    compute_dtypes: frozenset[DType]
    storage_dtypes: frozenset[DType]
    supports_non_blocking: bool = False
    supports_streams: bool = False
    total_memory: int | None = None

    def __post_init__(self) -> None:
        if self.total_memory is not None and self.total_memory < 0:
            raise ValueError("total_memory must be >= 0")


@dataclass(frozen=True)
class PrecisionPlan:
    """The resolved answer to "how do these weights live and compute".

    ``manual_cast`` means storage differs from compute and weights are
    cast at use (ComfyUI's manual_cast ops path, reference:
    comfy/ops.py @ b78cec87).
    """

    storage: DType
    compute: DType

    @property
    def manual_cast(self) -> bool:
        return self.storage != self.compute


@dataclass(frozen=True)
class PrecisionRequest:
    """The facts a precision decision consumes, as one value.

    ``parameter_count`` and ``storage_bytes`` are both needed: packed
    quantized checkpoints make bytes a poor proxy for compute size
    (upstream reasons from parameter counts for FP8 decisions,
    comfy/model_management.py @ b78cec87). Live free-memory state is
    deliberately absent - dynamic memory policy is the governor's job
    (dinkster-memory); ``capabilities.total_memory`` carries stable
    hardware capacity.
    """

    weights_dtype: DType
    supported: frozenset[DType]
    capabilities: DeviceCapabilities
    parameter_count: int
    storage_bytes: int

    def __post_init__(self) -> None:
        if self.parameter_count < 0 or self.storage_bytes < 0:
            raise ValueError("parameter_count and storage_bytes must be >= 0")


__all__ = [
    "BFLOAT16",
    "BOOL",
    "COMPLEX64",
    "CPU",
    "DType",
    "DTypeKind",
    "DeviceCapabilities",
    "DeviceRef",
    "DtypeSupport",
    "FLOAT4",
    "FLOAT6_E2M3",
    "FLOAT6_E3M2",
    "FLOAT8_E4M3",
    "FLOAT8_E4M3FNUZ",
    "FLOAT8_E5M2",
    "FLOAT8_E5M2FNUZ",
    "FLOAT8_E8M0",
    "FLOAT16",
    "FLOAT32",
    "FLOAT64",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "NVFP4",
    "NVIDIA_10_SERIES",
    "NVIDIA_16_SERIES",
    "PrecisionPlan",
    "PrecisionRequest",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "nvidia_bf16_compute",
    "nvidia_compute_dtypes",
    "nvidia_fp16_support",
]
