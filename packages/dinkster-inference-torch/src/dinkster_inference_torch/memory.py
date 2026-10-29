"""Device memory introspection and the reserve policy.

Ports of comfy/model_management.py get_free_memory / get_total_memory
/ minimum_inference_memory / soft_empty_cache
@ b78cec87, reshaped per docs/native-inference-plan.md 1.1: pure
functions over an explicit device plus an injected ``MemoryPolicy``
value, instead of module globals populated by import-time hardware
probing and CLI parsing.

The load-bearing formula is the reference's effective-free-memory
math: free memory is what the driver reports free PLUS the torch
allocator's reclaimable reserve (``reserved - active`` bytes torch
holds but is not using - an ``empty_cache`` away from being free).
Ignoring that term makes a warm process look pathologically out of
memory.

Device coverage matches what Dinkster can execute on today: cuda (NVIDIA
and ROCm, which exposes AMD GPUs through the CUDA namespace), xpu, cpu,
and mps. MPS intersects the shared system-memory pool with Metal's
recommended working-set headroom. XPU prefers the device's own
``torch.xpu.mem_get_info`` report and falls back to the reference's
allocator-derived free view when the running torch does not expose it.
Other accelerator backends (npu/mlu/directml) are a ROADMAP item;
asking about them raises loudly instead of guessing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import torch
from dinkster_inference import GIBIBYTE, MEBIBYTE, ModelFamily
from dinkster_memory import (
    DEFAULT_ACCELERATOR_HEADROOM_BYTES,
    DEFAULT_INFERENCE_RESERVE_BYTES,
    AcceleratorMemoryPolicy,
    ResolvedAcceleratorMemoryPolicy,
    SystemMemorySnapshot,
    system_memory_snapshot,
)

from .dtype_policy import fp16_support

MiB = MEBIBYTE
GiB = GIBIBYTE


@dataclass(frozen=True)
class DeviceMemory:
    """One free-memory measurement (get_free_memory @ b78cec87).

    ``free_total`` includes ``free_torch``: driver-free bytes plus the
    torch allocator's reclaimable reserve. ``free_torch`` alone is the
    reserve - what ``soft_empty_cache`` would hand back to the driver.
    XPU samples retain the allocator's full reserved byte count so a
    configured hard budget can account for live and cached allocations.
    Policy projections retain any live allocator bytes above that budget
    as debt so an already-loaded mechanism can be reduced below the cap.
    """

    free_total: int
    free_torch: int
    allocator_reserved_bytes: int | None = field(default=None, repr=False, compare=False)
    allocator_budget_debt_bytes: int | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CudaMemorySnapshot:
    """One attributed CUDA memory sample.

    Driver free, torch allocator reserve, and evictable VBAR pages are
    disjoint reclaimable tiers. Pinned VBAR pages remain attribution only.
    """

    total_bytes: int
    driver_free_bytes: int
    allocator_reclaimable_bytes: int
    dynamic_evictable_bytes: int = 0
    dynamic_pinned_bytes: int = 0

    @property
    def free_bytes(self) -> int:
        return min(
            self.total_bytes,
            self.driver_free_bytes
            + self.allocator_reclaimable_bytes
            + self.dynamic_evictable_bytes,
        )

    def device_memory(self) -> DeviceMemory:
        return DeviceMemory(
            free_total=self.free_bytes,
            free_torch=self.allocator_reclaimable_bytes,
        )


def cuda_memory_snapshot(device: torch.device) -> CudaMemorySnapshot:
    """Read one fresh driver and torch-allocator CUDA snapshot."""
    if device.type != "cuda":
        raise ValueError(f"CUDA memory snapshot requires a CUDA device, got {device.type!r}")
    stats = torch.cuda.memory_stats_as_nested_dict(device)
    try:
        active = stats["active_bytes"]["all"]["current"]
    except (KeyError, TypeError):
        mem_active = 0
    else:
        mem_active = int(active)
    try:
        reserved = stats["reserved_bytes"]["all"]["current"]
    except (KeyError, TypeError):
        mem_reserved = 0
    else:
        mem_reserved = int(reserved)
    mem_free_backend, mem_total = torch.cuda.mem_get_info(device)
    return CudaMemorySnapshot(
        total_bytes=int(mem_total),
        driver_free_bytes=int(mem_free_backend),
        allocator_reclaimable_bytes=mem_reserved - mem_active,
    )


@dataclass(frozen=True)
class XpuMemorySnapshot:
    """One XPU memory sample.

    ``driver_reported`` records provenance: True when the driver's own
    ``torch.xpu.mem_get_info`` report supplied ``driver_free_bytes``,
    False when it was derived from allocator reserve alone (the
    reference's XPU view, which cannot see other processes' usage).
    """

    total_bytes: int
    driver_free_bytes: int
    allocator_reclaimable_bytes: int
    driver_reported: bool
    allocator_reserved_bytes: int | None = None

    @property
    def free_bytes(self) -> int:
        return min(self.total_bytes, self.driver_free_bytes + self.allocator_reclaimable_bytes)

    def device_memory(self) -> DeviceMemory:
        return DeviceMemory(
            free_total=self.driver_free_bytes + self.allocator_reclaimable_bytes,
            free_torch=self.allocator_reclaimable_bytes,
            allocator_reserved_bytes=self.allocator_reserved_bytes,
        )


def xpu_memory_snapshot(device: torch.device) -> XpuMemorySnapshot:
    """Read one fresh XPU memory snapshot through ``torch.xpu``.

    Total is always the device property, matching get_total_memory's XPU
    arithmetic @ b78cec87. Free prefers ``torch.xpu.mem_get_info`` (the
    device's own report); when the running torch does not expose it,
    falls back to get_free_memory's XPU arithmetic @ b78cec87 - device
    total minus allocator reserve - which overstates free memory when
    another process occupies the GPU. A fresh XPU allocator can return
    an empty stats mapping, so its direct counters supply the zero state.
    Missing stats on an unavailable runtime still raise rather than
    silently report the whole device free.
    """
    if device.type != "xpu":
        raise ValueError(f"XPU memory snapshot requires an XPU device, got {device.type!r}")
    stats = torch.xpu.memory_stats(device)
    try:
        mem_active = int(stats["active_bytes.all.current"])
        mem_reserved = int(stats["reserved_bytes.all.current"])
    except KeyError:
        if not torch.xpu.is_available():
            raise
        mem_active = int(torch.xpu.memory_allocated(device))
        mem_reserved = int(torch.xpu.memory_reserved(device))
    reclaimable = mem_reserved - mem_active
    mem_total = int(torch.xpu.get_device_properties(device).total_memory)
    mem_get_info = getattr(torch.xpu, "mem_get_info", None)
    if callable(mem_get_info):
        mem_free_backend, _ = cast("tuple[int, int]", mem_get_info(device))
        return XpuMemorySnapshot(
            total_bytes=mem_total,
            driver_free_bytes=int(mem_free_backend),
            allocator_reclaimable_bytes=reclaimable,
            driver_reported=True,
            allocator_reserved_bytes=mem_reserved,
        )
    return XpuMemorySnapshot(
        total_bytes=mem_total,
        driver_free_bytes=max(0, mem_total - mem_reserved),
        allocator_reclaimable_bytes=reclaimable,
        driver_reported=False,
        allocator_reserved_bytes=mem_reserved,
    )


def _mps_memory_status() -> tuple[int, int, int]:
    return (
        int(torch.mps.recommended_max_memory()),
        int(torch.mps.driver_allocated_memory()),
        int(torch.mps.current_allocated_memory()),
    )


def _validated_mps_memory_status(
    query: Callable[[], tuple[int, int, int]],
) -> tuple[int, int, int]:
    recommended, driver_allocated, current_allocated = query()
    if recommended <= 0 or driver_allocated < 0 or current_allocated < 0:
        raise RuntimeError("MPS allocator returned invalid memory values")
    return recommended, driver_allocated, current_allocated


@dataclass(frozen=True)
class MpsMemorySnapshot:
    """One attributed MPS unified-memory sample.

    MPS shares physical memory with the whole system, so its budget is
    the intersection of two clamps: Metal's recommended working-set
    headroom (what the driver will let the process wire) and the
    container-aware system availability (what the machine actually has
    left). Metal's driver-allocated residual includes live MPS/MPSGraph
    allocations, so no portion is provably reclaimable by
    ``soft_empty_cache``; ``current_allocated_bytes`` (live tensor
    bytes) is attribution only.
    """

    recommended_max_bytes: int
    driver_allocated_bytes: int
    current_allocated_bytes: int
    system_total_bytes: int
    system_available_bytes: int

    @property
    def metal_headroom_bytes(self) -> int:
        return max(0, self.recommended_max_bytes - self.driver_allocated_bytes)

    @property
    def free_bytes(self) -> int:
        return min(self.system_available_bytes, self.metal_headroom_bytes)

    @property
    def total_bytes(self) -> int:
        return min(self.system_total_bytes, self.recommended_max_bytes)

    def device_memory(self) -> DeviceMemory:
        return DeviceMemory(free_total=self.free_bytes, free_torch=0)

    def describe(self) -> str:
        """One-line budget provenance for refusal messages and logs."""
        return (
            f"Metal working-set budget {self.recommended_max_bytes} bytes"
            f" with {self.driver_allocated_bytes} bytes driver-allocated"
            f" ({self.current_allocated_bytes} bytes in live tensors)"
            f" leaves {self.metal_headroom_bytes} bytes of Metal headroom;"
            f" system has {self.system_available_bytes} bytes available"
            f" of {self.system_total_bytes} bytes total"
        )


def mps_memory_snapshot(
    device: torch.device,
    *,
    system_memory: Callable[[], SystemMemorySnapshot] = system_memory_snapshot,
    mps_memory: Callable[[], tuple[int, int, int]] = _mps_memory_status,
) -> MpsMemorySnapshot:
    """Read one fresh Metal-allocator and system-memory MPS snapshot."""
    if device.type != "mps":
        raise ValueError(f"MPS memory snapshot requires an MPS device, got {device.type!r}")
    recommended, driver_allocated, current_allocated = _validated_mps_memory_status(mps_memory)
    system = system_memory()
    return MpsMemorySnapshot(
        recommended_max_bytes=recommended,
        driver_allocated_bytes=driver_allocated,
        current_allocated_bytes=current_allocated,
        system_total_bytes=system.effective_total_bytes,
        system_available_bytes=system.effective_available_bytes,
    )


def lora_compute_dtype(device: torch.device) -> torch.dtype:
    """lora_compute_dtype @ b78cec87: the dtype LoRA weight patching
    computes in - float16 where fp16 kernels run natively, float32
    otherwise.

    XPU deviates from the reference deliberately: upstream patches in
    fp16 whenever the device reports fp16 kernels, but runtime fp16
    patching on XPU has produced NaN output where an offline CPU
    float32 merge of the same LoRA works (ComfyUI issue #14720), so
    XPU patches compute in float32 until real-device evidence proves
    fp16 safe. The reference's per-device memo cache is dropped: this
    is a pure function of the device's stable properties.
    """
    if device.type == "xpu":
        return torch.float32
    return torch.float16 if fp16_support(device).compute else torch.float32


def get_free_memory(
    device: torch.device,
    *,
    system_memory: Callable[[], SystemMemorySnapshot] = system_memory_snapshot,
    mps_memory: Callable[[], tuple[int, int, int]] = _mps_memory_status,
    cuda_memory: Callable[[torch.device], CudaMemorySnapshot] = cuda_memory_snapshot,
    xpu_memory: Callable[[torch.device], XpuMemorySnapshot] = xpu_memory_snapshot,
) -> DeviceMemory:
    """get_free_memory(device, torch_free_too=True) @ b78cec87.

    MPS deliberately tightens the reference's system-shared view with
    Metal working-set headroom while retaining the container-aware system
    clamp.
    """
    if device.type == "cpu":
        available = system_memory().effective_available_bytes
        return DeviceMemory(free_total=available, free_torch=available)
    if device.type == "mps":
        return mps_memory_snapshot(
            device, system_memory=system_memory, mps_memory=mps_memory
        ).device_memory()
    if device.type == "cuda":
        memory = cuda_memory(device)
        free_torch = memory.allocator_reclaimable_bytes
        # Keep the pinned ComfyUI projection byte-identical. The structured
        # dynamic path separately caps its derived available value to total.
        return DeviceMemory(
            free_total=memory.driver_free_bytes + free_torch,
            free_torch=free_torch,
        )
    if device.type == "xpu":
        return xpu_memory(device).device_memory()
    raise ValueError(
        f"no memory introspection for device type {device.type!r}"
        " (cuda/cpu/mps/xpu only; other backends are a ROADMAP item)"
    )


def get_total_memory(
    device: torch.device,
    *,
    system_memory: Callable[[], SystemMemorySnapshot] = system_memory_snapshot,
    mps_memory: Callable[[], tuple[int, int, int]] = _mps_memory_status,
    xpu_memory: Callable[[torch.device], XpuMemorySnapshot] = xpu_memory_snapshot,
) -> int:
    """get_total_memory @ b78cec87, with the same MPS deviation as
    ``get_free_memory``."""
    if device.type == "cpu":
        return system_memory().effective_total_bytes
    if device.type == "mps":
        return mps_memory_snapshot(
            device, system_memory=system_memory, mps_memory=mps_memory
        ).total_bytes
    if device.type == "cuda":
        _, mem_total = torch.cuda.mem_get_info(device)
        return int(mem_total)
    if device.type == "xpu":
        return xpu_memory(device).total_bytes
    raise ValueError(
        f"no memory introspection for device type {device.type!r}"
        " (cuda/cpu/mps/xpu only; other backends are a ROADMAP item)"
    )


def regional_working_memory(
    family: ModelFamily,
    *,
    batch: int,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> float:
    """Pinned accelerated-attention BaseModel memory estimate.

    This is intentionally limited to the five native families supported by
    the regional conditioning executor. The formula is spatial batch area times
    compute dtype bytes times 0.01 MiB times the exact family factor.
    """

    if family.engine.regional_memory_factor is None:
        raise ValueError(f"unsupported regional memory family {family.id!r}")
    factor = family.engine.regional_memory_factor
    if any(type(value) is not int or value < 1 for value in (batch, height, width)):
        raise ValueError("regional memory dimensions must be positive integers")
    if not dtype.is_floating_point:
        raise ValueError("regional memory dtype must be floating point")
    element_size = torch.empty((), dtype=dtype).element_size()
    return batch * height * width * element_size * 0.01 * MiB * factor


def soft_empty_cache(device: torch.device) -> None:
    """soft_empty_cache @ b78cec87: return the allocator's reclaimable
    reserve to the driver. A no-op off-accelerator."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@dataclass(frozen=True)
class MemoryPolicy:
    """One injected accelerator policy for classic and dynamic residency.

    - ``inference_reserve``: the 0.8 GiB working-memory floor of
      minimum_inference_memory().
    - ``physical_headroom``: the operator-visible physical free floor.
      Dinkster deliberately uses its visible 256 MiB server default instead
      of ComfyUI @ b78cec87's hidden 400 MiB EXTRA_RESERVED_VRAM default.
      ComfyUI uses 600-700 MiB on Windows; WDDM operators who need that
      larger cushion raise the same visible setting.
    - ``hard_budgets``: optional per-device operator caps (``cuda:N`` or
      ``xpu:N``). Residency subtracts the physical bytes outside a cap
      from measured free memory.
    - ``min_weight_memory_ratio``: optional MIN_WEIGHT_MEMORY_RATIO
      override. The default is selected per supported accelerator:
      0.0 on NVIDIA CUDA and 0.4 on ROCm, XPU, and MPS.
    - ``load_inflation``: the 1.1 safety factor load_models_gpu
      applies to the bytes it frees ahead of a load.
    """

    inference_reserve: int = DEFAULT_INFERENCE_RESERVE_BYTES
    physical_headroom: int = DEFAULT_ACCELERATOR_HEADROOM_BYTES
    hard_budgets: Mapping[str, int] = field(default_factory=dict)
    min_weight_memory_ratio: float | None = None
    load_inflation: float = 1.1

    def __post_init__(self) -> None:
        AcceleratorMemoryPolicy(
            physical_headroom_bytes=self.physical_headroom,
            inference_reserve_bytes=self.inference_reserve,
        )
        budgets: dict[str, int] = {}
        for device, value in self.hard_budgets.items():
            family, _, index = device.partition(":")
            if family not in ("cuda", "xpu") or not index.isdigit():
                raise ValueError(f"hard budget device must be cuda:N or xpu:N, got {device!r}")
            if type(value) is not int or value < 0:
                raise ValueError("hard accelerator budgets must be non-negative integers")
            budgets[device] = value
        object.__setattr__(self, "hard_budgets", MappingProxyType(budgets))

    def minimum_inference_memory(self) -> int:
        """Inference working reserve plus physical accelerator headroom."""
        return self.inference_reserve + self.physical_headroom

    def weight_memory_ratio(self, device: torch.device, *, rocm: bool = False) -> float:
        if self.min_weight_memory_ratio is not None:
            return self.min_weight_memory_ratio
        if device.type == "cuda":
            return 0.4 if rocm else 0.0
        if device.type in ("xpu", "mps"):
            return 0.4
        return 0.0

    def hard_budget(self, device: torch.device) -> int | None:
        if device.type not in ("cuda", "xpu"):
            return None
        index = cast("int | None", device.index)
        if index is None:
            namespace = torch.cuda if device.type == "cuda" else torch.xpu
            index = namespace.current_device()
        return self.hard_budgets.get(f"{device.type}:{index}")

    def resolve(
        self,
        device: torch.device,
        total_bytes: int,
    ) -> ResolvedAcceleratorMemoryPolicy:
        return AcceleratorMemoryPolicy(
            physical_headroom_bytes=self.physical_headroom,
            inference_reserve_bytes=self.inference_reserve,
        ).resolve(total_bytes, self.hard_budget(device))


__all__ = [
    "CudaMemorySnapshot",
    "DeviceMemory",
    "GiB",
    "MemoryPolicy",
    "MiB",
    "MpsMemorySnapshot",
    "XpuMemorySnapshot",
    "cuda_memory_snapshot",
    "get_free_memory",
    "get_total_memory",
    "lora_compute_dtype",
    "mps_memory_snapshot",
    "regional_working_memory",
    "soft_empty_cache",
    "xpu_memory_snapshot",
]
