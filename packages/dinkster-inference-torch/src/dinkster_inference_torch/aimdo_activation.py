"""Process-wide post-torch comfy-aimdo device activation.

The worker bootstrap owns ``comfy_aimdo.control.init()`` because that
native-library load must happen before torch is imported. This module
owns the single later ``init_devices`` attempt. Readiness is proven by
``get_devctx`` for every requested index: comfy-aimdo 0.4.13 assigns
``control.lib`` before all native symbols are bound, so ``lib is not
None`` is not a valid activation test.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from .memory import (
    CudaMemorySnapshot,
    DeviceMemory,
    MemoryPolicy,
    cuda_memory_snapshot,
    get_free_memory,
)

if TYPE_CHECKING:
    import torch


_AimdoDeviceEntry = int | tuple[int, int]


class AimdoUnavailableError(RuntimeError):
    """comfy-aimdo cannot serve the requested CUDA device."""


@dataclass(frozen=True)
class AimdoMemoryStatus:
    """Classified Dinkster-owned VBAR residency for one CUDA device."""

    evictable_bytes: int
    pinned_bytes: int

    @property
    def resident_bytes(self) -> int:
        return self.evictable_bytes + self.pinned_bytes


class _AimdoControl(Protocol):
    lib: object

    def get_devctx(self, device_id: int) -> object: ...

    def init_devices(self, device_ids: Sequence[_AimdoDeviceEntry]) -> bool: ...


_activation_lock = threading.RLock()
_native_init_attempted = False
_native_ready = False


def _load_control() -> _AimdoControl:
    return cast(
        "_AimdoControl",
        importlib.import_module("comfy_aimdo.control"),
    )


def _ready(control: _AimdoControl, device_indices: Sequence[int]) -> bool:
    try:
        for index in device_indices:
            if control.get_devctx(index) is None:
                return False
    except Exception:
        return False
    return True


def _normalize_device_entries(
    device_entries: Sequence[_AimdoDeviceEntry],
) -> tuple[tuple[_AimdoDeviceEntry, ...], tuple[int, ...]]:
    requested: list[_AimdoDeviceEntry] = []
    indices: list[int] = []
    headrooms: dict[int, int] = {}
    for entry in device_entries:
        if isinstance(entry, tuple):
            if len(entry) != 2:
                raise ValueError(
                    "aimdo device entry must be an index or (index, extra_vram_headroom_bytes)"
                )
            raw_index, raw_headroom = entry
            index = int(raw_index)
            headroom = int(raw_headroom)
            forwarded: _AimdoDeviceEntry = (index, headroom)
        else:
            index = int(entry)
            headroom = 0
            forwarded = index
        if headroom < 0:
            raise ValueError("aimdo extra_vram_headroom_bytes must be non-negative")
        if index in headrooms:
            if headrooms[index] != headroom:
                raise ValueError(
                    f"aimdo device {index} has conflicting headroom values"
                    f" {headrooms[index]} and {headroom}"
                )
            continue
        headrooms[index] = headroom
        requested.append(forwarded)
        indices.append(index)
    return tuple(requested), tuple(indices)


def ensure_aimdo_devices(device_indices: Sequence[_AimdoDeviceEntry]) -> bool:
    """Ensure all requested devices have native aimdo contexts.

    Existing contexts are accepted without an init call. Otherwise the
    process receives exactly one ``init_devices`` attempt; a later request
    for another uninitialized device fails cleanly rather than attempting
    to reinitialize native process-global state. Entries may be an index or
    ``(index, extra_vram_headroom_bytes)`` with non-negative headroom.
    Headroom binds at the one-shot native init and cannot be adjusted later.
    Process-global ``simple_vram_headroom`` remains backend-owned at the
    pre-torch ``control.init()`` bootstrap.
    """
    requested, indices = _normalize_device_entries(device_indices)
    if not indices:
        return False

    global _native_init_attempted  # noqa: PLW0603 - process activation state
    global _native_ready  # noqa: PLW0603 - process activation state
    with _activation_lock:
        try:
            control = _load_control()
        except ImportError:
            return False
        if _native_init_attempted:
            return _native_ready and _ready(control, indices)
        _native_init_attempted = True
        if _ready(control, indices):
            _native_ready = True
            return True
        if any(_ready(control, (index,)) for index in indices):
            return False
        try:
            initialized = control.init_devices(requested)
        except Exception:
            return False
        _native_ready = bool(initialized) and _ready(control, indices)
        return _native_ready


def ensure_visible_aimdo_devices(policy: MemoryPolicy | None = None) -> bool:
    """Admit the whole process-visible CUDA namespace in one native attempt.

    Device budgets are caps, not an allowlist. Admission must not depend on
    which user's model requests a device first. Headroom binds at the first
    initialization; later requests only verify the existing contexts.
    """
    import torch

    indices = tuple(range(torch.cuda.device_count()))
    with _activation_lock:
        if policy is None or _native_init_attempted:
            return ensure_aimdo_devices(indices)
        entries: list[_AimdoDeviceEntry] = []
        for index in indices:
            device = torch.device("cuda", index)
            headroom = 0
            if policy.hard_budget(device) is not None:
                total = int(torch.cuda.get_device_properties(index).total_memory)
                headroom = policy.resolve(device, total).budget_headroom_bytes
            entries.append((index, headroom))
        return ensure_aimdo_devices(entries)


def aimdo_resident_bytes(device: torch.device) -> int:
    """Return process-global resident VBAR page bytes for a CUDA device.

    This upstream-compatible observability total includes pinned and
    evictable pages. It is not a free-memory value; policy must use the
    classified Dinkster-owned status instead.

    Returns zero for non-CUDA devices, unavailable comfy-aimdo, or a device
    whose native aimdo context has not been initialized.
    """
    if device.type != "cuda":
        return 0
    index = cast("int | None", device.index)
    if index is None:
        import torch

        index = torch.cuda.current_device()
    try:
        control = _load_control()
        if not _ready(control, (index,)):
            return 0
        model_vbar = importlib.import_module("comfy_aimdo.model_vbar")
    except (ImportError, AttributeError):
        return 0
    return int(model_vbar.vbars_analyze(index))


def aimdo_memory_status(device: torch.device) -> AimdoMemoryStatus:
    """Classify Dinkster-owned VBAR pages without waiting for GPU work."""
    if device.type != "cuda":
        return AimdoMemoryStatus(0, 0)
    index = cast("int | None", device.index)
    if index is None:
        import torch

        index = torch.cuda.current_device()
    try:
        control = _load_control()
        if not _ready(control, (index,)):
            return AimdoMemoryStatus(0, 0)
    except (ImportError, AttributeError):
        return AimdoMemoryStatus(0, 0)

    from .aimdo_residency import production_vbar_memory

    evictable, pinned = production_vbar_memory(index)
    return AimdoMemoryStatus(evictable, pinned)


def dynamic_cuda_memory_snapshot(
    device: torch.device,
    *,
    cuda_memory: Callable[[torch.device], CudaMemorySnapshot] = cuda_memory_snapshot,
    aimdo_memory: Callable[[torch.device], AimdoMemoryStatus] = aimdo_memory_status,
) -> CudaMemorySnapshot:
    """Add only proven-evictable VBAR pages to CUDA capacity.

    This deliberately tightens the pinned ComfyUI behavior: resident pages
    with active pins cannot be shed and therefore are not free memory.
    """
    memory = cuda_memory(device)
    dynamic = aimdo_memory(device)
    return replace(
        memory,
        dynamic_evictable_bytes=dynamic.evictable_bytes,
        dynamic_pinned_bytes=dynamic.pinned_bytes,
    )


def dynamic_free_memory(
    device: torch.device,
    *,
    cuda_memory: Callable[[torch.device], CudaMemorySnapshot] = cuda_memory_snapshot,
    aimdo_memory: Callable[[torch.device], AimdoMemoryStatus] = aimdo_memory_status,
    free_memory: Callable[[torch.device], DeviceMemory] = get_free_memory,
) -> DeviceMemory:
    """Project classified dynamic CUDA capacity into ``DeviceMemory``."""
    if device.type != "cuda":
        return free_memory(device)
    return dynamic_cuda_memory_snapshot(
        device,
        cuda_memory=cuda_memory,
        aimdo_memory=aimdo_memory,
    ).device_memory()


def set_simple_vram_headroom(bytes: int) -> bool:
    """Set upstream's process-global simple headroom after successful init.

    This is the upstream ``--reserve-vram`` equivalent and applies to all
    devices. It is callable any time after successful aimdo activation. The
    pre-torch ``control.init(simple_vram_headroom=...)`` default remains
    backend-owned; this wrapper lets later reservation mirroring avoid
    touching comfy-aimdo control state directly.
    """
    value = int(bytes)
    if value < 0:
        raise ValueError("aimdo simple_vram_headroom bytes must be non-negative")
    with _activation_lock:
        if not _native_ready:
            return False
        try:
            control = _load_control()
            setter = cast(Any, control.lib).set_simple_vram_headroom
        except (ImportError, AttributeError):
            return False
        setter(value)
        return True


__all__ = [
    "AimdoMemoryStatus",
    "AimdoUnavailableError",
    "aimdo_memory_status",
    "aimdo_resident_bytes",
    "dynamic_cuda_memory_snapshot",
    "dynamic_free_memory",
    "ensure_aimdo_devices",
    "ensure_visible_aimdo_devices",
    "set_simple_vram_headroom",
]
