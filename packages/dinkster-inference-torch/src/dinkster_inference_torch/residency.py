"""Residency: which weights live on the load device, and who decides.

Port of the residency mechanism of comfy/model_management.py
(LoadedModel, load_models_gpu, free_memory) and comfy/model_patcher.py
(load, partially_load, partially_unload) @ b78cec87, reshaped per
docs/native-inference-plan.md 1.1/1.2: explicit constructed state
instead of a process-global registry, key-group units instead of
nn.Module scans (stage-5 modules map one module -> one unit), and int
byte budgets instead of the reference's float sentinels (0.1 = "load
nothing", 1e32 = "everything"; Dinkster: ``None`` = unlimited, an int is
a real byte count, negative shrinks).

Two layers, matching the reference split:

- ``ResidentWeights`` is one model's mechanism (the ModelPatcher
  residency half): per-unit placement over a weight store. A RESIDENT
  unit holds patched storage on the load device with patched originals,
  and optionally all offload storage, retained as backup; an OFFLOADED
  unit holds original (never-patched) storage on the offload device and
  applies ordinary patches at cast time via ``DeferredPatch``. Packed
  INT8/NVFP4 weights patch and requantize on demand so their quantized
  execution path remains available.
- ``ResidencyManager`` is the fleet policy (the load_models_gpu /
  free_memory half): an MRU registry, the free-ahead-of-load pass with
  the 1.1 inflation, the verbatim low-VRAM budget formula, and
  eviction ordered by (most-offloaded, smallest, newest) with
  partial-unload-before-detach.

The manager consumes mechanisms only through ``ResidencyMechanism`` -
that protocol is the aimdo seam (see aimdo.py): a VBAR-backed dynamic
mechanism drops in beside ``ResidentWeights`` without the manager
knowing.

Deliberate deviations from the reference (beyond the sentinel/int
reshaping above), all recorded in docs/native-inference-plan.md:

- The manager keeps requested models out of its own pre-load eviction
  pass; upstream can evict a model it is about to reload.
- Re-loading a registered model moves its registry entry to the front;
  upstream re-inserts and duplicates the entry
  (docs/comfyui-issues/comfyui-load-models-gpu-duplicate-registry.md).
- The eviction sort drops upstream's ``sys.getrefcount`` term - Dinkster
  has no clone economy whose liveness that term approximates.
- CUDA partial-residency loads use one bounded reusable pinned staging ring per
  device and run on a producer stream; full-model loads stay on the current
  stream, matching upstream. Prefetch remains the dynamic-residency mechanism's job.
- Unit ordering estimates patch overhead at STORAGE itemsize; the
  reference uses the model compute dtype, unknown at this layer.
- ``force_patch_weights`` (patch offloaded keys at load time for
  consumers that cannot run weight functions) is deferred (ROADMAP).
- The VRAMState machine (DISABLED..HIGH_VRAM/SHARED) is host policy,
  not mechanism; the manager always behaves like NORMAL_VRAM with
  smart memory. Hosts express NO_VRAM-style modes through
  ``MemoryPolicy`` and budgets when they arrive (ROADMAP).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Generator, MutableMapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

import torch
from dinkster_inference.patches import PatchEntry, PatchSet

from . import pinned_host
from .apply import (
    PatchApplyError,
    StoredWeight,
    _component_tensors,  # pyright: ignore[reportPrivateUsage]
    _preserve_parameter_registration,  # pyright: ignore[reportPrivateUsage]
    move_stored,
    patch_stored_weight,
)
from .memory import (
    DeviceMemory,
    MemoryPolicy,
    MpsMemorySnapshot,
    get_free_memory,
    get_total_memory,
    mps_memory_snapshot,
    soft_empty_cache,
)
from .ops import DeferredPatch, WeightFunction, cast_weight
from .quant import Fp8ScaledWeight, Int8PackedWeight, Nvfp4PackedWeight
from .residency_timing import (
    DEQUANT,
    EXPOSED_STALL,
    TRANSFER,
    PartialResidencyTiming,
    active_partial_residency_timing,
    timed_phase,
)
from .tensor_ops import use_transfer_stager

LOWVRAM_PATCH_ESTIMATE_FACTOR = 2
"""comfy/model_patcher.py LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR
@ b78cec87: patched keys cost roughly two extra weight-sized buffers
when applied at cast time; the estimate orders units so patch-heavy
ones prefer residency."""

_MPS_OOM_PREFIX = "MPS backend out of memory"
logger = logging.getLogger(__name__)


def stored_nbytes(stored: StoredWeight) -> int:
    """Bytes a stored weight occupies, including packed quantization state."""
    if isinstance(stored, Fp8ScaledWeight):
        return stored.qdata.nbytes + stored.scale.nbytes
    if isinstance(stored, Int8PackedWeight):
        return stored.qdata.nbytes + stored.scale.nbytes
    if isinstance(stored, Nvfp4PackedWeight):
        return stored.qdata.nbytes + stored.block_scale.nbytes + stored.tensor_scale.nbytes
    return stored.nbytes


def _stored_numel(stored: StoredWeight) -> int:
    if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
        return stored.qdata.numel()
    return stored.numel()


def _stored_itemsize(stored: StoredWeight) -> int:
    if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
        return stored.qdata.itemsize
    return stored.itemsize


@dataclass(frozen=True)
class ResidencyUnit:
    """The granularity of placement: a named group of store keys that
    load and offload together (the analog of one nn.Module in the
    reference's _load_list; stage-5 modules map one module -> one
    unit). Sparse expert units are admitted after shared state and
    evicted before it under constrained eager placement."""

    name: str
    keys: tuple[str, ...]
    expert: bool = False


class ResidencyMechanism(Protocol):
    """What the manager needs from one model's residency mechanism -
    the seam behind which ``ResidentWeights`` (eager moves) and
    ``AimdoWeights`` (demand-paged VBAR storage) are interchangeable.
    Byte counts are of weights currently on / off the load device;
    ``partially_load(None)`` means load fully, a negative extra
    shrinks (LoadedModel + ModelPatcher.partially_load/partially_unload
    @ b78cec87)."""

    @property
    def load_device(self) -> torch.device: ...

    @property
    def demand_paged(self) -> bool:
        """Whether weight residency is populated on demand at use time."""
        ...

    def total_bytes(self) -> int: ...

    def loaded_bytes(self) -> int: ...

    def automatically_reclaimable_bytes(self) -> int:
        """Loaded bytes the runtime can reuse without explicit eviction."""
        ...

    def offloaded_bytes(self) -> int: ...

    def partially_load(self, extra_memory: int | None) -> int: ...

    def partially_unload(self, memory_to_free: int) -> int: ...

    def unload(self) -> None: ...

    def release_working_buffers(self) -> bool:
        """Release idle execution-scoped buffers and report whether allocator cleanup is useful."""
        ...

    def working_set_reservation_bytes(self) -> int: ...

    def reserve_working_set(self) -> AbstractContextManager[None]: ...


@runtime_checkable
class DiscardableResidency(Protocol):
    """Optional terminal cleanup when the owner guarantees store destruction."""

    def discard(self) -> None: ...


class WeightLease(Protocol):
    """Weights available for one module forward.

    The consuming operation must execute before the lease closes.
    Values returned by ``get`` and ``get_stored`` must not be stashed
    or used after that point, although output tensors produced by the
    operation may escape the bracket.
    """

    def get(self, key: str, *, dtype: torch.dtype) -> torch.Tensor: ...

    def get_stored(self, key: str) -> StoredWeight:
        """Raw packed storage for a quantized hardware matmul.

        The result is moved to the mechanism's load device without a
        dtype cast or weight functions. Dynamic mechanisms use this as
        their device-materialization point for uncast qdata and scale
        storage.
        """
        ...

    def timing_collector(self) -> PartialResidencyTiming | None:
        """The collector this lease's transfers record into, or None
        when collection is inactive or the mechanism's transfers are
        not instrumented. Layer-level phases must record into this
        collector (never the thread-local state directly) so a module
        receipt stays coherent with its lease's transfers.
        """
        ...


class EagerTransferHooks(Protocol):
    """CUDA loading and producer/consumer ordering for eager weights."""

    @property
    def non_blocking(self) -> bool: ...

    def loading_context(self) -> AbstractContextManager[None]: ...

    def consumer_wait_for_loading(self) -> None: ...

    def producer_context(self) -> AbstractContextManager[None]: ...

    def consumer_wait_for_producer(self) -> None: ...

    def producer_wait_for_consumer(self) -> None: ...


class _SynchronousTransferHooks:
    non_blocking = False

    def loading_context(self) -> AbstractContextManager[None]:
        return nullcontext()

    def consumer_wait_for_loading(self) -> None:
        return None

    def producer_context(self) -> AbstractContextManager[None]:
        return nullcontext()

    def consumer_wait_for_producer(self) -> None:
        return None

    def producer_wait_for_consumer(self) -> None:
        return None


def _cuda_host_register(ptr: int, size: int) -> bool:
    try:
        return cast(Any, torch.cuda.cudart()).cudaHostRegister(ptr, size, 1) == 0
    except RuntimeError:
        return False


def _cuda_host_unregister(ptr: int) -> bool:
    try:
        return cast(Any, torch.cuda.cudart()).cudaHostUnregister(ptr) == 0
    except RuntimeError:
        return False


def _discard_cuda_async_error(device: torch.device) -> None:
    """Consume the sticky async error a failed cudart call can leave
    on the device so later kernels do not inherit it."""
    try:
        with torch.cuda.device(device):
            torch.ones(1, dtype=torch.uint8, device=device).add_(1)
            torch.cuda.synchronize(device)
    except RuntimeError:
        pass


_CUDA_STAGING_SLOT_BYTES = (96 * 1024**2, 32 * 1024**2)


class _CudaTransferStager:
    """One pageable-to-CUDA staging ring and producer stream per device."""

    def __init__(
        self,
        device: torch.device,
        *,
        slot_bytes: tuple[int, ...] = _CUDA_STAGING_SLOT_BYTES,
    ) -> None:
        self.device = device
        self.slot_bytes = slot_bytes
        self.lock = threading.RLock()
        self._stream: torch.cuda.Stream | None = None
        self._buffers: list[torch.Tensor | None] = [None] * len(slot_bytes)
        self._registered = [False] * len(slot_bytes)
        self._ready: list[torch.cuda.Event | None] = [None] * len(slot_bytes)
        self._next = 0
        self.pin_active = False

    @property
    def allocated_bytes(self) -> int:
        return sum(0 if buffer is None else buffer.nbytes for buffer in self._buffers)

    @property
    def capacity_bytes(self) -> int:
        return sum(self.slot_bytes)

    def pin_debug_label(self) -> str:
        return f"CudaTransferStager(device={self.device})"

    def producer_stream(self) -> torch.cuda.Stream:
        with self.lock:
            if self._stream is None:
                self._stream = torch.cuda.Stream(device=self.device)
            return self._stream

    @contextmanager
    def active(self) -> Generator[None]:
        with self.lock:
            was_active = self.pin_active
            self.pin_active = True
            try:
                yield
            finally:
                self.pin_active = was_active

    def _register_buffer(self, index: int, buffer: torch.Tensor) -> bool:
        if self._registered[index]:
            return True
        size = buffer.nbytes
        if not pinned_host.ensure_pin_budget(size) or not pinned_host.ensure_pin_registerable(
            size, evict_active=False
        ):
            return False
        if not _cuda_host_register(buffer.data_ptr(), size):
            _discard_cuda_async_error(self.device)
            pinned_host.free_registrations(size)
            if not _cuda_host_register(buffer.data_ptr(), size):
                _discard_cuda_async_error(self.device)
                return False
        self._registered[index] = True
        pinned_host.account(size)
        return True

    def _unregister_buffer(self, index: int, buffer: torch.Tensor) -> bool:
        if not self._registered[index]:
            return True
        if not _cuda_host_unregister(buffer.data_ptr()):
            _discard_cuda_async_error(self.device)
            return False
        self._registered[index] = False
        pinned_host.account(-buffer.nbytes)
        return True

    def _drop_buffer(self, index: int) -> int:
        buffer = self._buffers[index]
        if buffer is None:
            return 0
        ready = self._ready[index]
        if ready is not None:
            ready.synchronize()
        if not self._unregister_buffer(index, buffer):
            return 0
        size = buffer.nbytes
        self._buffers[index] = None
        self._ready[index] = None
        pinned_host.account_storage(self, -size)
        pinned_host.discard_owner_if_empty(self)
        return size

    def free_pins(self, size: int) -> int:
        needed = max(0, int(size))
        freed = 0
        if not self.lock.acquire(blocking=False):
            return 0
        try:
            for offset in range(len(self._buffers)):
                if freed >= needed:
                    break
                index = (self._next + offset) % len(self._buffers)
                freed += self._drop_buffer(index)
        finally:
            self.lock.release()
        return freed

    def free_registrations(self, size: int) -> int:
        needed = max(0, int(size))
        freed = 0
        if not self.lock.acquire(blocking=False):
            return 0
        try:
            for offset in range(len(self._buffers)):
                if freed >= needed:
                    break
                index = (self._next + offset) % len(self._buffers)
                buffer = self._buffers[index]
                if buffer is None or not self._registered[index]:
                    continue
                ready = self._ready[index]
                if ready is not None:
                    ready.synchronize()
                if self._unregister_buffer(index, buffer):
                    self._ready[index] = None
                    freed += buffer.nbytes
        finally:
            self.lock.release()
        return freed

    def transfer(
        self,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        with self.active():
            index = None
            for offset in range(len(self._buffers)):
                candidate = (self._next + offset) % len(self._buffers)
                if tensor.nbytes <= self.slot_bytes[candidate]:
                    index = candidate
                    break
            if index is None:
                return tensor.to(device=self.device, dtype=dtype, non_blocking=False)
            buffer = self._buffers[index]
            if buffer is None:
                if pinned_host.DISABLED:
                    return tensor.to(device=self.device, dtype=dtype, non_blocking=False)
                size = self.slot_bytes[index]
                if not pinned_host.reserve_storage(self, size):
                    pinned_host.discard_owner_if_empty(self)
                    return tensor.to(device=self.device, dtype=dtype, non_blocking=False)
                try:
                    buffer = torch.empty(size, dtype=torch.uint8)
                except RuntimeError:
                    pinned_host.account_storage(self, -size)
                    pinned_host.discard_owner_if_empty(self)
                    return tensor.to(device=self.device, dtype=dtype, non_blocking=False)
                self._buffers[index] = buffer
            if not self._register_buffer(index, buffer):
                self._drop_buffer(index)
                return tensor.to(device=self.device, dtype=dtype, non_blocking=False)
            ready = self._ready[index]
            if ready is not None:
                ready.synchronize()
            source = buffer[: tensor.nbytes].view(tensor.dtype).view(tensor.shape)
            with torch.no_grad():
                source.copy_(tensor)
                result = torch.empty_like(tensor, device=self.device, dtype=dtype)
                result.copy_(source, non_blocking=True)
            if ready is None:
                ready = torch.cuda.Event()
                self._ready[index] = ready
            ready.record(torch.cuda.current_stream(self.device))
            self._next = (index + 1) % len(self._buffers)
            result.requires_grad_(tensor.requires_grad)
            return result


_CUDA_STAGERS_LOCK = threading.Lock()
_CUDA_STAGERS: dict[int, _CudaTransferStager] = {}


def _cuda_transfer_stager(device: torch.device) -> _CudaTransferStager:
    index = device.index
    if index is None:  # pyright: ignore[reportUnnecessaryComparison]
        index = torch.cuda.current_device()
    with _CUDA_STAGERS_LOCK:
        stager = _CUDA_STAGERS.get(index)
        if stager is None:
            stager = _CudaTransferStager(torch.device("cuda", index))
            _CUDA_STAGERS[index] = stager
        return stager


class _StoredSourcePins:
    """In-place CUDA host registrations over offloaded stored weights
    (ComfyUI's pin_memory @ b78cec87).

    Registering the store's own CPU tensors makes every copy that reads
    them - fused casts, split lease moves, packed-weight components, and
    unit loads - a true async DMA instead of a pageable bounce. Pins
    hold strong references, so registered pages stay alive until they
    are unregistered here. Registered bytes are accounted against the
    shared registration budget; the pins own no physical staging
    storage, so this owner leaves the registry via ``unregister_owner``,
    never ``discard_owner_if_empty``.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.lock = threading.RLock()
        self._pinned: dict[int, torch.Tensor] = {}
        self._active = 0
        self.pin_active = False

    def pin_debug_label(self) -> str:
        return f"StoredSourcePins(device={self.device})"

    def acquire_active(self) -> None:
        with self.lock:
            self._active += 1
            self.pin_active = True

    def release_active(self) -> None:
        with self.lock:
            self._active -= 1
            self.pin_active = self._active > 0

    def ensure(self, stored: StoredWeight) -> None:
        """Pin ``stored``'s component tensors in place when eligible.

        Refusals (budget, registration ceiling, cudart failure) leave
        the tensor pageable; every consumer path works unchanged, only
        slower. Components are deduplicated by data pointer, so tied
        keys sharing storage pin once.
        """
        for tensor in _component_tensors(stored):
            self._ensure_tensor(tensor)

    def _ensure_tensor(self, tensor: torch.Tensor) -> None:
        if (
            tensor.device.type != "cpu"
            or tensor.nbytes == 0
            or not tensor.is_contiguous()
            or tensor.is_pinned()
        ):
            return
        ptr = tensor.data_ptr()
        if ptr == 0:
            return
        with self.lock:
            if ptr in self._pinned:
                return
            size = tensor.nbytes
            if not pinned_host.ensure_pin_budget(size) or not pinned_host.ensure_pin_registerable(
                size, evict_active=False
            ):
                return
            if not _cuda_host_register(ptr, size):
                _discard_cuda_async_error(self.device)
                pinned_host.free_registrations(size)
                if not _cuda_host_register(ptr, size):
                    _discard_cuda_async_error(self.device)
                    return
            pinned_host.register_owner(self)
            pinned_host.account(size)
            self._pinned[ptr] = tensor

    def _release_pointers(self, pointers: Sequence[int], *, needed: int | None = None) -> int:
        """Unregister the given pins under the held lock, synchronizing
        the device once before the first unregistration so in-flight
        copies out of the registered pages complete first. Returns the
        bytes freed; stops early once ``needed`` bytes are freed."""
        freed = 0
        synced = False
        for ptr in pointers:
            if needed is not None and freed >= needed:
                break
            tensor = self._pinned.get(ptr)
            if tensor is None:
                continue
            if not synced:
                torch.cuda.synchronize(self.device)
                synced = True
            if not _cuda_host_unregister(ptr):
                _discard_cuda_async_error(self.device)
                continue
            pinned_host.account(-tensor.nbytes)
            del self._pinned[ptr]
            freed += tensor.nbytes
        return freed

    def release(self, stored: StoredWeight) -> None:
        """Unpin ``stored``'s components; unpinned components are skipped."""
        with self.lock:
            self._release_pointers([tensor.data_ptr() for tensor in _component_tensors(stored)])

    def release_all(self) -> None:
        with self.lock:
            self._release_pointers(list(self._pinned))
            if self._pinned:
                raise RuntimeError("failed source host unregistration; pinned tensors remain owned")
            pinned_host.unregister_owner(self)

    def free_pins(self, size: int) -> int:
        # In-place pins allocate no reclaimable host storage.
        return 0

    def free_registrations(self, size: int) -> int:
        needed = max(0, int(size))
        if needed == 0 or not self.lock.acquire(blocking=False):
            return 0
        try:
            if self._active > 0:
                return 0
            return self._release_pointers(list(self._pinned), needed=needed)
        finally:
            self.lock.release()


class _CudaTransferHooks:
    non_blocking = True

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._stager = _cuda_transfer_stager(device)

    @property
    def _producer(self) -> torch.cuda.Stream:
        return self._stager.producer_stream()

    @contextmanager
    def loading_context(self) -> Generator[None]:
        with self._stager.active(), use_transfer_stager(self._stager):
            yield

    def consumer_wait_for_loading(self) -> None:
        return None

    @contextmanager
    def producer_context(self) -> Generator[None]:
        with (
            self._stager.active(),
            torch.cuda.stream(self._producer),
            use_transfer_stager(self._stager),
        ):
            yield

    def consumer_wait_for_producer(self) -> None:
        current = torch.cuda.current_stream(self._device)
        current.wait_stream(self._producer)

    def producer_wait_for_consumer(self) -> None:
        current = torch.cuda.current_stream(self._device)
        self._producer.wait_stream(current)


@dataclass(slots=True)
class ResidencyUnitState:
    """Live placement state shared with bound module forwards."""

    loaded: bool = False


class UnitResidency(Protocol):
    """The mechanism-facing per-forward weight-consumption seam."""

    @property
    def load_device(self) -> torch.device:
        """The device where leased weights are materialized."""
        ...

    def is_loaded(self, unit: str) -> bool: ...

    def unit_state(self, unit: str) -> ResidencyUnitState: ...

    def weight_functions(self, key: str) -> tuple[WeightFunction, ...]: ...

    def lease(self, unit: str) -> AbstractContextManager[WeightLease]: ...


def _validate_residency_layout(
    weights: MutableMapping[str, StoredWeight],
    patch_set: PatchSet[torch.Tensor] | None,
    units: Sequence[ResidencyUnit] | None,
) -> tuple[
    tuple[ResidencyUnit, ...],
    dict[str, str],
    dict[str, tuple[PatchEntry[torch.Tensor], ...]],
]:
    """Validate and snapshot the constructor inputs shared by residency
    mechanisms.

    Unit coverage and patch-target validation are one contract. Keeping
    them here prevents eager and demand-paged mechanisms from drifting.
    """
    if units is None:
        units = tuple(ResidencyUnit(key, (key,)) for key in weights)
    validated_units = tuple(units)
    unit_names: set[str] = set()
    unit_of: dict[str, str] = {}
    for unit in validated_units:
        if unit.name in unit_names:
            raise PatchApplyError(f"unit name {unit.name!r} appears more than once")
        unit_names.add(unit.name)
        for key in unit.keys:
            if key not in weights:
                raise PatchApplyError(
                    f"unit {unit.name!r} names key {key!r} which is not in the weight store"
                )
            if key in unit_of:
                raise PatchApplyError(
                    f"key {key!r} appears in units {unit_of[key]!r} and {unit.name!r}"
                )
            unit_of[key] = unit.name
    uncovered = weights.keys() - unit_of.keys()
    if uncovered:
        raise PatchApplyError(f"units do not cover store keys: {sorted(uncovered)!r}")

    entries_by_key: dict[str, tuple[PatchEntry[torch.Tensor], ...]] = {}
    if patch_set is not None:
        for key in patch_set.keys():
            entries = patch_set.entries(key)
            if not entries:
                continue
            if key not in weights:
                raise PatchApplyError(f"patch target {key!r} is not in the weight store")
            entries_by_key[key] = entries
    return validated_units, unit_of, entries_by_key


class _EagerWeightLease:
    def __init__(
        self,
        mechanism: ResidentWeights,
        weights: MutableMapping[str, StoredWeight],
        load_device: torch.device,
        unit: str,
        transfer_hooks: EagerTransferHooks,
    ) -> None:
        self._mechanism = mechanism
        self._weights = weights
        self._load_device = load_device
        self._unit = unit
        self._transfer_hooks = transfer_hooks
        self._materialized: dict[tuple[str, torch.dtype | None], StoredWeight] = {}
        self._used_transfer = False
        self._closed = False
        self._collector = active_partial_residency_timing()

    def timing_collector(self) -> PartialResidencyTiming | None:
        return self._collector

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"lease for unit {self._unit!r} is closed")

    def _transfer(
        self,
        move: Callable[[], StoredWeight],
        finish: Callable[[StoredWeight], StoredWeight],
        fused: Callable[[], StoredWeight] | None = None,
    ) -> StoredWeight:
        """Enqueue one producer-stream transfer, timed when collected.

        ``move`` performs only the device copy of the stored
        representation; ``finish`` performs the on-device work that
        turns it into the leased value (cast-at-use, dequantization,
        deferred patches). Uncollected, ``fused`` runs instead when
        given (the mechanism's fused move-and-finish call, byte-
        identical to the pre-receipt behavior); otherwise
        ``finish(move())``.

        While collecting, the transfer phase brackets only ``move`` and
        the dequant phase brackets ``finish``, so transfer time counts
        copies and transfer bytes count the stored representation that
        crossed devices, not the finished value. The exposed-stall
        phase brackets the consumer stream's wait on the producer, so
        transfer time hidden under compute never appears in it; off
        CUDA the copy is synchronous and fully exposed, so the stall
        phase records the same span as the move.
        """
        collector = self._collector
        device = self._load_device
        if collector is None:
            with self._transfer_hooks.producer_context():
                value = finish(move()) if fused is None else fused()
            self._used_transfer = True
            self._transfer_hooks.consumer_wait_for_producer()
            return value
        if device.type == "cuda":
            with self._transfer_hooks.producer_context():
                with timed_phase(collector, TRANSFER, device):
                    moved = move()
                with timed_phase(collector, DEQUANT, device):
                    value = finish(moved)
            self._used_transfer = True
            with timed_phase(collector, EXPOSED_STALL, device):
                self._transfer_hooks.consumer_wait_for_producer()
        else:
            with self._transfer_hooks.producer_context():
                with (
                    timed_phase(collector, TRANSFER, device),
                    timed_phase(collector, EXPOSED_STALL, device),
                ):
                    moved = move()
                with timed_phase(collector, DEQUANT, device):
                    value = finish(moved)
            self._used_transfer = True
            self._transfer_hooks.consumer_wait_for_producer()
        collector.count_transfer(stored_nbytes(moved))
        return value

    def _consume_prefetched(self, cache_key: tuple[str, torch.dtype | None]) -> StoredWeight | None:
        """Adopt a mechanism-prefetched value: its copy already ran on
        the producer stream, so the lease pays only the wait."""
        value = self._mechanism._peek_prefetched(cache_key)  # pyright: ignore[reportPrivateUsage]
        if value is None:
            return None
        collector = self._collector
        if collector is None:
            self._transfer_hooks.consumer_wait_for_producer()
        else:
            with timed_phase(collector, EXPOSED_STALL, self._load_device):
                self._transfer_hooks.consumer_wait_for_producer()
        self._used_transfer = True
        self._materialized[cache_key] = value
        return value

    def get(self, key: str, *, dtype: torch.dtype) -> torch.Tensor:
        self._ensure_open()
        if self._mechanism._is_key_loaded(key):  # pyright: ignore[reportPrivateUsage]
            return self._mechanism.use(key, dtype=dtype)
        cache_key = (key, dtype)
        cached = self._materialized.get(cache_key)
        if cached is not None:
            assert isinstance(cached, torch.Tensor)
            return cached
        prefetched = self._consume_prefetched(cache_key)
        if prefetched is not None:
            assert isinstance(prefetched, torch.Tensor)
            return prefetched
        non_blocking = self._transfer_hooks.non_blocking
        value = self._transfer(
            lambda: self._mechanism._move_for_lease(  # pyright: ignore[reportPrivateUsage]
                key,
                non_blocking=non_blocking,
            ),
            lambda moved: self._mechanism._finish_cast(  # pyright: ignore[reportPrivateUsage]
                key,
                moved,
                dtype=dtype,
            ),
            fused=lambda: self._mechanism.use(key, dtype=dtype, non_blocking=non_blocking),
        )
        assert isinstance(value, torch.Tensor)
        self._materialized[cache_key] = value
        return value

    def get_stored(self, key: str) -> StoredWeight:
        self._ensure_open()
        if self._mechanism._is_key_loaded(key):  # pyright: ignore[reportPrivateUsage]
            return move_stored(self._weights[key], self._load_device)
        cache_key = (key, None)
        cached = self._materialized.get(cache_key)
        if cached is not None:
            return cached
        prefetched = self._consume_prefetched(cache_key)
        if prefetched is not None:
            return prefetched
        non_blocking = self._transfer_hooks.non_blocking
        value = self._transfer(
            lambda: self._mechanism._move_for_lease(  # pyright: ignore[reportPrivateUsage]
                key,
                non_blocking=non_blocking,
            ),
            lambda moved: self._mechanism._finish_stored(  # pyright: ignore[reportPrivateUsage]
                key,
                moved,
            ),
        )
        self._materialized[cache_key] = value
        return value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        materialized = self._materialized
        self._materialized = {}
        try:
            if self._used_transfer:
                self._transfer_hooks.producer_wait_for_consumer()
        finally:
            materialized.clear()


class _EagerPrefetch:
    """Handle over values a mechanism prefetch staged ahead of leases.

    Closing releases exactly the entries this handle staged, after
    ordering the producer stream behind the consumer stream, so staged
    device memory is never reused while a consumer still reads it.
    """

    def __init__(
        self,
        mechanism: ResidentWeights,
        requests: tuple[tuple[str, torch.dtype | None], ...],
    ) -> None:
        self._mechanism = mechanism
        self._requests = requests
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mechanism._release_prefetched(self._requests)  # pyright: ignore[reportPrivateUsage]


class ResidentWeights:
    """One model's residency mechanism over a weight store.

    The store is OWNED by this object between construction and
    ``unload()``: values are replaced with patched/moved forms and
    restored from backup on the way out. Offloaded consumers open one
    :class:`WeightLease` per module forward; its ``get`` method delegates
    to ``use()`` (the cast-at-use path).

    Changing the patch set means ``unload()`` and construct anew - the
    same full unpatch/repatch work the reference does on a
    patches_uuid change, made explicit.
    """

    def __init__(
        self,
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
        patch_weight_dtype: torch.dtype | None = None,
        patch_key_prefix: str = "",
        transfer_hooks: EagerTransferHooks | None = None,
    ) -> None:
        self.load_device = torch.device(load_device)
        self.offload_device = torch.device(offload_device)
        self._weights = weights
        self._intermediate_dtype = intermediate_dtype
        self._patch_weight_dtype = patch_weight_dtype
        self._patch_key_prefix = patch_key_prefix
        self._transfer_hooks = (
            transfer_hooks
            if transfer_hooks is not None
            else (
                _CudaTransferHooks(self.load_device)
                if self.load_device.type == "cuda"
                else _SynchronousTransferHooks()
            )
        )
        self._source_pins = (
            _StoredSourcePins(self.load_device)
            if self.load_device.type == "cuda" and torch.cuda.is_available()
            else None
        )

        self._units, self._unit_of, self._entries = _validate_residency_layout(
            weights, patch_set, units
        )

        self._loaded: set[str] = set()
        self._unit_states = {unit.name: ResidencyUnitState() for unit in self._units}
        self._backup: dict[str, StoredWeight] = {}
        self._prefetched: dict[tuple[str, torch.dtype | None], StoredWeight] = {}
        self._retain_offload_storage = False
        self._unit_byte_counts = {
            unit.name: sum(stored_nbytes(self._weights[key]) for key in unit.keys)
            for unit in self._units
        }
        self._total_byte_count = sum(self._unit_byte_counts.values())
        self._loaded_byte_count = 0

    # -- accounting ---------------------------------------------------

    @property
    def demand_paged(self) -> bool:
        return False

    def execution_context(self) -> AbstractContextManager[None]:
        return nullcontext()

    def reserve_working_set(self) -> AbstractContextManager[None]:
        return nullcontext()

    def working_set_reservation_bytes(self) -> int:
        return 0

    def total_bytes(self) -> int:
        return self._total_byte_count

    def loaded_bytes(self) -> int:
        return self._loaded_byte_count

    def automatically_reclaimable_bytes(self) -> int:
        return 0

    def partial_unload_capacity(self) -> int:
        """Bytes that partial unloading can reclaim before terminal unload."""
        return self.loaded_bytes()

    def offloaded_bytes(self) -> int:
        return self.total_bytes() - self.loaded_bytes()

    def loaded_unit_names(self) -> frozenset[str]:
        return frozenset(self._loaded)

    def residency_units(self) -> tuple[ResidencyUnit, ...]:
        """The immutable placement units owned by this mechanism."""
        return self._units

    def unit_bytes(self, unit: str) -> int:
        """Current stored bytes belonging to one placement unit."""
        return self._unit_byte_counts[unit]

    def is_loaded(self, unit: str) -> bool:
        """Whether ``unit`` is resident on the load device."""
        state = self._unit_states.get(unit)
        return False if state is None else state.loaded

    def unit_state(self, unit: str) -> ResidencyUnitState:
        """Return the stable state object observed by bound forwards.

        Unknown names remain permanently offloaded, matching ``is_loaded``.
        """
        state = self._unit_states.get(unit)
        if state is None:
            state = ResidencyUnitState()
            self._unit_states[unit] = state
        return state

    def _is_key_loaded(self, key: str) -> bool:
        """Whether ``key`` belongs to a resident unit."""
        return self._unit_of[key] in self._loaded

    def _unit_bytes(self, unit: ResidencyUnit) -> int:
        return self._unit_byte_counts[unit.name]

    def _unit_offload_estimate(self, unit: ResidencyUnit) -> int:
        """_load_list's module_offload_mem @ b78cec87: unit bytes plus
        the cast-time patch overhead estimate for patched keys."""
        estimate = 0
        for key in unit.keys:
            stored = self._weights[key]
            estimate += stored_nbytes(stored)
            if key in self._entries:
                estimate += (
                    _stored_numel(stored) * _stored_itemsize(stored) * LOWVRAM_PATCH_ESTIMATE_FACTOR
                )
        return estimate

    def _unit_sort_key(self, unit: ResidencyUnit) -> tuple[int, int, int, str]:
        return (
            0 if unit.expert else 1,
            self._unit_offload_estimate(unit),
            self._unit_bytes(unit),
            unit.name,
        )

    # -- placement ----------------------------------------------------

    def _load_unit(self, unit: ResidencyUnit, *, async_transfer: bool) -> None:
        done: list[tuple[str, StoredWeight]] = []
        sources = {key: self._weights[key] for key in unit.keys}
        source_counts: dict[int, int] = {}
        for source in sources.values():
            source_counts[id(source)] = source_counts.get(id(source), 0) + 1
        moved_aliases: dict[int, StoredWeight] = {}
        try:
            transfer_context = (
                self._transfer_hooks.producer_context()
                if async_transfer
                else self._transfer_hooks.loading_context()
            )
            with transfer_context:
                for key in unit.keys:
                    original = sources[key]
                    entries = self._entries.get(key)
                    tied = entries is None and source_counts[id(original)] > 1
                    moved = (
                        moved_aliases[id(original)]
                        if tied and id(original) in moved_aliases
                        else (
                            _preserve_parameter_registration(
                                original,
                                move_stored(
                                    original,
                                    self.load_device,
                                    non_blocking=(
                                        self._transfer_hooks.non_blocking
                                        if async_transfer
                                        else False
                                    ),
                                ),
                            )
                            if tied
                            else move_stored(
                                original,
                                self.load_device,
                                non_blocking=(
                                    self._transfer_hooks.non_blocking if async_transfer else False
                                ),
                            )
                        )
                    )
                    if tied:
                        moved_aliases[id(original)] = moved
                    if entries is not None:
                        moved = patch_stored_weight(
                            moved,
                            entries,
                            key=f"{self._patch_key_prefix}{key}",
                            intermediate_dtype=self._intermediate_dtype,
                            weight_dtype=self._patch_weight_dtype,
                        )
                    if entries is not None or self._retain_offload_storage:
                        # Set only after patching succeeds so a failed key
                        # never leaves retained state.
                        self._backup[key] = original
                    self._weights[key] = moved
                    done.append((key, original))
            if async_transfer:
                self._transfer_hooks.consumer_wait_for_producer()
            else:
                self._transfer_hooks.consumer_wait_for_loading()
        except BaseException as error:
            try:
                if async_transfer:
                    self._transfer_hooks.consumer_wait_for_producer()
                else:
                    self._transfer_hooks.consumer_wait_for_loading()
            except BaseException as cleanup:
                if async_transfer:
                    error.add_note(f"producer-stream cleanup also failed: {cleanup!r}")
                else:
                    error.add_note(f"loading cleanup also failed: {cleanup!r}")
            # roll the half-loaded unit back so the store never leaks
            # moved/patched storage for a unit that is not loaded;
            # restoring the retained originals costs no device traffic
            for key, original in done:
                self._backup.pop(key, None)
                self._weights[key] = original
            raise
        if self._source_pins is not None:
            # Unpin loaded sources (pin_weight_to_device's unpin-on-load
            # @ b78cec87): a resident unit's CPU storage no longer
            # serves lease copies, so its registration budget returns
            # to the offloaded working set.
            for _key, original in done:
                self._source_pins.release(original)
        previous_bytes = self._unit_byte_counts[unit.name]
        current_bytes = sum(stored_nbytes(self._weights[key]) for key in unit.keys)
        self._unit_byte_counts[unit.name] = current_bytes
        self._total_byte_count += current_bytes - previous_bytes
        self._loaded_byte_count += current_bytes
        self._loaded.add(unit.name)
        self._unit_states[unit.name].loaded = True

    def _unload_unit(self, unit: ResidencyUnit) -> None:
        loaded_bytes = self._unit_byte_counts[unit.name]
        sources: dict[str, StoredWeight] = {}
        for key in unit.keys:
            sources[key] = self._backup.get(key, self._weights[key])
        source_counts: dict[int, int] = {}
        for source in sources.values():
            source_counts[id(source)] = source_counts.get(id(source), 0) + 1
        moved_aliases: dict[int, StoredWeight] = {}
        for key in unit.keys:
            self._backup.pop(key, None)
            stored = sources[key]
            tied = source_counts[id(stored)] > 1
            moved = moved_aliases.get(id(stored)) if tied else None
            if moved is None:
                moved = (
                    _preserve_parameter_registration(
                        stored, move_stored(stored, self.offload_device)
                    )
                    if tied
                    else move_stored(stored, self.offload_device)
                )
                if tied:
                    moved_aliases[id(stored)] = moved
            self._weights[key] = moved
        current_bytes = sum(stored_nbytes(self._weights[key]) for key in unit.keys)
        self._unit_byte_counts[unit.name] = current_bytes
        self._total_byte_count += current_bytes - loaded_bytes
        self._loaded_byte_count -= loaded_bytes
        self._loaded.discard(unit.name)
        self._unit_states[unit.name].loaded = False

    def _settle(self, budget: int | None) -> None:
        """ModelPatcher.load @ b78cec87: walk units from most to least
        expensive-to-offload, with shared state ahead of sparse experts,
        and keep/load those that fit the budget (strict <, like the
        reference). Offload the rest."""
        order = sorted(self._units, key=self._unit_sort_key, reverse=True)
        mem_counter = 0
        for unit in order:
            unit_bytes = self._unit_bytes(unit)
            fits = budget is None or mem_counter + unit_bytes < budget
            if fits:
                if unit.name not in self._loaded:
                    self._load_unit(unit, async_transfer=budget is not None)
                # re-measure: patching can change stored size
                # (pad_weight diffs grow the weight)
                mem_counter += self._unit_bytes(unit)
            elif unit.name in self._loaded:
                self._unload_unit(unit)

    def partially_load(self, extra_memory: int | None) -> int:
        """ModelPatcher.partially_load @ b78cec87: grow residency by
        an extra byte allowance; ``None`` or an allowance exceeding the
        whole model (strict >) means full load. A negative allowance
        routes through ``partially_unload`` exactly like the reference
        (smallest-first shrink, never a re-settle). Returns the change
        in loaded bytes."""
        if extra_memory is not None and extra_memory < 0:
            return -self.partially_unload(-extra_memory)
        before = self.loaded_bytes()
        budget: int | None
        if extra_memory is None:
            budget = None
        else:
            budget = before + extra_memory
            if budget > self.total_bytes():
                budget = None
        self._settle(budget)
        return self.loaded_bytes() - before

    def partially_unload(self, memory_to_free: int) -> int:
        """ModelPatcher.partially_unload @ b78cec87: offload loaded
        expert units first, then remaining units from least to most
        expensive-to-offload, until ``memory_to_free`` bytes have left
        the load device. Returns bytes freed."""
        freed = 0
        order = sorted(
            (u for u in self._units if u.name in self._loaded),
            key=self._unit_sort_key,
        )
        for unit in order:
            if freed >= memory_to_free:
                break
            unit_bytes = self._unit_bytes(unit)
            self._unload_unit(unit)
            freed += unit_bytes
        return freed

    def unload(self) -> None:
        """Full detach (LoadedModel.model_unload's terminal branch
        @ b78cec87): restore every patched key's original and move
        everything to the offload device."""
        for unit in self._units:
            if unit.name in self._loaded:
                self._unload_unit(unit)
        if self._source_pins is not None:
            self._source_pins.release_all()

    def discard(self) -> None:
        """Release residency resources without restoring a store that will die.

        The owner must close forward/prefetch leases and destroy the store;
        its tensors retain their current placement and patched values.
        """
        self._transfer_hooks.producer_wait_for_consumer()
        if self._source_pins is not None:
            self._source_pins.release_all()
        self._backup.clear()
        self._loaded.clear()
        for state in self._unit_states.values():
            state.loaded = False
        self._loaded_byte_count = 0

    def release_working_buffers(self) -> bool:
        return False

    def retain_offload_storage(self) -> None:
        """Keep authoritative offload tensors so later unloads avoid copies."""
        if self._loaded and not self._retain_offload_storage:
            raise RuntimeError("offload storage retention must be enabled before loading")
        self._retain_offload_storage = True

    # -- consumption ----------------------------------------------------

    def lease(self, unit: str) -> AbstractContextManager[WeightLease]:
        """Bracket one module forward's weight consumption.

        The consuming operation must execute inside the bracket.
        Tensors obtained from the lease must not be stashed or used
        after it closes; output tensors produced by the operation may
        escape. Offloaded casts are retained once per key and dtype until
        the operation finishes, then their producer stream is ordered after
        the consumer before the lease releases them.

        Unit names are deliberately tolerant, matching ``is_loaded``;
        key lookup through the lease retains ``use``'s ``KeyError``.
        The active timing collector is captured when the bracket is
        entered, not when this context manager is constructed, so a
        deferred entry lands in whichever collection window surrounds
        the forward itself.
        """

        @contextmanager
        def bracket() -> Generator[WeightLease]:
            if self._source_pins is not None:
                self._source_pins.acquire_active()
            try:
                eager = _EagerWeightLease(
                    self,
                    self._weights,
                    self.load_device,
                    unit,
                    self._transfer_hooks,
                )
                try:
                    yield eager
                finally:
                    eager.close()
            finally:
                if self._source_pins is not None:
                    self._source_pins.release_active()

        return bracket()

    def prefetch_enabled(self) -> bool:
        return not torch.compiler.is_compiling()

    def prefetch(self, requests: Sequence[tuple[str, torch.dtype | None]]) -> _EagerPrefetch | None:
        """Stage offloaded weights on the producer stream ahead of the
        leases that will consume them, so their copies run under the
        preceding compute instead of stalling the consuming forward.

        Each request is a store key with the ``dtype`` a lease will
        ``get`` it at, or ``None`` for the stored representation a
        lease will ``get_stored``. Loaded keys, duplicates, and
        requests already staged by an open handle are skipped; the
        returned handle keeps the staged values alive until its
        ``close``, which orders the producer stream behind consumers
        before releasing. Values are produced by the same move, cast,
        and patch calls the lease itself would make, so consuming a
        staged value is bit-identical to an unstaged lease.
        """
        if not self.prefetch_enabled():
            return None
        batch: list[tuple[str, torch.dtype | None]] = []
        seen: set[tuple[str, torch.dtype | None]] = set()
        for request in requests:
            key, _dtype = request
            if request in seen or request in self._prefetched or self._is_key_loaded(key):
                continue
            seen.add(request)
            batch.append(request)
        if not batch:
            return None
        collector = active_partial_residency_timing()
        non_blocking = self._transfer_hooks.non_blocking
        staged: dict[tuple[str, torch.dtype | None], StoredWeight] = {}
        source_pins = self._source_pins
        if source_pins is not None:
            source_pins.acquire_active()
        try:
            with self._transfer_hooks.producer_context():
                for key, dtype in batch:
                    if collector is None:
                        staged[(key, dtype)] = (
                            self.use_stored(key, non_blocking=non_blocking)
                            if dtype is None
                            else self.use(key, dtype=dtype, non_blocking=non_blocking)
                        )
                        continue
                    with timed_phase(collector, TRANSFER, self.load_device):
                        moved = self._move_for_lease(key, non_blocking=non_blocking)
                    with timed_phase(collector, DEQUANT, self.load_device):
                        staged[(key, dtype)] = (
                            self._finish_stored(key, moved)
                            if dtype is None
                            else self._finish_cast(key, moved, dtype=dtype)
                        )
                    collector.count_prefetch(stored_nbytes(moved))
            handle = _EagerPrefetch(self, tuple(staged))
            self._prefetched.update(staged)
            return handle
        except BaseException:
            for request, value in staged.items():
                if self._prefetched.get(request) is value:
                    del self._prefetched[request]
            if source_pins is not None:
                source_pins.release_active()
            raise

    def _peek_prefetched(self, request: tuple[str, torch.dtype | None]) -> StoredWeight | None:
        return self._prefetched.get(request)

    def _release_prefetched(self, requests: tuple[tuple[str, torch.dtype | None], ...]) -> None:
        held = [
            value
            for request in requests
            if (value := self._prefetched.pop(request, None)) is not None
        ]
        try:
            if held:
                self._transfer_hooks.producer_wait_for_consumer()
        finally:
            held.clear()
            if self._source_pins is not None:
                self._source_pins.release_active()

    def weight_functions(self, key: str) -> tuple[WeightFunction, ...]:
        """The cast-time functions for ``key``: a ``DeferredPatch``
        when its unit is offloaded and an ordinary stored value has
        patches (the LowVramPatch attachment @ b78cec87). Packed
        INT8/NVFP4 values patch and requantize through ``use_stored``;
        resident keys are already patched in storage."""
        if self._unit_of[key] in self._loaded:
            return ()
        entries = self._entries.get(key)
        if entries is None:
            return ()
        if isinstance(self._weights[key], Int8PackedWeight | Nvfp4PackedWeight):
            return ()
        return (DeferredPatch(key=key, entries=entries),)

    def _stored_for_transfer(self, key: str) -> StoredWeight:
        """``key``'s stored value, host-pinned in place first when an
        offloaded copy toward a CUDA load device is about to read it."""
        stored = self._weights[key]
        if self._source_pins is not None and self._unit_of[key] not in self._loaded:
            self._source_pins.ensure(stored)
        return stored

    def _move_for_lease(self, key: str, *, non_blocking: bool = False) -> StoredWeight:
        """The device copy alone: ``key``'s stored representation on
        the load device, before any cast or patch work. Timed leases
        split ``use``/``use_stored`` here so transfer receipts bracket
        only the copy."""
        return move_stored(
            self._stored_for_transfer(key),
            self.load_device,
            non_blocking=non_blocking,
        )

    def _finish_stored(self, key: str, moved: StoredWeight) -> StoredWeight:
        """The on-device tail of ``use_stored`` for an already-moved
        stored weight."""
        entries = self._entries.get(key)
        if self._unit_of[key] in self._loaded or entries is None:
            return moved
        if not isinstance(moved, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            raise TypeError("stored patch materialization requires packed storage")
        return patch_stored_weight(
            moved,
            entries,
            key=f"{self._patch_key_prefix}{key}",
            intermediate_dtype=self._intermediate_dtype,
            weight_dtype=self._patch_weight_dtype,
        )

    def _finish_cast(self, key: str, moved: StoredWeight, *, dtype: torch.dtype) -> torch.Tensor:
        """The on-device tail of ``use`` for an already-moved stored
        weight: cast to ``dtype`` with deferred patches applied. Values
        match ``use``; when patch functions are present the separate
        move costs one extra device-local copy, paid only while a
        timed lease is splitting phases."""
        return cast_weight(moved, dtype=dtype, functions=self.weight_functions(key))

    def use_stored(self, key: str, *, non_blocking: bool = False) -> StoredWeight:
        return self._finish_stored(key, self._move_for_lease(key, non_blocking=non_blocking))

    def use(
        self,
        key: str,
        *,
        dtype: torch.dtype,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Cast-at-use for ``key`` (comfy/ops.py cast_bias_weight
        @ b78cec87): the stored weight, on the load device, at the
        compute ``dtype``, with deferred patches applied."""
        return cast_weight(
            self._stored_for_transfer(key),
            dtype=dtype,
            device=self.load_device,
            functions=self.weight_functions(key),
            non_blocking=non_blocking,
        )


@dataclass(frozen=True)
class ResidencyManager:
    """The fleet half: load_models_gpu + free_memory @ b78cec87 over
    explicit, injected state. Memory queries and policy are injectable
    so reserve and hard-budget math are testable without CUDA. The reference's
    ``currently_used`` flag (feeding loaded_models(
    only_currently_used=True) for controlnet-style reloads) has no
    Dinkster consumer yet and is deliberately omitted (ROADMAP)."""

    policy: MemoryPolicy = field(default_factory=MemoryPolicy)
    policy_provider: Callable[[], MemoryPolicy] | None = None
    free_memory: Callable[[torch.device], DeviceMemory] = get_free_memory
    total_memory: Callable[[torch.device], int] = get_total_memory
    empty_cache: Callable[[torch.device], None] = soft_empty_cache
    mps_snapshot: Callable[[torch.device], MpsMemorySnapshot] = mps_memory_snapshot
    _registry: list[ResidencyMechanism] = field(default_factory=list)
    _policy_totals: dict[torch.device, int] = field(default_factory=dict)

    def current_policy(self) -> MemoryPolicy:
        return self.policy if self.policy_provider is None else self.policy_provider()

    def policy_memory(
        self,
        device: torch.device,
        *,
        policy: MemoryPolicy | None = None,
    ) -> DeviceMemory:
        """Project physical free bytes into the configured hard budget."""
        measured = self.free_memory(device)
        active_policy = self.current_policy() if policy is None else policy
        hard_budget = active_policy.hard_budget(device)
        if device.type == "cpu" or hard_budget is None:
            return measured
        total = self._policy_totals.get(device)
        if total is None:
            total = self.total_memory(device)
            self._policy_totals[device] = total
        resolved = active_policy.resolve(device, total)
        available = max(0, measured.free_total - resolved.budget_headroom_bytes)
        allocator_budget_debt: int | None = None
        if device.type == "xpu" and measured.allocator_reserved_bytes is not None:
            # Cached reserve is already reclaimable in free_torch, so add it
            # back exactly once when deriving room below the allocator cap.
            allocator_available = (
                hard_budget - measured.allocator_reserved_bytes + measured.free_torch
            )
            allocator_budget_debt = max(0, -allocator_available)
            available = min(available, max(0, allocator_available))
        return DeviceMemory(
            free_total=available,
            free_torch=min(measured.free_torch, available),
            allocator_reserved_bytes=measured.allocator_reserved_bytes,
            allocator_budget_debt_bytes=allocator_budget_debt,
        )

    def registered(self) -> tuple[ResidencyMechanism, ...]:
        """Registry order: most recently loaded first (the reference's
        current_loaded_models with insert(0, ...))."""
        return tuple(self._registry)

    @contextmanager
    def reserve_working_sets(self, mechanisms: Sequence[ResidencyMechanism]) -> Generator[None]:
        policy = self.current_policy()
        required: dict[torch.device, int] = {}
        for mechanism in mechanisms:
            device = mechanism.load_device
            required[device] = required.get(device, 0) + mechanism.working_set_reservation_bytes()
        reserve = policy.minimum_inference_memory()
        eligible = {
            device
            for device, size in required.items()
            if device.type == "cpu"
            or size <= max(0, self.policy_memory(device, policy=policy).free_total - reserve)
        }
        with ExitStack() as stack:
            for mechanism in mechanisms:
                if mechanism.load_device in eligible:
                    stack.enter_context(mechanism.reserve_working_set())
            yield

    def load(
        self,
        mechanisms: Sequence[ResidencyMechanism],
        *,
        memory_required: int = 0,
        minimum_memory: int | None = None,
        force_full_load: bool = False,
    ) -> None:
        """load_models_gpu @ b78cec87: make room (1.1-inflated plus
        the working reserve), then give each model a low-VRAM byte
        budget from the verbatim reference formula and let its
        mechanism settle to it. ``memory_required`` is the expected
        inference working memory; ``minimum_memory`` optionally pins a
        floor of free bytes that must remain."""
        policy = self.current_policy()
        inference = policy.minimum_inference_memory()
        extra_mem = max(inference, memory_required + policy.physical_headroom)
        if minimum_memory is None:
            minimum_required = extra_mem
        else:
            minimum_required = max(inference, minimum_memory + policy.physical_headroom)

        todo: list[ResidencyMechanism] = []
        seen: set[int] = set()
        for mechanism in mechanisms:
            if id(mechanism) not in seen:
                seen.add(id(mechanism))
                todo.append(mechanism)
        # the reference reverses after dedup so later-listed dependents
        # settle first and end up newest in the registry
        todo.reverse()

        total_required: dict[torch.device, int] = {}
        for mechanism in todo:
            device = mechanism.load_device
            total_required[device] = total_required.get(device, 0) + mechanism.offloaded_bytes()

        for device, required in total_required.items():
            if device.type == "cpu":
                continue
            demand_paged = all(
                mechanism.demand_paged for mechanism in todo if mechanism.load_device == device
            )
            self.free(
                int(required * policy.load_inflation) + extra_mem,
                device,
                keep=todo,
                skip_demand_paged=demand_paged,
                _policy=policy,
            )
            if self.policy_memory(device, policy=policy).free_total < minimum_required:
                self.free(
                    minimum_required,
                    device,
                    keep=todo,
                    skip_demand_paged=demand_paged,
                    _policy=policy,
                )
            if device.type == "mps" and required > 0:
                measured = self.policy_memory(device, policy=policy).free_total
                if measured < minimum_required:
                    logger.warning(
                        f"MPS memory pressure: loading {required} bytes of weights"
                        f" leaves {measured} bytes free after eviction, below the"
                        f" {minimum_required}-byte inference reserve"
                        f" ({self.mps_snapshot(device).describe()});"
                        " attempting memory-budgeted loading"
                    )

        for mechanism in todo:
            device = mechanism.load_device
            projected: DeviceMemory | None = None
            if mechanism.offloaded_bytes() == 0:
                projected = self.policy_memory(device, policy=policy)
                if not projected.allocator_budget_debt_bytes:
                    self._touch(mechanism)
                    continue
            budget: int | None
            xpu_hard_cap = device.type == "xpu" and policy.hard_budget(device) is not None
            if device.type == "cpu" or (force_full_load and not xpu_hard_cap):
                budget = None
            else:
                loaded = mechanism.loaded_bytes()
                if projected is None:
                    projected = self.policy_memory(device, policy=policy)
                current_free = (
                    projected.free_total + loaded - (projected.allocator_budget_debt_bytes or 0)
                )
                lowvram = max(
                    0,
                    current_free - minimum_required,
                    min(
                        int(
                            current_free
                            * policy.weight_memory_ratio(
                                device,
                                rocm=device.type == "cuda" and torch.version.hip is not None,
                            )
                        ),
                        current_free - inference,
                    ),
                )
                budget = lowvram - loaded
            try:
                residency_change = mechanism.partially_load(budget)
            except RuntimeError as error:
                self._annotate_mps_oom(error, device)
                raise
            if (
                projected is not None
                and projected.allocator_budget_debt_bytes
                and residency_change < 0
            ):
                self.empty_cache(device)
            self._touch(mechanism)

    def _annotate_mps_oom(self, error: RuntimeError, device: torch.device) -> None:
        """Attach the measured unified-memory budget to torch's opaque
        MPS out-of-memory error so the failure cites real numbers."""
        if device.type != "mps" or not str(error).startswith(_MPS_OOM_PREFIX):
            return
        try:
            note = self.mps_snapshot(device).describe()
        except Exception:  # noqa: BLE001 - a failed snapshot must not mask the OOM
            return
        error.add_note(f"MPS unified-memory budget at failure: {note}")

    def _touch(self, mechanism: ResidencyMechanism) -> None:
        for i, registered in enumerate(self._registry):
            if registered is mechanism:
                self._registry.insert(0, self._registry.pop(i))
                return
        self._registry.insert(0, mechanism)

    def free(
        self,
        memory_required: int,
        device: torch.device,
        keep: Sequence[ResidencyMechanism] = (),
        *,
        skip_demand_paged: bool = False,
        _policy: MemoryPolicy | None = None,
    ) -> None:
        """free_memory @ b78cec87: evict from ``device`` until
        ``memory_required`` bytes are free. Candidates in
        (most-already-offloaded, smallest, newest) order; each is
        partially unloaded first and detached only when partial
        unloading cannot satisfy the shortfall."""
        policy = self.current_policy() if _policy is None else _policy
        keep_ids = {id(m) for m in keep}
        candidates: list[tuple[int, int, int]] = []
        for i, mechanism in enumerate(self._registry):
            if mechanism.load_device != device or id(mechanism) in keep_ids:
                continue
            candidates.append((-mechanism.offloaded_bytes(), mechanism.total_bytes(), i))
        candidates.sort()

        detached: list[int] = []
        adjusted_required = memory_required
        for _neg_offloaded, _size, i in candidates:
            shortfall = adjusted_required - self.policy_memory(device, policy=policy).free_total
            if shortfall <= 0:
                break
            mechanism = self._registry[i]
            partial_capacity = getattr(mechanism, "partial_unload_capacity", None)
            reclaimable = (
                mechanism.loaded_bytes() if partial_capacity is None else int(partial_capacity())
            )
            if skip_demand_paged and mechanism.demand_paged:
                # Dynamic-for-dynamic loads can reuse demand-paged bytes.
                # Explicitly demote fixed tiers when reusable pages alone
                # cannot satisfy the request, without detaching the victim.
                automatic_reclaimable = mechanism.automatically_reclaimable_bytes()
                adjusted_required -= automatic_reclaimable
                shortfall = adjusted_required - self.policy_memory(device, policy=policy).free_total
                if shortfall <= 0:
                    continue
                explicit_reclaimable = max(0, reclaimable - automatic_reclaimable)
                if explicit_reclaimable > 0:
                    partial_target = automatic_reclaimable + min(shortfall, explicit_reclaimable)
                    reclaimed = mechanism.partially_unload(partial_target)
                    adjusted_required += min(automatic_reclaimable, reclaimed)
                continue
            can_fully_offload = getattr(mechanism, "can_fully_offload", None)
            fully_offloadable = callable(can_fully_offload) and bool(can_fully_offload())
            use_partial_unload = shortfall < reclaimable or fully_offloadable
            if use_partial_unload:
                partial_target = min(shortfall, reclaimable) if fully_offloadable else shortfall
                reclaimed = mechanism.partially_unload(partial_target)
                if reclaimed >= shortfall or (fully_offloadable and reclaimed >= reclaimable):
                    continue
            mechanism.unload()
            detached.append(i)

        for i in sorted(detached, reverse=True):
            self._registry.pop(i)

        if detached:
            self.empty_cache(device)
        else:
            measured = self.policy_memory(device, policy=policy)
            # MPS has no trustworthy byte count for its reclaimable allocator cache.
            if device.type == "mps" and measured.free_total < adjusted_required:
                self.empty_cache(device)
            elif measured.free_torch > measured.free_total * 0.25:
                self.empty_cache(device)

    def remove(
        self,
        mechanisms: Sequence[ResidencyMechanism],
        *,
        unload: bool = True,
        discard: bool = False,
    ) -> None:
        """Terminal deregistration. Deviation rationale: the
        reference's process-global current_loaded_models relies on
        refcount-driven eviction for terminal release; Dinkster's
        explicit-state manager needs explicit deregistration (the
        deregistration half that free_memory's detach path implies).

        With ``unload=True`` every distinct mechanism passed is
        unloaded, registered or not (``unload()`` is idempotent; this
        is the defensive terminal-release behavior the compat pool
        wants); ``unload=False`` leaves placement untouched and only
        deregisters. Registry removal is identity-based, matching
        ``_touch``/``free``; unregistered mechanisms are silently
        ignored because ``free`` can autonomously detach-and-pop a
        mechanism during eviction, so a terminal-release path must
        tolerate already-removed entries. Mirroring ``free``'s detach
        branch, the cache is emptied once per distinct non-cpu load
        device that held loaded bytes before the unload.

        ``discard=True`` requires owner-guaranteed store destruction and
        uses optional terminal discard instead of restoring placement.
        Mechanisms without that capability retain full unload cleanup.
        Actual discard does not empty the cache: scoped module references
        can still hold its tensors until the invocation returns.
        """
        if discard and not unload:
            raise ValueError("discard requires unload=True")
        distinct: list[ResidencyMechanism] = []
        seen: set[int] = set()
        for mechanism in mechanisms:
            if id(mechanism) not in seen:
                seen.add(id(mechanism))
                distinct.append(mechanism)

        devices_to_clear: list[torch.device] = []
        if unload:
            for mechanism in distinct:
                was_loaded = mechanism.loaded_bytes() > 0
                device = mechanism.load_device
                if discard and isinstance(mechanism, DiscardableResidency):
                    mechanism.discard()
                else:
                    mechanism.unload()
                    if was_loaded and device.type != "cpu" and device not in devices_to_clear:
                        devices_to_clear.append(device)

        self._registry[:] = [
            registered for registered in self._registry if id(registered) not in seen
        ]

        for device in devices_to_clear:
            self.empty_cache(device)


__all__ = [
    "LOWVRAM_PATCH_ESTIMATE_FACTOR",
    "ResidencyManager",
    "ResidencyMechanism",
    "ResidencyUnit",
    "ResidentWeights",
    "UnitResidency",
    "WeightLease",
    "move_stored",
    "stored_nbytes",
]
