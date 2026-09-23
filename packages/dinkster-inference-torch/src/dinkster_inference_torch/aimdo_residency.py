"""Demand-paged dinkster-aimdo residency for eager inference forwards.

``AimdoWeights`` retains immutable CPU originals and uses conventional
resident storage for units at or below 16 KiB. Latency-sensitive callers may
opt into budget-selected fixed residency for larger units. One ModelVBAR
remains a disposable CUDA cache for the other units. Demand-paged routed
operations open a lease, fault the keys they consume, and unpin every
successful fault when the operation finishes.
This is inference-mode, eager-forward-only machinery: a view saved for
delayed autograd backward may already be unmapped. Conventional units report
loaded and use the same resident fast path as ``ResidentWeights``.

Allocation order follows ComfyUI's dynamic ``_load_list`` policy: keys
below 64 KiB are allocated first, then larger keys, and each class is
largest-first. ModelVBAR addresses encode eviction priority (higher
addresses evict first), so this keeps small latency-sensitive weights at
lower addresses while still preferring larger weights within each class.
"""

from __future__ import annotations

import bisect
import importlib
import threading
import time
import weakref
from collections.abc import Callable, Generator, MutableMapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, replace
from math import prod
from types import MappingProxyType
from typing import Any, BinaryIO, Literal, Protocol, cast

import torch
from dinkster_inference.patches import (
    AdapterPatch,
    DiffPatch,
    ModelAsLoraPatch,
    NestedPatch,
    PatchEntry,
    PatchSet,
    calculate_shape,
    patch_payloads,
)
from dinkster_memory import PageMap

from . import pinned_host
from .adapters import BOFTAdapter, GLoRAAdapter, LoHaAdapter, LoKrAdapter, LoRAAdapter, OFTAdapter
from .aimdo_activation import AimdoUnavailableError, ensure_visible_aimdo_devices
from .apply import StoredWeight, patch_stored_weight
from .memory import get_free_memory
from .ops import DeferredPatch, PreparedPatchSource, WeightFunction, cast_weight
from .quant import FP8_DTYPES, Fp8ScaledWeight, Int8PackedWeight, Nvfp4PackedWeight
from .residency import (
    ResidencyUnit,
    ResidencyUnitState,
    WeightLease,
    _preserve_parameter_registration,  # pyright: ignore[reportPrivateUsage]
    _validate_residency_layout,  # pyright: ignore[reportPrivateUsage]
    move_stored,
    stored_nbytes,
)
from .residency_timing import (
    DEQUANT,
    EXPOSED_STALL,
    TRANSFER,
    PartialResidencyTiming,
    active_partial_residency_timing,
    timed_phase,
)
from .sources import tensor_file_slice

_VBAR_ALIGNMENT = 512
_VBAR_PAGE_SIZE = 32 << 20
_RAW_SCALE_ALIGNMENT = 4
_EAGER_UNIT_LIMIT = 16 * 1024
_CAST_ARENA_RESERVATION_BYTES = 16 * 1024**3
_CAST_ARENA_ALIGNMENT = 1024
_PREFETCH_ADMISSION_CACHE_NS = 100_000_000
_BOUNDED_ADAPTER_TYPES = (
    LoRAAdapter,
    LoHaAdapter,
    LoKrAdapter,
    GLoRAAdapter,
    OFTAdapter,
    BOFTAdapter,
)
# These exceed the maximum concurrent max-sized results in the closed patch
# operations while separately charging every converted payload kept live.
_PATCH_VALUE_WORKSPACE_BUFFERS = 8
_ADAPTER_WORKSPACE_BUFFERS = 16
_production_vbars_guard = threading.Lock()
_production_vbars: dict[int, list[weakref.ReferenceType[object]]] = {}
_cuda_context_state = threading.local()


def _physical_free_bytes(device: torch.device) -> int:
    return get_free_memory(device).free_total


def _register_production_vbar(vbar: object, device_index: int) -> None:
    try:
        reference = weakref.ref(vbar)
    except TypeError:
        # Older extension objects may not expose weak-reference support.
        # Leaving them unclassified is conservative and does not retain them.
        return
    with _production_vbars_guard:
        _production_vbars.setdefault(device_index, []).append(reference)


class AimdoForceFullLoadError(RuntimeError):
    """Demand-paged aimdo residency cannot force all weights resident."""


class VbarBackend(Protocol):
    """The native VBAR operations used by :class:`AimdoWeights`.

    ``materialization_device`` lets the CPU test backend preserve a CUDA
    public mechanism identity while returning ordinary CPU uint8 tensors.
    The production backend always returns the requested CUDA device.
    """

    @property
    def requires_cuda_context(self) -> bool: ...

    def create_vbar(self, size: int, device_index: int) -> object: ...

    def alloc(self, vbar: object, size: int) -> object: ...

    def span(self, allocations: Sequence[object]) -> object: ...

    def fault(self, allocation: object) -> object | None: ...

    def signature_compare(self, left: object, right: object) -> bool: ...

    def unpin(self, allocation: object) -> None: ...

    def free_memory(self, vbar: object, size: int) -> int: ...

    def loaded_size(self, vbar: object) -> int: ...

    def prioritize(self, vbar: object) -> None: ...

    def deprioritize(self, vbar: object) -> None: ...

    def set_watermark_limit(self, vbar: object, size: int) -> None: ...

    def alloc_to_uint8_tensor(self, allocation: object, device: torch.device) -> torch.Tensor: ...

    def create_stream(self, device: torch.device) -> object: ...

    def current_stream(self, device: torch.device) -> object: ...

    def stream_wait_stream(self, stream: object, other: object) -> None: ...

    def stream_context(self, stream: object) -> AbstractContextManager[None]: ...

    def synchronize_stream(self, stream: object) -> None: ...

    def record_event(self, stream: object) -> object: ...

    def event_query(self, event: object) -> bool: ...

    def synchronize_event(self, event: object) -> None: ...

    def create_cast_arena(self, size: int, device_index: int) -> object: ...

    def cast_arena_size(self, arena: object) -> int: ...

    def cast_arena_to_uint8_tensor(
        self,
        arena: object,
        size: int,
        offset: int,
        device: torch.device,
    ) -> torch.Tensor: ...

    def materialization_device(self, device: torch.device) -> torch.device: ...

    def create_host_buffer(self, prewarm: int, max_grow_size: int) -> object: ...
    def host_buffer_size(self, host_buffer: object) -> int: ...
    def extend_host_buffer(self, host_buffer: object, size: int) -> None: ...
    def host_buffer_tensor(self, host_buffer: object) -> torch.Tensor: ...
    def truncate_host_buffer(self, host_buffer: object, size: int, unregister: bool) -> None: ...
    def read_file_slice(
        self,
        file: BinaryIO,
        offset: int,
        target: torch.Tensor,
        stream: object | None,
    ) -> None: ...
    def cleanup_file_reader(self) -> None: ...
    def register_host_memory(self, tensor: torch.Tensor) -> bool: ...
    def unregister_host_memory(self, tensor: torch.Tensor) -> bool: ...
    def discard_cuda_async_error(self) -> None: ...
    def is_pinned(self, tensor: torch.Tensor) -> bool: ...


class ComfyAimdoBackend:
    """Lazy production wrapper around dinkster-aimdo's public Python API."""

    def __init__(self) -> None:
        # dinkster-aimdo's torch bridge builds tensors from
        # __cuda_array_interface__ holders, which torch.as_tensor rejects
        # without numpy. Fail with the typed error instead of a deep
        # RuntimeError inside dinkster_aimdo when the environment lacks it.
        try:
            importlib.import_module("numpy")
        except ImportError as error:
            raise AimdoUnavailableError(
                "dinkster-aimdo tensor views require numpy; install numpy in"
                " the inference worker environment"
            ) from error
        # Import only after ensure_aimdo_devices proved get_devctx. In
        # Upstream comfy-aimdo 0.4.13 model_vbar captures control.lib at import.
        from dinkster_aimdo import (  # pyright: ignore[reportMissingTypeStubs]
            host_buffer,
            model_vbar,
            vram_buffer,
        )
        from dinkster_aimdo import (  # pyright: ignore[reportMissingTypeStubs]
            torch as aimdo_torch,
        )

        self._model_vbar = model_vbar
        self._aimdo_torch = aimdo_torch
        self._vram_buffer = vram_buffer
        self._host_buffer = host_buffer

    @property
    def requires_cuda_context(self) -> bool:
        return True

    def create_vbar(self, size: int, device_index: int) -> object:
        vbar = self._model_vbar.ModelVBAR(size, device_index)
        _register_production_vbar(vbar, device_index)
        return vbar

    def alloc(self, vbar: object, size: int) -> object:
        return cast(Any, vbar).alloc(size)

    def span(self, allocations: Sequence[object]) -> object:
        if not allocations:
            raise ValueError("VBAR span requires at least one allocation")
        if len(allocations) == 1:
            return allocations[0]
        parts = [cast("tuple[object, int, int]", allocation) for allocation in allocations]
        vbar = parts[0][0]
        if any(part[0] is not vbar for part in parts[1:]):
            raise ValueError("VBAR span allocations must share one VBAR")
        start = parts[0][1]
        end = max(address + size for _, address, size in parts)
        return vbar, start, end - start

    def fault(self, allocation: object) -> object | None:
        return self._model_vbar.vbar_fault(allocation)

    def signature_compare(self, left: object, right: object) -> bool:
        if len(cast(Any, left)) != len(cast(Any, right)):
            raise ValueError("VBAR signatures have mismatched lengths")
        return bytes(cast(Any, left)) == bytes(cast(Any, right))

    def unpin(self, allocation: object) -> None:
        self._model_vbar.vbar_unpin(allocation)

    def free_memory(self, vbar: object, size: int) -> int:
        return int(cast(Any, vbar).free_memory(size))

    def loaded_size(self, vbar: object) -> int:
        return int(cast(Any, vbar).loaded_size())

    def prioritize(self, vbar: object) -> None:
        cast(Any, vbar).prioritize()

    def deprioritize(self, vbar: object) -> None:
        cast(Any, vbar).deprioritize()

    def set_watermark_limit(self, vbar: object, size: int) -> None:
        cast(Any, vbar).set_watermark_limit(size)

    def alloc_to_uint8_tensor(self, allocation: object, device: torch.device) -> torch.Tensor:
        return self._aimdo_torch.aimdo_to_tensor(allocation, device)

    def create_stream(self, device: torch.device) -> object:
        return torch.cuda.Stream(device=device, priority=0)

    def current_stream(self, device: torch.device) -> object:
        stream_data = torch._C._cuda_getCurrentStream(device.index)  # pyright: ignore[reportPrivateUsage]
        cached = getattr(_cuda_context_state, "current_stream", None)
        if cached is not None and cached[0] == stream_data:
            return cached[1]
        stream = torch.cuda.current_stream(device)
        _cuda_context_state.current_stream = (stream_data, stream)
        return stream

    def stream_wait_stream(self, stream: object, other: object) -> None:
        cast(Any, stream).wait_stream(other)

    def stream_context(self, stream: object) -> AbstractContextManager[None]:
        return torch.cuda.stream(cast("torch.cuda.Stream", stream))

    def synchronize_stream(self, stream: object) -> None:
        cast(Any, stream).synchronize()

    def record_event(self, stream: object) -> object:
        event = torch.cuda.Event()
        cast(Any, event).record(stream)
        return event

    def event_query(self, event: object) -> bool:
        return bool(cast(Any, event).query())

    def synchronize_event(self, event: object) -> None:
        cast(Any, event).synchronize()

    def create_cast_arena(self, size: int, device_index: int) -> object:
        return self._vram_buffer.VRAMBuffer(size, device_index)

    def cast_arena_size(self, arena: object) -> int:
        return int(cast(Any, arena).size())

    def cast_arena_to_uint8_tensor(
        self,
        arena: object,
        size: int,
        offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        allocation = cast(Any, arena).get(size, offset)
        return self._aimdo_torch.aimdo_to_tensor(allocation, device)

    def materialization_device(self, device: torch.device) -> torch.device:
        return device

    def create_host_buffer(self, prewarm: int, max_grow_size: int) -> object:
        return self._host_buffer.HostBuffer(0, prewarm, max_grow_size)

    def host_buffer_size(self, host_buffer: object) -> int:
        return int(cast(Any, host_buffer).size)

    def extend_host_buffer(self, host_buffer: object, size: int) -> None:
        cast(Any, host_buffer).extend(size, register=False)

    def host_buffer_tensor(self, host_buffer: object) -> torch.Tensor:
        tensor = self._aimdo_torch.hostbuf_to_tensor(host_buffer)
        cast(Any, tensor.untyped_storage())._dinkster_hostbuf = host_buffer
        return tensor

    def truncate_host_buffer(self, host_buffer: object, size: int, unregister: bool) -> None:
        cast(Any, host_buffer).truncate(size, do_unregister=unregister)

    def read_file_slice(
        self,
        file: BinaryIO,
        offset: int,
        target: torch.Tensor,
        stream: object | None,
    ) -> None:
        stream_ptr = getattr(stream, "cuda_stream", 0) if stream is not None else 0
        if target.device.type == "cpu":
            hostbuf = getattr(target.untyped_storage(), "_dinkster_hostbuf", None)
            if hostbuf is None:
                raise RuntimeError("direct file read requires an Aimdo host buffer")
            hostbuf.read_file_slice(
                file,
                offset,
                target.nbytes,
                offset=target.data_ptr() - hostbuf.get_raw_address(),
                stream=stream_ptr,
            )
            return
        self._host_buffer.read_file_to_device(
            file,
            offset,
            target.nbytes,
            stream_ptr,
            target.data_ptr(),
            target.device.index,
            mark_cold=True,
        )

    def cleanup_file_reader(self) -> None:
        self._host_buffer.cleanup_file_reader()

    def register_host_memory(self, tensor: torch.Tensor) -> bool:
        return (
            cast(Any, torch.cuda.cudart()).cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 1)
            == 0
        )

    def unregister_host_memory(self, tensor: torch.Tensor) -> bool:
        return cast(Any, torch.cuda.cudart()).cudaHostUnregister(tensor.data_ptr()) == 0

    def discard_cuda_async_error(self) -> None:
        try:
            device = torch.device("cuda", torch.cuda.current_device())
            torch.ones(1, dtype=torch.uint8, device=device).add_(1)
            torch.cuda.synchronize(device)
        except RuntimeError:
            pass

    def is_pinned(self, tensor: torch.Tensor) -> bool:
        return tensor.is_pinned()


@dataclass(frozen=True)
class _Geometry:
    shape: tuple[int, ...]
    cast_bytes: int
    raw_bytes: int
    allocation_bytes: int


@dataclass(frozen=True)
class _CacheEntry:
    signature: object
    value: StoredWeight


_CacheForm = Literal["get", "get_stored"]
_CacheKey = tuple[bytes, torch.dtype, _CacheForm, str]


@dataclass(frozen=True, slots=True)
class _BatchRequest:
    key: str
    dtype: torch.dtype
    form: _CacheForm


@dataclass(frozen=True, slots=True)
class _ArenaLayout:
    request_offsets: dict[int, int]
    patch_offsets: dict[int, int]
    end: int


@dataclass(frozen=True, slots=True)
class _PatchRequest:
    key: str
    index: int
    source_id: int
    dtype: torch.dtype
    shape: tuple[int, ...]


_PinSubset = Literal["weights", "patches"]
_PinRequest = _BatchRequest | _PatchRequest
_PinIdentity = tuple[_PinSubset, object]


@dataclass
class _Pin:
    request: _PinRequest
    identity: _PinIdentity
    tensor: torch.Tensor
    value: StoredWeight
    offset: int
    registered: bool
    stack_index: int
    priority: int
    bucket_entry: list[Any]


def _bit_reverse_16(value: int) -> int:
    return int(f"{value & 0xFFFF:016b}"[::-1], 2)


@dataclass
class _StreamState:
    backend: VbarBackend
    device: torch.device
    stream_count: int
    streams: list[object]
    arenas: dict[int, object]
    counter: int = 0
    started: bool = False
    largest_ref: tuple[int, str] | None = None
    largest_size: int = 0

    def rotate(self) -> object:
        if not self.streams:
            self.streams = [
                self.backend.create_stream(self.device) for _ in range(self.stream_count)
            ]
        if not self.started:
            self.started = True
            return self.streams[0]
        outgoing = self.streams[self.counter]
        self.backend.stream_wait_stream(outgoing, self.backend.current_stream(self.device))
        self.counter = (self.counter + 1) % len(self.streams)
        return self.streams[self.counter]

    def arena(self, stream: object) -> object:
        key = id(stream)
        arena = self.arenas.get(key)
        if arena is None:
            raw_index = self.device.index
            assert raw_index is not None
            arena = self.backend.create_cast_arena(_CAST_ARENA_RESERVATION_BYTES, raw_index)
            self.arenas[key] = arena
        return arena

    def reset_arenas(self) -> None:
        for stream in self.streams:
            self.backend.synchronize_stream(stream)
        self.arenas.clear()
        self.largest_ref = None
        self.largest_size = 0


_stream_states_guard = threading.Lock()
_stream_states: dict[tuple[int, int, int], _StreamState] = {}


def _stream_state(backend: VbarBackend, device: torch.device, stream_count: int) -> _StreamState:
    raw_index = device.index
    assert raw_index is not None
    backend_key = 0 if isinstance(backend, ComfyAimdoBackend) else id(backend)
    key = (backend_key, raw_index, stream_count)
    with _stream_states_guard:
        state = _stream_states.get(key)
        if state is None:
            state = _StreamState(backend, device, stream_count, [], {})
            _stream_states[key] = state
        return state


def _is_compiling() -> bool:
    return bool(torch.compiler.is_compiling())


_device_locks_guard = threading.Lock()
_device_locks: dict[int, threading.RLock] = {}


@dataclass(frozen=True, slots=True)
class _PendingUnpins:
    owner: object
    backend: VbarBackend
    event: object
    allocations: list[object]


@dataclass(frozen=True, slots=True)
class _PendingStreamUnpins:
    stream: object
    entries: list[_PendingUnpins]


# Graph-cache eviction can retire a mechanism before its consumer events complete.
_device_unpins: dict[tuple[int, int], list[_PendingStreamUnpins]] = {}


def _device_lock(device_index: int) -> threading.RLock:
    with _device_locks_guard:
        return _device_locks.setdefault(device_index, threading.RLock())


def _reap_device_unpins(
    key: tuple[int, int],
    *,
    wait_owner: object | None,
) -> None:
    pending = _device_unpins.get(key)
    if not pending:
        return
    retained_streams: list[_PendingStreamUnpins] = []
    for stream_pending in pending:
        entries = stream_pending.entries
        start = 0
        if wait_owner is not None:
            for index in range(len(entries) - 1, -1, -1):
                if entries[index].owner is wait_owner:
                    entries[index].backend.synchronize_event(entries[index].event)
                    for entry in entries[: index + 1]:
                        for allocation in entry.allocations:
                            entry.backend.unpin(allocation)
                    start = index + 1
                    break
        for index in range(start, len(entries)):
            entry = entries[index]
            if not entry.backend.event_query(entry.event):
                if index == 0:
                    retained_streams.append(stream_pending)
                else:
                    retained_streams.append(
                        _PendingStreamUnpins(stream_pending.stream, entries[index:])
                    )
                break
            for allocation in entry.allocations:
                entry.backend.unpin(allocation)
    if retained_streams:
        _device_unpins[key] = retained_streams
    else:
        _device_unpins.pop(key, None)


def _vbar_page_flags(statuses: Sequence[object]) -> tuple[int, ...]:
    flags: list[int] = []
    for value in statuses:
        if type(value) is not int or value < 0 or value & ~3 or value == 2:
            raise RuntimeError(f"dinkster-aimdo returned invalid VBAR page status {value!r}")
        flags.append(value)
    return tuple(flags)


def _classify_vbar_pages(statuses: Sequence[object]) -> tuple[int, int]:
    evictable_pages = 0
    pinned_pages = 0
    for value in _vbar_page_flags(statuses):
        if value == 1:
            evictable_pages += 1
        elif value == 3:
            pinned_pages += 1
    return evictable_pages * _VBAR_PAGE_SIZE, pinned_pages * _VBAR_PAGE_SIZE


def production_vbar_memory(device_index: int) -> tuple[int, int]:
    """Return evictable and pinned bytes for live Dinkster-owned VBARs."""
    with _device_lock(device_index):
        # Native unpin can synchronize and evict above a pressure-lowered
        # watermark. Mechanism paths reap; measurements may under-report
        # evictable pages until the next mechanism operation.
        with _production_vbars_guard:
            references = _production_vbars.get(device_index, ())
            live = [(reference, vbar) for reference in references if (vbar := reference())]
            if live:
                _production_vbars[device_index] = [reference for reference, _ in live]
            else:
                _production_vbars.pop(device_index, None)

        evictable = 0
        pinned = 0
        for _, vbar in live:
            try:
                query = getattr(vbar, "get_residency", None)
            except (AttributeError, ImportError):
                continue
            if not callable(query):
                continue
            try:
                statuses = cast("Sequence[object]", query())
            except (AttributeError, ImportError):
                # Older dinkster-aimdo builds cannot classify pages. Unknown
                # residency stays absent rather than becoming fake free.
                continue
            vbar_evictable, vbar_pinned = _classify_vbar_pages(statuses)
            evictable += vbar_evictable
            pinned += vbar_pinned
        return evictable, pinned


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _stored_shape(stored: StoredWeight) -> tuple[int, ...]:
    if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
        return stored.shape
    return tuple(stored.shape)


def _packed_tensors(
    stored: Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight,
) -> tuple[torch.Tensor, ...]:
    if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight):
        return stored.qdata, stored.scale
    return stored.qdata, stored.block_scale, stored.tensor_scale


def _stored_tensors(stored: StoredWeight) -> tuple[torch.Tensor, ...]:
    if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
        return _packed_tensors(stored)
    return (stored,)


def _has_file_slices(stored: StoredWeight) -> bool:
    return all(tensor_file_slice(tensor) is not None for tensor in _stored_tensors(stored))


def _raw_packed_bytes(
    stored: Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight,
) -> int:
    offset = 0
    for tensor in _packed_tensors(stored):
        offset = _align_up(offset, _RAW_SCALE_ALIGNMENT) + tensor.nbytes
    return offset


def _packed_from_backing(
    stored: Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight,
    backing: torch.Tensor,
) -> Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight:
    values: list[torch.Tensor] = []
    offset = 0
    for tensor in _packed_tensors(stored):
        offset = _align_up(offset, _RAW_SCALE_ALIGNMENT)
        values.append(
            backing[offset : offset + tensor.nbytes].view(tensor.dtype).reshape(tensor.shape)
        )
        offset += tensor.nbytes
    if isinstance(stored, Fp8ScaledWeight):
        return Fp8ScaledWeight(values[0], values[1], stored.orig_dtype)
    if isinstance(stored, Int8PackedWeight):
        return replace(stored, qdata=values[0], scale=values[1])
    return replace(stored, qdata=values[0], block_scale=values[1], tensor_scale=values[2])


def _raw_request_dtype(stored: StoredWeight) -> torch.dtype:
    """Storage dtype transported by a raw (``dtype=None``) request.

    Packed storage streams its packed component tensors; a plain
    integer tensor (GGUF encoded blocks) streams its exact bytes.
    Plain floating-point state has no raw consumers and fails closed.
    """
    if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
        return stored.qdata.dtype
    if stored.is_floating_point() or stored.is_complex():
        raise TypeError("AimdoWeights.get_stored requires packed or integer storage")
    return stored.dtype


def _patch_admission_bound(
    entries: Sequence[PatchEntry[torch.Tensor]],
    base_shape: tuple[int, ...],
) -> tuple[int, int, int] | None:
    intermediate_itemsize = 0
    workspace_buffers = 0
    largest_intermediate = 0
    current_shape = base_shape
    for entry in entries:
        if entry.function is not None:
            return None
        value = entry.value
        active_shape = current_shape
        if entry.offset is not None:
            if entry.offset.dim >= len(active_shape):
                return None
            active = list(active_shape)
            active[entry.offset.dim] = entry.offset.length
            active_shape = tuple(active)
        if isinstance(value, AdapterPatch):
            adapter = value.adapter
            if type(adapter) not in _BOUNDED_ADAPTER_TYPES:
                return None
            target_shape = adapter.target_shape(active_shape)
            adapter_dtype = cast(Any, adapter).intermediate_dtype
            intermediate_numel = _adapter_intermediate_numel(adapter, target_shape)
            if intermediate_numel is None:
                return None
            intermediate_itemsize = max(
                intermediate_itemsize,
                adapter_dtype.itemsize,
            )
            largest_intermediate = max(
                largest_intermediate,
                intermediate_numel * adapter_dtype.itemsize,
            )
            workspace_buffers += _ADAPTER_WORKSPACE_BUFFERS
            if entry.offset is None:
                current_shape = target_shape
        elif isinstance(value, NestedPatch):
            if value.convert is not None:
                return None
            nested = _patch_admission_bound(value.entries, tuple(value.base.shape))
            if nested is None:
                return None
            intermediate_itemsize = max(intermediate_itemsize, nested[0])
            workspace_buffers += nested[1] + _PATCH_VALUE_WORKSPACE_BUFFERS
            largest_intermediate = max(largest_intermediate, nested[2])
        elif isinstance(value, ModelAsLoraPatch):
            return None
        else:
            workspace_buffers += _PATCH_VALUE_WORKSPACE_BUFFERS
            if entry.offset is None and isinstance(value, DiffPatch) and value.pad_weight:
                current_shape = tuple(value.value.shape)
    return intermediate_itemsize, workspace_buffers, largest_intermediate


def _adapter_intermediate_numel(
    adapter: object,
    target_shape: tuple[int, ...],
) -> int | None:
    if cast(Any, adapter).dora_scale is not None:
        return None
    largest = prod(target_shape)
    if type(adapter) is LoRAAdapter:
        lora = adapter
        if lora.up.dim() >= 2 and lora.down.dim() >= 2:
            final_columns = prod(lora.down.shape[1:])
            if lora.mid is not None and lora.mid.dim() >= 4:
                largest = max(
                    largest,
                    lora.down.shape[1] * lora.mid.shape[0] * prod(lora.mid.shape[2:]),
                )
                final_columns = lora.down.shape[1] * lora.mid.shape[2] * lora.mid.shape[3]
            largest = max(largest, lora.up.shape[0] * final_columns)
    elif type(adapter) is LoHaAdapter:
        loha = adapter
        if loha.t1 is not None and loha.t2 is not None:
            return None
        if all(payload.dim() >= 2 for payload in (loha.w1_a, loha.w1_b, loha.w2_a, loha.w2_b)):
            m1_shape = (loha.w1_a.shape[0], loha.w1_b.shape[1])
            m2_shape = (loha.w2_a.shape[0], loha.w2_b.shape[1])
            if m1_shape != m2_shape or prod(m1_shape) != prod(target_shape):
                return None
            largest = max(largest, prod(m1_shape))
        else:
            return None
    elif type(adapter) is LoKrAdapter:
        lokr = adapter
        if lokr.w2 is None and lokr.t2 is not None:
            return None
        w1_numel = 0
        if lokr.w1 is not None:
            w1_numel = lokr.w1.numel()
        elif (
            lokr.w1_a is not None
            and lokr.w1_b is not None
            and lokr.w1_a.dim() >= 2
            and lokr.w1_b.dim() >= 2
        ):
            w1_numel = lokr.w1_a.shape[0] * lokr.w1_b.shape[1]
        else:
            return None
        w2_numel = 0
        if lokr.w2 is not None:
            w2_numel = lokr.w2.numel()
        elif (
            lokr.w2_a is not None
            and lokr.w2_b is not None
            and lokr.w2_a.dim() >= 2
            and lokr.w2_b.dim() >= 2
        ):
            w2_numel = lokr.w2_a.shape[0] * lokr.w2_b.shape[1]
        else:
            return None
        largest = max(largest, w1_numel, w2_numel, w1_numel * w2_numel)
    elif type(adapter) is GLoRAAdapter:
        glora = adapter
        if not target_shape or any(
            payload.dim() < 2 for payload in (glora.a1, glora.a2, glora.b1, glora.b2)
        ):
            return None
        spatial = prod(target_shape[2:])
        a1_columns = prod(glora.a1.shape[1:])
        a2_columns = prod(glora.a2.shape[1:])
        b1_columns = prod(glora.b1.shape[1:])
        b2_columns = prod(glora.b2.shape[1:])
        target_rows = target_shape[0] if target_shape else 0
        target_columns = target_shape[1] if len(target_shape) >= 2 else None
        old_glora = glora.b2.shape[1] == glora.b1.shape[0] == glora.a1.shape[0] == glora.a2.shape[1]
        new_glora = glora.b2.shape[0] == glora.b1.shape[1] == glora.a1.shape[1] == glora.a2.shape[0]
        if new_glora and not (
            old_glora and glora.a2.shape[0] == target_rows and target_rows == target_columns
        ):
            old_glora = False
        target_numel = prod(target_shape)
        if old_glora:
            b_shape = (glora.b2.shape[0], b1_columns)
            a_shape = (target_rows, a1_columns)
            if b_shape != a_shape or prod(a_shape) != target_numel:
                return None
            largest = max(
                largest,
                target_rows * a2_columns,
                prod(a_shape),
            )
        else:
            a_numel = target_rows * a2_columns * spatial
            b_numel = glora.b1.shape[0] * b2_columns
            if a_numel != target_numel or b_numel != target_numel:
                return None
            largest = max(largest, target_rows * a1_columns * spatial)
    elif type(adapter) is OFTAdapter:
        oft = adapter
        if oft.blocks.dim() != 3 or oft.blocks.shape[1] != oft.blocks.shape[2]:
            return None
    elif type(adapter) is BOFTAdapter:
        boft = adapter
        if boft.blocks.dim() != 4 or boft.blocks.shape[2] != boft.blocks.shape[3]:
            return None
        if boft.rescale is not None:
            try:
                rescaled_shape = torch.broadcast_shapes(target_shape, tuple(boft.rescale.shape))
            except RuntimeError:
                return None
            if tuple(rescaled_shape) != target_shape:
                return None
    return largest


def _signature_bytes(signature: object) -> bytes:
    try:
        return memoryview(cast(Any, signature)).tobytes()
    except TypeError:
        try:
            return bytes(cast(Any, signature))
        except Exception as error:
            raise TypeError("aimdo fault signature does not expose bytes") from error


class _AimdoWeightLease:
    """Private per-forward fault-and-transfer scope.

    The timing ``collector`` binds when the mechanism constructs the
    lease at bracket entry (or when ``prefetch`` starts staging), so a
    lease participates in exactly the collection window active when it
    opens. ``prefetching`` attributes this lease's transfers to the
    prefetch counters instead of the lease counters.
    """

    __slots__ = (
        "_mechanism",
        "_unit",
        "_successful_faults",
        "_faulted_units",
        "_stream",
        "_arena_offset",
        "_closed",
        "_collector",
        "_prefetching",
    )

    def __init__(
        self,
        mechanism: AimdoWeights,
        unit: str,
        *,
        collector: PartialResidencyTiming | None = None,
        prefetching: bool = False,
    ) -> None:
        self._mechanism = mechanism
        self._unit = unit
        self._successful_faults: list[object] = []
        self._faulted_units: dict[str, object | None] = {}
        self._stream: object | None = None
        self._arena_offset = 0
        self._closed = False
        self._collector = collector
        self._prefetching = prefetching

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"lease for unit {self._unit!r} is closed")

    def get(self, key: str, *, dtype: torch.dtype) -> torch.Tensor:
        self._ensure_open()
        value = self._mechanism._lease_get_many(  # pyright: ignore[reportPrivateUsage]
            self,
            self._mechanism._singleton_request(  # pyright: ignore[reportPrivateUsage]
                key, dtype
            ),
        )[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("cached cast form is not a tensor")
        return value

    def get_many(self, requests: Sequence[tuple[str, torch.dtype | None]]) -> list[StoredWeight]:
        """Fault and transfer several keys as one private lease batch.

        A ``None`` dtype requests the raw stored form (packed components
        or a plain integer tensor's exact bytes). The method is
        intentionally absent from ``WeightLease`` until prefetch consumers
        land; future remote/multi-device work can drive this concrete
        mechanism across unit boundaries.
        """
        self._ensure_open()
        batch: list[_BatchRequest] = []
        for key, dtype in requests:
            batch.extend(
                self._mechanism._singleton_request(  # pyright: ignore[reportPrivateUsage]
                    key, dtype
                )
            )
        return self._mechanism._lease_get_many(  # pyright: ignore[reportPrivateUsage]
            self, batch
        )

    def get_stored(self, key: str) -> StoredWeight:
        self._ensure_open()
        return self._mechanism._lease_get_many(  # pyright: ignore[reportPrivateUsage]
            self,
            self._mechanism._singleton_request(  # pyright: ignore[reportPrivateUsage]
                key, None
            ),
        )[0]

    def timing_collector(self) -> PartialResidencyTiming | None:
        return self._collector

    def pin(self, allocation: object) -> None:
        self._successful_faults.append(allocation)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        current_stream: object | None = None
        if self._stream is not None:
            try:
                current_stream = self._mechanism._backend.current_stream(  # pyright: ignore[reportPrivateUsage]
                    self._mechanism.load_device
                )
                self._mechanism._backend.stream_wait_stream(  # pyright: ignore[reportPrivateUsage]
                    self._stream,
                    current_stream,
                )
            except BaseException as error:
                first_error = error
        if self._successful_faults:
            try:
                if self._prefetching:
                    for allocation in self._successful_faults:
                        self._mechanism._backend.unpin(  # pyright: ignore[reportPrivateUsage]
                            allocation
                        )
                else:
                    if current_stream is None:
                        current_stream = self._mechanism._backend.current_stream(  # pyright: ignore[reportPrivateUsage]
                            self._mechanism.load_device
                        )
                    event = self._mechanism._backend.record_event(  # pyright: ignore[reportPrivateUsage]
                        current_stream
                    )
                    self._mechanism._defer_unpins(  # pyright: ignore[reportPrivateUsage]
                        current_stream, event, self._successful_faults
                    )
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


class _AimdoPrefetch:
    __slots__ = ("_mechanism", "_lease", "_requests", "_closed")

    def __init__(
        self,
        mechanism: AimdoWeights,
        lease: _AimdoWeightLease,
        requests: list[_BatchRequest],
    ) -> None:
        self._mechanism = mechanism
        self._lease = lease
        self._requests = requests
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        mechanism = self._mechanism
        try:
            with mechanism._cuda_context():  # pyright: ignore[reportPrivateUsage]
                for request in self._requests:
                    prefetched = mechanism._prefetched.get(request)  # pyright: ignore[reportPrivateUsage]
                    if prefetched is not None and prefetched[0] is self:
                        del mechanism._prefetched[request]  # pyright: ignore[reportPrivateUsage]
                self._lease.close()
        finally:
            mechanism.pin_active = False
            mechanism._lock.release()  # pyright: ignore[reportPrivateUsage]


class AimdoWeights:
    """Hybrid eager-tiny and demand-paged residency over CPU originals."""

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
        backend: VbarBackend | None = None,
        stream_count: int = 2,
        pin_all_sources: bool = False,
        physical_free_memory: Callable[[torch.device], int] | None = None,
        fixed_promotion: bool = False,
        promote_non_fp8_raw: bool = False,
    ) -> None:
        if stream_count < 0:
            raise ValueError("AimdoWeights stream_count must be non-negative")
        if type(fixed_promotion) is not bool:
            raise TypeError("AimdoWeights fixed_promotion must be a bool")
        if type(promote_non_fp8_raw) is not bool:
            raise TypeError("AimdoWeights promote_non_fp8_raw must be a bool")
        device = torch.device(load_device)
        if device.type != "cuda":
            raise ValueError(f"AimdoWeights requires a CUDA load device, got {device.type!r}")
        raw_index = cast("int | None", device.index)
        index = torch.cuda.current_device() if raw_index is None else raw_index
        self.load_device = torch.device("cuda", index)
        self.offload_device = torch.device(offload_device)
        self._weights = weights
        self._intermediate_dtype = intermediate_dtype
        self._patch_weight_dtype = patch_weight_dtype
        self._patch_key_prefix = patch_key_prefix
        self._units, self._unit_of, self._entries = _validate_residency_layout(
            weights, patch_set, units
        )
        self._units_by_name = MappingProxyType({unit.name: unit for unit in self._units})
        raw_residency = getattr(weights, "uses_raw_residency", None)
        if raw_residency is not None and not callable(raw_residency):
            raise TypeError("uses_raw_residency must be callable")
        self._raw_residency_keys = frozenset(
            key
            for key in weights
            if raw_residency is not None and raw_residency(key) and not self._entries.get(key)
        )
        itemsize_capability = getattr(weights, "max_materialized_itemsize", None)
        if itemsize_capability is not None and not callable(itemsize_capability):
            raise TypeError("max_materialized_itemsize must be callable")
        materialized_itemsizes: dict[str, int] = {}
        for key in weights:
            itemsize = (
                torch.float32.itemsize if itemsize_capability is None else itemsize_capability(key)
            )
            if type(itemsize) is not int or not 1 <= itemsize <= torch.float32.itemsize:
                raise ValueError(
                    f"max materialized itemsize for {key!r} must be a positive integer"
                    f" no greater than {torch.float32.itemsize}, got {itemsize!r}"
                )
            materialized_itemsizes[key] = itemsize
        self._max_materialized_itemsizes = MappingProxyType(materialized_itemsizes)
        self._patch_revision = (
            "" if patch_set is None else (patch_set.structural_digest or patch_set.revision)
        )
        self._functions: dict[str, tuple[WeightFunction, ...]] = {}
        self._prepared_sources: dict[str, PreparedPatchSource] = {}
        for key in weights:
            entries = self._entries.get(key)
            if entries is None:
                self._functions[key] = ()
                continue
            deferred = DeferredPatch(key=key, entries=entries)
            self._functions[key] = (deferred,)
            self._prepared_sources[key] = PreparedPatchSource(
                deferred, alignment=_CAST_ARENA_ALIGNMENT
            )
        self._geometry = {key: self._geometry_for(key, stored) for key, stored in weights.items()}
        self._unit_vbar_sizes = MappingProxyType(
            {
                unit.name: sum(
                    _align_up(self._geometry[key].allocation_bytes, _VBAR_ALIGNMENT)
                    for key in unit.keys
                )
                for unit in self._units
            }
        )
        self._eager_units = {
            unit.name
            for unit in self._units
            if sum(stored_nbytes(weights[key]) for key in unit.keys) <= _EAGER_UNIT_LIMIT
        }
        self._fixed_promotion = fixed_promotion
        self._promote_non_fp8_raw = promote_non_fp8_raw
        self._promoted_units: set[str] = set()
        self._eager_loaded: set[str] = set()
        self._unit_states = {unit.name: ResidencyUnitState() for unit in self._units}
        self._eager_backup: dict[str, StoredWeight] = {}
        self._working_set_reservations = 0
        demand_units = tuple(unit for unit in self._units if unit.name not in self._eager_units)
        demand_keys = {key for unit in demand_units for key in unit.keys}
        self._reservation_bytes = sum(self._unit_vbar_bytes(unit) for unit in demand_units)
        self._lock = _device_lock(index)
        self._cache: dict[str, dict[_CacheKey, _CacheEntry]] = {}
        self._prefetched: dict[_BatchRequest, tuple[_AimdoPrefetch, StoredWeight]] = {}
        self._singleton_requests: dict[tuple[str, torch.dtype | None], tuple[_BatchRequest]] = {}
        self._request_size_cache: dict[_BatchRequest, int] = {}
        self._preflight_request_cache: set[tuple[_BatchRequest, ...]] = set()
        self._prefetch_peak_cache: dict[tuple[_BatchRequest, ...], int | None] = {}
        self._prefetch_admission: tuple[int, int] | None = None
        self._unpin_owner = object()
        self._pin_all_sources = pin_all_sources
        self._pin_state: dict[
            str,
            tuple[
                object,
                list[tuple[_Pin, int]],
                list[int],
                list[int],
                list[int],
                dict[int, list[list[Any]]],
            ],
        ] = {}
        self._pins: dict[_PinIdentity, _Pin] = {}
        self._pin_priorities: dict[_PinIdentity, int] = {}
        self.pin_active = False

        with self._lock:
            if backend is None:
                if not ensure_visible_aimdo_devices():
                    raise AimdoUnavailableError(f"dinkster-aimdo device {index} is not initialized")
                backend = ComfyAimdoBackend()
            self._backend = backend
            self._physical_free_memory = (
                _physical_free_bytes
                if physical_free_memory is None and backend.requires_cuda_context
                else physical_free_memory
            )
            patch_payload_geometry: dict[str, tuple[tuple[int, int], ...]] = {}
            patch_staging_bytes: dict[str, int] = {}
            patch_math_itemsize: dict[str, int] = {}
            patch_workspace_buffers: dict[str, int] = {}
            patch_largest_intermediate: dict[str, int] = {}
            unbounded_patch_keys: set[str] = set()
            if self._physical_free_memory is not None:
                for key, entries in self._entries.items():
                    bound = _patch_admission_bound(entries, tuple(self._weights[key].shape))
                    if bound is None:
                        unbounded_patch_keys.add(key)
                        continue
                    unsupported: list[object] = []
                    payloads = patch_payloads(entries, unsupported=unsupported.append)
                    geometry = tuple(
                        (payload.numel(), payload.dtype.itemsize) for payload in payloads
                    )
                    patch_payload_geometry[key] = geometry
                    patch_staging_bytes[key] = sum(
                        _align_up(numel * itemsize, _CAST_ARENA_ALIGNMENT)
                        for numel, itemsize in geometry
                    )
                    if (
                        unsupported
                        or not payloads
                        or any(payload.numel() == 0 for payload in payloads)
                    ):
                        unbounded_patch_keys.add(key)
                    else:
                        (
                            patch_math_itemsize[key],
                            patch_workspace_buffers[key],
                            patch_largest_intermediate[key],
                        ) = bound
            self._patch_payload_geometry = MappingProxyType(patch_payload_geometry)
            self._patch_staging_bytes = MappingProxyType(patch_staging_bytes)
            self._patch_math_itemsize = MappingProxyType(patch_math_itemsize)
            self._patch_workspace_buffers = MappingProxyType(patch_workspace_buffers)
            self._patch_largest_intermediate = MappingProxyType(patch_largest_intermediate)
            self._unbounded_patch_keys = frozenset(unbounded_patch_keys)
            backend_key = 0 if isinstance(backend, ComfyAimdoBackend) else id(backend)
            self._unpin_key = (backend_key, index)
            self._requires_cuda_context = backend.requires_cuda_context
            self._stream_count = stream_count
            self._stream_state = (
                None
                if stream_count == 0
                else _stream_state(backend, self.load_device, stream_count)
            )
            with self._cuda_context():
                self._vbar = (
                    backend.create_vbar(self._reservation_bytes, index) if demand_keys else None
                )
                order = tuple(
                    key
                    for unit in self._units
                    if unit.name not in self._eager_units
                    for key in unit.keys
                )
                self._allocations = {
                    key: backend.alloc(self._vbar, self._geometry[key].allocation_bytes)
                    for key in order
                }
                self._unit_allocations = {
                    unit.name: backend.span(tuple(self._allocations[key] for key in unit.keys))
                    for unit in self._units
                    if unit.name not in self._eager_units
                }

    def _initialize_pins(self) -> None:
        if self._pin_state or pinned_host.DISABLED:
            return
        maximum = pinned_host.pinned_hostbuf_size(self.total_bytes())
        self._pin_state = {
            "weights": (
                self._backend.create_host_buffer(64 * 1024**2, maximum),
                [],
                [-1],
                [0],
                [0],
                {},
            ),
            "patches": (
                self._backend.create_host_buffer(8 * 1024**2, maximum),
                [],
                [-1],
                [0],
                [0],
                {},
            ),
        }

    def pin_debug_label(self) -> str:
        keys = tuple(self._weights)
        sample = ",".join(keys[:2])
        return f"AimdoWeights(device={self.load_device},keys={sample},count={len(keys)})"

    @staticmethod
    def _pin_identity(subset: _PinSubset, request: _PinRequest) -> _PinIdentity:
        source: object = (
            request.key
            if isinstance(request, _BatchRequest)
            else (request.key, request.index, request.source_id)
        )
        return subset, source

    def _priority_for(self, subset: _PinSubset, request: _PinRequest) -> int:
        identity = self._pin_identity(subset, request)
        existing = self._pin_priorities.get(identity)
        if existing is not None:
            return existing
        counter = self._pin_state[subset][4]
        priority = _bit_reverse_16(counter[0])
        counter[0] += 1
        self._pin_priorities[identity] = priority
        return priority

    def _pin_for(
        self,
        request: _PinRequest,
        source: StoredWeight,
        *,
        subset: _PinSubset = "weights",
    ) -> StoredWeight | None:
        existing = self._existing_pin(request, subset=subset)
        if existing is not None:
            return existing
        self._initialize_pins()
        if not self._pin_state:
            return None
        priority = self._priority_for(subset, request)
        size = self._host_size(request)
        allocation_size = _align_up(size, _CAST_ARENA_ALIGNMENT) if subset == "patches" else size
        if not pinned_host.reserve_storage(self, allocation_size):
            pinned_host.discard_owner_if_empty(self)
            return self._steal_pin(request, source, allocation_size, priority, subset=subset)
        if not pinned_host.ensure_pin_budget(
            allocation_size
        ) or not pinned_host.ensure_pin_registerable(allocation_size):
            pinned_host.account_storage(self, -allocation_size)
            pinned_host.discard_owner_if_empty(self)
            return self._steal_pin(request, source, allocation_size, priority, subset=subset)
        hostbuf, stack, split, pinned_size, _, buckets = self._pin_state[subset]
        truncate_offset = self._backend.host_buffer_size(hostbuf)
        raw: torch.Tensor | None = None
        registered = False
        try:
            self._backend.extend_host_buffer(hostbuf, allocation_size)
            raw = self._backend.host_buffer_tensor(hostbuf)[
                truncate_offset : truncate_offset + allocation_size
            ]
            registered = self._backend.register_host_memory(raw)
            if not registered:
                self._backend.discard_cuda_async_error()
                pinned_host.free_registrations(allocation_size)
                registered = self._backend.register_host_memory(raw)
                if not registered:
                    self._backend.discard_cuda_async_error()
        except RuntimeError:
            # The active traceback references the extend/register frames'
            # buffers until the handler exits, so the truncate and the steal
            # fallback run after it. This path runs under host pinned-memory
            # exhaustion, where that retention matters.
            registered = False
        if not registered or raw is None:
            self._backend.truncate_host_buffer(hostbuf, truncate_offset, False)
            pinned_host.account_storage(self, -allocation_size)
            pinned_host.discard_owner_if_empty(self)
            return self._steal_pin(request, source, allocation_size, priority, subset=subset)
        identity = self._pin_identity(subset, request)
        entry: list[Any] = [-priority, 0, None]
        value = self._host_value_from_backing(request, raw)
        pin = _Pin(
            request, identity, raw, value, truncate_offset, True, len(stack), priority, entry
        )
        entry[1:] = [id(entry), pin]
        bisect.insort(buckets.setdefault(allocation_size, []), entry)
        stack.append((pin, truncate_offset))
        split[0] = max(split[0], pin.stack_index)
        pinned_size[0] += allocation_size
        pinned_host.account(allocation_size)
        self._pins[identity] = pin
        if not self._read_file_value(value, source, None):
            self._copy_value(value, source, non_blocking=False)
        return value

    def _existing_pin(
        self, request: _PinRequest, *, subset: _PinSubset = "weights"
    ) -> StoredWeight | None:
        pin = self._pins.get(self._pin_identity(subset, request))
        if pin is None:
            return None
        pinned_host.register_owner(self)
        if not pin.registered and pinned_host.ensure_pin_registerable(pin.tensor.nbytes):
            try:
                pin.registered = self._backend.register_host_memory(pin.tensor)
            except RuntimeError:
                pin.registered = False
            if not pin.registered:
                self._backend.discard_cuda_async_error()
            else:
                state = self._pin_state[subset]
                state[2][0] = max(state[2][0], pin.stack_index)
                state[3][0] += pin.tensor.nbytes
                pinned_host.account(pin.tensor.nbytes)
        if isinstance(request, _BatchRequest) or request == pin.request:
            return pin.value
        return self._host_value_from_backing(request, pin.tensor)

    def _synchronize_transfer_streams(self) -> None:
        state = self._stream_state
        if state is not None:
            for stream in state.streams:
                self._backend.synchronize_stream(stream)

    def _steal_pin(
        self,
        request: _PinRequest,
        source: StoredWeight,
        size: int,
        priority: int,
        *,
        subset: _PinSubset = "weights",
    ) -> StoredWeight | None:
        _, stack, _, _, _, buckets = self._pin_state[subset]
        bucket = buckets.get(size)
        while bucket and bucket[-1][-1] is None:
            bucket.pop()
        if not bucket or priority <= -bucket[-1][0]:
            return None
        self._synchronize_transfer_streams()
        victim = cast(_Pin, bucket.pop()[-1])
        self._pins.pop(victim.identity, None)
        identity = self._pin_identity(subset, request)
        entry: list[Any] = [-priority, 0, None]
        value = self._host_value_from_backing(request, victim.tensor)
        pin = _Pin(
            request,
            identity,
            victim.tensor,
            value,
            victim.offset,
            victim.registered,
            victim.stack_index,
            priority,
            entry,
        )
        entry[1:] = [id(entry), pin]
        bisect.insort(bucket, entry)
        stack[pin.stack_index] = (pin, pin.offset)
        self._pins[identity] = pin
        if not self._read_file_value(value, source, None):
            self._copy_value(value, source, non_blocking=False)
        return value

    @contextmanager
    def _cuda_context(self) -> Generator[None]:
        """Make this VBAR's CUDA context current on the calling thread.

        Native ModelVBAR selects aimdo's ``AimdoContext`` but calls CUDA
        driver APIs against the thread-current CUDA context. CUDA currency
        is thread-local, so construction, leases, and eviction operations
        must bind the target device on every calling thread before its first
        use. ``current_device`` can report the target index even when that
        thread has no current CUDA context yet.
        """
        if not self._requires_cuda_context:
            yield
            return
        device_index = self.load_device.index
        if getattr(_cuda_context_state, "active_device", None) == device_index:
            yield
            return
        previous = torch.cuda.current_device()
        initialized = getattr(_cuda_context_state, "initialized", None)
        if initialized is None:
            initialized = set()
            _cuda_context_state.initialized = initialized
        first_use = device_index not in initialized
        if first_use or previous != device_index:
            torch.cuda.set_device(self.load_device)
        if first_use:
            # A logical current-device selection alone need not instantiate
            # and bind torch's primary context on a fresh Python thread.
            torch.empty(0, device=self.load_device)
            initialized.add(device_index)
        try:
            yield
        finally:
            if previous != device_index:
                torch.cuda.set_device(previous)

    @contextmanager
    def execution_context(self) -> Generator[None]:
        with self._cuda_context():
            previous = getattr(_cuda_context_state, "active_device", None)
            _cuda_context_state.active_device = self.load_device.index
            try:
                yield
            finally:
                _cuda_context_state.active_device = previous

    def _geometry_for(self, key: str, stored: StoredWeight) -> _Geometry:
        # Unlike upstream's resizing-LoRA force-load, VBAR geometry is
        # planned from the patched target shape, so a large resized unit
        # remains safe to demand-page. The regression test pins this.
        shape = calculate_shape(_stored_shape(stored), self._entries.get(key, ()))
        if key in self._raw_residency_keys:
            raw_bytes = (
                _raw_packed_bytes(stored)
                if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight)
                else stored.nbytes
            )
            return _Geometry(shape, raw_bytes, raw_bytes, raw_bytes)
        cast_bytes = prod(shape) * self._max_materialized_itemsizes[key]
        if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            raw_bytes = _raw_packed_bytes(stored)
        elif stored.is_floating_point() or stored.is_complex():
            # Plain float state has no raw consumers; keeping raw_bytes
            # at zero keeps its allocation sized by the cast ceiling.
            raw_bytes = 0
        else:
            raw_bytes = stored.nbytes
        return _Geometry(shape, cast_bytes, raw_bytes, max(cast_bytes, raw_bytes))

    @property
    def demand_paged(self) -> bool:
        return True

    def _unit_bytes(self, unit: ResidencyUnit) -> int:
        return sum(stored_nbytes(self._weights[key]) for key in unit.keys)

    def _unit_working_bytes(self, unit: ResidencyUnit) -> int:
        return sum(
            max(
                self._geometry[key].allocation_bytes,
                prod(self._geometry[key].shape) * self._max_materialized_itemsizes[key],
            )
            if key in self._raw_residency_keys
            else self._geometry[key].allocation_bytes
            for key in unit.keys
        )

    def _unit_vbar_bytes(self, unit: ResidencyUnit) -> int:
        return self._unit_vbar_sizes[unit.name]

    def _eager_loaded_bytes(self) -> int:
        return sum(
            self._unit_bytes(unit) for unit in self._units if unit.name in self._eager_loaded
        )

    def _demand_reservation_bytes(self) -> int:
        return self._reservation_bytes - sum(
            self._unit_vbar_sizes[name] for name in self._promoted_units
        )

    def working_set_reservation_bytes(self) -> int:
        with self._lock, self._cuda_context():
            loaded = 0 if self._vbar is None else self._backend.loaded_size(self._vbar)
            capacity = _align_up(self._demand_reservation_bytes(), _VBAR_PAGE_SIZE)
            return max(0, capacity - loaded)

    @contextmanager
    def reserve_working_set(self) -> Generator[None]:
        if self._vbar is None:
            yield
            return
        with self._lock, self._cuda_context():
            if self._working_set_reservations == 0:
                self._backend.set_watermark_limit(self._vbar, self._demand_reservation_bytes())
            self._working_set_reservations += 1
        try:
            yield
        finally:
            with self._lock, self._cuda_context():
                self._working_set_reservations -= 1
                if self._working_set_reservations == 0:
                    self._backend.set_watermark_limit(self._vbar, 0)

    def total_bytes(self) -> int:
        return sum(stored_nbytes(stored) for stored in self._weights.values())

    def loaded_bytes(self) -> int:
        """Conventional eager bytes plus clamped resident VBAR bytes.

        dinkster-aimdo accounts 32 MiB pages, and fp32 cast allocations can
        exceed compact storage, so native loaded size may exceed the model's
        logical demand-paged total. Tiny conventional units compose beside
        that value like upstream's ``model_loaded_weight_memory`` and remain
        resident until ``unload()``.
        """
        with self._lock:
            eager = self._eager_loaded_bytes()
            demand_total = sum(
                self._unit_bytes(unit)
                for unit in self._units
                if unit.name not in self._eager_loaded
            )
            vbar = (
                0
                if self._vbar is None
                else min(self._backend.loaded_size(self._vbar), demand_total)
            )
            return eager + vbar

    def page_map(self) -> PageMap | None:
        """Report the VBAR's existing page-residency flags with their geometry."""
        with self._lock, self._cuda_context():
            if self._vbar is None:
                return None
            try:
                query = cast(Any, self._vbar).get_residency
            except (AttributeError, ImportError):
                return None
            if not callable(query):
                return None
            try:
                raw = cast("Sequence[object]", query())
            except (AttributeError, ImportError):
                return None
            return PageMap(page_bytes=_VBAR_PAGE_SIZE, flags=_vbar_page_flags(raw))

    def automatically_reclaimable_bytes(self) -> int:
        """Resident VBAR pages reusable by another prioritized VBAR."""
        with self._lock:
            return 0 if self._vbar is None else self._backend.loaded_size(self._vbar)

    def partial_unload_capacity(self) -> int:
        """VBAR pages and promoted units reclaimable without terminal unload."""
        with self._lock:
            promoted = sum(
                self._unit_bytes(unit) for unit in self._units if unit.name in self._promoted_units
            )
            return self.automatically_reclaimable_bytes() + promoted

    def offloaded_bytes(self) -> int:
        return self.total_bytes() - self.loaded_bytes()

    def is_loaded(self, unit: str) -> bool:
        state = self._unit_states.get(unit)
        return False if state is None else state.loaded

    def unit_state(self, unit: str) -> ResidencyUnitState:
        state = self._unit_states.get(unit)
        if state is None:
            state = ResidencyUnitState()
            self._unit_states[unit] = state
        return state

    def weight_functions(self, key: str) -> tuple[WeightFunction, ...]:
        if self._unit_of[key] in self._eager_loaded:
            return ()
        if isinstance(
            self._weights[key], Int8PackedWeight | Nvfp4PackedWeight
        ) and self._entries.get(key):
            return ()
        return self._functions[key]

    def _load_eager_unit(
        self,
        unit: ResidencyUnit,
        *,
        collector: PartialResidencyTiming | None = None,
        prefetching: bool = False,
    ) -> None:
        """Move every key of a hybrid eager unit onto the load device.

        With a ``collector``, each physical move is bracketed as
        transfer work and its stored bytes are counted into the
        transfer or prefetch counters, and patch application is
        bracketed as dequantization, so a lease that triggers the
        load reports the whole unit's movement rather than a partial
        receipt.
        """
        if unit.name in self._eager_loaded:
            return
        device = self._backend.materialization_device(self.load_device)
        done: list[tuple[str, StoredWeight]] = []
        sources = {key: self._weights[key] for key in unit.keys}
        source_counts: dict[int, int] = {}
        for source in sources.values():
            source_counts[id(source)] = source_counts.get(id(source), 0) + 1
        moved_aliases: dict[int, StoredWeight] = {}
        try:
            for key in unit.keys:
                original = sources[key]
                entries = self._entries.get(key)
                tied = entries is None and source_counts[id(original)] > 1
                if tied and id(original) in moved_aliases:
                    moved = moved_aliases[id(original)]
                else:
                    with timed_phase(collector, TRANSFER, device):
                        moved = (
                            _preserve_parameter_registration(
                                original, move_stored(original, device)
                            )
                            if tied
                            else move_stored(original, device)
                        )
                    if collector is not None:
                        if prefetching:
                            collector.count_prefetch(stored_nbytes(original))
                        else:
                            collector.count_transfer(stored_nbytes(original))
                if tied:
                    moved_aliases[id(original)] = moved
                if entries is not None:
                    with timed_phase(collector, DEQUANT, device):
                        moved = patch_stored_weight(
                            moved,
                            entries,
                            key=f"{self._patch_key_prefix}{key}",
                            intermediate_dtype=self._intermediate_dtype,
                            weight_dtype=self._patch_weight_dtype,
                        )
                    self._eager_backup[key] = original
                elif moved is not original:
                    self._eager_backup[key] = original
                self._weights[key] = moved
                done.append((key, original))
        except Exception:
            for key, original in done:
                self._eager_backup.pop(key, None)
                self._weights[key] = original
            raise
        self._eager_loaded.add(unit.name)
        self._unit_states[unit.name].loaded = True
        self._request_size_cache.clear()
        self._preflight_request_cache.clear()
        self._prefetch_peak_cache.clear()

    def _load_eager_units(self) -> None:
        for unit in self._units:
            if unit.name in self._eager_units:
                self._load_eager_unit(unit)

    def _unload_eager_unit(self, unit: ResidencyUnit) -> None:
        if unit.name not in self._eager_loaded:
            return
        sources = {key: self._eager_backup.get(key, self._weights[key]) for key in unit.keys}
        source_counts: dict[int, int] = {}
        for source in sources.values():
            source_counts[id(source)] = source_counts.get(id(source), 0) + 1
        moved_aliases: dict[int, StoredWeight] = {}
        for key in unit.keys:
            self._eager_backup.pop(key, None)
            original = sources[key]
            tied = source_counts[id(original)] > 1
            moved = moved_aliases.get(id(original)) if tied else None
            if moved is None:
                moved = (
                    _preserve_parameter_registration(
                        original, move_stored(original, self.offload_device)
                    )
                    if tied
                    else move_stored(original, self.offload_device)
                )
                if tied:
                    moved_aliases[id(original)] = moved
            self._weights[key] = moved
        self._eager_loaded.remove(unit.name)
        self._unit_states[unit.name].loaded = False
        self._request_size_cache.clear()
        self._preflight_request_cache.clear()
        self._prefetch_peak_cache.clear()

    def _unload_eager_units(self) -> None:
        for unit in self._units:
            self._unload_eager_unit(unit)

    def _promote_units(self, extra_memory: int, eager_before: int) -> None:
        if not self._fixed_promotion:
            return
        if self._vbar is not None and self._backend.loaded_size(self._vbar) > 0:
            # Promoting a unit whose VBAR allocation may still be resident
            # would retain two physical copies with no per-allocation eviction.
            return
        if (
            self._raw_residency_keys
            and not self._promote_non_fp8_raw
            and not any(
                isinstance(stored := self._weights[key], Fp8ScaledWeight)
                or isinstance(stored, torch.Tensor)
                and stored.dtype in FP8_DTYPES
                for key in self._raw_residency_keys
            )
        ):
            # Fixed promotion offsets raw FP8 routing overhead. Other raw
            # components retain their demand-paged memory ceiling.
            return
        budget = eager_before + extra_memory
        resident = self._eager_loaded_bytes()
        physical_free: int | None = None
        if self._physical_free_memory is not None:
            physical_free = self._physical_free_memory(self.load_device)
            # The manager's dynamic free-memory view already counts evictable
            # VBAR pages. Cap the fixed tier against physical capacity so
            # loaded_bytes does not count those pages a second time.
            budget = min(budget, resident + physical_free)
        unit_bytes = {
            unit.name: self._unit_bytes(unit)
            for unit in self._units
            if unit.name not in self._eager_units
        }
        working_bytes = {
            unit.name: self._unit_working_bytes(unit)
            for unit in self._units
            if unit.name not in self._eager_units
        }
        order = sorted(
            (unit for unit in self._units if unit.name not in self._eager_units),
            key=lambda unit: (0 if unit.expert else 1, unit_bytes[unit.name], unit.name),
            reverse=True,
        )
        allocation_counts: dict[int, int] = {}
        for unit in order:
            if unit.name not in self._promoted_units:
                size = working_bytes[unit.name]
                allocation_counts[size] = allocation_counts.get(size, 0) + 1
        allocation_sizes = sorted(allocation_counts)
        for unit in order:
            if unit.name in self._promoted_units:
                continue
            stored_size = unit_bytes[unit.name]
            allocation_size = working_bytes[unit.name]
            largest_demand = allocation_sizes[-1] if allocation_sizes else 0
            if allocation_counts.get(allocation_size) == 1 and allocation_size == largest_demand:
                largest_demand = allocation_sizes[-2] if len(allocation_sizes) > 1 else 0
            # Demand paging can overlap one active unit with each transfer
            # stream while routed operations need another same-sized torch
            # allocation. Keep that workspace outside fixed residency.
            working_reserve = largest_demand * max(2, self._stream_count + 2)
            if resident + stored_size + working_reserve >= budget:
                continue
            self._load_eager_unit(unit)
            self._promoted_units.add(unit.name)
            loaded_size = self._unit_bytes(unit)
            resident += loaded_size
            allocation_counts[allocation_size] -= 1
            if allocation_counts[allocation_size] == 0:
                del allocation_counts[allocation_size]
                allocation_sizes.pop(bisect.bisect_left(allocation_sizes, allocation_size))

    def partially_load(self, extra_memory: int | None) -> int:
        if extra_memory is None:
            raise AimdoForceFullLoadError(
                "AimdoWeights cannot force a full load; residency is demand-paged"
            )
        if extra_memory < 0:
            return -self.partially_unload(-extra_memory)
        with self._lock, self._cuda_context():
            self._initialize_pins()
            before = self.loaded_bytes()
            eager_before = self._eager_loaded_bytes()
            eager_bytes = sum(
                stored_nbytes(self._weights[key])
                for unit in self._units
                if unit.name in self._eager_units
                for key in unit.keys
            )
            pinned_host.ensure_pin_budget(eager_bytes)
            pinned_host.ensure_pin_registerable(eager_bytes)
            self._load_eager_units()
            self._promote_units(extra_memory, eager_before)
            # Native free_memory and cross-VBAR pressure lower a VBAR's
            # watermark. Every manager load must restore it, including a
            # zero-byte load budget, or all later faults above that watermark
            # fall back forever. This mirrors ModelPatcherDynamic.load.
            if self._vbar is not None:
                self._backend.prioritize(self._vbar)
            return self.loaded_bytes() - before

    def partially_unload(self, memory_to_free: int) -> int:
        with self._lock, self._cuda_context():
            self._prefetch_admission = None
            freed = 0
            if self._vbar is not None:
                self._reap_unpins(wait=False)
                freed = self._backend.free_memory(self._vbar, memory_to_free)
            order = sorted(
                (unit for unit in self._units if unit.name in self._promoted_units),
                key=lambda unit: (0 if unit.expert else 1, self._unit_bytes(unit), unit.name),
            )
            for unit in order:
                if freed >= memory_to_free:
                    break
                unit_bytes = self._unit_bytes(unit)
                self._unload_eager_unit(unit)
                self._promoted_units.remove(unit.name)
                freed += unit_bytes
            return freed

    def unload(self) -> None:
        with self._lock, self._cuda_context():
            for prefetch in {owner for owner, _value in self._prefetched.values()}:
                prefetch.close()
            self._prefetched.clear()
            self._prefetch_admission = None
            self._reap_unpins(wait=True)
            self._backend.cleanup_file_reader()
            if self._stream_state is not None:
                self._stream_state.reset_arenas()
            if self._vbar is not None:
                loaded = self._backend.loaded_size(self._vbar)
                if loaded > 0:
                    self._backend.free_memory(self._vbar, loaded)
                self._backend.deprioritize(self._vbar)
            self._unload_eager_units()
            self._promoted_units.clear()
            self._cache.clear()
            self._request_size_cache.clear()
            self._preflight_request_cache.clear()
            self._prefetch_peak_cache.clear()
            self.free_pins(1 << 63)
            self._pin_state.clear()
            self._pins.clear()
            self._pin_priorities.clear()
            self.pin_active = False
            pinned_host.unregister_owner(self)

    def release_working_buffers(self) -> bool:
        with self._lock, self._cuda_context():
            self._reap_unpins(wait=True)
            if self._stream_state is None or not self._stream_state.arenas:
                return False
            self._stream_state.reset_arenas()
            return True

    def retain_offload_storage(self) -> None:
        """Aimdo already keeps CPU sources authoritative while pages are loaded."""

    @contextmanager
    def _try_pin_release(self) -> Generator[bool, None, None]:
        acquired = self._lock.acquire(blocking=False)
        try:
            if not acquired:
                yield False
            else:
                with self._cuda_context():
                    yield True
        finally:
            if acquired:
                self._lock.release()

    def free_registrations(self, size: int) -> int:
        with self._try_pin_release() as acquired:
            if not acquired:
                return 0
            freed = 0
            synchronized = False
            for subset in ("weights", "patches"):
                state = self._pin_state.get(subset)
                if state is None:
                    continue
                _, stack, split, pinned_size, _, _ = state
                index = split[0]
                while index >= 0 and freed < size:
                    pin = stack[index][0]
                    index -= 1
                    split[0] = index
                    if not pin.registered:
                        continue
                    if not synchronized:
                        self._synchronize_transfer_streams()
                        synchronized = True
                    try:
                        unregistered = self._backend.unregister_host_memory(pin.tensor)
                    except RuntimeError:
                        unregistered = False
                    if not unregistered:
                        self._backend.discard_cuda_async_error()
                        continue
                    pin.registered = False
                    pinned_size[0] -= pin.tensor.nbytes
                    pinned_host.account(-pin.tensor.nbytes)
                    freed += pin.tensor.nbytes
            return freed

    def free_pins(self, size: int) -> int:
        with self._try_pin_release() as acquired:
            if not acquired:
                return 0
            return self._free_pins_locked(size)

    def _free_pins_locked(self, size: int) -> int:
        freed = 0
        synchronized = False
        for subset in ("weights", "patches"):
            state = self._pin_state.get(subset)
            if state is None:
                continue
            hostbuf, stack, split, pinned_size, _, _ = state
            while stack and freed < size:
                if not synchronized:
                    self._synchronize_transfer_streams()
                    synchronized = True
                pin, offset = stack.pop()
                pin.bucket_entry[-1] = None
                self._pins.pop(pin.identity, None)
                self._backend.truncate_host_buffer(hostbuf, offset, pin.registered)
                pinned_host.account_storage(self, -pin.tensor.nbytes)
                split[0] = min(split[0], len(stack) - 1)
                if pin.registered:
                    pinned_size[0] -= pin.tensor.nbytes
                    pinned_host.account(-pin.tensor.nbytes)
                freed += pin.tensor.nbytes
        if not self._pins:
            pinned_host.discard_owner_if_empty(self)
        return freed

    def lease(self, unit: str) -> AbstractContextManager[WeightLease]:
        @contextmanager
        def bracket() -> Generator[WeightLease]:
            self._lock.acquire()
            self.pin_active = True
            try:
                pinned_host.ensure_storage_reserve()
                lease = _AimdoWeightLease(self, unit, collector=active_partial_residency_timing())
                if getattr(_cuda_context_state, "active_device", None) == self.load_device.index:
                    try:
                        yield lease
                    finally:
                        lease.close()
                else:
                    with self._cuda_context():
                        try:
                            yield lease
                        finally:
                            lease.close()
            finally:
                self.pin_active = False
                self._lock.release()

        return bracket()

    def prefetch_enabled(self) -> bool:
        if self._stream_state is None or _is_compiling():
            return False
        return (
            self._vbar is None
            or self._backend.loaded_size(self._vbar) < self._demand_reservation_bytes()
        )

    def _prefetch_free_memory(self, required: int) -> int:
        query = self._physical_free_memory
        if query is None:
            raise RuntimeError("prefetch memory query is unavailable")
        # A fully resident VBAR has stable page demand, so briefly share a
        # successful driver check instead of synchronizing every routed layer.
        cacheable = (
            query is _physical_free_bytes
            and self._vbar is not None
            and self._backend.loaded_size(self._vbar) >= self._demand_reservation_bytes()
        )
        if not cacheable:
            self._prefetch_admission = None
            return query(self.load_device)
        now = time.monotonic_ns()
        if (
            self._prefetch_admission is not None
            and now - self._prefetch_admission[0] < _PREFETCH_ADMISSION_CACHE_NS
        ):
            return self._prefetch_admission[1]
        free = query(self.load_device)
        self._prefetch_admission = (now, free) if required <= free else None
        return free

    def prefetch(self, requests: Sequence[tuple[str, torch.dtype | None]]) -> _AimdoPrefetch | None:
        if self._stream_state is None or _is_compiling():
            return None
        self._lock.acquire()
        lease: _AimdoWeightLease | None = None
        try:
            normalized: list[_BatchRequest] = []
            for key, dtype in requests:
                normalized.extend(self._singleton_request(key, dtype))
            self._preflight_requests(normalized)

            batch: list[_BatchRequest] = []
            seen: set[_BatchRequest] = set()
            for request in normalized:
                if request not in seen and request not in self._prefetched:
                    seen.add(request)
                    batch.append(request)
            if not batch:
                self._lock.release()
                return None

            if self._physical_free_memory is not None:
                projected_peak = self._prefetch_peak_bytes(batch)
                if projected_peak is None:
                    self._lock.release()
                    return None
                physical_free = self._prefetch_free_memory(projected_peak)
                if projected_peak > physical_free:
                    self._lock.release()
                    return None

            self.pin_active = True
            pinned_host.ensure_storage_reserve()
            with self._cuda_context():
                registerable = 0
                for request in batch:
                    registerable += _align_up(
                        stored_nbytes(self._weights[request.key]),
                        _CAST_ARENA_ALIGNMENT,
                    )
                    prepared = self._prepared_sources.get(request.key)
                    if request.form == "get" and prepared is not None:
                        registerable += prepared.memory_required()
                pinned_host.ensure_pin_registerable(registerable)
                lease = _AimdoWeightLease(
                    self,
                    self._unit_of[batch[0].key],
                    collector=active_partial_residency_timing(),
                    prefetching=True,
                )
                values = self._lease_get_many(lease, batch)
                prefetch = _AimdoPrefetch(self, lease, batch)
                for request, value in zip(batch, values, strict=True):
                    self._prefetched[request] = (prefetch, value)
                return prefetch
        except BaseException as error:
            if lease is not None:
                try:
                    with self._cuda_context():
                        lease.close()
                except BaseException as cleanup_error:
                    error.add_note(f"prefetch rollback also failed: {cleanup_error!r}")
            self.pin_active = False
            self._lock.release()
            raise

    def _fault(self, lease: _AimdoWeightLease, key: str) -> tuple[object, object] | None:
        unit = self._unit_of[key]
        if unit in lease._faulted_units:  # pyright: ignore[reportPrivateUsage]
            signature = lease._faulted_units[unit]  # pyright: ignore[reportPrivateUsage]
        else:
            self._reap_unpins(wait=False)
            unit_allocation = self._unit_allocations[unit]
            signature = self._backend.fault(unit_allocation)
            lease._faulted_units[unit] = signature  # pyright: ignore[reportPrivateUsage]
            if signature is not None:
                lease.pin(unit_allocation)
        return None if signature is None else (self._allocations[key], signature)

    def _defer_unpins(self, stream: object, event: object, allocations: list[object]) -> None:
        pending = _device_unpins.setdefault(self._unpin_key, [])
        for stream_pending in pending:
            if stream_pending.stream == stream:
                break
        else:
            stream_pending = _PendingStreamUnpins(stream, [])
            pending.append(stream_pending)
        stream_pending.entries.append(
            _PendingUnpins(self._unpin_owner, self._backend, event, allocations)
        )

    def _reap_unpins(self, *, wait: bool) -> None:
        _reap_device_unpins(
            self._unpin_key,
            wait_owner=self._unpin_owner if wait else None,
        )

    def _cache_hit(
        self,
        key: str,
        signature: object,
        dtype: torch.dtype,
        form: _CacheForm,
    ) -> StoredWeight | None:
        key_cache = self._cache.get(key)
        if not key_cache:
            return None
        cache_key, entry = next(iter(key_cache.items()))
        if cache_key[1:] != (
            dtype,
            form,
            self._patch_revision,
        ) or not self._backend.signature_compare(signature, entry.signature):
            return None
        return entry.value

    def _cache_occupied(self, key: str, signature: object) -> bool:
        """True when the key's allocation still holds a live typed
        representation: its bytes may have enqueued readers or writers."""
        signature_bytes = _signature_bytes(signature)
        return any(cache_key[0] == signature_bytes for cache_key in self._cache.get(key, {}))

    def _cache_store(
        self,
        key: str,
        signature: object,
        dtype: torch.dtype,
        form: _CacheForm,
        value: StoredWeight,
    ) -> None:
        signature_bytes = _signature_bytes(signature)
        cache_key = (signature_bytes, dtype, form, self._patch_revision)
        key_cache = self._cache.setdefault(key, {})
        # One allocation can hold one typed representation at a time.
        # A dtype/form/revision miss overwrites its bytes, so every prior
        # view for this key becomes stale even if its page signature did not.
        key_cache.clear()
        key_cache[cache_key] = _CacheEntry(signature, value)
        # Unconsumed prefetched views of this key alias the overwritten
        # bytes; drop them so a later consumer re-faults instead of
        # reading the new representation through the old view. The
        # prefetch handle still owns its pins until it closes.
        stale = [request for request in self._prefetched if request.key == key]
        for request in stale:
            del self._prefetched[request]

    def _request_size(self, request: _BatchRequest) -> int:
        cached = self._request_size_cache.get(request)
        if cached is not None:
            return cached
        if request.form == "get":
            size = prod(self._geometry[request.key].shape) * request.dtype.itemsize
        else:
            stored = self._weights[request.key]
            if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
                size = _raw_packed_bytes(stored)
            else:
                _raw_request_dtype(stored)
                size = stored.nbytes
        self._request_size_cache[request] = size
        return size

    def _can_copy_directly(self, request: _BatchRequest) -> bool:
        stored = self._weights[request.key]
        return (
            request.form == "get"
            and isinstance(stored, torch.Tensor)
            and stored.dtype == request.dtype
            and not self._functions[request.key]
        )

    def _arena_layout(
        self,
        requests: Sequence[tuple[int, _BatchRequest, bool]],
        *,
        start: int = 0,
        admission: bool = False,
    ) -> _ArenaLayout:
        request_offsets: dict[int, int] = {}
        patch_offsets: dict[int, int] = {}
        offset = start
        for index, request, stage_request in requests:
            if stage_request:
                offset = _align_up(offset, _CAST_ARENA_ALIGNMENT)
                request_offsets[index] = offset
                request_size = self._request_size(request)
                file_backed = _has_file_slices(self._weights[request.key])
                compact_file_stage = (
                    file_backed
                    and not self._pin_all_sources
                    and self._pin_identity("weights", request) not in self._pins
                )
                stage_size = self._host_size(request) if compact_file_stage else request_size
                if file_backed and not compact_file_stage:
                    stage_size = max(stage_size, self._host_size(request))
                offset += stage_size
            prepared = self._prepared_sources.get(request.key)
            if prepared is not None:
                patch_size = (
                    self._patch_staging_bytes[request.key]
                    if admission
                    else prepared.memory_required()
                )
                if patch_size:
                    offset = _align_up(offset, _CAST_ARENA_ALIGNMENT)
                    patch_offsets[index] = offset
                    offset += patch_size
        return _ArenaLayout(request_offsets, patch_offsets, offset)

    def _patch_temporary_bytes(self, request: _BatchRequest, dtype: torch.dtype) -> int:
        geometry = self._patch_payload_geometry.get(request.key)
        if not geometry:
            return 0
        itemsize = max(dtype.itemsize, self._patch_math_itemsize[request.key])
        converted = tuple(
            numel * max(storage_itemsize, itemsize) for numel, storage_itemsize in geometry
        )
        output = prod(self._geometry[request.key].shape) * itemsize
        largest = max(output, max(converted), self._patch_largest_intermediate[request.key])
        return sum(converted) + self._patch_workspace_buffers[request.key] * largest

    def _demand_transient_bytes(self, request: _BatchRequest) -> int:
        if self._can_copy_directly(request) and request.key not in self._raw_residency_keys:
            return 0
        stored = self._weights[request.key]
        stored_bytes = stored_nbytes(stored)
        request_bytes = self._request_size(request)
        prepared = self._prepared_sources.get(request.key)
        if request.form == "get_stored" and prepared is not None:
            dtype = self._patch_weight_dtype or self._intermediate_dtype
            intermediate = prod(self._geometry[request.key].shape) * dtype.itemsize
            return (
                stored_bytes
                + request_bytes
                + 3 * intermediate
                + self._patch_temporary_bytes(request, dtype)
            )

        transient = stored_bytes + request_bytes
        if request.key in self._raw_residency_keys:
            transient += (
                prod(self._geometry[request.key].shape)
                * self._max_materialized_itemsizes[request.key]
            )
        if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            transient += request_bytes
        if prepared is not None:
            transient += request_bytes + self._patch_temporary_bytes(request, request.dtype)
        return transient

    def _eager_unit_target_bytes(self, unit: ResidencyUnit) -> int:
        seen: set[int] = set()
        total = 0
        for key in unit.keys:
            stored = self._weights[key]
            if self._entries.get(key) is None and id(stored) in seen:
                continue
            seen.add(id(stored))
            if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
                total += stored_nbytes(stored)
            else:
                total += prod(self._geometry[key].shape) * stored.dtype.itemsize
        return total

    def _eager_unit_transient_bytes(self, unit: ResidencyUnit) -> int:
        largest = 0
        for key in unit.keys:
            stored = self._weights[key]
            if not self._entries.get(key):
                continue
            if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
                dtype = self._patch_weight_dtype or self._intermediate_dtype
                intermediate = prod(self._geometry[key].shape) * dtype.itemsize
                transient = (
                    stored_nbytes(stored)
                    + 3 * intermediate
                    + self._patch_temporary_bytes(
                        _BatchRequest(key, _raw_request_dtype(stored), "get_stored"), dtype
                    )
                )
            else:
                dtype = self._patch_weight_dtype or self._intermediate_dtype
                request = _BatchRequest(key, dtype, "get")
                transient = (
                    2 * stored_nbytes(stored)
                    + self._request_size(request)
                    + self._patch_temporary_bytes(request, dtype)
                )
            largest = max(largest, transient)
        return largest

    def _eager_request_target_and_transient(self, request: _BatchRequest) -> tuple[int, int]:
        if request.form == "get_stored":
            return 0, 0
        stored = self._weights[request.key]
        request_bytes = self._request_size(request)
        if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            return request_bytes, request_bytes
        if stored.dtype != request.dtype:
            return request_bytes, 0
        return 0, 0

    def _singleton_request(self, key: str, dtype: torch.dtype | None) -> tuple[_BatchRequest]:
        cache_key = (key, dtype)
        cached = self._singleton_requests.get(cache_key)
        if cached is not None:
            return cached
        stored = self._weights[key]
        request = _BatchRequest(
            key,
            _raw_request_dtype(stored) if dtype is None else dtype,
            "get_stored" if dtype is None else "get",
        )
        cached = (request,)
        self._singleton_requests[cache_key] = cached
        return cached

    def _prefetch_peak_bytes(self, requests: Sequence[_BatchRequest]) -> int | None:
        """Bound additional target, batch-arena, and peak temporary bytes."""
        cache_key = tuple(requests)
        if cache_key in self._prefetch_peak_cache:
            return self._prefetch_peak_cache[cache_key]
        if any(request.key in self._unbounded_patch_keys for request in requests):
            self._prefetch_peak_cache[cache_key] = None
            return None
        demand: list[tuple[int, _BatchRequest, bool]] = []
        demand_units: set[str] = set()
        eager_units: dict[str, ResidencyUnit] = {}
        eager_request_target = 0
        largest_transient = 0

        for index, request in enumerate(requests):
            unit_name = self._unit_of[request.key]
            if unit_name in self._eager_loaded or unit_name in self._eager_units:
                if unit_name not in self._eager_loaded:
                    eager_units[unit_name] = self._units_by_name[unit_name]
                target, transient = self._eager_request_target_and_transient(request)
                eager_request_target += target
                largest_transient = max(largest_transient, transient)
                continue
            demand.append((index, request, not self._can_copy_directly(request)))
            demand_units.add(unit_name)
            largest_transient = max(largest_transient, self._demand_transient_bytes(request))

        demand_target = sum(
            _align_up(
                self._unit_vbar_sizes[unit_name] + _VBAR_PAGE_SIZE - _VBAR_ALIGNMENT,
                _VBAR_PAGE_SIZE,
            )
            for unit_name in demand_units
        )
        if any(self._unbounded_patch_keys.intersection(unit.keys) for unit in eager_units.values()):
            self._prefetch_peak_cache[cache_key] = None
            return None
        eager_target = sum(self._eager_unit_target_bytes(unit) for unit in eager_units.values())
        for unit in eager_units.values():
            largest_transient = max(largest_transient, self._eager_unit_transient_bytes(unit))
        layout = self._arena_layout(demand, admission=True)
        peak = demand_target + eager_target + eager_request_target + layout.end + largest_transient
        # Pinning a compact file source can enlarge its staging arena, so cache only
        # after that transition cannot increase the projected peak.
        cacheable = all(
            not stage_request
            or self._pin_all_sources
            or not _has_file_slices(self._weights[request.key])
            or self._request_size(request) <= self._host_size(request)
            or self._pin_identity("weights", request) in self._pins
            for _index, request, stage_request in demand
        )
        if cacheable:
            self._prefetch_peak_cache[cache_key] = peak
        return peak

    def _preflight_requests(self, requests: Sequence[_BatchRequest]) -> None:
        cache_key = tuple(requests)
        if cache_key in self._preflight_request_cache:
            return
        request_sizes = tuple(self._request_size(request) for request in requests)
        for request, request_size in zip(requests, request_sizes, strict=True):
            allocation_size = self._geometry[request.key].allocation_bytes
            if request_size > allocation_size:
                raise ValueError(
                    f"AimdoWeights request for {request.key!r} requires {request_size} bytes;"
                    f" allocation is {allocation_size} bytes"
                )
            ceiling = self._max_materialized_itemsizes[request.key]
            if request.form == "get" and request.dtype.itemsize > ceiling:
                raise ValueError(
                    f"AimdoWeights request for {request.key!r} uses"
                    f" {request.dtype.itemsize} bytes per element;"
                    f" materialization ceiling is {ceiling}"
                )
        self._preflight_request_cache.add(cache_key)

    def _lease_get_many(
        self,
        lease: _AimdoWeightLease,
        requests: Sequence[_BatchRequest],
    ) -> list[StoredWeight]:
        self._preflight_requests(requests)
        collector = lease._collector  # pyright: ignore[reportPrivateUsage]

        results: list[StoredWeight | None] = [None] * len(requests)
        misses: list[tuple[int, _BatchRequest, object | None, object | None]] = []
        overwrites_live = False
        for index, request in enumerate(requests):
            prefetched = self._prefetched.get(request)
            if prefetched is not None:
                results[index] = prefetched[1]
                continue
            unit_name = self._unit_of[request.key]
            if unit_name in self._eager_loaded or unit_name in self._eager_units:
                if unit_name not in self._eager_loaded:
                    self._load_eager_unit(
                        self._units_by_name[unit_name],
                        collector=collector,
                        prefetching=lease._prefetching,  # pyright: ignore[reportPrivateUsage]
                    )
                results[index] = self._eager_value(request)
                continue

            faulted = self._fault(lease, request.key)
            if faulted is None:
                misses.append((index, request, None, None))
                continue
            allocation, signature = faulted
            cached = self._cache_hit(request.key, signature, request.dtype, request.form)
            if cached is not None:
                results[index] = cached
            else:
                misses.append((index, request, allocation, signature))
                if not overwrites_live:
                    overwrites_live = self._cache_occupied(request.key, signature)

        if not misses:
            return cast("list[StoredWeight]", results)

        stream: object | None = None
        state = self._stream_state
        if state is not None and not _is_compiling():
            if lease._stream is None:  # pyright: ignore[reportPrivateUsage]
                lease._stream = state.rotate()  # pyright: ignore[reportPrivateUsage]
            stream = lease._stream  # pyright: ignore[reportPrivateUsage]

        arena: object | None = None
        arena_start = lease._arena_offset  # pyright: ignore[reportPrivateUsage]
        layout = _ArenaLayout({}, {}, arena_start)
        if stream is not None:
            layout = self._arena_layout(
                tuple(
                    (
                        index,
                        request,
                        not self._can_copy_directly(request)
                        and (
                            allocation is not None or _has_file_slices(self._weights[request.key])
                        ),
                    )
                    for index, request, allocation, _ in misses
                ),
                start=arena_start,
            )
        total_size = layout.end - arena_start
        if stream is not None and total_size:
            if len(requests) == 1 and lease._arena_offset == 0:  # pyright: ignore[reportPrivateUsage]
                request = requests[0]
                reference = (id(self), request.key)
                existing = state.arenas.get(id(stream)) if state is not None else None
                existing_size = 0 if existing is None else self._backend.cast_arena_size(existing)
                if (
                    state is not None
                    and state.largest_ref == reference
                    and existing_size < total_size
                ):
                    stream = state.rotate()
                    lease._stream = stream  # pyright: ignore[reportPrivateUsage]
                if state is not None and total_size > state.largest_size:
                    state.largest_ref = reference
                    state.largest_size = total_size
            assert state is not None
            arena = state.arena(stream)
            lease._arena_offset = layout.end  # pyright: ignore[reportPrivateUsage]

        if stream is not None and overwrites_live:
            # Rewriting an allocation that still holds a live typed
            # representation must run after every already-enqueued reader
            # and writer of its bytes. All of them chain through the
            # consumer stream: each miss batch ends with the consumer
            # stream waiting on its transfer stream, and exposed views are
            # only read by work enqueued on the consumer stream after that
            # wait. Waiting on the consumer stream here therefore orders
            # this batch after all of them; first fills write fresh bytes
            # and skip the edge so cold transfers keep overlapping compute.
            self._backend.stream_wait_stream(stream, self._backend.current_stream(self.load_device))
        context = nullcontext() if stream is None else self._backend.stream_context(stream)
        try:
            with context, torch.inference_mode():
                for index, request, allocation, signature in misses:
                    host_source = self._host_value(request)
                    direct = allocation is not None and self._can_copy_directly(request)
                    pinned = self._existing_pin(request)
                    if pinned is None and (allocation is None or self._pin_all_sources or direct):
                        pinned = self._pin_for(request, host_source)
                    file_target = direct and pinned is None and _has_file_slices(host_source)

                    prepared = self._prepared_sources.get(request.key)
                    functions: tuple[WeightFunction, ...] | None = None
                    if arena is not None and index in layout.patch_offsets and prepared is not None:
                        functions = self._prepare_patch_functions(
                            request,
                            prepared,
                            arena,
                            layout.patch_offsets[index],
                            create_pins=allocation is None or self._pin_all_sources,
                        )
                    try:
                        selected_source = host_source if pinned is None else pinned
                        file_staged = False
                        if (
                            not direct
                            and pinned is None
                            and arena is not None
                            and index in layout.request_offsets
                            and _has_file_slices(host_source)
                        ):
                            raw = self._backend.cast_arena_to_uint8_tensor(
                                arena,
                                self._host_size(request),
                                layout.request_offsets[index],
                                self.load_device,
                            )
                            staged = self._host_value_from_backing(request, raw)
                            if not self._read_file_value(staged, host_source, stream):
                                raise RuntimeError(
                                    "mapped file source changed during Aimdo transfer"
                                )
                            selected_source = staged
                            file_staged = True
                        if direct:
                            source = selected_source
                        elif functions is None:
                            source = self._computed_from_source(
                                request,
                                selected_source,
                                non_blocking=stream is not None,
                                collector=collector,
                            )
                        else:
                            source = self._computed_from_source(
                                request,
                                selected_source,
                                functions=functions,
                                non_blocking=stream is not None,
                                collector=collector,
                            )
                    finally:
                        if prepared is not None:
                            prepared.clear_prepared()
                    if collector is not None:
                        moved_bytes = stored_nbytes(self._weights[request.key])
                        if lease._prefetching:  # pyright: ignore[reportPrivateUsage]
                            collector.count_prefetch(moved_bytes)
                        else:
                            collector.count_transfer(moved_bytes)

                    if allocation is None:
                        results[index] = source
                        continue
                    target = self._allocation_value(request, allocation)
                    if collector is None:
                        if file_target:
                            if not self._read_file_value(target, host_source, stream):
                                raise RuntimeError(
                                    "mapped file source changed during Aimdo transfer"
                                )
                        elif arena is not None and not direct and not file_staged:
                            scratch = self._arena_value(
                                request, arena, layout.request_offsets[index]
                            )
                            self._copy_value(scratch, source, non_blocking=True)
                            source = scratch
                        if not file_target:
                            self._copy_value(target, source, non_blocking=stream is not None)
                    else:
                        # The on-device staging copies into the arena and
                        # the vbar allocation are transfer work: they move
                        # the produced representation into its resident
                        # location on the producing stream.
                        timing_device = self._backend.materialization_device(self.load_device)
                        with timed_phase(collector, TRANSFER, timing_device):
                            if file_target:
                                if not self._read_file_value(target, host_source, stream):
                                    raise RuntimeError(
                                        "mapped file source changed during Aimdo transfer"
                                    )
                            elif arena is not None and not direct and not file_staged:
                                scratch = self._arena_value(
                                    request, arena, layout.request_offsets[index]
                                )
                                self._copy_value(scratch, source, non_blocking=True)
                                source = scratch
                            if not file_target:
                                self._copy_value(target, source, non_blocking=stream is not None)
                    assert signature is not None
                    self._cache_store(
                        request.key,
                        signature,
                        request.dtype,
                        request.form,
                        target,
                    )
                    results[index] = target
        except BaseException:
            if stream is not None:
                self._backend.synchronize_stream(stream)
            raise

        if stream is not None:
            current = self._backend.current_stream(self.load_device)
            if collector is not None and not lease._prefetching:  # pyright: ignore[reportPrivateUsage]
                # The consumer stream waiting on the transfer stream is the
                # stall the lease exposes to compute; prefetching leases wait
                # here too, but that wait is overlap setup, not consumer stall.
                stall_device = self._backend.materialization_device(self.load_device)
                with timed_phase(collector, EXPOSED_STALL, stall_device):
                    self._backend.stream_wait_stream(current, stream)
            else:
                self._backend.stream_wait_stream(current, stream)
        if any(value is None for value in results):
            raise RuntimeError("AimdoWeights batch left an unresolved value")
        return cast("list[StoredWeight]", results)

    def _prepare_patch_functions(
        self,
        request: _BatchRequest,
        prepared: PreparedPatchSource,
        arena: object,
        offset: int,
        *,
        create_pins: bool,
    ) -> tuple[WeightFunction, ...]:
        payload_sources: list[torch.Tensor] = []
        for index, payload in enumerate(prepared.payload_tensors()):
            patch_request = _PatchRequest(
                request.key,
                index,
                id(payload),
                payload.dtype,
                tuple(payload.shape),
            )
            pinned = self._existing_pin(patch_request, subset="patches")
            if pinned is None and create_pins:
                pinned = self._pin_for(patch_request, payload, subset="patches")
            if pinned is not None and not isinstance(pinned, torch.Tensor):
                raise TypeError("patch payload pin must be a tensor")
            payload_sources.append(payload if pinned is None else pinned)
        destination = self._backend.cast_arena_to_uint8_tensor(
            arena,
            prepared.memory_required(),
            offset,
            self.load_device,
        )
        entries = prepared.prepare(
            destination,
            payload_sources=payload_sources,
            non_blocking=True,
        )
        prepared.commit(entries)
        return (prepared,)

    def _eager_value(self, request: _BatchRequest) -> StoredWeight:
        stored = self._weights[request.key]
        if request.form == "get_stored":
            _raw_request_dtype(stored)
            return stored
        return cast_weight(stored, dtype=request.dtype)

    def _uncached_value(self, request: _BatchRequest) -> StoredWeight:
        if request.form == "get":
            return self._materialize_get(request.key, dtype=request.dtype, copy=True)
        stored = self._weights[request.key]
        _raw_request_dtype(stored)
        return self._materialize_stored(request.key, stored, copy=True)

    def _host_value(self, request: _BatchRequest) -> StoredWeight:
        return move_stored(self._weights[request.key], self.offload_device)

    def _host_size(self, request: _PinRequest) -> int:
        if isinstance(request, _PatchRequest):
            return prod(request.shape) * request.dtype.itemsize
        stored = self._weights[request.key]
        if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            return _raw_packed_bytes(stored)
        return stored.nbytes

    def _host_value_from_backing(self, request: _PinRequest, backing: torch.Tensor) -> StoredWeight:
        if isinstance(request, _PatchRequest):
            size = prod(request.shape) * request.dtype.itemsize
            return backing[:size].view(request.dtype).reshape(request.shape)
        stored = self._weights[request.key]
        if not isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            return backing[: stored.nbytes].view(stored.dtype).reshape(stored.shape)
        return _packed_from_backing(stored, backing)

    def _read_file_value(
        self,
        target: StoredWeight,
        source: StoredWeight,
        stream: object | None,
    ) -> bool:
        if not (isinstance(target, torch.Tensor) and isinstance(source, torch.Tensor)) and type(
            target
        ) is not type(source):
            return False
        target_tensors = _stored_tensors(target)
        source_tensors = _stored_tensors(source)
        slices = tuple(tensor_file_slice(tensor) for tensor in source_tensors)
        if any(info is None for info in slices):
            return False
        if any(
            target_tensor.shape != source_tensor.shape
            or target_tensor.dtype != source_tensor.dtype
            or not target_tensor.is_contiguous()
            or target_tensor.nbytes != info.size
            for target_tensor, source_tensor, info in zip(
                target_tensors,
                source_tensors,
                slices,
                strict=True,
            )
            if info is not None
        ):
            return False
        for target_tensor, info in zip(target_tensors, slices, strict=True):
            assert info is not None
            with info.lock:
                self._backend.read_file_slice(info.file, info.offset, target_tensor, stream)
        return True

    def _computed_from_source(
        self,
        request: _BatchRequest,
        source: StoredWeight,
        *,
        functions: Sequence[WeightFunction] | None = None,
        non_blocking: bool = False,
        collector: PartialResidencyTiming | None = None,
    ) -> StoredWeight:
        """Produce the requested value from a host source.

        With a ``collector``, the stored-representation move is split
        from the on-device finish work (the exact move-then-finish
        composition ``cast_weight`` fuses) so the transfer phase
        brackets only the copy and the dequant phase brackets cast and
        patch work. Without one, the fused path runs unchanged.
        """
        device = self._backend.materialization_device(self.load_device)
        if request.form == "get":
            request_functions = self._functions[request.key] if functions is None else functions
            if collector is None:
                return cast_weight(
                    source,
                    dtype=request.dtype,
                    device=device,
                    functions=request_functions,
                    non_blocking=non_blocking,
                )
            with timed_phase(collector, TRANSFER, device):
                moved = move_stored(source, device, non_blocking=non_blocking)
            with timed_phase(collector, DEQUANT, device):
                return cast_weight(moved, dtype=request.dtype, functions=request_functions)
        if collector is None:
            moved = move_stored(source, device, non_blocking=non_blocking)
        else:
            with timed_phase(collector, TRANSFER, device):
                moved = move_stored(source, device, non_blocking=non_blocking)
        prepared = self._prepared_sources.get(request.key)
        if prepared is None:
            return moved
        if not isinstance(moved, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            raise TypeError("stored patch materialization requires packed storage")
        entries = prepared.take_prepared() if functions is not None else self._entries[request.key]
        if collector is None:
            return patch_stored_weight(
                moved,
                entries,
                key=f"{self._patch_key_prefix}{request.key}",
                intermediate_dtype=self._intermediate_dtype,
                weight_dtype=self._patch_weight_dtype,
            )
        with timed_phase(collector, DEQUANT, device):
            return patch_stored_weight(
                moved,
                entries,
                key=f"{self._patch_key_prefix}{request.key}",
                intermediate_dtype=self._intermediate_dtype,
                weight_dtype=self._patch_weight_dtype,
            )

    def _computed_value(self, request: _BatchRequest, *, non_blocking: bool) -> StoredWeight:
        if request.form == "get":
            return self._materialize_get(
                request.key,
                dtype=request.dtype,
                copy=False,
                non_blocking=non_blocking,
            )
        stored = self._weights[request.key]
        _raw_request_dtype(stored)
        return self._materialize_stored(
            request.key,
            stored,
            copy=False,
            non_blocking=non_blocking,
        )

    def _allocation_value(self, request: _BatchRequest, allocation: object) -> StoredWeight:
        backing = self._backend.alloc_to_uint8_tensor(allocation, self.load_device)
        return self._value_from_backing(request, backing)

    def _arena_value(self, request: _BatchRequest, arena: object, offset: int) -> StoredWeight:
        size = self._request_size(request)
        backing = self._backend.cast_arena_to_uint8_tensor(arena, size, offset, self.load_device)
        return self._value_from_backing(request, backing)

    def _value_from_backing(self, request: _BatchRequest, backing: torch.Tensor) -> StoredWeight:
        stored = self._weights[request.key]
        if request.form == "get":
            shape = self._geometry[request.key].shape
            needed = prod(shape) * request.dtype.itemsize
            return backing[:needed].view(request.dtype).reshape(shape)
        if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            return _packed_from_backing(stored, backing)
        return backing[: stored.nbytes].view(stored.dtype).reshape(stored.shape)

    @staticmethod
    def _copy_value(
        target: StoredWeight,
        source: StoredWeight,
        *,
        non_blocking: bool,
    ) -> None:
        if isinstance(target, Fp8ScaledWeight):
            if not isinstance(source, Fp8ScaledWeight):
                raise TypeError("fp8 target requires fp8 source")
            target.qdata.copy_(source.qdata, non_blocking=non_blocking)
            target.scale.copy_(source.scale, non_blocking=non_blocking)
            return
        if isinstance(target, Int8PackedWeight):
            if not isinstance(source, Int8PackedWeight):
                raise TypeError("INT8 target requires INT8 source")
            target.qdata.copy_(source.qdata, non_blocking=non_blocking)
            target.scale.copy_(source.scale, non_blocking=non_blocking)
            return
        if isinstance(target, Nvfp4PackedWeight):
            if not isinstance(source, Nvfp4PackedWeight):
                raise TypeError("NVFP4 target requires NVFP4 source")
            target.qdata.copy_(source.qdata, non_blocking=non_blocking)
            target.block_scale.copy_(source.block_scale, non_blocking=non_blocking)
            target.tensor_scale.copy_(source.tensor_scale, non_blocking=non_blocking)
            return
        if not isinstance(source, torch.Tensor):
            raise TypeError("tensor target requires tensor source")
        target.copy_(source, non_blocking=non_blocking)

    def _materialize_get(
        self,
        key: str,
        *,
        dtype: torch.dtype,
        copy: bool,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        device = self._backend.materialization_device(self.load_device)
        with torch.inference_mode():
            value = cast_weight(
                self._weights[key],
                dtype=dtype,
                device=device,
                functions=self._functions[key],
                non_blocking=non_blocking,
            )
            return value.clone() if copy else value

    def _materialize_stored(
        self,
        key: str,
        stored: StoredWeight,
        *,
        copy: bool,
        non_blocking: bool = False,
    ) -> StoredWeight:
        device = self._backend.materialization_device(self.load_device)
        with torch.inference_mode():
            moved = move_stored(stored, device, non_blocking=non_blocking)
            entries = self._entries.get(key)
            if entries:
                if not isinstance(moved, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
                    raise TypeError("stored patch materialization requires packed storage")
                moved = patch_stored_weight(
                    moved,
                    entries,
                    key=f"{self._patch_key_prefix}{key}",
                    intermediate_dtype=self._intermediate_dtype,
                    weight_dtype=self._patch_weight_dtype,
                )
            if not copy:
                return moved
            if isinstance(moved, Fp8ScaledWeight):
                return Fp8ScaledWeight(moved.qdata.clone(), moved.scale.clone(), moved.orig_dtype)
            if isinstance(moved, Int8PackedWeight):
                return replace(moved, qdata=moved.qdata.clone(), scale=moved.scale.clone())
            if isinstance(moved, Nvfp4PackedWeight):
                return replace(
                    moved,
                    qdata=moved.qdata.clone(),
                    block_scale=moved.block_scale.clone(),
                    tensor_scale=moved.tensor_scale.clone(),
                )
            return moved.clone()


__all__ = [
    "AimdoForceFullLoadError",
    "AimdoUnavailableError",
    "AimdoWeights",
    "ComfyAimdoBackend",
    "VbarBackend",
]
