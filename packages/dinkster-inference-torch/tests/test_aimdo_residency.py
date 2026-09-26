"""CPU-fake proofs for demand-paged aimdo residency."""

from __future__ import annotations

import ctypes
import gc
import json
import os
import struct
import sys
import threading
import time
import weakref
from collections.abc import Callable, Generator, MutableMapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import BinaryIO, cast

import pytest
import torch
from dinkster_inference import SD15
from dinkster_inference.patches import (
    AdapterPatch,
    DiffPatch,
    ModelAsLoraPatch,
    PatchEntry,
    PatchSet,
)
from dinkster_inference_torch import (
    INITLESS,
    AimdoForceFullLoadError,
    AimdoMemoryStatus,
    AimdoUnavailableError,
    AimdoWeights,
    AssembledSD,
    AutoencoderKL,
    BOFTAdapter,
    CastOperations,
    ClipTextModel,
    ComfyAimdoBackend,
    CudaMemorySnapshot,
    DeviceMemory,
    EnrolledResidency,
    Fp8Linear,
    Fp8ScaledWeight,
    GLoRAAdapter,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    MemoryPolicy,
    ModuleStateStore,
    OFTAdapter,
    PartialResidencyTiming,
    ResidencyManager,
    ResidencyUnit,
    ResidentWeights,
    UNetModel,
    collect_partial_residency_timing,
    declare_residency_materialization_ceilings,
    enroll_assembled,
    enroll_component,
    pinned_host,
)
from dinkster_inference_torch import aimdo_activation as activation
from dinkster_inference_torch import aimdo_residency as aimdo_mod
from dinkster_inference_torch import model_prefetch as prefetch_mod
from dinkster_inference_torch import module_residency as module_residency_mod
from dinkster_inference_torch.apply import StoredWeight, patch_stored_weight
from dinkster_inference_torch.gguf_linear import GgufEncodedLinear
from dinkster_inference_torch.model_prefetch import (
    PrefetchPlan,
    cleanup_prefetch_queues,
    close_prefetch_queue,
    make_prefetch_queue,
    prefetch_queue_pop,
)
from dinkster_inference_torch.quant import Int8PackedWeight
from dinkster_inference_torch.quant_linear import Int8Linear, Nvfp4Linear
from dinkster_inference_torch.sources import load_tensors, tensor_file_slice
from dinkster_memory import PageMap

CPU = torch.device("cpu")
CUDA0 = torch.device("cuda:0")
AimdoDeviceEntry = int | tuple[int, int]


@dataclass
class _FakeVbar:
    size: int
    device_index: int
    offset: int = 0


@dataclass(eq=False)
class _FakeAllocation:
    size: int
    offset: int
    tensor: torch.Tensor
    signature: bytes
    pins: int = 0
    loaded: bool = False


@dataclass(eq=False)
class _FakeSpan:
    allocations: tuple[_FakeAllocation, ...]


@dataclass(eq=False)
class _FakeStream:
    name: str


@dataclass(eq=False)
class _FakeEvent:
    complete: bool = False


@dataclass
class _FakeArena:
    max_size: int
    device_index: int
    tensor: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.uint8))


@dataclass
class _FakeHostBuffer:
    prewarm: int
    max_grow_size: int
    size: int = 0
    backing: torch.Tensor = field(init=False)
    commits: list[int] = field(default_factory=list)
    truncations: list[tuple[int, bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.backing = torch.empty(self.max_grow_size, dtype=torch.uint8)


@dataclass
class FakeVbarBackend:
    """ModelVBAR semantics over ordinary CPU uint8 tensors."""

    vbars: list[_FakeVbar] = field(default_factory=list)
    allocations: list[_FakeAllocation] = field(default_factory=list)
    fault_calls: list[_FakeAllocation | _FakeSpan] = field(default_factory=list)
    unpins: list[_FakeAllocation | _FakeSpan] = field(default_factory=list)
    tensor_calls: list[_FakeAllocation] = field(default_factory=list)
    free_calls: list[int] = field(default_factory=list)
    prioritize_calls: int = 0
    deprioritize_calls: int = 0
    watermark_limits: list[int] = field(default_factory=list)
    streams: list[_FakeStream] = field(default_factory=list)
    arenas: list[_FakeArena] = field(default_factory=list)
    events: list[tuple[object, ...]] = field(default_factory=list)
    oom: bool = False
    oom_offsets: set[int] = field(default_factory=set)
    generation: int = 1
    reported_loaded_size: int | None = None
    register_results: list[bool] = field(default_factory=list)
    unregister_results: list[bool] = field(default_factory=list)
    discarded_errors: int = 0
    registered_ptrs: set[int] = field(default_factory=set)
    complete_events_on_record: bool = True
    recorded_events: list[_FakeEvent] = field(default_factory=list)
    file_reader_cleanups: int = 0
    current: _FakeStream | None = None

    @property
    def requires_cuda_context(self) -> bool:
        return False

    def create_vbar(self, size: int, device_index: int) -> object:
        vbar = _FakeVbar(size, device_index)
        self.vbars.append(vbar)
        return vbar

    def alloc(self, vbar: object, size: int) -> object:
        concrete = cast(_FakeVbar, vbar)
        concrete.offset = (concrete.offset + 511) & ~511
        allocation = _FakeAllocation(
            size,
            concrete.offset,
            torch.empty(size, dtype=torch.uint8),
            self.generation.to_bytes(8, "little"),
        )
        concrete.offset += size
        self.generation += 1
        self.allocations.append(allocation)
        return allocation

    def span(self, allocations: Sequence[object]) -> object:
        concrete = tuple(cast(_FakeAllocation, allocation) for allocation in allocations)
        if not concrete:
            raise ValueError("VBAR span requires at least one allocation")
        return concrete[0] if len(concrete) == 1 else _FakeSpan(concrete)

    def fault(self, allocation: object) -> object | None:
        concrete = cast(_FakeAllocation | _FakeSpan, allocation)
        self.fault_calls.append(concrete)
        allocations = concrete.allocations if isinstance(concrete, _FakeSpan) else (concrete,)
        if self.oom or any(item.offset in self.oom_offsets for item in allocations):
            return None
        for item in allocations:
            item.pins += 1
            item.loaded = True
        return b"".join(item.signature for item in allocations)

    def signature_compare(self, left: object, right: object) -> bool:
        return left == right

    def unpin(self, allocation: object) -> None:
        concrete = cast(_FakeAllocation | _FakeSpan, allocation)
        allocations = concrete.allocations if isinstance(concrete, _FakeSpan) else (concrete,)
        assert all(item.pins > 0 for item in allocations)
        for item in allocations:
            item.pins -= 1
        self.unpins.append(concrete)

    def free_memory(self, vbar: object, size: int) -> int:
        cast(_FakeVbar, vbar)
        self.free_calls.append(size)
        freed = 0
        for allocation in reversed(self.allocations):
            if freed >= size:
                break
            if allocation.loaded and allocation.pins == 0:
                allocation.loaded = False
                allocation.tensor.zero_()
                allocation.signature = self.generation.to_bytes(8, "little")
                self.generation += 1
                freed += allocation.size
        return freed

    def loaded_size(self, vbar: object) -> int:
        cast(_FakeVbar, vbar)
        if self.reported_loaded_size is not None:
            return self.reported_loaded_size
        return sum(a.size for a in self.allocations if a.loaded)

    def prioritize(self, vbar: object) -> None:
        cast(_FakeVbar, vbar)
        self.prioritize_calls += 1

    def deprioritize(self, vbar: object) -> None:
        cast(_FakeVbar, vbar)
        self.deprioritize_calls += 1

    def set_watermark_limit(self, vbar: object, size: int) -> None:
        cast(_FakeVbar, vbar)
        self.watermark_limits.append(size)

    def alloc_to_uint8_tensor(self, allocation: object, device: torch.device) -> torch.Tensor:
        assert device.type == "cuda"
        concrete = cast(_FakeAllocation, allocation)
        self.tensor_calls.append(concrete)
        return concrete.tensor

    def create_stream(self, device: torch.device) -> object:
        stream = _FakeStream(f"transfer-{len(self.streams)}")
        self.streams.append(stream)
        self.events.append(("create-stream", stream.name, str(device)))
        return stream

    def current_stream(self, device: torch.device) -> object:
        assert device.type == "cuda"
        return self.current if self.current is not None else _FakeStream("current")

    def stream_wait_stream(self, stream: object, other: object) -> None:
        self.events.append(
            (
                "wait",
                cast(_FakeStream, stream).name,
                cast(_FakeStream, other).name,
            )
        )

    def stream_context(self, stream: object) -> AbstractContextManager[None]:
        self.events.append(("context", cast(_FakeStream, stream).name))
        return nullcontext()

    def synchronize_stream(self, stream: object) -> None:
        self.events.append(("synchronize", cast(_FakeStream, stream).name))

    def record_event(self, stream: object) -> object:
        event = _FakeEvent(self.complete_events_on_record)
        self.recorded_events.append(event)
        self.events.append(("record-event", cast(_FakeStream, stream).name))
        return event

    def event_query(self, event: object) -> bool:
        return cast(_FakeEvent, event).complete

    def synchronize_event(self, event: object) -> None:
        cast(_FakeEvent, event).complete = True
        self.events.append(("synchronize-event",))

    def create_cast_arena(self, size: int, device_index: int) -> object:
        arena = _FakeArena(size, device_index)
        self.arenas.append(arena)
        self.events.append(("create-arena", len(self.arenas) - 1, size))
        return arena

    def cast_arena_size(self, arena: object) -> int:
        return cast(_FakeArena, arena).tensor.numel()

    def cast_arena_to_uint8_tensor(
        self,
        arena: object,
        size: int,
        offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        assert device.type == "cuda"
        concrete = cast(_FakeArena, arena)
        required = size + offset
        if required > concrete.max_size:
            raise RuntimeError(f"VRAM grow failed: {required} bytes")
        arena_index = next(
            index for index, candidate in enumerate(self.arenas) if candidate is concrete
        )
        if required > concrete.tensor.numel():
            self.events.append(("arena-evict", arena_index, required))
            grown = torch.empty(required, dtype=torch.uint8)
            grown[: concrete.tensor.numel()].copy_(concrete.tensor)
            concrete.tensor = grown
            self.events.append(("arena-grow", arena_index, required))
        self.events.append(("arena-get", arena_index, size, offset))
        return concrete.tensor[offset:required]

    def materialization_device(self, device: torch.device) -> torch.device:
        assert device.type == "cuda"
        return CPU

    def create_host_buffer(self, prewarm: int, max_grow_size: int) -> object:
        return _FakeHostBuffer(prewarm, max_grow_size)

    def host_buffer_size(self, host_buffer: object) -> int:
        return cast(_FakeHostBuffer, host_buffer).size

    def extend_host_buffer(self, host_buffer: object, size: int) -> None:
        concrete = cast(_FakeHostBuffer, host_buffer)
        if concrete.size + size > concrete.max_grow_size:
            raise RuntimeError("host buffer exhausted")
        concrete.size += size
        concrete.commits.append(concrete.size)

    def host_buffer_tensor(self, host_buffer: object) -> torch.Tensor:
        concrete = cast(_FakeHostBuffer, host_buffer)
        return concrete.backing[: concrete.size]

    def truncate_host_buffer(self, host_buffer: object, size: int, unregister: bool) -> None:
        concrete = cast(_FakeHostBuffer, host_buffer)
        concrete.truncations.append((size, unregister))
        concrete.size = size

    def read_file_slice(
        self,
        file: BinaryIO,
        offset: int,
        target: torch.Tensor,
        stream: object | None,
    ) -> None:
        data = os.pread(  # pyright: ignore[reportAttributeAccessIssue]
            file.fileno(), target.nbytes, offset
        )
        if len(data) != target.nbytes:
            raise RuntimeError("direct file read was truncated")
        target.reshape(-1).view(torch.uint8).copy_(
            torch.frombuffer(bytearray(data), dtype=torch.uint8)
        )
        stream_name = None if stream is None else cast(_FakeStream, stream).name
        self.events.append(("read-file", offset, target.nbytes, stream_name))

    def cleanup_file_reader(self) -> None:
        self.file_reader_cleanups += 1

    def register_host_memory(self, tensor: torch.Tensor) -> bool:
        result = self.register_results.pop(0) if self.register_results else True
        if result:
            self.registered_ptrs.add(tensor.data_ptr())
        return result

    def unregister_host_memory(self, tensor: torch.Tensor) -> bool:
        result = self.unregister_results.pop(0) if self.unregister_results else True
        if result:
            self.registered_ptrs.discard(tensor.data_ptr())
        return result

    def discard_cuda_async_error(self) -> None:
        self.discarded_errors += 1

    def is_pinned(self, tensor: torch.Tensor) -> bool:
        return tensor.data_ptr() in self.registered_ptrs

    def change_signature(self, allocation: object) -> None:
        concrete = cast(_FakeAllocation, allocation)
        concrete.signature = self.generation.to_bytes(8, "little")
        self.generation += 1


def _aimdo(
    weights: MutableMapping[str, StoredWeight],
    *,
    backend: FakeVbarBackend | None = None,
    patch_set: PatchSet[torch.Tensor] | None = None,
    units: Sequence[ResidencyUnit] | None = None,
    stream_count: int = 2,
    pin_all_sources: bool = False,
    physical_free_memory: Callable[[torch.device], int] | None = None,
    fixed_promotion: bool = False,
    promote_non_fp8_raw: bool = False,
) -> tuple[AimdoWeights, FakeVbarBackend]:
    backend = FakeVbarBackend() if backend is None else backend
    mechanism = AimdoWeights(
        weights,
        load_device=CUDA0,
        offload_device=CPU,
        patch_set=patch_set,
        units=units,
        backend=backend,
        stream_count=stream_count,
        pin_all_sources=pin_all_sources,
        physical_free_memory=physical_free_memory,
        fixed_promotion=fixed_promotion,
        promote_non_fp8_raw=promote_non_fp8_raw,
    )
    return mechanism, backend


def _mapped_weights(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    dtype_names = {
        torch.float32: "F32",
        torch.float8_e4m3fn: "F8_E4M3",
    }
    header: dict[str, object] = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        contiguous = tensor.contiguous()
        data = bytes(contiguous.reshape(-1).view(torch.uint8).tolist())
        header[name] = {
            "dtype": dtype_names[contiguous.dtype],
            "shape": list(contiguous.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    encoded = json.dumps(header).encode()
    path = tmp_path / "weights.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return load_tensors(path)


def test_driver_context_binds_once_per_thread_and_restores_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism, _ = _aimdo({"weight": torch.ones(5000)})
    mechanism._requires_cuda_context = True  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(aimdo_mod, "_cuda_context_state", threading.local())
    current = [0]
    current_checks = 0
    sets: list[torch.device | str | int] = []
    allocations: list[torch.device | str | int | None] = []

    def current_device() -> int:
        nonlocal current_checks
        current_checks += 1
        return current[0]

    monkeypatch.setattr(torch.cuda, "current_device", current_device)

    def set_device(device: torch.device | str | int) -> None:
        sets.append(device)
        current[0] = torch.device(device).index if not isinstance(device, int) else device

    def empty(_size: int, *, device: torch.device | str | int | None = None) -> object:
        allocations.append(device)
        return object()

    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch, "empty", empty)

    with mechanism._cuda_context():  # pyright: ignore[reportPrivateUsage]
        assert current[0] == 0
    with mechanism._cuda_context():  # pyright: ignore[reportPrivateUsage]
        assert current[0] == 0
    assert sets == [CUDA0]
    assert allocations == [CUDA0]
    assert current_checks == 2

    with mechanism.execution_context():
        with mechanism._cuda_context():  # pyright: ignore[reportPrivateUsage]
            assert current[0] == 0
        with mechanism._cuda_context():  # pyright: ignore[reportPrivateUsage]
            assert current[0] == 0
    assert current_checks == 3

    current[0] = 1
    with mechanism._cuda_context():  # pyright: ignore[reportPrivateUsage]
        assert current[0] == 0
    assert sets == [CUDA0, CUDA0, 1]
    assert allocations == [CUDA0]
    assert current[0] == 1
    assert current_checks == 4


def test_driver_context_forces_binding_on_first_use_in_fresh_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism, _ = _aimdo({"weight": torch.ones(5000)})
    mechanism._requires_cuda_context = True  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(aimdo_mod, "_cuda_context_state", threading.local())
    sets: list[int] = []
    allocations: list[int] = []
    errors: list[BaseException] = []

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def record_set_device(_device: torch.device | str | int) -> None:
        sets.append(threading.get_ident())

    def record_allocation(_size: int, *, device: torch.device | str | int | None = None) -> object:
        allocations.append(threading.get_ident())
        return object()

    monkeypatch.setattr(torch.cuda, "set_device", record_set_device)
    monkeypatch.setattr(torch, "empty", record_allocation)

    def use_context() -> None:
        try:
            with mechanism._cuda_context():  # pyright: ignore[reportPrivateUsage]
                pass
        except BaseException as error:
            errors.append(error)

    threads = (threading.Thread(target=use_context), threading.Thread(target=use_context))
    for thread in threads:
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()

    assert not errors
    assert len(sets) == 2
    assert len(allocations) == 2


def test_driver_context_retries_failed_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    mechanism, _ = _aimdo({"weight": torch.ones(5000)})
    mechanism._requires_cuda_context = True  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(aimdo_mod, "_cuda_context_state", threading.local())
    sets = 0
    allocations = 0

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def record_set_device(_device: torch.device | str | int) -> None:
        nonlocal sets
        sets += 1

    def fail_once(_size: int, *, device: torch.device | str | int | None = None) -> object:
        nonlocal allocations
        allocations += 1
        if allocations == 1:
            raise RuntimeError("context allocation failed")
        return object()

    monkeypatch.setattr(torch.cuda, "set_device", record_set_device)
    monkeypatch.setattr(torch, "empty", fail_once)

    with pytest.raises(RuntimeError, match="context allocation failed"):
        with mechanism._cuda_context():  # pyright: ignore[reportPrivateUsage]
            pass
    with mechanism._cuda_context():  # pyright: ignore[reportPrivateUsage]
        pass
    assert sets == 2
    assert allocations == 2


def test_production_backend_caches_current_stream_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ComfyAimdoBackend.__new__(ComfyAimdoBackend)
    monkeypatch.setattr(aimdo_mod, "_cuda_context_state", threading.local())
    stream_data = [(0, 0, 1)]
    streams: list[object] = []
    indexes: list[int | None] = []

    def get_current_stream(_index: int | None) -> tuple[int, int, int]:
        indexes.append(_index)
        return stream_data[0]

    monkeypatch.setattr(
        torch._C,  # pyright: ignore[reportPrivateUsage]
        "_cuda_getCurrentStream",
        get_current_stream,
        raising=False,
    )

    def current_stream(_device: torch.device) -> object:
        stream = object()
        streams.append(stream)
        return stream

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)

    first = backend.current_stream(CUDA0)
    assert backend.current_stream(CUDA0) is first
    assert streams == [first]

    stream_data[0] = (1, 0, 1)
    second = backend.current_stream(CUDA0)
    assert second is not first
    assert streams == [first, second]
    assert indexes == [0, 0, 0]


def test_production_backend_compares_signature_bytes() -> None:
    backend = ComfyAimdoBackend.__new__(ComfyAimdoBackend)
    signature = (ctypes.c_uint32 * 2)(1, 2)

    assert backend.signature_compare(signature, (ctypes.c_uint32 * 2)(1, 2))
    assert not backend.signature_compare(signature, (ctypes.c_uint32 * 2)(1, 3))
    with pytest.raises(ValueError, match="mismatched lengths"):
        backend.signature_compare(signature, (ctypes.c_uint32 * 1)(1))


@pytest.fixture(autouse=True)
def _reset_pinned_host_globals(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None]:
    old_max = pinned_host.MAX_PINNED_MEMORY
    old_storage_max = pinned_host.MAX_PINNED_STORAGE
    memory_status = pinned_host.memory_status

    def stable_memory_status(
        platform: str | None = None,
        *,
        windows_query: Callable[[], tuple[int, int]] | None = None,
        linux_cgroup_query: Callable[[], tuple[int | None, int | None]] | None = None,
        sysconf_query: Callable[[str], int] | None = None,
    ) -> tuple[int, int]:
        if (
            platform is None
            and windows_query is None
            and linux_cgroup_query is None
            and sysconf_query is None
        ):
            return 1 << 60, 1 << 60
        if windows_query is None:
            assert linux_cgroup_query is not None
            if sysconf_query is not None:
                return memory_status(
                    platform,
                    linux_cgroup_query=linux_cgroup_query,
                    sysconf_query=sysconf_query,
                )
            return memory_status(platform, linux_cgroup_query=linux_cgroup_query)
        if linux_cgroup_query is None:
            return memory_status(platform, windows_query=windows_query)
        if sysconf_query is not None:
            return memory_status(
                platform,
                windows_query=windows_query,
                linux_cgroup_query=linux_cgroup_query,
                sysconf_query=sysconf_query,
            )
        return memory_status(
            platform,
            windows_query=windows_query,
            linux_cgroup_query=linux_cgroup_query,
        )

    monkeypatch.setattr(pinned_host, "memory_status", stable_memory_status)
    pinned_host.configure(disabled=False, maximum=1 << 60, storage_maximum=1 << 60)
    pinned_host.TOTAL_PINNED_MEMORY = 0  # pyright: ignore[reportConstantRedefinition]
    pinned_host.TOTAL_PINNED_STORAGE = 0  # pyright: ignore[reportConstantRedefinition]
    pinned_host._owners.clear()  # pyright: ignore[reportPrivateUsage]
    pinned_host._storage_by_owner.clear()  # pyright: ignore[reportPrivateUsage]
    pinned_host._warned_pin_refusals.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    pinned_host._owners.clear()  # pyright: ignore[reportPrivateUsage]
    pinned_host._storage_by_owner.clear()  # pyright: ignore[reportPrivateUsage]
    pinned_host._warned_pin_refusals.clear()  # pyright: ignore[reportPrivateUsage]
    pinned_host.TOTAL_PINNED_MEMORY = 0  # pyright: ignore[reportConstantRedefinition]
    pinned_host.TOTAL_PINNED_STORAGE = 0  # pyright: ignore[reportConstantRedefinition]
    pinned_host.configure(disabled=False, maximum=old_max, storage_maximum=old_storage_max)


def _fp8(shape: tuple[int, ...] = (1,)) -> Fp8ScaledWeight:
    return Fp8ScaledWeight(
        torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
        .reshape(shape)
        .to(torch.float8_e4m3fn),
        torch.tensor(0.5),
        torch.float32,
    )


def _prefetch_block(
    *,
    backend: FakeVbarBackend | None = None,
    patch_set: PatchSet[torch.Tensor] | None = None,
    stream_count: int = 2,
    pin_all_sources: bool = False,
    physical_free_memory: Callable[[torch.device], int] | None = None,
) -> tuple[torch.nn.Sequential, AimdoWeights, FakeVbarBackend]:
    block = torch.nn.Sequential(INITLESS.linear(100, 50), INITLESS.linear(50, 100))
    block.load_state_dict(
        {
            "0.weight": torch.arange(5000, dtype=torch.float32).reshape(50, 100),
            "0.bias": torch.arange(50, dtype=torch.float32),
            "1.weight": torch.arange(5000, dtype=torch.float32).reshape(100, 50),
            "1.bias": torch.arange(100, dtype=torch.float32),
        },
        strict=True,
        assign=True,
    )
    concrete = FakeVbarBackend() if backend is None else backend

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> AimdoWeights:
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            backend=concrete,
            stream_count=stream_count,
            pin_all_sources=pin_all_sources,
            physical_free_memory=physical_free_memory,
        )

    mechanism = enroll_component(
        block,
        load_device=CUDA0,
        offload_device=CPU,
        patch_set=patch_set,
        mechanism_factory=factory,
    )
    assert isinstance(mechanism, AimdoWeights)
    return block, mechanism, concrete


def _enroll_int8(
    layer: Int8Linear,
    *,
    backend: FakeVbarBackend | None = None,
    patch_set: PatchSet[torch.Tensor] | None = None,
    patch_weight_dtype: torch.dtype | None = None,
    patch_key_prefix: str = "",
) -> tuple[AimdoWeights, FakeVbarBackend]:
    concrete = FakeVbarBackend() if backend is None else backend

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
        patch_weight_dtype: torch.dtype | None = None,
        patch_key_prefix: str = "",
    ) -> AimdoWeights:
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            patch_weight_dtype=patch_weight_dtype,
            patch_key_prefix=patch_key_prefix,
            backend=concrete,
        )

    mechanism = enroll_component(
        layer,
        load_device=CUDA0,
        offload_device=CPU,
        patch_set=patch_set,
        patch_weight_dtype=patch_weight_dtype,
        patch_key_prefix=patch_key_prefix,
        mechanism_factory=factory,
    )
    assert isinstance(mechanism, AimdoWeights)
    return mechanism, concrete


def test_production_backend_builds_one_contiguous_vbar_span() -> None:
    backend = object.__new__(ComfyAimdoBackend)
    vbar = object()
    first = (vbar, 1024, 500)
    second = (vbar, 2048, 100)

    assert backend.span((first,)) is first
    assert backend.span((first, second)) == (vbar, 1024, 1124)
    with pytest.raises(ValueError, match="at least one allocation"):
        backend.span(())
    with pytest.raises(ValueError, match="share one VBAR"):
        backend.span((first, (object(), 2048, 100)))


def test_classify_vbar_pages_splits_evictable_and_pinned() -> None:
    page = 32 << 20
    assert aimdo_mod._classify_vbar_pages([0, 1, 3]) == (page, page)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(RuntimeError, match="invalid VBAR page status"):
        aimdo_mod._classify_vbar_pages([4])  # pyright: ignore[reportPrivateUsage]


def test_aimdo_weights_reports_existing_vbar_page_map() -> None:
    mechanism, backend = _aimdo({"weight": torch.ones(5000)})
    assert mechanism.page_map() is None

    backend.vbars[0].get_residency = lambda: [0, 1, 3]  # type: ignore[attr-defined]
    assert mechanism.page_map() == PageMap(page_bytes=32 << 20, flags=(0, 1, 3))

    backend.vbars[0].get_residency = lambda: [2]  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="invalid VBAR page status 2"):
        mechanism.page_map()


def test_production_vbar_registry_is_weak_and_capability_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aimdo_mod, "_production_vbars", {})
    monkeypatch.setattr(aimdo_mod, "_device_unpins", {})

    class _Tracked:
        def get_residency(self) -> list[int]:
            return [1, 3]

    class _Legacy:
        pass

    class _MissingCapability:
        def get_residency(self) -> list[int]:
            raise AttributeError("get_residency")

    class _MissingProperty:
        @property
        def get_residency(self) -> object:
            raise ImportError("get_residency")

    class _NoWeakReference:
        __slots__ = ()

    tracked = _Tracked()
    tracked_ref = weakref.ref(tracked)
    legacy = _Legacy()
    missing = _MissingCapability()
    missing_property = _MissingProperty()
    no_weak_reference = _NoWeakReference()
    aimdo_mod._register_production_vbar(tracked, 71)  # pyright: ignore[reportPrivateUsage]
    aimdo_mod._register_production_vbar(legacy, 71)  # pyright: ignore[reportPrivateUsage]
    aimdo_mod._register_production_vbar(missing, 71)  # pyright: ignore[reportPrivateUsage]
    aimdo_mod._register_production_vbar(missing_property, 71)  # pyright: ignore[reportPrivateUsage]
    aimdo_mod._register_production_vbar(no_weak_reference, 71)  # pyright: ignore[reportPrivateUsage]

    page = 32 << 20
    assert aimdo_mod.production_vbar_memory(71) == (page, page)

    del tracked
    gc.collect()
    assert tracked_ref() is None
    assert aimdo_mod.production_vbar_memory(71) == (0, 0)


def test_production_vbar_memory_does_not_touch_pending_unpins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"query": 0, "sync": 0, "unpin": 0}
    pending_event = object()

    class _Backend:
        def event_query(self, event: object) -> bool:
            assert event is pending_event
            calls["query"] += 1
            return False

        def synchronize_event(self, event: object) -> None:
            calls["sync"] += 1

        def unpin(self, allocation: object) -> None:
            calls["unpin"] += 1

    pending = aimdo_mod._PendingUnpins(  # pyright: ignore[reportPrivateUsage]
        owner=object(),
        backend=cast("aimdo_mod.VbarBackend", _Backend()),
        event=pending_event,
        allocations=[object()],
    )
    stream_pending = aimdo_mod._PendingStreamUnpins(  # pyright: ignore[reportPrivateUsage]
        stream=object(), entries=[pending]
    )
    monkeypatch.setattr(aimdo_mod, "_production_vbars", {})
    monkeypatch.setattr(aimdo_mod, "_device_unpins", {(0, 72): [stream_pending]})

    assert aimdo_mod.production_vbar_memory(72) == (0, 0)
    assert calls == {"query": 0, "sync": 0, "unpin": 0}
    assert aimdo_mod._device_unpins[(0, 72)] == [stream_pending]  # pyright: ignore[reportPrivateUsage]


def test_completed_deferred_page_stays_pinned_until_mechanism_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tracked:
        statuses = [3]

        def get_residency(self) -> list[int]:
            return list(self.statuses)

    tracked = _Tracked()
    backend = FakeVbarBackend(complete_events_on_record=False)
    original_unpin = backend.unpin

    def unpin(allocation: object) -> None:
        original_unpin(allocation)
        tracked.statuses = [1]

    monkeypatch.setattr(aimdo_mod, "_production_vbars", {})
    monkeypatch.setattr(aimdo_mod, "_device_unpins", {})
    monkeypatch.setattr(backend, "unpin", unpin)
    aimdo_mod._register_production_vbar(tracked, 0)  # pyright: ignore[reportPrivateUsage]
    mechanism, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend, stream_count=0)

    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    assert len(backend.recorded_events) == 1
    assert aimdo_mod.production_vbar_memory(0) == (0, 32 << 20)

    backend.recorded_events[0].complete = True
    assert aimdo_mod.production_vbar_memory(0) == (0, 32 << 20)
    assert mechanism.partially_unload(0) == 0
    assert aimdo_mod.production_vbar_memory(0) == (32 << 20, 0)


def test_production_vbar_memory_propagates_runtime_error_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("driver failure")

    class _Broken:
        def get_residency(self) -> list[int]:
            raise error

    broken = _Broken()
    monkeypatch.setattr(aimdo_mod, "_production_vbars", {})
    monkeypatch.setattr(aimdo_mod, "_device_unpins", {})
    aimdo_mod._register_production_vbar(broken, 73)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(RuntimeError) as caught:
        aimdo_mod.production_vbar_memory(73)
    assert caught.value is error


def test_geometry_uses_patched_shape_fp8_raw_alignment_and_vbar_padding() -> None:
    patch = PatchSet({"patched": (PatchEntry(DiffPatch(torch.ones(3, 5), pad_weight=True)),)})
    store: dict[str, StoredWeight] = {
        "patched": torch.ones(2, 2, dtype=torch.float16),
        "fp8": _fp8(),
        "large": torch.ones(20_000, dtype=torch.float16),
    }
    mechanism, backend = _aimdo(store, patch_set=patch)

    geometry = mechanism._geometry  # pyright: ignore[reportPrivateUsage]
    assert geometry["patched"].shape == (3, 5)
    assert geometry["patched"].cast_bytes == 3 * 5 * 4
    # One fp8 byte, three alignment bytes, then one float32 scale.
    assert geometry["fp8"].raw_bytes == 8
    assert geometry["fp8"].allocation_bytes == 8
    assert geometry["large"].allocation_bytes == 20_000 * 4
    assert backend.vbars[0].size == 80_384

    # Eager-tier keys have no VBAR allocation; the large key remains paged.
    allocations = mechanism._allocations  # pyright: ignore[reportPrivateUsage]
    assert set(allocations) == {"large"}
    assert allocations["large"] is backend.allocations[0]
    assert backend.allocations[0].offset == 0


def test_vbar_allocations_preserve_execution_unit_order() -> None:
    store: dict[str, StoredWeight] = {
        "first.weight": torch.ones(20_000),
        "first.bias": torch.ones(100),
        "second.weight": torch.ones(30_000),
        "second.bias": torch.ones(200),
    }
    units = (
        ResidencyUnit("first", ("first.weight", "first.bias")),
        ResidencyUnit("second", ("second.weight", "second.bias")),
    )
    mechanism, backend = _aimdo(store, units=units)

    allocations = mechanism._allocations  # pyright: ignore[reportPrivateUsage]
    assert backend.allocations == [allocations[key] for unit in units for key in unit.keys]


def test_int8_store_geometry_uses_route_owned_itemsize_ceiling() -> None:
    layer = Int8Linear(
        256,
        128,
        bias=True,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
    )
    layer.load_state_dict(
        {
            "weight": torch.ones((128, 256), dtype=torch.int8),
            "weight_scale": torch.ones((128, 1), dtype=torch.float32),
            "bias": torch.ones(128, dtype=torch.bfloat16),
        },
        strict=True,
        assign=True,
    )
    mechanism, backend = _enroll_int8(layer)

    geometry = mechanism._geometry  # pyright: ignore[reportPrivateUsage]
    assert geometry["weight"].cast_bytes == 128 * 256 * 2
    assert geometry["weight"].raw_bytes == 128 * 256 + 128 * 4
    assert geometry["weight"].allocation_bytes == 128 * 256 * 2
    assert geometry["bias"].cast_bytes == 128 * 4
    assert backend.vbars[0].size == 128 * 256 * 2 + 512
    allocations = mechanism._allocations  # pyright: ignore[reportPrivateUsage]
    assert cast(_FakeAllocation, allocations["weight"]).size == 128 * 256 * 2
    assert cast(_FakeAllocation, allocations["bias"]).size == 128 * 4


@pytest.mark.parametrize("itemsize", [0, -1, 5, 1.5, True])
def test_materialization_ceiling_capability_is_validated(itemsize: object) -> None:
    class Store(dict[str, StoredWeight]):
        def max_materialized_itemsize(self, key: str) -> object:
            assert key == "weight"
            return itemsize

    with pytest.raises(ValueError, match="positive integer.*no greater than 4"):
        _aimdo(Store(weight=torch.ones(5000)))


def test_materialization_ceiling_is_snapshotted_immutably() -> None:
    class Store(dict[str, StoredWeight]):
        itemsize = 2

        def max_materialized_itemsize(self, key: str) -> int:
            assert key == "weight"
            return self.itemsize

    store = Store(weight=torch.ones(5000))
    mechanism, _ = _aimdo(store)
    store.itemsize = 1

    assert mechanism._geometry["weight"].cast_bytes == 10_000  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(TypeError):
        mechanism._max_materialized_itemsizes["weight"] = 1  # pyright: ignore[reportPrivateUsage, reportIndexIssue]


def test_declared_bf16_route_right_sizes_vbar_and_rejects_wider_requests() -> None:
    module = INITLESS.linear(10_000, 1, bias=False)
    module.load_state_dict(
        {"weight": torch.ones(1, 10_000, dtype=torch.bfloat16)}, strict=True, assign=True
    )
    declare_residency_materialization_ceilings(module, {"weight": 2})
    mechanism, backend = _aimdo(ModuleStateStore(module))

    assert mechanism._geometry["weight"].cast_bytes == 20_000  # pyright: ignore[reportPrivateUsage]
    assert backend.vbars[0].size == 20_480
    with mechanism.lease("weight") as lease:
        with pytest.raises(ValueError, match="requires 40000 bytes; allocation is 20000 bytes"):
            lease.get("weight", dtype=torch.float32)


def test_materialization_ceiling_rejects_wider_cast_when_packed_storage_is_larger() -> None:
    class Store(dict[str, StoredWeight]):
        def max_materialized_itemsize(self, key: str) -> int:
            assert key == "weight"
            return 4

    mechanism, _ = _aimdo(Store(weight=_fp8()))
    assert mechanism._geometry["weight"].allocation_bytes == 8  # pyright: ignore[reportPrivateUsage]

    with mechanism.lease("weight") as lease:
        with pytest.raises(ValueError, match="uses 8 bytes per element.*ceiling is 4"):
            lease.get("weight", dtype=torch.float64)


def test_constructor_reuses_eager_unit_and_patch_validation() -> None:
    store: dict[str, StoredWeight] = {"weight": torch.ones(2, 2)}
    with pytest.raises(Exception, match="do not cover store keys"):
        _aimdo(store, units=())
    with pytest.raises(Exception, match="not in the weight store"):
        _aimdo(
            store,
            patch_set=PatchSet({"missing": (PatchEntry(DiffPatch(torch.ones(2, 2))),)}),
        )


def test_store_is_immutable_and_patch_functions_are_precomputed() -> None:
    original = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    patch = PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(2, 2))),)})
    store: dict[str, StoredWeight] = {"weight": original}
    mechanism, _ = _aimdo(store, patch_set=patch)
    functions = mechanism.weight_functions("weight")
    assert functions is mechanism.weight_functions("weight")
    assert not mechanism.is_loaded("weight")
    with mechanism.lease("weight") as lease:
        leased = lease.get("weight", dtype=torch.float32)
        assert torch.equal(leased, original.detach() + 1)
        assert mechanism.is_loaded("weight")
        assert torch.equal(original, torch.arange(4, dtype=torch.float32).reshape(2, 2))
    mechanism.unload()
    assert store["weight"] is original
    assert torch.equal(original, torch.arange(4, dtype=torch.float32).reshape(2, 2))


def test_cache_keys_signature_dtype_form_and_patch_revision() -> None:
    store: dict[str, StoredWeight] = {
        "plain": torch.arange(5000, dtype=torch.float32),
        "fp8": _fp8((17_000,)),
    }
    mechanism, backend = _aimdo(store)

    with mechanism.lease("plain") as lease:
        first = lease.get("plain", dtype=torch.float32)
        again = lease.get("plain", dtype=torch.float32)
        assert first.data_ptr() == again.data_ptr()
    mechanism._reap_unpins(wait=False)  # pyright: ignore[reportPrivateUsage]
    assert len(backend.tensor_calls) == 1
    assert len(backend.fault_calls) == len(backend.unpins) == 1

    with mechanism.lease("plain") as lease:
        lease.get("plain", dtype=torch.float16)
    assert len(backend.tensor_calls) == 2
    assert len(mechanism._cache["plain"]) == 1  # pyright: ignore[reportPrivateUsage]
    assert next(iter(mechanism._cache["plain"]))[1:3] == (  # pyright: ignore[reportPrivateUsage]
        torch.float16,
        "get",
    )

    mechanism._patch_revision = "new-revision"  # pyright: ignore[reportPrivateUsage]
    with mechanism.lease("plain") as lease:
        lease.get("plain", dtype=torch.float16)
    assert len(backend.tensor_calls) == 3

    plain_allocation = mechanism._allocations[  # pyright: ignore[reportPrivateUsage]
        "plain"
    ]
    backend.change_signature(plain_allocation)
    with mechanism.lease("plain") as lease:
        lease.get("plain", dtype=torch.float16)
    assert len(backend.tensor_calls) == 4

    with mechanism.lease("fp8") as lease:
        cast_value = lease.get("fp8", dtype=torch.float32)
        cast_snapshot = cast_value.clone()
        raw_value = lease.get_stored("fp8")
        raw_again = lease.get_stored("fp8")
    assert isinstance(raw_value, Fp8ScaledWeight)
    assert raw_value is raw_again
    stored_fp8 = cast(Fp8ScaledWeight, store["fp8"])
    assert torch.equal(cast_snapshot, stored_fp8.dequantize(torch.float32))
    assert len(backend.tensor_calls) == 6  # get and get_stored are distinct forms
    assert len(mechanism._cache["fp8"]) == 1  # pyright: ignore[reportPrivateUsage]
    assert next(iter(mechanism._cache["fp8"]))[2] == "get_stored"  # pyright: ignore[reportPrivateUsage]


def test_cache_hit_compares_the_live_signature_without_copying_its_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism, backend = _aimdo({"weight": torch.arange(5000, dtype=torch.float32)})
    with mechanism.lease("weight") as lease:
        expected = lease.get("weight", dtype=torch.float32)

    def reject_signature_copy(_signature: object) -> bytes:
        raise AssertionError("a cache hit must not copy signature bytes")

    monkeypatch.setattr(aimdo_mod, "_signature_bytes", reject_signature_copy)
    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)

    assert actual is expected
    assert len(backend.tensor_calls) == 1


def test_cache_hit_skips_materialization_context(monkeypatch: pytest.MonkeyPatch) -> None:
    mechanism, _ = _aimdo({"weight": torch.ones(5000)})
    with mechanism.lease("weight") as lease:
        expected = lease.get("weight", dtype=torch.float32)

    def refuse_inference_mode() -> AbstractContextManager[None]:
        raise AssertionError("cache hit entered materialization context")

    monkeypatch.setattr(torch, "inference_mode", refuse_inference_mode)
    with mechanism.lease("weight") as lease:
        assert lease.get("weight", dtype=torch.float32) is expected


def test_declared_structural_digest_replaces_random_revision_in_cache_identity() -> None:
    digest = "a" * 64
    patch = PatchSet(
        {"weight": (PatchEntry(DiffPatch(torch.ones(2, 2))),)},
        structural_digest=digest,
    )
    mechanism, _ = _aimdo({"weight": torch.zeros(2, 2)}, patch_set=patch)
    assert mechanism._patch_revision == digest  # pyright: ignore[reportPrivateUsage]


def test_undeclared_patch_set_keeps_random_revision_cache_identity() -> None:
    patch = PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(2, 2))),)})
    mechanism, _ = _aimdo({"weight": torch.zeros(2, 2)}, patch_set=patch)
    assert mechanism._patch_revision == patch.revision  # pyright: ignore[reportPrivateUsage]


def test_get_stored_materializes_raw_fp8_bytes_exactly() -> None:
    stored = _fp8((2, 3))
    mechanism, _ = _aimdo({"weight": stored})
    with mechanism.lease("weight") as lease:
        actual = lease.get_stored("weight")
    assert isinstance(actual, Fp8ScaledWeight)
    assert torch.equal(actual.qdata.view(torch.uint8), stored.qdata.view(torch.uint8))
    assert torch.equal(actual.scale, stored.scale)


def test_get_stored_streams_plain_integer_blocks_exactly() -> None:
    generator = torch.Generator().manual_seed(7)
    blocks = torch.randint(0, 256, (20_000,), dtype=torch.uint8, generator=generator)
    mechanism, backend = _aimdo({"blocks": blocks.clone()})

    geometry = mechanism._geometry  # pyright: ignore[reportPrivateUsage]
    assert geometry["blocks"].raw_bytes == blocks.nbytes

    with mechanism.lease("blocks") as lease:
        actual = lease.get_stored("blocks")
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == torch.uint8
    assert torch.equal(actual, blocks)
    assert backend.fault_calls  # streamed through a VBAR allocation

    prefetching, prefetch_backend = _aimdo({"blocks": blocks.clone()})
    handle = prefetching.prefetch((("blocks", None),))
    assert handle is not None
    faults_after_prefetch = len(prefetch_backend.fault_calls)
    with prefetching.lease("blocks") as lease:
        adopted = lease.get_stored("blocks")
    handle.close()
    assert isinstance(adopted, torch.Tensor)
    assert torch.equal(adopted, blocks)
    assert len(prefetch_backend.fault_calls) == faults_after_prefetch


def test_get_stored_plain_float_fails_closed() -> None:
    mechanism, _ = _aimdo({"weight": torch.ones(5000, dtype=torch.float32)})
    geometry = mechanism._geometry  # pyright: ignore[reportPrivateUsage]
    # Plain float state has no raw consumers; its allocation is sized
    # by the cast ceiling alone.
    assert geometry["weight"].raw_bytes == 0
    with mechanism.lease("weight") as lease:
        with pytest.raises(TypeError, match="packed or integer storage"):
            lease.get_stored("weight")
    with pytest.raises(TypeError, match="packed or integer storage"):
        mechanism.prefetch((("weight", None),))


def test_gguf_encoded_linear_streams_raw_blocks_and_reports_receipts_under_aimdo() -> None:
    generator = torch.Generator().manual_seed(11)
    qs = torch.randint(-127, 128, (512, 32), dtype=torch.int8, generator=generator)
    scales = torch.full((512, 1), 0.25, dtype=torch.float16)
    blocks = torch.cat((scales.view(torch.uint8), qs.view(torch.uint8)), dim=1)
    bias = torch.randn(64, generator=generator)
    module = GgufEncodedLinear(256, 64, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})
    x = torch.randn(3, 256, generator=generator)
    expected = module(x)

    concrete = FakeVbarBackend()

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> AimdoWeights:
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            backend=concrete,
        )

    mechanism = enroll_component(
        module,
        load_device=CUDA0,
        offload_device=CPU,
        mechanism_factory=factory,
    )
    assert isinstance(mechanism, AimdoWeights)

    # The GGUF route owns a one-byte materialization ceiling: encoded
    # blocks are never widened, so the VBAR allocation is exactly the
    # stored block bytes rather than a float32-sized ceiling.
    geometry = mechanism._geometry  # pyright: ignore[reportPrivateUsage]
    assert geometry["weight_blocks"].cast_bytes == blocks.numel()
    assert geometry["weight_blocks"].raw_bytes == blocks.nbytes
    assert geometry["weight_blocks"].allocation_bytes == blocks.nbytes

    with collect_partial_residency_timing() as timing:
        assert torch.equal(module(x), expected)
    report = timing.report()
    assert report.leased_forwards == 1
    assert report.leased_transfers == 2  # weight_blocks and bias
    assert report.transfer_bytes == blocks.nbytes + bias.nbytes
    assert report.dequant_ms > 0.0
    assert report.compute_ms > 0.0

    # The uncollected forward takes the fused path, still bit-equal.
    assert torch.equal(module(x), expected)


def test_oom_fallback_is_uncached_and_reuses_host_pin() -> None:
    backend = FakeVbarBackend(oom=True)
    expected = torch.arange(5000, dtype=torch.float32)
    mechanism, _ = _aimdo({"weight": expected}, backend=backend)
    with mechanism.lease("weight") as lease:
        first = lease.get("weight", dtype=torch.float32)
        second = lease.get("weight", dtype=torch.float32)
    assert torch.equal(first, expected) and torch.equal(second, expected)
    assert first is second
    assert first.data_ptr() == second.data_ptr()
    assert not first.requires_grad and not second.requires_grad
    assert backend.tensor_calls == []
    assert backend.unpins == []
    assert len(mechanism._pins) == 1  # pyright: ignore[reportPrivateUsage]


def test_all_fault_miss_uses_pinned_non_blocking_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeVbarBackend(oom=True)
    expected = torch.arange(5000, dtype=torch.float32)
    mechanism, _ = _aimdo(
        {"weight": expected},
        backend=backend,
        stream_count=1,
    )
    calls: list[tuple[str, bool]] = []
    original = mechanism._computed_from_source  # pyright: ignore[reportPrivateUsage]

    def record(
        request: aimdo_mod._BatchRequest,  # pyright: ignore[reportPrivateUsage]
        source: StoredWeight,
        *,
        non_blocking: bool = False,
        collector: PartialResidencyTiming | None = None,
    ) -> StoredWeight:
        calls.append((request.key, non_blocking))
        return original(request, source, non_blocking=non_blocking, collector=collector)

    monkeypatch.setattr(mechanism, "_computed_from_source", record)
    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)

    pin = mechanism._pins[("weights", "weight")]  # pyright: ignore[reportPrivateUsage]
    assert torch.equal(actual, expected)
    assert backend.is_pinned(pin.tensor)
    assert calls == [("weight", True)]
    assert [stream.name for stream in backend.streams] == ["transfer-0"]
    assert [event for event in backend.events if event[0] == "context"] == [
        ("context", "transfer-0")
    ]
    assert [event for event in backend.events if event[0] == "wait"] == [
        ("wait", "current", "transfer-0"),
        ("wait", "transfer-0", "current"),
    ]


def test_pin_steal_waits_for_pending_transfer_before_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeVbarBackend(oom=True)
    first_source = torch.ones(5000)
    second_source = torch.full((5000,), 2.0)
    mechanism, _ = _aimdo(
        {"first": first_source, "second": second_source},
        backend=backend,
        stream_count=1,
    )
    pending: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_synchronize = backend.synchronize_stream

    def defer_transfer(
        _request: aimdo_mod._BatchRequest,  # pyright: ignore[reportPrivateUsage]
        source: StoredWeight,
        *,
        non_blocking: bool = False,
        collector: PartialResidencyTiming | None = None,
    ) -> StoredWeight:
        assert non_blocking and isinstance(source, torch.Tensor)
        target = torch.empty_like(source)
        pending.append((target, source))
        return target

    def synchronize(stream: object) -> None:
        with torch.inference_mode():
            for target, source in pending:
                target.copy_(source)
        pending.clear()
        original_synchronize(stream)

    monkeypatch.setattr(mechanism, "_computed_from_source", defer_transfer)
    monkeypatch.setattr(backend, "synchronize_stream", synchronize)
    with mechanism.lease("first") as lease:
        first = lease.get("first", dtype=torch.float32)

    def no_pin_budget(_size: int) -> bool:
        return False

    monkeypatch.setattr(pinned_host, "ensure_pin_budget", no_pin_budget)
    with mechanism.lease("second") as lease:
        second = lease.get("second", dtype=torch.float32)
    backend.synchronize_stream(backend.streams[0])

    assert torch.equal(first, first_source)
    assert torch.equal(second, second_source)


def test_pin_host_buffer_failure_steals_outside_the_exception_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RuntimeError traceback holds the extend/register frames' buffers
    while the handler runs, so the steal fallback must execute with no
    active exception."""
    backend = FakeVbarBackend(oom=True)
    first_source = torch.ones(5000)
    second_source = torch.full((5000,), 2.0)
    mechanism, _ = _aimdo(
        {"first": first_source, "second": second_source},
        backend=backend,
        stream_count=1,
    )
    with mechanism.lease("first") as lease:
        first = lease.get("first", dtype=torch.float32)
    assert torch.equal(first, first_source)

    def exhausted_extend(host_buffer: object, size: int) -> None:
        del host_buffer, size
        raise RuntimeError("host buffer exhausted")

    monkeypatch.setattr(backend, "extend_host_buffer", exhausted_extend)

    active_exceptions: list[BaseException | None] = []
    original_steal = mechanism._steal_pin  # pyright: ignore[reportPrivateUsage]

    def recording_steal(
        request: aimdo_mod._PinRequest,  # pyright: ignore[reportPrivateUsage]
        source: StoredWeight,
        size: int,
        priority: int,
        *,
        subset: aimdo_mod._PinSubset = "weights",  # pyright: ignore[reportPrivateUsage]
    ) -> StoredWeight | None:
        active_exceptions.append(sys.exc_info()[1])
        return original_steal(request, source, size, priority, subset=subset)

    monkeypatch.setattr(mechanism, "_steal_pin", recording_steal)
    with mechanism.lease("second") as lease:
        second = lease.get("second", dtype=torch.float32)
    backend.synchronize_stream(backend.streams[0])

    assert torch.equal(second, second_source)
    assert active_exceptions
    assert all(active is None for active in active_exceptions)


def test_tight_registration_budget_synchronizes_and_reregisters_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeVbarBackend(oom=True)
    first_source = torch.arange(5000, dtype=torch.float32)
    second_source = first_source + 10_000
    mechanism, _ = _aimdo(
        {"first": first_source, "second": second_source},
        backend=backend,
        pin_all_sources=True,
        stream_count=2,
    )
    pinned_host.configure(maximum=first_source.nbytes)
    pending: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_synchronize = backend.synchronize_stream

    def defer_transfer(
        _request: aimdo_mod._BatchRequest,  # pyright: ignore[reportPrivateUsage]
        selected: StoredWeight,
        *,
        non_blocking: bool = False,
        collector: PartialResidencyTiming | None = None,
    ) -> StoredWeight:
        assert non_blocking and isinstance(selected, torch.Tensor)
        target = torch.full_like(selected, -1)
        pending.append((target, selected))
        return target

    def synchronize(stream: object) -> None:
        with torch.inference_mode():
            for target, selected in pending:
                target.copy_(selected)
        pending.clear()
        original_synchronize(stream)

    monkeypatch.setattr(mechanism, "_computed_from_source", defer_transfer)
    monkeypatch.setattr(backend, "synchronize_stream", synchronize)
    with mechanism.lease("first") as lease:
        first = lease.get("first", dtype=torch.float32)
    with mechanism.lease("second") as lease:
        second = lease.get("second", dtype=torch.float32)
    with mechanism.lease("first") as lease:
        first_again = lease.get("first", dtype=torch.float32)
    for stream in backend.streams:
        backend.synchronize_stream(stream)

    assert len(mechanism._pins) == 1  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_MEMORY == first_source.nbytes
    assert pinned_host.TOTAL_PINNED_STORAGE == first_source.nbytes
    assert torch.equal(first, first_source)
    assert torch.equal(second, second_source)
    assert torch.equal(first_again, first_source)


def test_pin_eviction_waits_for_pending_transfer_before_truncating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeVbarBackend(oom=True)
    source = torch.arange(5000, dtype=torch.float32)
    mechanism, _ = _aimdo({"weight": source}, backend=backend, pin_all_sources=True, stream_count=2)
    pending: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_synchronize = backend.synchronize_stream

    def defer_transfer(
        _request: aimdo_mod._BatchRequest,  # pyright: ignore[reportPrivateUsage]
        selected: StoredWeight,
        *,
        non_blocking: bool = False,
        collector: PartialResidencyTiming | None = None,
    ) -> StoredWeight:
        assert non_blocking and isinstance(selected, torch.Tensor)
        target = torch.full_like(selected, -1)
        pending.append((target, selected))
        return target

    def synchronize(stream: object) -> None:
        with torch.inference_mode():
            for target, selected in pending:
                target.copy_(selected)
        pending.clear()
        original_synchronize(stream)

    monkeypatch.setattr(mechanism, "_computed_from_source", defer_transfer)
    monkeypatch.setattr(backend, "synchronize_stream", synchronize)
    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)

    hostbuf = cast(
        _FakeHostBuffer,
        mechanism._pin_state["weights"][0],  # pyright: ignore[reportPrivateUsage]
    )
    incoming = AimdoWeights(
        {"other": torch.ones(5000)},
        load_device=torch.device("cuda:1"),
        offload_device=CPU,
        backend=backend,
        pin_all_sources=True,
    )
    pinned_host.configure(maximum=source.nbytes)
    with incoming.lease("other") as lease:
        lease.get("other", dtype=torch.float32)

    assert torch.equal(actual, source)
    assert pending == []
    assert hostbuf.size == 0
    assert not mechanism.pin_active


def test_repeated_retired_owners_keep_total_pins_bounded() -> None:
    mechanisms: list[AimdoWeights] = []
    hostbufs: list[_FakeHostBuffer] = []
    pinned_host.configure(maximum=20_000)

    for value in range(6):
        mechanism, _ = _aimdo(
            {"weight": torch.full((5000,), float(value))},
            backend=FakeVbarBackend(oom=True),
            pin_all_sources=True,
        )
        with mechanism.lease("weight") as lease:
            lease.get("weight", dtype=torch.float32)
        mechanisms.append(mechanism)
        hostbufs.append(
            cast(
                _FakeHostBuffer,
                mechanism._pin_state["weights"][0],  # pyright: ignore[reportPrivateUsage]
            )
        )
        assert pinned_host.TOTAL_PINNED_MEMORY == 20_000

    assert all(hostbuf.size == 0 for hostbuf in hostbufs[:-1])
    assert hostbufs[-1].size == 20_000


def test_reused_retired_owner_reregisters_and_rebuilds_pin() -> None:
    source = torch.arange(5000, dtype=torch.float32)
    retired, _ = _aimdo({"weight": source}, backend=FakeVbarBackend(oom=True), pin_all_sources=True)
    retired.partially_load(0)
    with retired.lease("weight") as lease:
        assert torch.equal(lease.get("weight", dtype=torch.float32), source)
    hostbuf = cast(
        _FakeHostBuffer,
        retired._pin_state["weights"][0],  # pyright: ignore[reportPrivateUsage]
    )

    incoming, _ = _aimdo(
        {"other": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    pinned_host.configure(maximum=source.nbytes)
    with incoming.lease("other") as lease:
        lease.get("other", dtype=torch.float32)
    assert hostbuf.size == 0
    assert retired not in pinned_host._owners  # pyright: ignore[reportPrivateUsage]

    with retired.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)

    assert torch.equal(actual, source)
    assert hostbuf.size == source.nbytes
    assert retired in pinned_host._owners  # pyright: ignore[reportPrivateUsage]


def test_new_owner_admission_skips_a_busy_owner() -> None:
    active, _ = _aimdo(
        {"weight": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    active.partially_load(0)
    with active.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    hostbuf = cast(
        _FakeHostBuffer,
        active._pin_state["weights"][0],  # pyright: ignore[reportPrivateUsage]
    )
    entered = threading.Event()
    release = threading.Event()

    def hold_active_owner() -> None:
        with active._lock:  # pyright: ignore[reportPrivateUsage]
            active.pin_active = True
            entered.set()
            assert release.wait(timeout=5)
            active.pin_active = False

    thread = threading.Thread(target=hold_active_owner)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        pinned_host.configure(maximum=20_000)
        incoming = AimdoWeights(
            {"other": torch.ones(5000)},
            load_device=torch.device("cuda:1"),
            offload_device=CPU,
            backend=FakeVbarBackend(oom=True),
            pin_all_sources=True,
        )
        with incoming.lease("other") as lease:
            lease.get("other", dtype=torch.float32)

        assert hostbuf.size == 20_000
        assert pinned_host.TOTAL_PINNED_STORAGE == 20_000
        assert not incoming._pins  # pyright: ignore[reportPrivateUsage]
        assert active.pin_active
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_production_order_keeps_every_allocating_owner_in_ledger() -> None:
    names = ("conditioner", "dit_rank", "video_vae", "audio_vae")
    mechanisms = {
        name: _aimdo(
            {"weight": torch.ones(5000)},
            backend=FakeVbarBackend(oom=True),
            pin_all_sources=True,
        )[0]
        for name in names
    }

    # Compat admits text before the nested VAE stage. ResidencyManager reverses
    # the VAE tuple, so audio admission precedes video admission.
    mechanisms["conditioner"].partially_load(0)
    mechanisms["audio_vae"].partially_load(0)
    mechanisms["video_vae"].partially_load(0)
    for name in ("conditioner", "video_vae", "audio_vae"):
        with mechanisms[name].lease("weight") as lease:
            lease.get("weight", dtype=torch.float32)
    with mechanisms["dit_rank"].lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)

    assert set(pinned_host._owners) == set(mechanisms.values())  # pyright: ignore[reportPrivateUsage]
    assert all(
        pinned_host._storage_by_owner[id(mechanism)] == 20_000  # pyright: ignore[reportPrivateUsage]
        for mechanism in mechanisms.values()
    )
    assert pinned_host.TOTAL_PINNED_STORAGE == 80_000

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def condition() -> None:
        try:
            barrier.wait()
            mechanisms["conditioner"].partially_load(0)
            mechanisms["audio_vae"].partially_load(0)
            mechanisms["video_vae"].partially_load(0)
            with mechanisms["conditioner"].lease("weight") as lease:
                lease.get("weight", dtype=torch.float32)
        except BaseException as error:
            errors.append(error)

    conditions = [threading.Thread(target=condition) for _ in range(2)]
    for thread in conditions:
        thread.start()
    for thread in conditions:
        thread.join(timeout=5)

    assert not errors
    assert all(not thread.is_alive() for thread in conditions)
    assert set(pinned_host._owners) == set(mechanisms.values())  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_STORAGE == 80_000


def test_registration_pressure_and_storage_eviction_overlap_without_double_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _ = _aimdo(
        {"weight": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    second_backend = FakeVbarBackend(oom=True)
    second = AimdoWeights(
        {"weight": torch.ones(5000)},
        load_device=torch.device("cuda:1"),
        offload_device=CPU,
        backend=second_backend,
        pin_all_sources=True,
    )
    for mechanism in (first, second):
        with mechanism.lease("weight") as lease:
            lease.get("weight", dtype=torch.float32)
    pinned_host.configure(maximum=40_000)
    registration_entered = threading.Event()
    storage_entered = threading.Event()
    registration_released: list[int] = []
    storage_released: list[int] = []
    storage_reserved: list[bool] = []
    errors: list[BaseException] = []
    original_registration_release = second.free_registrations
    original_storage_release = first.free_pins

    def skip_first_registration(_size: int) -> int:
        return 0

    monkeypatch.setattr(first, "free_registrations", skip_first_registration)

    def release_registration(size: int) -> int:
        with second._lock:  # pyright: ignore[reportPrivateUsage]
            registration_entered.set()
            assert storage_entered.wait(timeout=5)
            released = original_registration_release(size)
            registration_released.append(released)
            return released

    def release_storage(size: int) -> int:
        with first._lock:  # pyright: ignore[reportPrivateUsage]
            storage_entered.set()
            assert registration_entered.wait(timeout=5)
            released = original_storage_release(size)
            storage_released.append(released)
            return released

    monkeypatch.setattr(second, "free_registrations", release_registration)
    monkeypatch.setattr(first, "free_pins", release_storage)

    def registration_pressure() -> None:
        try:
            pinned_host.free_registrations(1)
        except BaseException as error:
            errors.append(error)

    def storage_pressure() -> None:
        try:
            reserved = pinned_host.reserve_storage(second, 20_000)
            storage_reserved.append(reserved)
            if reserved:
                pinned_host.account_storage(second, -20_000)
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=registration_pressure),
        threading.Thread(target=storage_pressure),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert registration_entered.is_set() and storage_entered.is_set()
    assert registration_released == [20_000]
    assert storage_released == [20_000]
    assert storage_reserved == [True]
    first_hostbuf = cast(
        _FakeHostBuffer,
        first._pin_state["weights"][0],  # pyright: ignore[reportPrivateUsage]
    )
    assert first_hostbuf.truncations[-1] == (0, True)
    assert not first._pins  # pyright: ignore[reportPrivateUsage]
    second_pin = next(iter(second._pins.values()))  # pyright: ignore[reportPrivateUsage]
    assert not second_pin.registered
    assert not second_backend.registered_ptrs
    assert pinned_host.TOTAL_PINNED_MEMORY == 0
    assert pinned_host.TOTAL_PINNED_STORAGE == 20_000


def test_terminal_unload_removes_strong_owner_and_all_storage() -> None:
    mechanism, _ = _aimdo(
        {"first": torch.ones(5000), "second": torch.ones(7000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    for key in ("first", "second"):
        with mechanism.lease(key) as lease:
            lease.get(key, dtype=torch.float32)
    assert mechanism in pinned_host._owners  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_STORAGE == 48_000

    mechanism.unload()

    assert mechanism not in pinned_host._owners  # pyright: ignore[reportPrivateUsage]
    assert id(mechanism) not in pinned_host._storage_by_owner  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_STORAGE == 0
    assert pinned_host.TOTAL_PINNED_MEMORY == 0


def test_live_available_pressure_shrinks_storage_below_static_cap() -> None:
    mechanism, _ = _aimdo(
        {"weight": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    assert pinned_host.TOTAL_PINNED_STORAGE == 20_000

    assert pinned_host.ensure_pin_budget(
        1,
        available=lambda: pinned_host.AVAILABLE_RAM_FLOOR - 1,
    )
    assert pinned_host.TOTAL_PINNED_STORAGE == 0
    assert mechanism not in pinned_host._owners  # pyright: ignore[reportPrivateUsage]


def test_synchronous_pin_state_keeps_patches_subset_unstaged() -> None:
    patch = PatchSet({"patched": (PatchEntry(DiffPatch(torch.ones(5000))),)})
    mechanism, _ = _aimdo(
        {"plain": torch.ones(5000), "patched": torch.ones(5000)},
        patch_set=patch,
        pin_all_sources=True,
        stream_count=0,
    )
    with mechanism.lease("plain") as lease:
        lease.get("plain", dtype=torch.float32)
        lease.get("patched", dtype=torch.float32)
    weights = mechanism._pin_state["weights"]  # pyright: ignore[reportPrivateUsage]
    patches = mechanism._pin_state["patches"]  # pyright: ignore[reportPrivateUsage]
    assert len(weights[1]) == 2 and weights[2] == [1] and weights[3] == [40_000]
    assert weights[4] == [2] and list(weights[5]) == [20_000]
    assert patches[1] == [] and patches[2] == [-1] and patches[3] == [0]


def test_streamed_patch_payload_populates_reuses_and_frees_patches_subset() -> None:
    payload = torch.arange(5000, dtype=torch.float32)
    patch = PatchSet({"patched": (PatchEntry(DiffPatch(payload)),)})
    mechanism, backend = _aimdo(
        {"patched": torch.ones(5000)},
        patch_set=patch,
        pin_all_sources=True,
    )
    with mechanism.lease("patched") as lease:
        first = lease.get("patched", dtype=torch.float32)
    assert torch.equal(first, torch.ones(5000) + payload)
    patch_state = mechanism._pin_state["patches"]  # pyright: ignore[reportPrivateUsage]
    assert len(patch_state[1]) == 1
    assert patch_state[2] == [0] and patch_state[3] == [20_480]
    patch_identity = next(identity for identity in mechanism._pins if identity[0] == "patches")  # pyright: ignore[reportPrivateUsage]
    pin = mechanism._pins[patch_identity]  # pyright: ignore[reportPrivateUsage]
    assert backend.is_pinned(pin.tensor)
    request = cast(aimdo_mod._PatchRequest, pin.request)  # pyright: ignore[reportPrivateUsage]
    reused = mechanism._existing_pin(request, subset="patches")  # pyright: ignore[reportPrivateUsage]
    assert reused is pin.value
    reshaped_request = aimdo_mod._PatchRequest(  # pyright: ignore[reportPrivateUsage]
        request.key,
        request.index,
        request.source_id,
        request.dtype,
        (100, 50),
    )
    reshaped = mechanism._existing_pin(  # pyright: ignore[reportPrivateUsage]
        reshaped_request, subset="patches"
    )
    assert isinstance(reshaped, torch.Tensor) and reshaped is not pin.value
    assert reshaped.dtype == request.dtype and reshaped.shape == (100, 50)
    assert isinstance(pin.value, torch.Tensor) and reshaped.data_ptr() == pin.value.data_ptr()
    total = pinned_host.TOTAL_PINNED_MEMORY

    backend.change_signature(mechanism._allocations["patched"])  # pyright: ignore[reportPrivateUsage]
    with mechanism.lease("patched") as lease:
        second = lease.get("patched", dtype=torch.float32)
    assert torch.equal(second, first)
    assert mechanism._pins[patch_identity] is pin  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_MEMORY == total

    assert mechanism.free_registrations(40_480) == 40_480
    assert not pin.registered
    backend.change_signature(mechanism._allocations["patched"])  # pyright: ignore[reportPrivateUsage]
    with mechanism.lease("patched") as lease:
        third = lease.get("patched", dtype=torch.float32)
    assert torch.equal(third, first) and pin.registered
    assert (
        mechanism._existing_pin(  # pyright: ignore[reportPrivateUsage]
            request, subset="patches"
        )
        is pin.value
    )
    mechanism.unload()
    assert pinned_host.TOTAL_PINNED_MEMORY == 0
    assert patch_state[1] == []
    mechanism.partially_load(0)
    backend.change_signature(mechanism._allocations["patched"])  # pyright: ignore[reportPrivateUsage]
    with mechanism.lease("patched") as lease:
        assert torch.equal(lease.get("patched", dtype=torch.float32), first)
    assert len(mechanism._pin_state["patches"][1]) == 1  # pyright: ignore[reportPrivateUsage]


def test_patches_subset_steals_only_an_equal_size_patch_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = PatchSet(
        {
            "first": (PatchEntry(DiffPatch(torch.ones(5000))),),
            "second": (PatchEntry(DiffPatch(torch.full((5000,), 2.0))),),
        }
    )
    mechanism, _ = _aimdo(
        {"first": torch.ones(5000), "second": torch.ones(5000)},
        patch_set=patch,
        pin_all_sources=True,
    )
    with mechanism.lease("first") as lease:
        lease.get("first", dtype=torch.float32)
    first_identity = next(
        identity
        for identity in mechanism._pins  # pyright: ignore[reportPrivateUsage]
        if identity[0] == "patches"
    )
    victim = mechanism._pins[first_identity]  # pyright: ignore[reportPrivateUsage]

    def no_pin_budget(_size: int) -> bool:
        return False

    monkeypatch.setattr(pinned_host, "ensure_pin_budget", no_pin_budget)
    with mechanism.lease("second") as lease:
        actual = lease.get("second", dtype=torch.float32)
    patch_pins = {
        identity: pin
        for identity, pin in mechanism._pins.items()  # pyright: ignore[reportPrivateUsage]
        if identity[0] == "patches"
    }
    assert len(patch_pins) == 1 and first_identity not in patch_pins
    replacement = next(iter(patch_pins.values()))
    assert replacement.tensor.data_ptr() == victim.tensor.data_ptr()
    assert replacement.value is not victim.value
    assert isinstance(replacement.value, torch.Tensor)
    assert torch.equal(replacement.value, torch.full((5000,), 2.0))
    assert torch.equal(actual, torch.full((5000,), 3.0))
    assert ("weights", "second") in mechanism._pins  # pyright: ignore[reportPrivateUsage]


def test_lifo_unregister_reregister_and_failed_unregister_error_clear() -> None:
    expected = torch.arange(5000, dtype=torch.float32)
    mechanism, backend = _aimdo(
        {"first": expected, "second": expected + 1},
        pin_all_sources=True,
        stream_count=0,
    )
    with mechanism.lease("first") as lease:
        lease.get("first", dtype=torch.float32)
        lease.get("second", dtype=torch.float32)
    hostbuf = cast(_FakeHostBuffer, mechanism._pin_state["weights"][0])  # pyright: ignore[reportPrivateUsage]
    assert hostbuf.commits == [20_000, 40_000]
    backend.unregister_results = [False, False]
    assert mechanism.free_registrations(1) == 0
    assert backend.discarded_errors == 2
    # Reset the high-water mark to exercise the successful keep-data path
    # independently of upstream's no-retry treatment of failed unregisters.
    mechanism._pin_state["weights"][2][0] = 1  # pyright: ignore[reportPrivateUsage]
    backend.unregister_results = [True]
    assert mechanism.free_registrations(1) == 20_000
    pin = mechanism._pins[("weights", "second")]  # pyright: ignore[reportPrivateUsage]
    snapshot = pin.tensor.clone()
    backend.change_signature(mechanism._allocations["second"])  # pyright: ignore[reportPrivateUsage]
    with mechanism.lease("second") as lease:
        lease.get("second", dtype=torch.float32)
    assert pin.registered and torch.equal(pin.tensor, snapshot)
    assert mechanism.free_pins(1) == 20_000
    assert hostbuf.truncations[-1] == (20_000, True)


@pytest.mark.parametrize("release_kind", ["unregister", "truncate"])
def test_cross_device_pressure_serializes_transfer_and_pin_release(
    monkeypatch: pytest.MonkeyPatch, release_kind: str
) -> None:
    backend = FakeVbarBackend(oom=True)
    source = torch.arange(5000, dtype=torch.float32)
    mechanism, _ = _aimdo({"weight": source}, backend=backend, pin_all_sources=True, stream_count=2)
    other = AimdoWeights(
        {"other": torch.ones(5000)},
        load_device=torch.device("cuda:1"),
        offload_device=CPU,
        backend=backend,
        stream_count=2,
    )
    other._initialize_pins()  # pyright: ignore[reportPrivateUsage]
    pending: list[tuple[torch.Tensor, torch.Tensor]] = []
    invalid: set[int] = set()
    order: list[tuple[str, str]] = []
    pressure_sync = threading.Event()
    allow_sync = threading.Event()
    read_attempted = threading.Event()
    read_enqueued = threading.Event()

    def defer_transfer(
        _request: aimdo_mod._BatchRequest,  # pyright: ignore[reportPrivateUsage]
        selected: StoredWeight,
        *,
        non_blocking: bool = False,
        collector: PartialResidencyTiming | None = None,
    ) -> StoredWeight:
        assert non_blocking and isinstance(selected, torch.Tensor)
        target = torch.full_like(selected, -1)
        pending.append((target, selected))
        if read_attempted.is_set():
            read_enqueued.set()
        return target

    original_synchronize = backend.synchronize_stream

    def synchronize(stream: object) -> None:
        name = cast(_FakeStream, stream).name
        order.append(("synchronize", name))
        submitted = tuple(pending)
        del pending[:]
        with torch.inference_mode():
            for target, selected in submitted:
                if selected.data_ptr() in invalid:
                    target.fill_(-2)
                else:
                    target.copy_(selected)
        if threading.current_thread().name == "gpu1-pressure" and not pressure_sync.is_set():
            pressure_sync.set()
            assert allow_sync.wait(5)
        original_synchronize(stream)

    original_register = backend.register_host_memory

    def register(tensor: torch.Tensor) -> bool:
        registered = original_register(tensor)
        if registered:
            invalid.discard(tensor.data_ptr())
        return registered

    original_unregister = backend.unregister_host_memory

    def unregister(tensor: torch.Tensor) -> bool:
        order.append(("release", "unregister"))
        invalid.add(tensor.data_ptr())
        return original_unregister(tensor)

    original_truncate = backend.truncate_host_buffer

    def truncate(host_buffer: object, size: int, unregister_pin: bool) -> None:
        order.append(("release", "truncate"))
        invalid.add(pin_pointer)
        original_truncate(host_buffer, size, unregister_pin)

    monkeypatch.setattr(mechanism, "_computed_from_source", defer_transfer)
    monkeypatch.setattr(backend, "synchronize_stream", synchronize)
    monkeypatch.setattr(backend, "register_host_memory", register)
    monkeypatch.setattr(backend, "unregister_host_memory", unregister)
    monkeypatch.setattr(backend, "truncate_host_buffer", truncate)
    with mechanism.lease("weight") as lease:
        first = lease.get("weight", dtype=torch.float32)
    pin_pointer = mechanism._pins[  # pyright: ignore[reportPrivateUsage]
        ("weights", "weight")
    ].tensor.data_ptr()

    pressure_result: list[int] = []

    def force_pressure(_size: int) -> bool:
        if pressure_result:
            return True
        if release_kind == "unregister":
            pressure_result.append(int(pinned_host.free_registrations(1)))
        else:
            pressure_result.append(pinned_host.free_pins(1))
        return False

    monkeypatch.setattr(pinned_host, "ensure_pin_budget", force_pressure)

    def apply_pressure() -> None:
        with other.lease("other") as lease:
            lease.get("other", dtype=torch.float32)

    pressure = threading.Thread(target=apply_pressure, name="gpu1-pressure")
    pressure.start()
    assert pressure_sync.wait(5)

    second_result: list[torch.Tensor] = []

    def read_again() -> None:
        read_attempted.set()
        with mechanism.lease("weight") as lease:
            second_result.append(lease.get("weight", dtype=torch.float32))

    reader = threading.Thread(target=read_again, name="gpu0-reader")
    reader.start()
    assert read_attempted.wait(5)
    try:
        assert not read_enqueued.wait(0.1)
    finally:
        allow_sync.set()
    pressure.join(5)
    reader.join(5)
    assert not pressure.is_alive() and not reader.is_alive()

    for stream in tuple(backend.streams):
        backend.synchronize_stream(stream)
    assert pressure_result == [1 if release_kind == "unregister" else 20_000]
    assert torch.equal(first, source)
    assert len(second_result) == 1 and torch.equal(second_result[0], source)
    release_index = order.index(("release", release_kind))
    assert {name for event, name in order[:release_index] if event == "synchronize"} >= {
        "transfer-0",
        "transfer-1",
    }


def test_concurrent_cross_device_pressure_skips_busy_owners() -> None:
    first, _ = _aimdo({"first": torch.ones(5000)})
    second = AimdoWeights(
        {"second": torch.ones(5000)},
        load_device=torch.device("cuda:1"),
        offload_device=CPU,
        backend=FakeVbarBackend(),
    )
    first._initialize_pins()  # pyright: ignore[reportPrivateUsage]
    second._initialize_pins()  # pyright: ignore[reportPrivateUsage]
    barrier = threading.Barrier(2)
    results: list[bool] = []

    def apply_pressure(owner: AimdoWeights) -> None:
        with owner._lock:  # pyright: ignore[reportPrivateUsage]
            barrier.wait()
            results.append(pinned_host.free_registrations(1))

    threads = [
        threading.Thread(target=apply_pressure, args=(owner,), daemon=True)
        for owner in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(1)

    assert not any(thread.is_alive() for thread in threads)
    assert results == [False, False]


def test_balancer_priority_persists_across_failed_attempts_and_steals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeVbarBackend(oom=True)
    mechanism, _ = _aimdo({"first": torch.ones(5000), "second": torch.ones(5000)}, backend=backend)
    original_budget = pinned_host.ensure_pin_budget

    def no_budget(_size: int) -> bool:
        return False

    monkeypatch.setattr(pinned_host, "ensure_pin_budget", no_budget)
    with mechanism.lease("first") as lease:
        lease.get("first", dtype=torch.float32)
        lease.get("first", dtype=torch.float32)
    assert mechanism._pin_priorities[("weights", "first")] == 0  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(pinned_host, "ensure_pin_budget", original_budget)
    with mechanism.lease("first") as lease:
        lease.get("first", dtype=torch.float32)
    victim = mechanism._pins[("weights", "first")]  # pyright: ignore[reportPrivateUsage]
    pinned_host.configure(maximum=0)
    with mechanism.lease("second") as lease:
        lease.get("second", dtype=torch.float32)
    replacement = mechanism._pins[("weights", "second")]  # pyright: ignore[reportPrivateUsage]
    assert replacement.tensor.data_ptr() == victim.tensor.data_ptr()
    assert replacement.value is not victim.value
    assert isinstance(replacement.value, torch.Tensor)
    assert torch.equal(replacement.value, torch.ones(5000))


def test_pin_budget_platform_memory_and_disable_math() -> None:
    assert pinned_host.PIN_PRESSURE_HYSTERESIS == 256 * 1024**2
    assert pinned_host.REGISTERABLE_PIN_HYSTERESIS == 2 * 1024**3
    assert pinned_host.memory_status("win32", windows_query=lambda: (1000, 250)) == (1000, 250)
    assert pinned_host.platform_pin_ratio("win32") == 0.40
    assert pinned_host.platform_pin_ratio("linux") == 0.90
    pinned_host.configure(maximum=100)
    assert pinned_host.pinned_hostbuf_size(80) == 160
    assert pinned_host.pinned_hostbuf_size(200) == 200
    pinned_host.configure(disabled=True)
    mechanism, _ = _aimdo({"weight": torch.ones(5000)}, backend=FakeVbarBackend(oom=True))
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    assert not mechanism._pin_state  # pyright: ignore[reportPrivateUsage]


def test_host_memory_status_does_not_apply_linux_cgroup_limits() -> None:
    def sysconf(name: str) -> int:
        return {"SC_PAGE_SIZE": 1, "SC_PHYS_PAGES": 2000, "SC_AVPHYS_PAGES": 900}[name]

    assert pinned_host.host_memory_status("linux", sysconf_query=sysconf) == (2000, 900)
    assert pinned_host.memory_status(
        "linux",
        linux_cgroup_query=lambda: (700, 500),
        sysconf_query=sysconf,
    ) == (700, 500)


def test_pin_refusal_warnings_are_reasoned_bounded_and_quiet_when_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pinned_host.configure(maximum=0, storage_maximum=0)

    assert not pinned_host.ensure_pin_budget(17, available=lambda: 0)
    assert not pinned_host.ensure_pin_budget(19, available=lambda: 0)
    assert not pinned_host.ensure_pin_registerable(23)
    assert not pinned_host.ensure_pin_registerable(29)

    class Owner:
        pin_active = False

        def free_pins(self, size: int) -> int:
            del size
            return 0

        def free_registrations(self, size: int) -> int:
            del size
            return 0

    owner = Owner()
    assert not pinned_host.reserve_storage(owner, 31)
    assert not pinned_host.reserve_storage(owner, 37)

    messages = [record.getMessage() for record in caplog.records]
    assert sum("reason=ram-budget" in message for message in messages) == 1
    assert sum("reason=registration-cap" in message for message in messages) == 1
    assert sum("reason=physical-storage-cap" in message for message in messages) == 1
    assert any(
        "requested_bytes=17 available_bytes=0" in message
        and f"floor_bytes={pinned_host.AVAILABLE_RAM_FLOOR}" in message
        and "reclaimed_bytes=0" in message
        for message in messages
    )
    assert any(
        "requested_bytes=23 registered_bytes=0 maximum_bytes=0" in message for message in messages
    )
    assert any(
        "requested_bytes=31 stored_bytes=0 maximum_bytes=0" in message for message in messages
    )

    caplog.clear()
    pinned_host.configure(disabled=True)
    pinned_host._warned_pin_refusals.clear()  # pyright: ignore[reportPrivateUsage]
    assert not pinned_host.ensure_pin_budget(41, available=lambda: 0)
    assert not pinned_host.ensure_pin_registerable(43)
    disabled_owner = Owner()
    assert not pinned_host.reserve_storage(disabled_owner, 47)
    assert not caplog.records


def test_successful_pin_admission_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    pinned_host.configure(maximum=100, storage_maximum=100)

    assert pinned_host.ensure_pin_budget(
        17,
        available=lambda: pinned_host.AVAILABLE_RAM_FLOOR + 17,
    )
    assert pinned_host.ensure_pin_registerable(23)

    class Owner:
        pin_active = False

        def free_pins(self, size: int) -> int:
            del size
            return 0

        def free_registrations(self, size: int) -> int:
            del size
            return 0

    owner = Owner()
    assert pinned_host.reserve_storage(owner, 31)
    assert not caplog.records


def test_initial_pin_budget_divides_shared_limit_between_job_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_WORLD_SIZE", "2")
    monkeypatch.setattr(pinned_host, "memory_status", lambda: (1000, 500))
    monkeypatch.setattr(pinned_host, "platform_pin_ratio", lambda: 0.9)

    assert pinned_host._initial_maximum() == 450  # pyright: ignore[reportPrivateUsage]


def test_initial_storage_budget_uses_conservative_tunable_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_WORLD_SIZE", "2")
    monkeypatch.setattr(pinned_host, "memory_status", lambda: (1000, 500))
    monkeypatch.setattr(pinned_host, "platform_pin_ratio", lambda: 0.9)

    assert pinned_host._initial_storage_maximum() == 150  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("DINKSTER_PINNED_STAGING_FRACTION", "0.2")
    assert pinned_host._initial_storage_maximum() == 100  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("DINKSTER_PINNED_STAGING_FRACTION", "0.95")
    assert pinned_host._initial_storage_maximum() == 450  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("DINKSTER_PINNED_STAGING_FRACTION", "invalid")
    assert pinned_host._initial_storage_maximum() == 150  # pyright: ignore[reportPrivateUsage]


def test_storage_budget_is_independent_of_windows_registration_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_WORLD_SIZE", "2")
    monkeypatch.setenv("DINKSTER_PINNED_STAGING_FRACTION", "0.80")
    monkeypatch.setattr(pinned_host, "memory_status", lambda: (1000, 500))
    monkeypatch.setattr(pinned_host, "platform_pin_ratio", lambda: 0.40)

    assert pinned_host._initial_maximum() == 200  # pyright: ignore[reportPrivateUsage]
    assert pinned_host._initial_storage_maximum() == 400  # pyright: ignore[reportPrivateUsage]


def test_storage_fraction_cap_evicts_inactive_lru_owner() -> None:
    first, _ = _aimdo(
        {"weight": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    second, _ = _aimdo(
        {"weight": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    with first.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    pinned_host.configure(maximum=1 << 60, storage_maximum=20_000)

    with second.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)

    assert not first._pins  # pyright: ignore[reportPrivateUsage]
    assert len(second._pins) == 1  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_STORAGE == 20_000


def test_lease_open_reclaims_for_live_reserve_without_storage_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, _ = _aimdo(
        {"weight": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    incoming, _ = _aimdo({"other": torch.ones(5000)})
    with retained.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    initial_storage = pinned_host.TOTAL_PINNED_STORAGE
    monkeypatch.setattr(pinned_host, "MINIMUM_STAGING_RESERVE", 0)
    monkeypatch.setattr(
        pinned_host,
        "memory_status",
        lambda: (
            100_000,
            10_000 + initial_storage - pinned_host.TOTAL_PINNED_STORAGE,
        ),
    )

    with incoming.lease("other"):
        pass

    assert not retained._pins  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_STORAGE == 0


def test_live_reserve_checks_are_coalesced_but_storage_growth_checks_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = 0

    def memory_status() -> tuple[int, int]:
        nonlocal queries
        queries += 1
        return 1 << 60, 1 << 60

    monkeypatch.setattr(pinned_host, "memory_status", memory_status)
    mechanism, _ = _aimdo({"weight": torch.ones(1)})

    with mechanism.lease("weight"):
        pass
    with mechanism.lease("weight"):
        pass
    assert queries == 1

    assert pinned_host.reserve_storage(mechanism, 1)
    assert queries == 2
    pinned_host.account_storage(mechanism, -1)
    pinned_host.discard_owner_if_empty(mechanism)


@pytest.mark.parametrize("operation", ["lease", "prefetch"])
def test_reserve_failure_releases_device_lock_and_clears_active_owner(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    retained, _ = _aimdo(
        {"weight": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    with retained.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    incoming, _ = _aimdo(
        {"other": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        stream_count=2,
    )
    monkeypatch.setattr(pinned_host, "MINIMUM_STAGING_RESERVE", 0)
    monkeypatch.setattr(pinned_host, "memory_status", lambda: (100_000, 0))

    def fail_release(_size: int) -> int:
        raise RuntimeError("simulated release failure")

    monkeypatch.setattr(retained, "free_pins", fail_release)

    with pytest.raises(RuntimeError, match="simulated release failure"):
        if operation == "lease":
            with incoming.lease("other"):
                pytest.fail("lease entered after reserve failure")
        else:
            incoming.prefetch((("other", torch.float32),))

    acquired: list[bool] = []

    def try_acquire() -> None:
        locked = incoming._lock.acquire(blocking=False)  # pyright: ignore[reportPrivateUsage]
        acquired.append(locked)
        if locked:
            incoming._lock.release()  # pyright: ignore[reportPrivateUsage]

    thread = threading.Thread(target=try_acquire)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert acquired == [True]
    assert not incoming.pin_active


def test_storage_admission_reclaims_for_live_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, _ = _aimdo(
        {"weight": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    incoming, _ = _aimdo({"other": torch.ones(5000)})
    with retained.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    initial_storage = pinned_host.TOTAL_PINNED_STORAGE
    monkeypatch.setattr(pinned_host, "MINIMUM_STAGING_RESERVE", 0)
    monkeypatch.setattr(
        pinned_host,
        "memory_status",
        lambda: (
            100_000,
            10_000 + initial_storage - pinned_host.TOTAL_PINNED_STORAGE,
        ),
    )

    assert pinned_host.reserve_storage(incoming, 1)
    assert not retained._pins  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_STORAGE == 1
    pinned_host.account_storage(incoming, -1)
    pinned_host.discard_owner_if_empty(incoming)


def test_storage_reserve_skips_busy_owners_and_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inactive, _ = _aimdo(
        {"inactive": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    busy, _ = _aimdo(
        {"busy": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    for mechanism, key in ((inactive, "inactive"), (busy, "busy")):
        with mechanism.lease(key) as lease:
            lease.get(key, dtype=torch.float32)
    busy.pin_active = True
    monkeypatch.setattr(pinned_host, "MINIMUM_STAGING_RESERVE", 0)
    monkeypatch.setattr(pinned_host, "memory_status", lambda: (100_000, 0))

    assert not pinned_host.ensure_storage_reserve()
    assert not inactive._pins  # pyright: ignore[reportPrivateUsage]
    assert len(busy._pins) == 1  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_STORAGE == 20_000


def test_storage_debug_row_reports_exact_owner_bytes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mechanism, _ = _aimdo(
        {"weight": torch.ones(5000)},
        backend=FakeVbarBackend(oom=True),
        pin_all_sources=True,
    )
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    monkeypatch.setenv("DINKSTER_PINNED_STORAGE_DEBUG", "1")

    pinned_host.log_storage_ledger("decode")

    row = capsys.readouterr().err
    assert "phase=decode total=20000 registered=20000" in row
    assert "pid=" in row
    assert '"bytes":20000' in row
    assert '"owner":"AimdoWeights(device=cuda:0,keys=weight,count=1)"' in row


def test_cgroup_available_ram_bounds_repeated_inactive_pin_owners() -> None:
    chunk = 256 * 1024**2
    cgroup_limit = 3 * 1024**3
    process_overhead = 512 * 1024**2

    class Owner:
        pin_active = False

        def __init__(self) -> None:
            self.pinned = chunk
            self.truncated = False

        def free_pins(self, size: int) -> int:
            del size
            freed = self.pinned
            self.pinned = 0
            self.truncated = True
            pinned_host.account(-freed)
            return freed

        def free_registrations(self, size: int) -> int:
            del size
            return 0

    owners: list[Owner] = []
    for _ in range(6):
        assert pinned_host.ensure_pin_budget(
            chunk,
            available=lambda: cgroup_limit - process_overhead - pinned_host.TOTAL_PINNED_MEMORY,
        )
        owner = Owner()
        owners.append(owner)
        pinned_host.register_owner(owner)
        pinned_host.account(chunk)

    assert pinned_host.TOTAL_PINNED_MEMORY <= 2 * chunk
    assert sum(owner.truncated for owner in owners) >= 4


def test_pin_registry_visits_inactive_lru_then_active_mru() -> None:
    calls: list[str] = []

    class Owner:
        def __init__(self, name: str, active: bool) -> None:
            self.name = name
            self.pin_active = active

        def free_pins(self, size: int) -> int:
            calls.append(self.name)
            return 1

        def free_registrations(self, size: int) -> int:
            calls.append(self.name)
            return 1

    owners = [Owner("old", False), Owner("active", True), Owner("new", False)]
    for owner in owners:
        pinned_host.register_owner(owner)
    assert pinned_host.free_pins(2) == 2
    assert calls == ["old", "new"]
    calls.clear()
    assert pinned_host.free_registrations(3)
    assert calls == ["old", "new", "active"]


def test_repeated_requests_share_one_unit_fault_and_unpin() -> None:
    mechanism, backend = _aimdo({"weight": torch.ones(5000)})
    with mechanism.lease("weight") as lease:
        for _ in range(4):
            lease.get("weight", dtype=torch.float32)
        allocation = backend.allocations[0]
        assert allocation.pins == 1
    mechanism._reap_unpins(wait=False)  # pyright: ignore[reportPrivateUsage]
    assert len(backend.fault_calls) == len(backend.unpins) == 1
    assert allocation.pins == 0


def test_vbar_unpin_waits_for_consumer_stream_event() -> None:
    backend = FakeVbarBackend(complete_events_on_record=False)
    mechanism, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend)

    with mechanism.lease("weight") as lease:
        concrete = cast("aimdo_mod._AimdoWeightLease", lease)  # pyright: ignore[reportPrivateUsage]
        lease.get("weight", dtype=torch.float32)
        successful_faults = concrete._successful_faults  # pyright: ignore[reportPrivateUsage]
    allocation = backend.allocations[0]
    assert allocation.pins == 1
    assert backend.unpins == []
    stream_pending = aimdo_mod._device_unpins[mechanism._unpin_key][0]  # pyright: ignore[reportPrivateUsage]
    pending = stream_pending.entries[0]
    assert pending.allocations is successful_faults
    assert not hasattr(concrete, "__dict__")
    assert not hasattr(pending, "__dict__")
    assert [event for event in backend.events if event[0] in {"wait", "record-event"}] == [
        ("wait", "current", "transfer-0"),
        ("wait", "transfer-0", "current"),
        ("record-event", "current"),
    ]

    backend.recorded_events[0].complete = True
    mechanism._reap_unpins(wait=False)  # pyright: ignore[reportPrivateUsage]
    assert backend.unpins == [allocation]
    assert allocation.pins == 0


def test_deferred_unpins_reap_on_the_next_fault_without_polling_on_close() -> None:
    backend = FakeVbarBackend(complete_events_on_record=False)
    first, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend)
    second, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend)
    with first.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)

    queries: list[object] = []
    event_query = backend.event_query

    def record_query(event: object) -> bool:
        queries.append(event)
        return event_query(event)

    backend.event_query = record_query
    with second.lease("weight") as lease:
        for _ in range(4):
            lease.get("weight", dtype=torch.float32)
        assert len(queries) == 1
    assert len(queries) == 1

    first.unload()
    second.unload()


def test_completed_unpins_reap_across_independent_streams() -> None:
    backend = FakeVbarBackend(complete_events_on_record=False)
    first, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend)
    second, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend)

    backend.current = _FakeStream("first")
    with first.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    backend.current = _FakeStream("second")
    with first.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    first_allocation = backend.allocations[0]
    assert first_allocation.pins == 2

    backend.recorded_events[1].complete = True
    with second.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    assert backend.unpins == [first_allocation]
    assert first_allocation.pins == 1

    backend.recorded_events[0].complete = True
    second._reap_unpins(wait=False)  # pyright: ignore[reportPrivateUsage]
    assert backend.unpins == [first_allocation, first_allocation]
    assert first_allocation.pins == 0
    second.unload()


def test_pending_unpins_stop_querying_after_an_incomplete_stream_event() -> None:
    backend = FakeVbarBackend(
        complete_events_on_record=False,
        current=_FakeStream("current"),
    )
    mechanism, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend)
    for _ in range(3):
        with mechanism.lease("weight") as lease:
            lease.get("weight", dtype=torch.float32)

    queries: list[object] = []
    event_query = backend.event_query

    def record_query(event: object) -> bool:
        queries.append(event)
        return event_query(event)

    backend.event_query = record_query
    mechanism._reap_unpins(wait=False)  # pyright: ignore[reportPrivateUsage]
    assert queries == [backend.recorded_events[0]]

    backend.recorded_events[0].complete = True
    mechanism._reap_unpins(wait=False)  # pyright: ignore[reportPrivateUsage]
    assert queries == [
        backend.recorded_events[0],
        backend.recorded_events[0],
        backend.recorded_events[1],
    ]
    assert len(backend.unpins) == 1

    backend.recorded_events[1].complete = True
    backend.recorded_events[2].complete = True
    mechanism._reap_unpins(wait=False)  # pyright: ignore[reportPrivateUsage]
    assert len(backend.unpins) == 3


def test_unload_waits_for_owner_and_reaps_earlier_same_stream_events() -> None:
    backend = FakeVbarBackend(
        complete_events_on_record=False,
        current=_FakeStream("current"),
    )
    first, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend)
    second, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend)
    with first.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    with second.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)

    first_allocation, second_allocation = backend.allocations
    second.unload()

    assert first_allocation.pins == 0
    assert second_allocation.pins == 0
    assert backend.recorded_events[1].complete is True
    assert [event for event in backend.events if event == ("synchronize-event",)] == [
        ("synchronize-event",)
    ]
    first.unload()


def test_stream_rotation_and_wait_edges_follow_upstream_order() -> None:
    mechanism, backend = _aimdo({"first": torch.ones(5000), "second": torch.ones(5001)})
    with mechanism.lease("first") as lease:
        lease.get("first", dtype=torch.float32)
    with mechanism.lease("second") as lease:
        lease.get("second", dtype=torch.float32)

    waits = [event for event in backend.events if event[0] == "wait"]
    assert waits == [
        ("wait", "current", "transfer-0"),
        ("wait", "transfer-0", "current"),
        ("wait", "transfer-0", "current"),
        ("wait", "current", "transfer-1"),
        ("wait", "transfer-1", "current"),
    ]


def test_overwriting_miss_waits_on_consumer_stream_before_transfer() -> None:
    expected = torch.arange(5000, dtype=torch.float32)
    mechanism, backend = _aimdo({"weight": expected.clone()})

    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    first_context_index = backend.events.index(("context", "transfer-0"))
    # A first fill writes fresh bytes nothing can be reading yet: the
    # transfer stream takes no producer edge, preserving cold overlap.
    assert ("wait", "transfer-0", "current") not in backend.events[:first_context_index]

    with mechanism.lease("weight") as lease:
        half = lease.get("weight", dtype=torch.float16)
        assert isinstance(half, torch.Tensor)
        assert torch.equal(half, expected.to(torch.float16))
    # The float16 miss overwrites the allocation bytes the float32 view
    # still exposes, so whichever transfer stream carries it must first
    # wait on the consumer stream, which chains every prior reader and
    # writer of those bytes.
    second_context_index = max(
        index for index, event in enumerate(backend.events) if event[0] == "context"
    )
    assert second_context_index > first_context_index
    stream_name = backend.events[second_context_index][1]
    assert backend.events[second_context_index - 1] == ("wait", stream_name, "current")


def test_dtype_flip_flop_drops_stale_prefetched_view_then_refaults() -> None:
    expected = torch.arange(5000, dtype=torch.float32)
    mechanism, _backend = _aimdo({"weight": expected.clone()})
    handle = mechanism.prefetch((("weight", torch.float32),))
    assert handle is not None
    assert len(mechanism._prefetched) == 1  # pyright: ignore[reportPrivateUsage]

    with mechanism.lease("weight") as lease:
        half = lease.get("weight", dtype=torch.float16)
        assert isinstance(half, torch.Tensor)
        assert torch.equal(half, expected.to(torch.float16))
    # The float16 overwrite made the unconsumed float32 prefetched view
    # stale; it must be dropped so a later consumer re-faults.
    assert mechanism._prefetched == {}  # pyright: ignore[reportPrivateUsage]

    with mechanism.lease("weight") as lease:
        refetched = lease.get("weight", dtype=torch.float32)
        assert isinstance(refetched, torch.Tensor)
        assert torch.equal(refetched, expected)
    handle.close()


def test_private_batch_reuses_request_and_result_lists() -> None:
    mechanism, _backend = _aimdo({"weight": torch.ones(5000)})
    target = AimdoWeights._lease_get_many.__code__  # pyright: ignore[reportPrivateUsage]
    observed: list[tuple[object, object, object, object]] = []

    def capture(frame: FrameType, event: str, arg: object) -> None:
        if frame.f_code is target and event == "return":
            observed.append(
                (
                    frame.f_locals["requests"],
                    frame.f_locals["results"],
                    frame.f_locals["layout"],
                    arg,
                )
            )

    previous = sys.getprofile()
    sys.setprofile(capture)
    try:
        with mechanism.lease("weight") as lease:
            batch_lease = cast("aimdo_mod._AimdoWeightLease", lease)  # pyright: ignore[reportPrivateUsage]
            actual = batch_lease.get_many((("weight", torch.float32),))
    finally:
        sys.setprofile(previous)

    assert len(observed) == 1
    requests, scratch, layout, returned = observed[0]
    assert isinstance(requests, list)
    assert actual is scratch
    assert returned is scratch
    assert not hasattr(requests[0], "__dict__")
    assert not hasattr(layout, "__dict__")
    assert not hasattr(batch_lease, "__dict__")


def test_batch_faults_multiple_units_and_suballocates_one_arena() -> None:
    mechanism, backend = _aimdo(
        {
            "first": torch.arange(10_000, dtype=torch.float16),
            "second": torch.arange(12_000, dtype=torch.float16),
        },
        units=(
            ResidencyUnit("unit-a", ("first",)),
            ResidencyUnit("unit-b", ("second",)),
        ),
    )
    with mechanism.lease("unit-a") as lease:
        batch_lease = cast("aimdo_mod._AimdoWeightLease", lease)  # pyright: ignore[reportPrivateUsage]
        values = batch_lease.get_many((("first", torch.float32), ("second", torch.float32)))
        assert isinstance(values, list)
        first, second = values
    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert torch.equal(first, torch.arange(10_000, dtype=torch.float16).float())
    assert torch.equal(second, torch.arange(12_000, dtype=torch.float16).float())
    allocations = mechanism._allocations  # pyright: ignore[reportPrivateUsage]
    assert backend.fault_calls == [allocations["first"], allocations["second"]]
    assert [event for event in backend.events if event[0] == "arena-get"] == [
        ("arena-get", 0, 40_000, 0),
        ("arena-get", 0, 48_000, 40_960),
    ]


def test_plain_same_dtype_batch_copies_directly_into_vbar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "first": torch.arange(5000, dtype=torch.float32),
        "second": torch.arange(6000, dtype=torch.bfloat16),
    }
    mechanism, backend = _aimdo(
        {key: value.clone() for key, value in expected.items()},
        units=(
            ResidencyUnit("unit-a", ("first",)),
            ResidencyUnit("unit-b", ("second",)),
        ),
    )

    def refuse_staging(*_args: object, **_kwargs: object) -> StoredWeight:
        pytest.fail("same-dtype plain weight used device staging")

    monkeypatch.setattr(mechanism, "_computed_from_source", refuse_staging)
    with collect_partial_residency_timing() as timing:
        with mechanism.lease("unit-a") as lease:
            batch = cast("aimdo_mod._AimdoWeightLease", lease)  # pyright: ignore[reportPrivateUsage]
            first, second = batch.get_many((("first", torch.float32), ("second", torch.bfloat16)))

    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert torch.equal(first, expected["first"])
    assert torch.equal(second, expected["second"])
    assert backend.tensor_calls == backend.allocations
    assert backend.registered_ptrs
    assert not [
        event for event in backend.events if event[0] in {"arena-evict", "arena-grow", "arena-get"}
    ]
    report = timing.report()
    assert report.transfer_bytes == sum(value.nbytes for value in expected.values())
    assert report.dequant_ms == 0.0


def test_mapped_same_dtype_weight_reads_file_into_host_pin(tmp_path: Path) -> None:
    expected = torch.arange(5000, dtype=torch.float32)
    mapped = _mapped_weights(tmp_path, {"weight": expected})["weight"]
    info = tensor_file_slice(mapped)
    assert info is not None
    mechanism, backend = _aimdo({"weight": mapped})

    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)

    assert torch.equal(actual, expected)
    assert [event for event in backend.events if event[0] == "read-file"] == [
        ("read-file", info.offset, info.size, None)
    ]


def test_mapped_fp8_weight_reads_file_into_device_staging(tmp_path: Path) -> None:
    qdata = torch.ones(20_000, dtype=torch.float8_e4m3fn)
    scale = torch.tensor(0.5, dtype=torch.float32)
    mapped = _mapped_weights(tmp_path, {"qdata": qdata, "scale": scale})
    qdata_info = tensor_file_slice(mapped["qdata"])
    scale_info = tensor_file_slice(mapped["scale"])
    assert qdata_info is not None
    assert scale_info is not None
    stored = Fp8ScaledWeight(mapped["qdata"], mapped["scale"], torch.float32)
    mechanism, backend = _aimdo({"weight": stored})

    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)

    assert torch.equal(actual, torch.full((20_000,), 0.5))
    assert [event for event in backend.events if event[0] == "read-file"] == [
        ("read-file", qdata_info.offset, qdata_info.size, "transfer-0"),
        ("read-file", scale_info.offset, scale_info.size, "transfer-0"),
    ]
    assert [event for event in backend.events if event[0] == "arena-get"] == [
        ("arena-get", 0, qdata_info.size + scale_info.size, 0)
    ]


def test_mapped_fp8_host_pin_retains_full_output_staging(tmp_path: Path) -> None:
    qdata = torch.ones(20_000, dtype=torch.float8_e4m3fn)
    scale = torch.tensor(0.5, dtype=torch.float32)
    mapped = _mapped_weights(tmp_path, {"qdata": qdata, "scale": scale})
    stored = Fp8ScaledWeight(mapped["qdata"], mapped["scale"], torch.float32)
    mechanism, backend = _aimdo({"weight": stored}, pin_all_sources=True)

    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)

    assert torch.equal(actual, torch.full((20_000,), 0.5))
    assert [event for event in backend.events if event[0] == "arena-get"] == [
        ("arena-get", 0, actual.nbytes, 0)
    ]


def test_aimdo_lease_binds_the_timing_collector_on_entry_not_construction() -> None:
    """The collection window is defined by when the lease bracket is
    entered: a lease context created before the window but entered
    inside it records into the active collector, and one created
    inside the window but entered after it records nothing."""
    expected = torch.arange(5000, dtype=torch.float32)
    mechanism, _backend = _aimdo({"weight": expected.clone()})
    constructed_outside = mechanism.lease("weight")
    with collect_partial_residency_timing() as timing:
        with constructed_outside as lease:
            assert lease.timing_collector() is timing
            actual = lease.get("weight", dtype=torch.float32)
    report = timing.report()
    assert torch.equal(actual, expected)
    assert report.leased_transfers == 1
    # Transfer bytes count the stored representation that crossed
    # devices, matching the eager lease contract.
    assert report.transfer_bytes == expected.nbytes
    assert report.prefetched_transfers == 0
    assert report.transfer_ms > 0.0
    assert report.exposed_stall_ms >= 0.0

    fresh, _fresh_backend = _aimdo({"weight": expected.clone()})
    with collect_partial_residency_timing() as ended:
        constructed_inside = fresh.lease("weight")
    with constructed_inside as lease:
        assert lease.timing_collector() is None
        lease.get("weight", dtype=torch.float32)
    assert ended.report().leased_transfers == 0


def test_aimdo_prefetch_counts_prefetch_and_consumption_adds_no_transfer() -> None:
    expected = torch.arange(5000, dtype=torch.float32)
    mechanism, backend = _aimdo({"weight": expected.clone()})
    with collect_partial_residency_timing() as timing:
        handle = mechanism.prefetch((("weight", torch.float32),))
        assert handle is not None
        faults_after_prefetch = len(backend.fault_calls)
        with mechanism.lease("weight") as lease:
            assert lease.timing_collector() is timing
            actual = lease.get("weight", dtype=torch.float32)
        handle.close()
    report = timing.report()
    assert torch.equal(actual, expected)
    assert len(backend.fault_calls) == faults_after_prefetch
    assert report.prefetched_transfers == 1
    assert report.prefetch_bytes == expected.nbytes
    assert report.transfer_bytes == expected.nbytes
    # The consuming lease adopted the prefetched value; only the
    # mechanism-prefetched move is counted, never a second transfer.
    assert report.leased_transfers == 0


def test_prefetch_admission_exact_boundary_recovers_and_preserves_demand_path() -> None:
    expected = torch.arange(5000, dtype=torch.float32)
    free = [(64 << 20) - 1]
    queries: list[torch.device] = []

    def physical_free(device: torch.device) -> int:
        queries.append(device)
        return free[0]

    mechanism, backend = _aimdo(
        {"weight": expected.clone()},
        physical_free_memory=physical_free,
    )
    requests = (("weight", torch.float32), ("weight", torch.float32))
    with collect_partial_residency_timing() as timing:
        assert mechanism.prefetch(requests) is None
    report = timing.report()
    assert queries == [CUDA0]
    assert not mechanism.pin_active
    assert not mechanism._prefetched  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._cache  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._pin_state  # pyright: ignore[reportPrivateUsage]
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []
    assert report.transfer_bytes == 0
    assert report.prefetched_transfers == 0

    acquired = threading.Event()

    def take_refused_lock() -> None:
        with mechanism._lock:  # pyright: ignore[reportPrivateUsage]
            acquired.set()

    thread = threading.Thread(target=take_refused_lock)
    thread.start()
    thread.join(timeout=1)
    assert acquired.is_set()
    assert not thread.is_alive()

    with mechanism.lease("weight") as lease:
        actual_after_refusal = lease.get("weight", dtype=torch.float32)
    assert torch.equal(actual_after_refusal, expected)
    assert len(backend.fault_calls) == 1

    backend.change_signature(backend.allocations[0])
    free[0] = 64 << 20
    handle = mechanism.prefetch(requests)
    assert handle is not None
    assert queries == [CUDA0, CUDA0]
    faults_after_prefetch = len(backend.fault_calls)
    assert mechanism.prefetch(requests) is None
    assert queries == [CUDA0, CUDA0]
    with mechanism.lease("weight") as lease:
        actual_at_boundary = lease.get("weight", dtype=torch.float32)
    assert torch.equal(actual_at_boundary, expected)
    assert len(backend.fault_calls) == faults_after_prefetch
    handle.close()
    assert not mechanism.pin_active


def test_prefetch_admission_caches_stable_peak_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism, _ = _aimdo(
        {"weight": torch.ones(5000)},
        physical_free_memory=lambda _device: 1 << 30,
    )
    request = aimdo_mod._BatchRequest(  # pyright: ignore[reportPrivateUsage]
        "weight", torch.float32, "get"
    )

    expected = mechanism._prefetch_peak_bytes((request,))  # pyright: ignore[reportPrivateUsage]
    assert expected is not None
    assert mechanism._prefetch_peak_cache == {  # pyright: ignore[reportPrivateUsage]
        (request,): expected
    }

    def poisoned(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cached peak was recomputed")

    monkeypatch.setattr(mechanism, "_arena_layout", poisoned)
    assert mechanism._prefetch_peak_bytes((request,)) == expected  # pyright: ignore[reportPrivateUsage]
    mechanism.unload()
    assert not mechanism._prefetch_peak_cache  # pyright: ignore[reportPrivateUsage]


def test_prefetch_admission_waits_for_large_file_source_pin_before_caching(
    tmp_path: Path,
) -> None:
    mapped = _mapped_weights(
        tmp_path,
        {"weight": torch.ones(20_000, dtype=torch.float8_e4m3fn)},
    )["weight"]
    mechanism, _ = _aimdo(
        {"weight": mapped},
        physical_free_memory=lambda _device: 1 << 30,
    )
    request = aimdo_mod._BatchRequest(  # pyright: ignore[reportPrivateUsage]
        "weight", torch.float32, "get"
    )

    expected = mechanism._prefetch_peak_bytes((request,))  # pyright: ignore[reportPrivateUsage]
    assert expected is not None
    assert not mechanism._prefetch_peak_cache  # pyright: ignore[reportPrivateUsage]

    assert mechanism._pin_for(request, mapped) is not None  # pyright: ignore[reportPrivateUsage]
    pinned_peak = mechanism._prefetch_peak_bytes((request,))  # pyright: ignore[reportPrivateUsage]
    assert pinned_peak is not None and pinned_peak > expected
    assert mechanism._prefetch_peak_cache == {  # pyright: ignore[reportPrivateUsage]
        (request,): pinned_peak
    }


def test_resident_vbar_reuses_production_prefetch_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = 0
    clock = [0]
    free = [1 << 30]

    def physical_free(_device: torch.device) -> int:
        nonlocal queries
        queries += 1
        return free[0]

    monkeypatch.setattr(aimdo_mod, "_physical_free_bytes", physical_free)
    monkeypatch.setattr(aimdo_mod.time, "monotonic_ns", lambda: clock[0])
    mechanism, backend = _aimdo(
        {"weight": torch.ones(20_000)},
        physical_free_memory=physical_free,
    )
    requests = (("weight", torch.float32),)

    handle = mechanism.prefetch(requests)
    assert handle is not None
    handle.close()
    assert queries == 1
    backend.reported_loaded_size = mechanism._demand_reservation_bytes()  # pyright: ignore[reportPrivateUsage]
    for expected_queries in (2, 2):
        handle = mechanism.prefetch(requests)
        assert handle is not None
        handle.close()
        assert queries == expected_queries

    backend.reported_loaded_size = 0
    handle = mechanism.prefetch(requests)
    assert handle is not None
    handle.close()
    assert queries == 3
    backend.reported_loaded_size = mechanism._demand_reservation_bytes()  # pyright: ignore[reportPrivateUsage]
    handle = mechanism.prefetch(requests)
    assert handle is not None
    handle.close()
    assert queries == 4

    clock[0] = aimdo_mod._PREFETCH_ADMISSION_CACHE_NS  # pyright: ignore[reportPrivateUsage]
    free[0] = 0
    assert mechanism.prefetch(requests) is None
    assert queries == 5
    free[0] = 1 << 30
    handle = mechanism.prefetch(requests)
    assert handle is not None
    handle.close()
    assert queries == 6
    assert backend.fault_calls
    assert mechanism._prefetch_admission is not None  # pyright: ignore[reportPrivateUsage]
    mechanism.partially_unload(1)
    assert mechanism._prefetch_admission is None  # pyright: ignore[reportPrivateUsage]
    mechanism.unload()
    assert mechanism._prefetch_admission is None  # pyright: ignore[reportPrivateUsage]


def test_prefetch_admission_bounds_unknown_vbar_page_crossing() -> None:
    page_size = 32 << 20
    free = [2 * page_size - 1]
    mechanism, backend = _aimdo(
        {
            "prefix": torch.ones(128, dtype=torch.float32),
            "weight": torch.ones(page_size // 4, dtype=torch.float32),
        },
        units=(ResidencyUnit("unit", ("prefix", "weight")),),
        physical_free_memory=lambda _device: free[0],
    )
    allocation = cast(
        _FakeAllocation,
        mechanism._allocations["weight"],  # pyright: ignore[reportPrivateUsage]
    )
    assert allocation.offset == 512
    assert (allocation.offset % page_size + allocation.size + page_size - 1) // page_size == 2

    assert mechanism.prefetch((("weight", torch.float32),)) is None
    assert backend.fault_calls == []

    free[0] = 2 * page_size
    handle = mechanism.prefetch((("weight", torch.float32),))
    assert handle is not None
    assert backend.fault_calls == [
        mechanism._unit_allocations["unit"]  # pyright: ignore[reportPrivateUsage]
    ]
    handle.close()


def test_prefetch_admission_validates_before_query_and_propagates_query_error() -> None:
    queries = 0

    def physical_free(_device: torch.device) -> int:
        nonlocal queries
        queries += 1
        raise RuntimeError("free query failed")

    mechanism, backend = _aimdo(
        {"weight": torch.ones(5000)},
        physical_free_memory=physical_free,
    )
    with pytest.raises(
        ValueError,
        match=r"'weight' requires 40000 bytes; allocation is 20000 bytes",
    ):
        mechanism.prefetch((("weight", torch.float64),))
    assert queries == 0

    with pytest.raises(RuntimeError, match="free query failed") as raised:
        mechanism.prefetch((("weight", torch.float32),))
    assert str(raised.value) == "free query failed"
    assert queries == 1
    assert not mechanism.pin_active
    assert not mechanism._prefetched  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._cache  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._pin_state  # pyright: ignore[reportPrivateUsage]
    assert backend.fault_calls == []

    acquired = threading.Event()

    def take_lock() -> None:
        with mechanism._lock:  # pyright: ignore[reportPrivateUsage]
            acquired.set()

    thread = threading.Thread(target=take_lock)
    thread.start()
    thread.join(timeout=1)
    assert acquired.is_set()
    assert not thread.is_alive()


def test_prefetch_low_headroom_stops_before_downstream_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    patch_set = PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(5000))),)})
    mechanism, backend = _aimdo(
        {"weight": torch.ones(5000)},
        patch_set=patch_set,
        physical_free_memory=lambda _device: 0,
    )
    prepared = mechanism._prepared_sources["weight"]  # pyright: ignore[reportPrivateUsage]
    assert prepared._payloads is None  # pyright: ignore[reportPrivateUsage]
    assert prepared._unsupported == set()  # pyright: ignore[reportPrivateUsage]
    assert not prepared._diagnosed  # pyright: ignore[reportPrivateUsage]
    assert prepared._prepared is None  # pyright: ignore[reportPrivateUsage]

    def poisoned(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("downstream prefetch path ran")

    monkeypatch.setattr(pinned_host, "ensure_storage_reserve", poisoned)
    monkeypatch.setattr(pinned_host, "ensure_pin_registerable", poisoned)
    monkeypatch.setattr(mechanism, "_cuda_context", poisoned)
    monkeypatch.setattr(mechanism, "_fault", poisoned)
    monkeypatch.setattr(mechanism, "_pin_for", poisoned)
    monkeypatch.setattr(mechanism, "_prepare_patch_functions", poisoned)
    monkeypatch.setattr(mechanism, "_cache_store", poisoned)
    state = mechanism._stream_state  # pyright: ignore[reportPrivateUsage]
    assert state is not None
    monkeypatch.setattr(state, "rotate", poisoned)
    monkeypatch.setattr(state, "arena", poisoned)
    monkeypatch.setattr(backend, "fault", poisoned)
    monkeypatch.setattr(backend, "create_stream", poisoned)
    monkeypatch.setattr(backend, "create_cast_arena", poisoned)

    with collect_partial_residency_timing() as timing:
        assert mechanism.prefetch((("weight", torch.float32),)) is None
    report = timing.report()
    assert not mechanism.pin_active
    assert not mechanism._prefetched  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._cache  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._pin_state  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._pins  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._pin_priorities  # pyright: ignore[reportPrivateUsage]
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []
    assert backend.events == []
    assert state.counter == 0
    assert not state.started
    assert prepared._payloads is None  # pyright: ignore[reportPrivateUsage]
    assert prepared._unsupported == set()  # pyright: ignore[reportPrivateUsage]
    assert not prepared._diagnosed  # pyright: ignore[reportPrivateUsage]
    assert prepared._prepared is None  # pyright: ignore[reportPrivateUsage]
    assert not any("dinkster.patch_payload_unstaged" in record.message for record in caplog.records)
    assert report.transfer_bytes == 0
    assert report.prefetched_transfers == 0


def test_prefetch_admission_aggregates_keys_forms_arena_and_largest_transient() -> None:
    patch_payload = torch.ones(5001)
    free = [0]
    mechanism, backend = _aimdo(
        {
            "plain": torch.arange(5000, dtype=torch.float32),
            "fp8": _fp8((17_001,)),
            "patched": torch.ones(5001),
        },
        patch_set=PatchSet({"patched": (PatchEntry(DiffPatch(patch_payload)),)}),
        physical_free_memory=lambda _device: free[0],
    )
    requests = (
        ("plain", torch.float32),
        ("plain", torch.float16),
        ("plain", torch.float16),
        ("fp8", None),
        ("fp8", torch.float32),
        ("patched", torch.float32),
    )
    page_targets = 6 * (32 << 20)
    arena_bytes = 137_216
    largest_transient = 3 * 20_004 + 9 * 20_004
    expected_peak = page_targets + arena_bytes + largest_transient

    free[0] = expected_peak - 1
    assert mechanism.prefetch(requests) is None
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []

    free[0] = expected_peak
    handle = mechanism.prefetch(requests)
    assert handle is not None
    assert len(backend.fault_calls) == 3
    assert len({id(allocation) for allocation in backend.fault_calls}) == 3
    assert [event for event in backend.events if event[0] == "arena-get"] == [
        ("arena-get", 0, 10_000, 0),
        ("arena-get", 0, 17_008, 10_240),
        ("arena-get", 0, 68_004, 27_648),
        ("arena-get", 0, 20_480, 116_736),
        ("arena-get", 0, 20_004, 96_256),
    ]
    handle.close()


def test_prefetch_admission_bounds_packed_raw_patch_intermediates() -> None:
    stored = _fp8((17_001,))
    patch_set = PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(17_001))),)})
    expected = patch_stored_weight(
        stored,
        patch_set.entries("weight"),
        key="weight",
    )
    assert isinstance(expected, Fp8ScaledWeight)
    free = [0]
    mechanism, backend = _aimdo(
        {"weight": stored},
        patch_set=patch_set,
        physical_free_memory=lambda _device: free[0],
    )
    arena_bytes = 86_016
    transient_bytes = 17_005 + 17_008 + 3 * 68_004 + 9 * 68_004
    projected_peak = (64 << 20) + arena_bytes + transient_bytes

    free[0] = projected_peak - 1
    assert mechanism.prefetch((("weight", None),)) is None
    assert backend.fault_calls == []

    free[0] = projected_peak
    handle = mechanism.prefetch((("weight", None),))
    assert handle is not None
    with mechanism.lease("weight") as lease:
        actual = lease.get_stored("weight")
    handle.close()
    assert isinstance(actual, Fp8ScaledWeight)
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)


def test_prefetch_admission_bounds_all_lora_payloads_and_math_workspace() -> None:
    output_size = 1024
    rank = 10_000
    patch_set = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(
                        LoRAAdapter(
                            torch.ones((output_size, rank), dtype=torch.float16),
                            torch.ones((rank, output_size), dtype=torch.float16),
                        )
                    )
                ),
            )
        }
    )
    projected_peak = 862_126_080
    mechanism, backend = _aimdo(
        {"weight": torch.ones((output_size, output_size), dtype=torch.float32)},
        patch_set=patch_set,
        physical_free_memory=lambda _device: projected_peak - 1,
    )
    request = aimdo_mod._BatchRequest(  # pyright: ignore[reportPrivateUsage]
        "weight", torch.float32, "get"
    )
    assert mechanism._prefetch_peak_bytes((request,)) == projected_peak  # pyright: ignore[reportPrivateUsage]
    assert mechanism.prefetch((("weight", torch.float32),)) is None
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []


def test_prefetch_admission_bounds_lora_mid_shape_expansion() -> None:
    patch_set = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(
                        LoRAAdapter(
                            torch.ones((1, 32, 1, 1)),
                            torch.ones((32, 1024, 1, 1)),
                            mid=torch.ones((32, 32, 32, 32)),
                        )
                    )
                ),
            )
        }
    )
    projected_peak = 2_240_021_632
    mechanism, backend = _aimdo(
        {"weight": torch.ones((1, 1024, 32, 32))},
        patch_set=patch_set,
        physical_free_memory=lambda _device: projected_peak - 1,
    )
    request = aimdo_mod._BatchRequest(  # pyright: ignore[reportPrivateUsage]
        "weight", torch.float32, "get"
    )
    assert mechanism._patch_largest_intermediate["weight"] == 134_217_728  # pyright: ignore[reportPrivateUsage]
    assert mechanism._prefetch_peak_bytes((request,)) == projected_peak  # pyright: ignore[reportPrivateUsage]
    assert mechanism.prefetch((("weight", torch.float32),)) is None
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []


@pytest.mark.parametrize(
    "adapter",
    (
        LoHaAdapter(
            torch.ones((4096, 1)),
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            torch.ones((1, 4096)),
        ),
        GLoRAAdapter(
            torch.ones((1, 4096)),
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            torch.ones((4096, 1)),
        ),
    ),
    ids=("loha-broadcast", "old-glora-broadcast"),
)
def test_prefetch_admission_refuses_malformed_adapter_broadcast(
    adapter: LoHaAdapter | GLoRAAdapter,
) -> None:
    queries = 0

    def physical_free(_device: torch.device) -> int:
        nonlocal queries
        queries += 1
        return 1 << 20

    mechanism, backend = _aimdo(
        {"weight": torch.ones((1, 1))},
        patch_set=PatchSet({"weight": (PatchEntry(AdapterPatch(adapter)),)}),
        physical_free_memory=physical_free,
    )
    assert mechanism.prefetch((("weight", torch.float32),)) is None
    assert queries == 0
    assert not mechanism.is_loaded("weight")
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []


def test_prefetch_admission_refuses_adapter_without_payloads() -> None:
    queries = 0

    def physical_free(_device: torch.device) -> int:
        nonlocal queries
        queries += 1
        return 1 << 60

    mechanism, backend = _aimdo(
        {"weight": torch.ones((1, 1))},
        patch_set=PatchSet({"weight": (PatchEntry(AdapterPatch(LoKrAdapter())),)}),
        physical_free_memory=physical_free,
    )
    assert mechanism.prefetch((("weight", torch.float32),)) is None
    assert queries == 0
    assert not mechanism.is_loaded("weight")
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []


@pytest.mark.parametrize(
    ("weight", "patch"),
    (
        (
            torch.ones((1, 4096)),
            ModelAsLoraPatch(torch.ones((4096, 1))),
        ),
        (
            torch.ones((2, 1)),
            AdapterPatch(
                BOFTAdapter(
                    torch.ones((1, 1, 2, 2)),
                    rescale=torch.ones((4096, 1, 1)),
                )
            ),
        ),
    ),
    ids=("model-as-lora-broadcast", "boft-rescale-broadcast"),
)
def test_prefetch_admission_refuses_open_shape_patch_math(
    weight: torch.Tensor,
    patch: ModelAsLoraPatch[torch.Tensor] | AdapterPatch[torch.Tensor],
) -> None:
    queries = 0

    def physical_free(_device: torch.device) -> int:
        nonlocal queries
        queries += 1
        return 1 << 60

    mechanism, backend = _aimdo(
        {"weight": weight},
        patch_set=PatchSet({"weight": (PatchEntry(patch),)}),
        physical_free_memory=physical_free,
    )
    assert mechanism.prefetch((("weight", torch.float32),)) is None
    assert queries == 0
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []


@pytest.mark.parametrize(
    "adapter",
    (
        OFTAdapter(torch.ones((1, 1, 4096))),
        BOFTAdapter(torch.ones((1, 1, 1, 4096))),
        LoRAAdapter(
            torch.ones((1, 1)),
            torch.ones((1, 4096)),
            dora_scale=torch.ones((4096, 1)),
        ),
    ),
    ids=("oft-block-broadcast", "boft-block-broadcast", "dora-broadcast"),
)
def test_prefetch_admission_refuses_unbounded_builtin_broadcast(
    adapter: OFTAdapter | BOFTAdapter | LoRAAdapter,
) -> None:
    queries = 0

    def physical_free(_device: torch.device) -> int:
        nonlocal queries
        queries += 1
        return 1 << 20

    shape = (1, 4096) if isinstance(adapter, LoRAAdapter) else (1, 1)
    mechanism, backend = _aimdo(
        {"weight": torch.ones(shape)},
        patch_set=PatchSet({"weight": (PatchEntry(AdapterPatch(adapter)),)}),
        physical_free_memory=physical_free,
    )
    assert mechanism.prefetch((("weight", torch.float32),)) is None
    assert queries == 0
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []


def test_prefetch_admission_refuses_unbounded_adapter_without_diagnostic_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class LegacyAdapter:
        def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
            return base

        def calculate(
            self,
            weight: torch.Tensor,
            *,
            strength: float,
            function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        ) -> torch.Tensor:
            del strength, function
            return weight

    queries = 0

    def physical_free(_device: torch.device) -> int:
        nonlocal queries
        queries += 1
        return 1 << 60

    mechanism, backend = _aimdo(
        {"weight": torch.ones(5000)},
        patch_set=PatchSet({"weight": (PatchEntry(AdapterPatch(LegacyAdapter())),)}),
        physical_free_memory=physical_free,
    )
    prepared = mechanism._prepared_sources["weight"]  # pyright: ignore[reportPrivateUsage]

    assert mechanism.prefetch((("weight", torch.float32),)) is None
    assert queries == 0
    assert prepared._payloads is None  # pyright: ignore[reportPrivateUsage]
    assert prepared._unsupported == set()  # pyright: ignore[reportPrivateUsage]
    assert not prepared._diagnosed  # pyright: ignore[reportPrivateUsage]
    assert prepared._prepared is None  # pyright: ignore[reportPrivateUsage]
    assert not any("dinkster.patch_payload_unstaged" in record.message for record in caplog.records)
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []


def test_prefetch_admission_refuses_unbounded_eager_sibling_and_lease_runs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions = 0

    class UnknownAdapter:
        def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
            return base

        def payload_tensors(self) -> tuple[torch.Tensor, ...]:
            raise AssertionError("unbounded payload discovery ran")

        def rebuild_payloads(self, replacements: Sequence[torch.Tensor]) -> UnknownAdapter:
            raise AssertionError(f"unbounded payload rebuild ran for {len(replacements)} values")

        def calculate(
            self,
            weight: torch.Tensor,
            *,
            strength: float,
            function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        ) -> torch.Tensor:
            nonlocal executions
            assert function is None
            executions += 1
            return weight.add_(strength)

    queries = 0

    def physical_free(_device: torch.device) -> int:
        nonlocal queries
        queries += 1
        return 1 << 60

    unit = ResidencyUnit("unit", ("trigger", "custom"))
    mechanism, backend = _aimdo(
        {"trigger": torch.zeros(4), "custom": torch.ones(4)},
        units=(unit,),
        patch_set=PatchSet({"custom": (PatchEntry(AdapterPatch(UnknownAdapter())),)}),
        physical_free_memory=physical_free,
    )
    prepared = mechanism._prepared_sources["custom"]  # pyright: ignore[reportPrivateUsage]

    def poisoned(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("downstream prefetch path ran")

    with monkeypatch.context() as blocked:
        blocked.setattr(pinned_host, "ensure_storage_reserve", poisoned)
        blocked.setattr(pinned_host, "ensure_pin_registerable", poisoned)
        blocked.setattr(mechanism, "_cuda_context", poisoned)
        blocked.setattr(mechanism, "_fault", poisoned)
        blocked.setattr(mechanism, "_pin_for", poisoned)
        blocked.setattr(mechanism, "_prepare_patch_functions", poisoned)
        blocked.setattr(mechanism, "_cache_store", poisoned)
        state = mechanism._stream_state  # pyright: ignore[reportPrivateUsage]
        assert state is not None
        blocked.setattr(state, "rotate", poisoned)
        blocked.setattr(state, "arena", poisoned)
        blocked.setattr(backend, "fault", poisoned)
        blocked.setattr(backend, "create_stream", poisoned)
        blocked.setattr(backend, "create_cast_arena", poisoned)

        assert mechanism.prefetch((("trigger", torch.float32),)) is None

    assert queries == 0
    assert executions == 0
    assert not mechanism.pin_active
    assert not mechanism.is_loaded("unit")
    assert prepared._payloads is None  # pyright: ignore[reportPrivateUsage]
    assert prepared._unsupported == set()  # pyright: ignore[reportPrivateUsage]
    assert not prepared._diagnosed  # pyright: ignore[reportPrivateUsage]
    assert prepared._prepared is None  # pyright: ignore[reportPrivateUsage]
    assert backend.fault_calls == []
    assert backend.streams == []
    assert backend.arenas == []

    with mechanism.lease("trigger") as lease:
        trigger = lease.get("trigger", dtype=torch.float32)
    assert torch.equal(trigger, torch.zeros(4))
    assert executions == 1
    assert mechanism.is_loaded("unit")
    with mechanism.lease("custom") as lease:
        custom = lease.get("custom", dtype=torch.float32)
    assert torch.equal(custom, torch.full((4,), 2.0))


def test_prefetch_admission_counts_unloaded_eager_target_without_vbar_pages() -> None:
    free = [15]
    mechanism, backend = _aimdo(
        {"weight": torch.arange(4, dtype=torch.float32)},
        physical_free_memory=lambda _device: free[0],
    )
    request = (("weight", torch.float32),)
    assert mechanism.prefetch(request) is None
    assert not mechanism.is_loaded("weight")
    assert backend.fault_calls == []

    free[0] = 16
    handle = mechanism.prefetch(request)
    assert handle is not None
    assert mechanism.is_loaded("weight")
    handle.close()

    free[0] = 0
    loaded_handle = mechanism.prefetch(request)
    assert loaded_handle is not None
    loaded_handle.close()
    assert backend.fault_calls == []


def test_prefetch_admission_bounds_eager_patch_intermediate_dtype() -> None:
    free = [183]
    mechanism, backend = _aimdo(
        {"weight": torch.ones(4, dtype=torch.float16)},
        patch_set=PatchSet(
            {"weight": (PatchEntry(DiffPatch(torch.ones(4, dtype=torch.float16))),)}
        ),
        physical_free_memory=lambda _device: free[0],
    )
    request = (("weight", torch.float16),)
    assert mechanism.prefetch(request) is None
    assert not mechanism.is_loaded("weight")
    assert backend.streams == []

    free[0] = 184
    handle = mechanism.prefetch(request)
    assert handle is not None
    assert mechanism.is_loaded("weight")
    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float16)
    assert torch.equal(actual, torch.full((4,), 2.0, dtype=torch.float16))
    handle.close()


def test_prefetch_without_physical_query_preserves_existing_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism, backend = _aimdo({"weight": torch.arange(5000, dtype=torch.float32)})

    def poisoned(_requests: Sequence[aimdo_mod._BatchRequest]) -> int:  # pyright: ignore[reportPrivateUsage]
        raise AssertionError("admission estimator ran without a physical-free query")

    monkeypatch.setattr(mechanism, "_prefetch_peak_bytes", poisoned)
    handle = mechanism.prefetch((("weight", torch.float32),))
    assert handle is not None
    assert len(backend.fault_calls) == 1
    handle.close()


def test_prefetch_high_headroom_preserves_events_and_values() -> None:
    expected = torch.arange(5000, dtype=torch.float32)

    def run(
        physical_free_memory: Callable[[torch.device], int] | None,
    ) -> tuple[torch.Tensor, list[tuple[object, ...]]]:
        mechanism, backend = _aimdo(
            {"weight": expected.clone()},
            physical_free_memory=physical_free_memory,
        )
        handle = mechanism.prefetch((("weight", torch.float16),))
        assert handle is not None
        with mechanism.lease("weight") as lease:
            actual = lease.get("weight", dtype=torch.float16).clone()
        handle.close()
        return actual, list(backend.events)

    baseline, baseline_events = run(None)
    admitted, admitted_events = run(lambda _device: 1 << 60)
    assert torch.equal(admitted, baseline)
    assert torch.equal(admitted, expected.to(torch.float16))
    assert admitted_events == baseline_events


def test_lease_that_loads_a_hybrid_eager_unit_counts_the_whole_unit_transfer() -> None:
    """A lease whose request triggers a hybrid eager-unit load reports
    every key the load moved, not just the requested one, and a
    prefetch that triggers the load attributes the same movement to
    the prefetch counters."""
    weight = torch.arange(1024, dtype=torch.float32)
    bias = torch.arange(256, dtype=torch.float32)
    mechanism, _backend = _aimdo(
        {"weight": weight.clone(), "bias": bias.clone()},
        units=(ResidencyUnit("unit", ("weight", "bias")),),
    )
    assert not mechanism.is_loaded("unit")
    with collect_partial_residency_timing() as timing:
        with mechanism.lease("weight") as lease:
            assert lease.timing_collector() is timing
            actual = lease.get("weight", dtype=torch.float32)
    report = timing.report()
    assert torch.equal(actual, weight)
    assert mechanism.is_loaded("unit")
    assert report.leased_transfers == 2
    assert report.transfer_bytes == weight.nbytes + bias.nbytes
    assert report.prefetched_transfers == 0

    fresh, _fresh_backend = _aimdo(
        {"weight": weight.clone(), "bias": bias.clone()},
        units=(ResidencyUnit("unit", ("weight", "bias")),),
    )
    with collect_partial_residency_timing() as prefetch_timing:
        handle = fresh.prefetch((("bias", torch.float32),))
        assert handle is not None
        handle.close()
    prefetch_report = prefetch_timing.report()
    assert fresh.is_loaded("unit")
    assert prefetch_report.prefetched_transfers == 2
    assert prefetch_report.prefetch_bytes == weight.nbytes + bias.nbytes
    assert prefetch_report.leased_transfers == 0

    # An already-loaded unit moves nothing, so a later collected lease
    # records no transfer for it.
    with collect_partial_residency_timing() as loaded_timing:
        with mechanism.lease("weight") as lease:
            lease.get("weight", dtype=torch.float32)
    assert loaded_timing.report().leased_transfers == 0


def test_oversized_batch_request_fails_before_direct_or_prefetch_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = Int8Linear(
        256,
        128,
        bias=True,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
    )
    layer.load_state_dict(
        {
            "weight": torch.ones((128, 256), dtype=torch.int8),
            "weight_scale": torch.ones((128, 1), dtype=torch.float32),
            "bias": torch.ones(128, dtype=torch.bfloat16),
        },
        strict=True,
        assign=True,
    )
    mechanism, backend = _enroll_int8(layer)
    expected = r"'weight' requires 131072 bytes; allocation is 65536 bytes"

    with mechanism.lease("weight") as lease:
        batch = cast("aimdo_mod._AimdoWeightLease", lease)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(ValueError, match=expected):
            batch.get_many((("weight", torch.float32),))
    assert backend.fault_calls == []
    assert backend.tensor_calls == []
    assert backend.events == []
    assert not mechanism._cache  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._pin_state  # pyright: ignore[reportPrivateUsage]

    registration_checks: list[int] = []

    def record_registration_check(size: int) -> bool:
        registration_checks.append(size)
        return True

    monkeypatch.setattr(
        pinned_host,
        "ensure_pin_registerable",
        record_registration_check,
    )
    with pytest.raises(ValueError, match=expected):
        mechanism.prefetch((("weight", torch.float32),))
    assert backend.fault_calls == []
    assert backend.tensor_calls == []
    assert backend.events == []
    assert not mechanism._cache  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._prefetched  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._pin_state  # pyright: ignore[reportPrivateUsage]
    assert registration_checks == []


def test_int8_two_forward_cache_survives_exact_request_and_reloads_after_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = Int8Linear(
        5000,
        4,
        bias=True,
        compute_dtype=torch.float32,
        convrot=False,
        convrot_groupsize=0,
        full_precision_matmul=True,
    )
    layer.load_state_dict(
        {
            "weight": torch.ones((4, 5000), dtype=torch.int8),
            "weight_scale": torch.tensor(0.25),
            "bias": torch.arange(4, dtype=torch.float32),
        },
        strict=True,
        assign=True,
    )
    mechanism, backend = _enroll_int8(layer)
    copy_calls: list[int] = []
    original_copy = mechanism._copy_value  # pyright: ignore[reportPrivateUsage]

    def record_copy(
        target: StoredWeight,
        source: StoredWeight,
        *,
        non_blocking: bool,
    ) -> None:
        copy_calls.append(1)
        original_copy(target, source, non_blocking=non_blocking)

    monkeypatch.setattr(mechanism, "_copy_value", record_copy)
    input = torch.ones((1, 5000), dtype=torch.float32)

    first = layer(input)
    first_copies = len(copy_calls)
    first_tensor_calls = len(backend.tensor_calls)
    second = layer(input)
    assert torch.equal(second, first)
    assert len(copy_calls) == first_copies
    assert len(backend.tensor_calls) == first_tensor_calls
    assert mechanism._geometry["weight"].allocation_bytes == 80_000  # pyright: ignore[reportPrivateUsage]

    loaded = sum(allocation.size for allocation in backend.allocations)
    assert mechanism.partially_unload(loaded) == loaded
    third = layer(input)
    assert torch.equal(third, first)
    assert len(copy_calls) > first_copies
    assert len(backend.tensor_calls) > first_tensor_calls


def test_batch_arena_aligns_mixed_typed_and_raw_fp8_requests() -> None:
    mechanism, backend = _aimdo(
        {
            "bf16": torch.arange(9001, dtype=torch.bfloat16),
            "float": torch.arange(5001, dtype=torch.float32),
            "fp8": _fp8((17_001,)),
        }
    )
    with mechanism.lease("bf16") as lease:
        batch_lease = cast("aimdo_mod._AimdoWeightLease", lease)  # pyright: ignore[reportPrivateUsage]
        values = batch_lease.get_many(
            (
                ("bf16", torch.float32),
                ("float", torch.bfloat16),
                ("fp8", None),
            )
        )
    assert all(value is not None for value in values)
    gets = [event for event in backend.events if event[0] == "arena-get"]
    assert gets == [
        ("arena-get", 0, 36_004, 0),
        ("arena-get", 0, 10_002, 36_864),
        ("arena-get", 0, 17_008, 47_104),
    ]
    assert all(cast(int, event[3]) % 1024 == 0 for event in gets)


def test_cached_plus_oom_batch_waits_for_streamed_fallback() -> None:
    mechanism, backend = _aimdo({"cached": torch.ones(5000), "oom": torch.ones(5001)})
    with mechanism.lease("cached") as lease:
        lease.get("cached", dtype=torch.float32)
    allocations = mechanism._allocations  # pyright: ignore[reportPrivateUsage]
    backend.oom_offsets.add(cast(_FakeAllocation, allocations["oom"]).offset)
    backend.events.clear()
    with mechanism.lease("cached") as lease:
        batch_lease = cast("aimdo_mod._AimdoWeightLease", lease)  # pyright: ignore[reportPrivateUsage]
        cached, fallback = batch_lease.get_many((("cached", torch.float32), ("oom", torch.float32)))
    assert isinstance(cached, torch.Tensor)
    assert isinstance(fallback, torch.Tensor)
    assert torch.equal(fallback, torch.ones(5001))
    assert [event for event in backend.events if event[0] == "wait"] == [
        ("wait", "transfer-0", "current"),
        ("wait", "current", "transfer-1"),
        ("wait", "transfer-1", "current"),
    ]


def test_mixed_streamed_oom_raw_fp8_pin_keeps_non_blocking_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism, backend = _aimdo({"cached": torch.ones(5000), "oom": _fp8((17_000,))})
    with mechanism.lease("cached") as lease:
        lease.get("cached", dtype=torch.float32)
    backend.oom_offsets.add(
        cast(_FakeAllocation, mechanism._allocations["oom"]).offset  # pyright: ignore[reportPrivateUsage]
    )
    calls: list[tuple[str, bool]] = []
    original = mechanism._computed_from_source  # pyright: ignore[reportPrivateUsage]

    def record(
        request: aimdo_mod._BatchRequest,  # pyright: ignore[reportPrivateUsage]
        source: StoredWeight,
        *,
        non_blocking: bool = False,
        collector: PartialResidencyTiming | None = None,
    ) -> StoredWeight:
        calls.append((request.key, non_blocking))
        return original(request, source, non_blocking=non_blocking, collector=collector)

    monkeypatch.setattr(mechanism, "_computed_from_source", record)
    with mechanism.lease("cached") as lease:
        batch = cast("aimdo_mod._AimdoWeightLease", lease)  # pyright: ignore[reportPrivateUsage]
        batch.get_many((("cached", torch.float32), ("oom", None)))
    assert ("oom", True) in calls


def test_cast_arena_evicts_before_growth() -> None:
    mechanism, backend = _aimdo({"weight": torch.ones(5000)})
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float16)
    relevant = [
        event for event in backend.events if event[0] in {"arena-evict", "arena-grow", "arena-get"}
    ]
    assert relevant == [
        ("arena-evict", 0, 10_000),
        ("arena-grow", 0, 10_000),
        ("arena-get", 0, 10_000, 0),
    ]


def test_stream_count_zero_preserves_synchronous_materialization() -> None:
    expected = torch.arange(5000, dtype=torch.float32)
    mechanism, backend = _aimdo({"weight": expected}, stream_count=0)
    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)
    assert torch.equal(actual, expected)
    assert backend.tensor_calls == backend.allocations
    assert backend.streams == []
    assert backend.arenas == []
    assert backend.events == [("record-event", "current")]


def test_compiling_guard_returns_to_synchronous_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aimdo_mod, "_is_compiling", lambda: True)
    expected = torch.arange(5000, dtype=torch.float32)
    mechanism, backend = _aimdo({"weight": expected})
    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)
    assert torch.equal(actual, expected)
    assert backend.streams == []
    assert backend.arenas == []
    assert backend.events == [("record-event", "current")]


def test_prefetch_target_reuses_grouped_request_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block, mechanism, _backend = _prefetch_block()
    target_code = prefetch_mod._target_from_routes.__code__  # pyright: ignore[reportPrivateUsage]
    prefetch_code = AimdoWeights.prefetch.__code__
    pop_code = prefetch_mod.PrefetchQueue.pop.__code__
    observed: list[tuple[object, object]] = []
    observed_prefetch: list[tuple[object, object]] = []
    observed_prepared: list[tuple[object, object]] = []
    received: list[Sequence[tuple[str, torch.dtype | None]]] = []

    original_prefetch = mechanism.prefetch

    def record_prefetch(requests: Sequence[tuple[str, torch.dtype | None]]) -> object | None:
        received.append(requests)
        return original_prefetch(requests)

    monkeypatch.setattr(mechanism, "prefetch", record_prefetch)

    def capture(frame: FrameType, event: str, arg: object) -> None:
        if frame.f_code is target_code and event == "return":
            observed.append((frame.f_locals["grouped"], arg))
        elif frame.f_code is prefetch_code and event == "return" and arg is not None:
            observed_prefetch.append((frame.f_locals["batch"], arg))
        elif frame.f_code is pop_code and event == "return":
            prepared = frame.f_locals["self"]._entries[0]
            observed_prepared.append((frame.f_locals["handles"], prepared))

    previous = sys.getprofile()
    sys.setprofile(capture)
    try:
        target = prefetch_mod._target(block)  # pyright: ignore[reportPrivateUsage]
        queue = prefetch_mod.PrefetchQueue((target,))
        queue.pop(block)
    finally:
        sys.setprofile(previous)

    assert len(observed) == 1
    grouped, returned = observed[0]
    assert target is returned
    assert target.requests is grouped
    assert not hasattr(target, "__dict__")
    request_lists = list(target.requests.values())
    assert len(request_lists) == 1
    assert isinstance(request_lists[0], list)
    assert len(observed_prefetch) == 1
    batch, handle = observed_prefetch[0]
    concrete_handle = cast("aimdo_mod._AimdoPrefetch", handle)  # pyright: ignore[reportPrivateUsage]
    assert concrete_handle._requests is batch  # pyright: ignore[reportPrivateUsage]
    assert not hasattr(concrete_handle, "__dict__")
    assert not hasattr(concrete_handle._lease, "__dict__")  # pyright: ignore[reportPrivateUsage]
    assert len(observed_prepared) == 1
    handles, prepared = observed_prepared[0]
    concrete_prepared = cast("prefetch_mod._Prepared", prepared)  # pyright: ignore[reportPrivateUsage]
    assert concrete_prepared.handles is handles
    assert not hasattr(concrete_prepared, "__dict__")
    try:
        assert len(received) == 1
        assert received[0] is request_lists[0]
    finally:
        queue.close()


def test_prefetch_queue_batches_multi_unit_block_and_consumes_without_refault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block, mechanism, backend = _prefetch_block()
    cleanup_events: list[str] = []
    original_unpin = backend.unpin

    def record_unpin(allocation: object) -> None:
        cleanup_events.append("unpin")
        original_unpin(allocation)

    monkeypatch.setattr(backend, "unpin", record_unpin)
    queue = make_prefetch_queue((block,))
    assert queue is not None
    prefetch_queue_pop(queue, block)
    unit_allocations = mechanism._unit_allocations  # pyright: ignore[reportPrivateUsage]
    assert backend.fault_calls == [unit_allocations["0"], unit_allocations["1"]]
    fault_count = len(backend.fault_calls)
    block(torch.arange(200, dtype=torch.float32).reshape(2, 100))
    assert len(backend.fault_calls) == fault_count
    assert mechanism._prefetched  # pyright: ignore[reportPrivateUsage]

    events_before_cleanup = len(backend.events)
    prefetch_queue_pop(queue, None)
    cleanup_waits = [
        event for event in backend.events[events_before_cleanup:] if event[0] == "wait"
    ]
    assert cleanup_waits == [("wait", "transfer-0", "current")]
    assert cleanup_events == ["unpin"] * 2
    assert not mechanism._prefetched  # pyright: ignore[reportPrivateUsage]
    assert all(allocation.pins == 0 for allocation in backend.allocations)


def test_prefetch_cleanup_does_not_build_consumer_event_backlog() -> None:
    backend = FakeVbarBackend(complete_events_on_record=False)
    block, mechanism, _ = _prefetch_block(backend=backend)
    queue = make_prefetch_queue((block,))
    assert queue is not None

    prefetch_queue_pop(queue, block)
    block(torch.arange(200, dtype=torch.float32).reshape(2, 100))
    prefetch_queue_pop(queue, None)

    assert backend.recorded_events == []
    assert mechanism._unpin_key not in aimdo_mod._device_unpins  # pyright: ignore[reportPrivateUsage]
    assert len(backend.unpins) == len(mechanism._unit_allocations)  # pyright: ignore[reportPrivateUsage]
    assert all(allocation.pins == 0 for allocation in backend.allocations)


def test_prefetch_skips_fully_resident_vbar_and_reenables_after_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block, mechanism, backend = _prefetch_block()
    queue = make_prefetch_queue((block,))
    assert queue is not None
    prefetch_queue_pop(queue, block)
    prefetch_queue_pop(queue, None)

    route_calls = 0
    layer_type = type(block[0])
    residency_prefetch = cast(
        Callable[
            [object],
            tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None,
        ],
        layer_type.residency_prefetch,  # type: ignore[attr-defined]
    )

    def record_route(
        self: object,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        nonlocal route_calls
        route_calls += 1
        return residency_prefetch(self)

    monkeypatch.setattr(layer_type, "residency_prefetch", record_route)
    plan = PrefetchPlan((block,))
    backend.reported_loaded_size = (
        mechanism._demand_reservation_bytes()  # pyright: ignore[reportPrivateUsage]
    )
    assert not mechanism.prefetch_enabled()
    assert make_prefetch_queue((block,)) is None
    assert plan.make_queue() is None
    assert route_calls == 2
    assert plan.make_queue() is None
    assert route_calls == 2

    backend.reported_loaded_size = None
    assert mechanism.partially_unload(1) > 0
    assert mechanism.prefetch_enabled()
    queue = plan.make_queue()
    assert queue is not None
    assert route_calls == 4
    queue.close()


def test_prefetch_queue_reports_refused_batch() -> None:
    block, mechanism, backend = _prefetch_block(physical_free_memory=lambda _device: 0)
    queue = make_prefetch_queue((block,))
    assert queue is not None
    assert not prefetch_queue_pop(queue, block)
    assert backend.fault_calls == []
    close_prefetch_queue(queue)
    mechanism.unload()


def test_prefetch_cleanup_on_exception_clears_vbar_and_prepared_state() -> None:
    patch_set = PatchSet({"0.weight": (PatchEntry(DiffPatch(torch.ones(50, 100))),)})
    block, mechanism, backend = _prefetch_block(
        patch_set=patch_set,
        pin_all_sources=True,
    )
    queue = make_prefetch_queue((block,))
    assert queue is not None
    prefetch_queue_pop(queue, block)
    assert mechanism._prefetched  # pyright: ignore[reportPrivateUsage]
    prepared = mechanism._prepared_sources["0.weight"]  # pyright: ignore[reportPrivateUsage]
    assert prepared._prepared is None  # pyright: ignore[reportPrivateUsage]

    cleanup_prefetch_queues()
    assert not mechanism._prefetched  # pyright: ignore[reportPrivateUsage]
    assert prepared._prepared is None  # pyright: ignore[reportPrivateUsage]
    assert all(allocation.pins == 0 for allocation in backend.allocations)
    mechanism.unload()
    assert pinned_host.TOTAL_PINNED_MEMORY == 0


def test_prefetch_uses_m4_pins_m4b_staging_and_exact_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_set = PatchSet({"0.weight": (PatchEntry(DiffPatch(torch.ones(50, 100))),)})
    block, mechanism, _ = _prefetch_block(
        patch_set=patch_set,
        pin_all_sources=True,
    )
    pressure: list[int] = []
    original = pinned_host.ensure_pin_registerable

    def record(size: int) -> bool:
        pressure.append(size)
        return original(size)

    monkeypatch.setattr(pinned_host, "ensure_pin_registerable", record)
    queue = make_prefetch_queue((block,))
    assert queue is not None
    prefetch_queue_pop(queue, block)

    aligned_stored = sum(
        (mechanism._weights[key].nbytes + 1023) & ~1023  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        for key in ("0.weight", "0.bias", "1.weight", "1.bias")
    )
    prepared = mechanism._prepared_sources["0.weight"]  # pyright: ignore[reportPrivateUsage]
    assert pressure[0] == aligned_stored + prepared.memory_required()
    assert {identity[0] for identity in mechanism._pins} == {  # pyright: ignore[reportPrivateUsage]
        "weights",
        "patches",
    }
    assert prepared._prepared is None  # pyright: ignore[reportPrivateUsage]
    assert pinned_host.TOTAL_PINNED_MEMORY == sum(
        pin.tensor.nbytes
        for pin in mechanism._pins.values()  # pyright: ignore[reportPrivateUsage]
    )
    prefetch_queue_pop(queue, None)
    mechanism.unload()


def test_prefetch_gating_matrix_is_structural_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unbound = torch.nn.Sequential(INITLESS.linear(4, 4))
    assert make_prefetch_queue((unbound,)) is None

    resident = torch.nn.Sequential(INITLESS.linear(100, 50))
    resident.load_state_dict(
        {"0.weight": torch.ones(50, 100), "0.bias": torch.ones(50)},
        assign=True,
    )
    enroll_component(resident, load_device=CPU, offload_device=CPU)
    # Eager module residency serves the queue; the structural no-op
    # cases are unbound modules, streamless aimdo backends, and
    # compiled regions.
    eager_queue = make_prefetch_queue((resident,))
    assert eager_queue is not None
    eager_queue.close()

    sync_block, _, sync_backend = _prefetch_block(stream_count=0)
    assert make_prefetch_queue((sync_block,)) is None
    assert sync_backend.fault_calls == []

    compiled_block, _, compiled_backend = _prefetch_block()
    monkeypatch.setattr(aimdo_mod, "_is_compiling", lambda: True)
    assert make_prefetch_queue((compiled_block,)) is None
    assert compiled_backend.fault_calls == []


def test_prefetch_plan_evaluates_routes_after_construction() -> None:
    class Handle:
        def close(self) -> None:
            pass

    class Mechanism:
        calls = 0

        def prefetch_enabled(self) -> bool:
            return True

        def prefetch(self, requests: Sequence[tuple[str, torch.dtype | None]]) -> Handle:
            assert requests == [("weight", torch.float32)]
            self.calls += 1
            return Handle()

    class Routed(torch.nn.Module):
        active = False

        def __init__(self) -> None:
            super().__init__()
            self.mechanism = Mechanism()

        def residency_prefetch(
            self,
        ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
            if not self.active:
                return None
            return self.mechanism, (("weight", torch.float32),)

    block = Routed()
    plan = PrefetchPlan((block,))
    assert plan.make_queue() is None

    block.active = True
    queue = plan.make_queue()
    assert queue is not None
    prefetch_queue_pop(queue, block)
    assert block.mechanism.calls == 1
    prefetch_queue_pop(queue, None)


def test_prefetch_serializes_whole_block_lifetime_between_threads() -> None:
    block, mechanism, _ = _prefetch_block()
    first_ready = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_done = threading.Event()
    errors: list[BaseException] = []

    def first() -> None:
        try:
            queue = make_prefetch_queue((block,))
            assert queue is not None
            prefetch_queue_pop(queue, block)
            first_ready.set()
            release_first.wait()
            prefetch_queue_pop(queue, None)
        except BaseException as error:
            errors.append(error)

    def second() -> None:
        try:
            first_ready.wait()
            queue = make_prefetch_queue((block,))
            assert queue is not None
            second_attempted.set()
            prefetch_queue_pop(queue, block)
            second_done.set()
            prefetch_queue_pop(queue, None)
        except BaseException as error:
            errors.append(error)

    threads = (threading.Thread(target=first), threading.Thread(target=second))
    for thread in threads:
        thread.start()
    assert second_attempted.wait(timeout=1)
    time.sleep(0.05)
    assert not second_done.is_set()
    release_first.set()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert second_done.is_set()
    assert not errors
    assert not mechanism._prefetched  # pyright: ignore[reportPrivateUsage]


def test_prefetch_partial_multi_mechanism_failure_closes_prior_handles() -> None:
    closed: list[str] = []

    class Handle:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    class Mechanism:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def prefetch_enabled(self) -> bool:
            return True

        def prefetch(self, requests: Sequence[tuple[str, torch.dtype | None]]) -> Handle:
            assert requests
            if self.fail:
                raise RuntimeError(self.name)
            return Handle(self.name)

    class Routed(torch.nn.Module):
        def __init__(self, mechanism: Mechanism) -> None:
            super().__init__()
            self.mechanism = mechanism

        def residency_prefetch(
            self,
        ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]]:
            return self.mechanism, ((self.mechanism.name, torch.float32),)

    block = torch.nn.Sequential(Routed(Mechanism("first")), Routed(Mechanism("second", fail=True)))
    queue = make_prefetch_queue((block,))
    assert queue is not None
    with pytest.raises(RuntimeError, match="second"):
        prefetch_queue_pop(queue, block)
    assert closed == ["first"]
    queue.close()


def test_prefetch_cleanup_does_not_mask_model_error() -> None:
    class Handle:
        def close(self) -> None:
            raise RuntimeError("cleanup failed")

    class Mechanism:
        def prefetch_enabled(self) -> bool:
            return True

        def prefetch(self, requests: Sequence[tuple[str, torch.dtype | None]]) -> Handle:
            return Handle()

    class Routed(torch.nn.Module):
        def residency_prefetch(
            self,
        ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]]:
            return Mechanism(), (("weight", torch.float32),)

    block = Routed()
    queue = make_prefetch_queue((block,))
    assert queue is not None
    prefetch_queue_pop(queue, block)
    with pytest.raises(ValueError, match="model failed") as raised:
        try:
            raise ValueError("model failed")
        finally:
            close_prefetch_queue(queue)
    assert any("cleanup failed" in note for note in raised.value.__notes__)


def test_fp8_prefetch_requests_match_dequant_and_raw_consumers() -> None:
    layer = Fp8Linear(128, 128, compute_dtype=torch.float32)
    layer.load_state_dict(
        {
            "weight": torch.ones(128, 128).to(torch.float8_e4m3fn),
            "weight_scale": torch.tensor(0.5),
            "input_scale": torch.tensor(1.0),
            "bias": torch.ones(128),
        },
        assign=True,
    )
    backend = FakeVbarBackend()

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> AimdoWeights:
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            backend=backend,
            fixed_promotion=True,
        )

    mechanism = enroll_component(
        layer,
        load_device=CUDA0,
        offload_device=CPU,
        mechanism_factory=factory,
    )
    assert isinstance(mechanism, AimdoWeights)
    route = layer.residency_prefetch()
    assert route is not None
    _, requests = route
    assert requests == (("weight", None), ("bias", torch.float32))
    assert mechanism._geometry["weight"].allocation_bytes == 16_388  # pyright: ignore[reportPrivateUsage]
    layer.bind_fp8_matmul(True)
    route = layer.residency_prefetch()
    assert route is not None
    _, requests = route
    assert requests == (
        ("weight", None),
        ("input_scale", torch.float32),
        ("bias", torch.float32),
    )
    layer.bind_fp8_matmul(False)
    assert mechanism.partially_load(1 << 30) == 16_904
    assert mechanism.loaded_bytes() == 16_904
    assert layer.residency_prefetch() is None

    value = torch.arange(128, dtype=torch.float32).reshape(1, 1, 1, 128) / 128
    expected = torch.nn.functional.linear(value, layer.weight.float() * 0.5, layer.bias)
    assert torch.equal(layer(value), expected)

    layer.bind_fp8_matmul(True)
    assert layer.residency_prefetch() is None
    assert torch.equal(layer(value), expected)


def test_plain_fp8_linear_keeps_raw_vbar_geometry_for_every_forward_shape() -> None:
    layer = CastOperations(torch.float32).linear(128, 256, bias=False)
    layer.load_state_dict(
        {
            "weight": (torch.arange(256 * 128).reshape(256, 128) % 32)
            .div(32)
            .to(torch.float8_e4m3fn)
        },
        assign=True,
    )
    backend = FakeVbarBackend()

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> AimdoWeights:
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            backend=backend,
            fixed_promotion=True,
        )

    mechanism = enroll_component(
        layer,
        load_device=CUDA0,
        offload_device=CPU,
        mechanism_factory=factory,
    )
    assert isinstance(mechanism, AimdoWeights)
    assert mechanism._geometry["weight"].allocation_bytes == layer.weight.nbytes  # pyright: ignore[reportPrivateUsage]
    assert layer.residency_prefetch() == (  # type: ignore[attr-defined]
        mechanism,
        (("weight", torch.float8_e4m3fn),),
    )
    assert mechanism.partially_load(1 << 30) == layer.weight.nbytes
    assert mechanism.loaded_bytes() == layer.weight.nbytes

    value = torch.arange(128, dtype=torch.float32).reshape(1, 1, 1, 128) / 128
    expected = torch.nn.functional.linear(value, layer.weight.float())
    assert torch.equal(layer(value), expected)

    layer.bind_fp8_matmul(True)  # type: ignore[attr-defined]
    assert torch.equal(layer(value), expected)


def test_h3_int8_convrot_prefetch_uses_packed_storage() -> None:
    layer = Int8Linear(
        256,
        128,
        bias=True,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
    )
    layer.load_state_dict(
        {
            "weight": torch.ones((128, 256), dtype=torch.int8),
            "weight_scale": torch.ones((128, 1), dtype=torch.float32),
            "bias": torch.ones(128, dtype=torch.bfloat16),
        },
        strict=True,
        assign=True,
    )
    backend = FakeVbarBackend()

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> AimdoWeights:
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            backend=backend,
        )

    mechanism = cast(
        AimdoWeights,
        enroll_component(
            layer,
            load_device=CUDA0,
            offload_device=CPU,
            mechanism_factory=factory,
        ),
    )
    route = layer.residency_prefetch()
    assert route is not None
    assert route == (
        mechanism,
        (
            ("weight", None),
            ("bias", torch.bfloat16),
        ),
    )
    handle = mechanism.prefetch(route[1])
    assert handle is not None
    assert backend.allocations
    unit_allocations = mechanism._unit_allocations  # pyright: ignore[reportPrivateUsage]
    assert backend.fault_calls == list(unit_allocations.values())
    handle.close()
    assert backend.unpins == list(unit_allocations.values())
    assert all(allocation.pins == 0 for allocation in backend.allocations)
    source = ModuleStateStore(layer)["weight"]
    with mechanism.lease("weight") as lease:
        stored = lease.get_stored("weight")
    assert isinstance(source, Int8PackedWeight)
    assert isinstance(stored, Int8PackedWeight)
    assert torch.equal(stored.qdata, source.qdata)
    assert torch.equal(stored.scale, source.scale)
    assert stored.convrot == source.convrot
    assert stored.convrot_groupsize == source.convrot_groupsize


def test_patched_int8_prefetch_requantizes_and_keeps_packed_storage() -> None:
    layer = Int8Linear(
        256,
        128,
        bias=False,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
    )
    layer.load_state_dict(
        {
            "weight": torch.arange(128 * 256, dtype=torch.int32)
            .remainder(251)
            .sub(125)
            .to(torch.int8)
            .reshape(128, 256),
            "weight_scale": torch.linspace(0.002, 0.01, 128).reshape(128, 1),
        },
        strict=True,
        assign=True,
    )
    source = ModuleStateStore(layer)["weight"]
    assert isinstance(source, Int8PackedWeight)
    delta = torch.linspace(-0.015, 0.015, 128 * 256).reshape(128, 256)
    patch_set = PatchSet({"weight": (PatchEntry(DiffPatch(delta)),)})
    prefix = "diffusion_model.blocks.0.cross_attn.k."
    expected = patch_stored_weight(
        source,
        patch_set.entries("weight"),
        key=f"{prefix}weight",
        intermediate_dtype=torch.float32,
        weight_dtype=torch.float16,
    )
    assert isinstance(expected, Int8PackedWeight)
    mechanism, backend = _enroll_int8(
        layer,
        patch_set=patch_set,
        patch_weight_dtype=torch.float16,
        patch_key_prefix=prefix,
    )
    assert mechanism.weight_functions("weight") == ()
    route = layer.residency_prefetch()
    assert route is not None
    assert route == (mechanism, (("weight", None),))
    handle = mechanism.prefetch(route[1])
    assert handle is not None
    with mechanism.lease("weight") as lease:
        actual = lease.get_stored("weight")
    handle.close()
    assert backend.allocations
    assert isinstance(actual, Int8PackedWeight)
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    stored = ModuleStateStore(layer)["weight"]
    assert isinstance(stored, Int8PackedWeight)
    assert torch.equal(stored.qdata, source.qdata)


def test_nvfp4_aimdo_fake_tracks_exact_state_prefetch_and_cleanup() -> None:
    layer = Nvfp4Linear(
        512,
        512,
        bias=True,
        compute_dtype=torch.float32,
        pre_quant_scale=True,
    )
    layer.load_state_dict(
        {
            "weight": torch.zeros((512, 256), dtype=torch.uint8),
            "weight_scale": torch.ones((512, 32), dtype=torch.float8_e4m3fn),
            "weight_scale_2": torch.tensor(0.25),
            "input_scale": torch.tensor(0.5),
            "pre_quant_scale": torch.ones(512),
            "bias": torch.zeros(512),
        },
        strict=True,
        assign=True,
    )
    backend = FakeVbarBackend()

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> AimdoWeights:
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            backend=backend,
        )

    mechanism = cast(
        AimdoWeights,
        enroll_component(
            layer,
            load_device=CUDA0,
            offload_device=CPU,
            mechanism_factory=factory,
        ),
    )
    route = layer.residency_prefetch()
    assert route is not None and route[0] is mechanism
    assert route[1] == (
        ("weight", None),
        ("input_scale", torch.float32),
        ("pre_quant_scale", torch.float32),
        ("bias", torch.float32),
    )
    handle = mechanism.prefetch(route[1])
    assert handle is not None
    handle.close()
    assert mechanism.total_bytes() == sum(
        value.numel() * value.element_size() for value in layer.state_dict().values()
    )
    mechanism.unload()
    assert mechanism.loaded_bytes() == 0
    assert not backend.allocations or all(
        allocation.pins == 0 for allocation in backend.allocations
    )


def test_unload_synchronizes_and_clears_cast_arenas() -> None:
    mechanism, backend = _aimdo({"weight": torch.ones(5000)})
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float16)
    state = mechanism._stream_state  # pyright: ignore[reportPrivateUsage]
    assert state is not None and state.arenas
    mechanism.unload()
    assert state.arenas == {}
    assert backend.file_reader_cleanups == 1
    assert [event for event in backend.events if event[0] == "synchronize"] == [
        ("synchronize", "transfer-0"),
        ("synchronize", "transfer-1"),
    ]
    mechanism.partially_load(0)
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float16)
    assert state.arenas


def test_release_working_buffers_synchronizes_and_clears_cast_arenas() -> None:
    backend = FakeVbarBackend(complete_events_on_record=False)
    mechanism, _ = _aimdo({"weight": torch.ones(5000)}, backend=backend)
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float16)
    state = mechanism._stream_state  # pyright: ignore[reportPrivateUsage]
    assert state is not None and state.arenas
    allocation = backend.allocations[0]
    assert allocation.pins == 1

    assert mechanism.release_working_buffers()
    assert state.arenas == {}
    assert allocation.pins == 0
    assert backend.recorded_events[0].complete is True
    assert backend.file_reader_cleanups == 0
    assert [event for event in backend.events if event[0] == "synchronize"] == [
        ("synchronize", "transfer-0"),
        ("synchronize", "transfer-1"),
    ]
    assert not mechanism.release_working_buffers()


def test_single_giant_weight_reuses_its_existing_stream_arena() -> None:
    mechanism, backend = _aimdo({"weight": torch.ones(5000)})
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float16)
    backend.change_signature(backend.allocations[0])
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float16)
    assert len(backend.arenas) == 1
    assert [event for event in backend.events if event[0] == "arena-get"] == [
        ("arena-get", 0, 10_000, 0),
        ("arena-get", 0, 10_000, 0),
    ]


def test_lease_body_exception_still_unpins_every_successful_fault() -> None:
    mechanism, backend = _aimdo({"weight": torch.ones(5000)})
    with pytest.raises(ValueError, match="forward failed"):
        with mechanism.lease("weight") as lease:
            lease.get("weight", dtype=torch.float32)
            lease.get("weight", dtype=torch.float32)
            raise ValueError("forward failed")
    mechanism._reap_unpins(wait=False)  # pyright: ignore[reportPrivateUsage]
    assert len(backend.fault_calls) == len(backend.unpins) == 1
    assert backend.allocations[0].pins == 0


def test_closed_lease_invalidates_both_accessors() -> None:
    mechanism, _ = _aimdo({"weight": _fp8((2, 2))})
    with mechanism.lease("unit") as lease:
        lease.get("weight", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="lease for unit 'unit' is closed"):
        lease.get("weight", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="lease for unit 'unit' is closed"):
        lease.get_stored("weight")


def test_request_larger_than_allocation_is_rejected_loudly() -> None:
    original = torch.ones(2, 2)
    mechanism, backend = _aimdo({"weight": original})
    with mechanism.lease("weight") as lease:
        with pytest.raises(
            ValueError,
            match=r"'weight' requires 32 bytes; allocation is 16 bytes",
        ):
            lease.get("weight", dtype=torch.float64)
    assert backend.fault_calls == []
    assert not mechanism.is_loaded("weight")
    assert mechanism._weights["weight"] is original  # pyright: ignore[reportPrivateUsage]


def test_request_geometry_caches_clear_when_eager_storage_changes() -> None:
    mechanism, _ = _aimdo({"weight": torch.ones(4)})
    request = aimdo_mod._BatchRequest(  # pyright: ignore[reportPrivateUsage]
        "weight", torch.float32, "get"
    )

    mechanism._preflight_requests((request,))  # pyright: ignore[reportPrivateUsage]
    assert mechanism._request_size_cache == {  # pyright: ignore[reportPrivateUsage]
        request: 16
    }
    assert mechanism._preflight_request_cache == {  # pyright: ignore[reportPrivateUsage]
        (request,)
    }

    mechanism.partially_load(0)
    assert not mechanism._request_size_cache  # pyright: ignore[reportPrivateUsage]
    assert not mechanism._preflight_request_cache  # pyright: ignore[reportPrivateUsage]


def test_accounting_priority_partial_unload_and_idempotent_unload() -> None:
    mechanism, backend = _aimdo({"weight": torch.ones(5000)})
    assert mechanism.demand_paged
    assert mechanism.total_bytes() == 20_000
    assert mechanism.loaded_bytes() == 0
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    assert mechanism.loaded_bytes() == 20_000
    backend.reported_loaded_size = 32 * 1024**2
    assert mechanism.loaded_bytes() == mechanism.total_bytes()
    backend.reported_loaded_size = None

    assert mechanism.partially_load(1) == 0
    assert backend.prioritize_calls == 1
    assert mechanism.partially_load(0) == 0
    assert backend.prioritize_calls == 2
    assert mechanism.partially_unload(1) == 20_000
    assert mechanism.loaded_bytes() == 0
    with pytest.raises(AimdoForceFullLoadError, match="cannot force a full load"):
        mechanism.partially_load(None)

    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    assert mechanism._cache  # pyright: ignore[reportPrivateUsage]
    mechanism.unload()
    assert not mechanism._cache  # pyright: ignore[reportPrivateUsage]
    mechanism.unload()
    assert backend.deprioritize_calls == 2


def test_memory_accounting_separates_logical_weights_vbar_surplus_and_cast_arena() -> None:
    mechanism, backend = _aimdo({"weight": torch.ones(5000)})
    with mechanism.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
    demand_bytes = mechanism.total_bytes()
    backend.reported_loaded_size = demand_bytes + 4096

    accounting = mechanism.memory_accounting()

    assert accounting.weights == demand_bytes
    assert accounting.other_reclaimable == 4096
    assert accounting.allocator_weight_bytes == 0
    assert accounting.shared_workspace_id is not None
    assert accounting.shared_workspace_bytes == sum(
        backend.cast_arena_size(arena) for arena in backend.arenas
    )
    assert accounting.memory_compiler == "unavailable"


def test_working_set_reservation_holds_full_vbar_until_outer_release() -> None:
    mechanism, backend = _aimdo({"weight": torch.ones(5000)})
    reservation = mechanism._reservation_bytes  # pyright: ignore[reportPrivateUsage]
    assert mechanism.working_set_reservation_bytes() == 32 << 20

    with mechanism.reserve_working_set():
        assert backend.watermark_limits == [reservation]
        with mechanism.reserve_working_set():
            assert backend.watermark_limits == [reservation]
        assert backend.watermark_limits == [reservation]

    assert backend.watermark_limits == [reservation, 0]

    with pytest.raises(RuntimeError, match="stage failed"):
        with mechanism.reserve_working_set():
            raise RuntimeError("stage failed")
    assert backend.watermark_limits == [reservation, 0, reservation, 0]


def test_eager_tier_boundary_and_no_vbar_calls() -> None:
    exact_backend = FakeVbarBackend()
    exact, _ = _aimdo(
        {"weight": torch.ones(4096, dtype=torch.float32)},
        backend=exact_backend,
    )
    assert exact.partially_load(0) == 16 * 1024
    assert exact.is_loaded("weight")
    assert exact_backend.vbars == []
    assert exact_backend.allocations == []
    assert exact_backend.fault_calls == []

    over_backend = FakeVbarBackend()
    over, _ = _aimdo(
        {
            "weight": torch.ones(4096, dtype=torch.float32),
            "extra": torch.ones(1, dtype=torch.uint8),
        },
        backend=over_backend,
        units=(ResidencyUnit("unit", ("weight", "extra")),),
    )
    over.partially_load(0)
    assert not over.is_loaded("unit")
    assert len(over_backend.vbars) == 1
    assert len(over_backend.allocations) == 2


def test_eager_lease_is_resident_and_unload_is_idempotent() -> None:
    original = torch.arange(4, dtype=torch.float32)
    store: dict[str, StoredWeight] = {"weight": original}
    mechanism, backend = _aimdo(store)
    with mechanism.lease("weight") as lease:
        first = lease.get("weight", dtype=torch.float32)
        second = lease.get("weight", dtype=torch.float32)
        assert first.data_ptr() == second.data_ptr()
    assert mechanism.is_loaded("weight")
    assert backend.fault_calls == backend.unpins == []
    mechanism.unload()
    mechanism.unload()
    assert not mechanism.is_loaded("weight")
    assert store["weight"] is original
    assert mechanism.loaded_bytes() == 0


def test_eager_assignment_records_residency_generation() -> None:
    module = INITLESS.linear(2, 2, bias=False)
    module.load_state_dict(
        {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        strict=True,
        assign=True,
    )
    original = module.weight
    backend = FakeVbarBackend()

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> AimdoWeights:
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            backend=backend,
        )

    mechanism = enroll_component(
        module,
        load_device=CUDA0,
        offload_device=CPU,
        mechanism_factory=factory,
    )
    query = module_residency_mod._residency_assignment_generation  # pyright: ignore[reportPrivateUsage]
    assert query(module, "weight", original) is None

    mechanism.partially_load(0)
    loaded = module.weight
    generation = query(module, "weight", loaded)
    assert generation is not None

    mechanism.unload()
    assert module.weight is original
    restored_generation = query(module, "weight", original)
    assert restored_generation is not None
    assert restored_generation > generation


def test_mixed_tier_accounting_and_negative_budget_keep_eager_resident() -> None:
    mechanism, backend = _aimdo(
        {
            "tiny": torch.ones(4, dtype=torch.float32),
            "large": torch.ones(5000, dtype=torch.float32),
        }
    )
    assert mechanism.partially_load(0) == 16
    with mechanism.lease("large") as lease:
        lease.get("large", dtype=torch.float32)
    assert mechanism.loaded_bytes() == 20_016
    assert mechanism.automatically_reclaimable_bytes() == 20_000
    assert mechanism.partial_unload_capacity() == 20_000
    assert mechanism.partially_load(-1) == -20_000
    assert mechanism.loaded_bytes() == 16
    assert mechanism.automatically_reclaimable_bytes() == 0
    assert mechanism.is_loaded("tiny")
    assert backend.free_calls == [1]
    mechanism.unload()
    assert mechanism.loaded_bytes() == 0


def test_budget_keeps_large_units_demand_paged() -> None:
    originals = {
        "small": torch.arange(6000, dtype=torch.float32),
        "middle": torch.arange(8000, dtype=torch.float32),
        "large": torch.arange(10_000, dtype=torch.float32),
        "tiny": torch.arange(4, dtype=torch.float32),
    }
    store: dict[str, StoredWeight] = {key: value.clone() for key, value in originals.items()}
    sources = dict(store)
    mechanism, backend = _aimdo(
        store,
        units=tuple(ResidencyUnit(key, (key,)) for key in originals),
        stream_count=0,
    )

    assert mechanism.partially_load(1 << 60) == originals["tiny"].nbytes
    assert mechanism.loaded_bytes() == originals["tiny"].nbytes
    assert mechanism.is_loaded("tiny")
    assert not any(mechanism.is_loaded(name) for name in ("small", "middle", "large"))

    with mechanism.lease("small") as lease:
        assert torch.equal(lease.get("small", dtype=torch.float32), originals["small"])
    assert backend.fault_calls == [mechanism._allocations["small"]]  # pyright: ignore[reportPrivateUsage]

    mechanism.unload()
    assert mechanism.loaded_bytes() == 0
    assert all(store[key] is sources[key] for key in originals)
    for key, value in originals.items():
        stored = store[key]
        assert isinstance(stored, torch.Tensor)
        assert torch.equal(stored, value)


def test_budget_promotes_raw_and_ordinary_units_to_fixed_residency() -> None:
    class Store(dict[str, StoredWeight]):
        def uses_raw_residency(self, key: str) -> bool:
            return key == "raw"

    raw = torch.ones(20_000, dtype=torch.float8_e4m3fn)
    ordinary = torch.ones(20_000, dtype=torch.float32)
    store = Store(raw=raw, ordinary=ordinary)
    mechanism, _backend = _aimdo(
        store,
        units=(
            ResidencyUnit("raw", ("raw",)),
            ResidencyUnit("ordinary", ("ordinary",)),
        ),
        stream_count=0,
        fixed_promotion=True,
    )

    assert mechanism._geometry["raw"].allocation_bytes == 20_000  # pyright: ignore[reportPrivateUsage]
    assert mechanism.partially_load(1 << 30) == 100_000
    assert mechanism._promoted_units == {  # pyright: ignore[reportPrivateUsage]
        "ordinary",
        "raw",
    }
    with mechanism.lease("raw") as lease:
        loaded_raw = lease.get("raw", dtype=torch.float8_e4m3fn)
        assert isinstance(loaded_raw, torch.Tensor)
        assert torch.equal(loaded_raw.float(), raw.float())
    with mechanism.lease("ordinary") as lease:
        loaded_ordinary = lease.get("ordinary", dtype=torch.float32)
        assert isinstance(loaded_ordinary, torch.Tensor)
        assert torch.equal(loaded_ordinary, ordinary)
    assert _backend.fault_calls == []


def test_budget_keeps_non_fp8_raw_component_demand_paged() -> None:
    class Store(dict[str, StoredWeight]):
        def uses_raw_residency(self, key: str) -> bool:
            return key == "raw"

    mechanism, _backend = _aimdo(
        Store(
            raw=torch.ones(20_000, dtype=torch.bfloat16),
            ordinary=torch.ones(20_000, dtype=torch.float32),
        ),
        units=(
            ResidencyUnit("raw", ("raw",)),
            ResidencyUnit("ordinary", ("ordinary",)),
        ),
        stream_count=0,
        fixed_promotion=True,
    )

    assert mechanism.partially_load(1 << 30) == 0
    assert mechanism._promoted_units == set()  # pyright: ignore[reportPrivateUsage]


def test_budget_promotes_explicit_non_fp8_raw_component() -> None:
    class Store(dict[str, StoredWeight]):
        def uses_raw_residency(self, key: str) -> bool:
            return key == "raw"

    mechanism, _backend = _aimdo(
        Store(
            raw=torch.ones(20_000, dtype=torch.bfloat16),
            ordinary=torch.ones(20_000, dtype=torch.float32),
        ),
        units=(
            ResidencyUnit("raw", ("raw",)),
            ResidencyUnit("ordinary", ("ordinary",)),
        ),
        stream_count=0,
        fixed_promotion=True,
        promote_non_fp8_raw=True,
    )

    assert mechanism.partially_load(1 << 30) == 120_000
    assert mechanism._promoted_units == {  # pyright: ignore[reportPrivateUsage]
        "ordinary",
        "raw",
    }


def test_raw_promotion_reserves_widened_demand_materialization() -> None:
    class Store(dict[str, StoredWeight]):
        def uses_raw_residency(self, key: str) -> bool:
            return key == "raw"

    mechanism, _backend = _aimdo(
        Store(
            raw=torch.ones(20_000, dtype=torch.float8_e4m3fn),
            ordinary=torch.ones(20_000, dtype=torch.float32),
        ),
        units=(
            ResidencyUnit("raw", ("raw",)),
            ResidencyUnit("ordinary", ("ordinary",)),
        ),
        stream_count=0,
        fixed_promotion=True,
    )

    assert mechanism.partially_load(239_999) == 20_000
    assert mechanism._promoted_units == {"raw"}  # pyright: ignore[reportPrivateUsage]
    assert mechanism.partially_load(220_002) == 80_000
    assert mechanism._promoted_units == {  # pyright: ignore[reportPrivateUsage]
        "ordinary",
        "raw",
    }


def test_raw_residency_admission_includes_transient_materialization() -> None:
    class Store(dict[str, StoredWeight]):
        def uses_raw_residency(self, key: str) -> bool:
            return key == "raw"

    free = [0]
    mechanism, backend = _aimdo(
        Store(raw=torch.ones(20_000, dtype=torch.float8_e4m3fn)),
        physical_free_memory=lambda _device: free[0],
    )
    projected_peak = (64 << 20) + 120_000

    free[0] = projected_peak - 1
    assert mechanism.prefetch((("raw", torch.float8_e4m3fn),)) is None
    assert backend.fault_calls == []

    free[0] = projected_peak
    handle = mechanism.prefetch((("raw", torch.float8_e4m3fn),))
    assert handle is not None
    assert len(backend.fault_calls) == 1
    handle.close()


def test_patched_raw_capability_uses_materialized_geometry_and_promotion() -> None:
    class Store(dict[str, StoredWeight]):
        def uses_raw_residency(self, key: str) -> bool:
            return key == "weight"

    store = Store(weight=torch.ones(20_000, dtype=torch.float8_e4m3fn))
    patch = PatchSet({"weight": (PatchEntry(DiffPatch(torch.zeros(20_000))),)})
    mechanism, _backend = _aimdo(store, patch_set=patch, stream_count=0, fixed_promotion=True)

    assert mechanism._geometry["weight"].allocation_bytes == 80_000  # pyright: ignore[reportPrivateUsage]
    assert mechanism.partially_load(1 << 30) == 20_000
    assert mechanism._promoted_units == {"weight"}  # pyright: ignore[reportPrivateUsage]


def test_budget_promotes_shared_state_before_equal_experts() -> None:
    store: dict[str, StoredWeight] = {
        "expert-a": torch.ones(10_000, dtype=torch.float32),
        "expert-b": torch.ones(10_000, dtype=torch.float32),
        "shared": torch.ones(10_000, dtype=torch.float32),
    }
    mechanism, _backend = _aimdo(
        store,
        units=(
            ResidencyUnit("expert-a", ("expert-a",), expert=True),
            ResidencyUnit("expert-b", ("expert-b",), expert=True),
            ResidencyUnit("shared", ("shared",)),
        ),
        stream_count=0,
        fixed_promotion=True,
    )

    assert mechanism.partially_load(130_001) == 40_000
    assert mechanism._promoted_units == {"shared"}  # pyright: ignore[reportPrivateUsage]


def test_pressure_demotes_equal_experts_before_shared_state() -> None:
    store: dict[str, StoredWeight] = {
        "expert-a": torch.ones(10_000, dtype=torch.float32),
        "expert-b": torch.ones(10_000, dtype=torch.float32),
        "shared": torch.ones(10_000, dtype=torch.float32),
    }
    mechanism, _backend = _aimdo(
        store,
        units=(
            ResidencyUnit("expert-a", ("expert-a",), expert=True),
            ResidencyUnit("expert-b", ("expert-b",), expert=True),
            ResidencyUnit("shared", ("shared",)),
        ),
        stream_count=0,
        fixed_promotion=True,
    )
    mechanism.partially_load(1_000_000)

    assert mechanism.partially_unload(1) == 40_000
    assert "shared" in mechanism._promoted_units  # pyright: ignore[reportPrivateUsage]
    assert len(mechanism._promoted_units) == 2  # pyright: ignore[reportPrivateUsage]


def test_budget_does_not_double_count_demand_pages_as_fixed_capacity() -> None:
    store: dict[str, StoredWeight] = {
        "small": torch.ones(6000, dtype=torch.float32),
        "middle": torch.ones(8000, dtype=torch.float32),
        "large": torch.ones(10_000, dtype=torch.float32),
        "micro": torch.ones(5000, dtype=torch.float32),
    }
    physical_free = 130_001
    mechanism, _backend = _aimdo(
        store,
        units=tuple(ResidencyUnit(key, (key,)) for key in store),
        stream_count=0,
        physical_free_memory=lambda _device: physical_free,
        fixed_promotion=True,
    )

    assert mechanism.partially_load(130_001) == 72_000
    assert mechanism._promoted_units == {  # pyright: ignore[reportPrivateUsage]
        "middle",
        "large",
    }

    physical_free = 10_000
    assert mechanism.partially_load(130_001) == 0
    assert mechanism._promoted_units == {  # pyright: ignore[reportPrivateUsage]
        "middle",
        "large",
    }

    physical_free = 130_001
    with mechanism.lease("small") as lease:
        lease.get("small", dtype=torch.float32)

    assert mechanism.partially_load(130_001) == 0
    assert mechanism._promoted_units == {  # pyright: ignore[reportPrivateUsage]
        "middle",
        "large",
    }


def test_untied_eager_move_preserves_plain_move_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    store: dict[str, StoredWeight] = {"weight": original}
    mechanism, _ = _aimdo(store, units=(ResidencyUnit("unit", ("weight",)),))
    moved = original.detach().clone()

    def cloning_move(
        _stored: StoredWeight,
        _device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> StoredWeight:
        del non_blocking
        return moved

    monkeypatch.setattr(aimdo_mod, "move_stored", cloning_move)

    mechanism.partially_load(0)
    assert store["weight"] is moved


def test_large_shape_changing_patch_remains_demand_paged() -> None:
    patch = PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(5001), pad_weight=True)),)})
    mechanism, backend = _aimdo({"weight": torch.ones(5000)}, patch_set=patch)
    assert mechanism._geometry["weight"].shape == (5001,)  # pyright: ignore[reportPrivateUsage]
    assert not mechanism.is_loaded("weight")
    with mechanism.lease("weight") as lease:
        actual = lease.get("weight", dtype=torch.float32)
    mechanism._reap_unpins(wait=False)  # pyright: ignore[reportPrivateUsage]
    assert actual.shape == (5001,)
    assert len(backend.fault_calls) == len(backend.unpins) == 1


def test_fixed_promotion_requires_bool() -> None:
    with pytest.raises(TypeError, match="fixed_promotion must be a bool"):
        AimdoWeights(
            {"weight": torch.ones(1)},
            load_device=CUDA0,
            offload_device=CPU,
            backend=FakeVbarBackend(),
            fixed_promotion=1,  # type: ignore[arg-type]
        )


def test_non_fp8_raw_promotion_requires_bool() -> None:
    with pytest.raises(TypeError, match="promote_non_fp8_raw must be a bool"):
        AimdoWeights(
            {"weight": torch.ones(1)},
            load_device=CUDA0,
            offload_device=CPU,
            backend=FakeVbarBackend(),
            promote_non_fp8_raw=1,  # type: ignore[arg-type]
        )


def test_non_cuda_rejected_and_unindexed_cuda_is_canonicalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="requires a CUDA load device"):
        AimdoWeights(
            {"weight": torch.ones(1)},
            load_device=CPU,
            offload_device=CPU,
            backend=FakeVbarBackend(),
        )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    mechanism = AimdoWeights(
        {"weight": torch.ones(1)},
        load_device="cuda",
        offload_device=CPU,
        backend=FakeVbarBackend(),
    )
    assert mechanism.load_device == torch.device("cuda:3")


class _Control:
    def __init__(self, *, ready: set[int] | None = None, init_ok: bool = True) -> None:
        self.lib: object = _NativeControl()
        self.ready = set() if ready is None else ready
        self.init_ok = init_ok
        self.init_calls: list[tuple[AimdoDeviceEntry, ...]] = []

    def get_devctx(self, device_id: int) -> object:
        if device_id not in self.ready:
            raise RuntimeError("not initialized")
        return object()

    def init_devices(self, device_ids: Sequence[AimdoDeviceEntry]) -> bool:
        entries = tuple(device_ids)
        self.init_calls.append(entries)
        if self.init_ok:
            self.ready.update(entry[0] if isinstance(entry, tuple) else entry for entry in entries)
        return self.init_ok


class _NativeControl:
    def __init__(self) -> None:
        self.headrooms: list[int] = []

    def set_simple_vram_headroom(self, value: int) -> None:
        self.headrooms.append(value)


def _reset_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(activation, "_native_init_attempted", False)
    monkeypatch.setattr(activation, "_native_ready", False)


def test_activation_missing_module_fails_without_native_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_activation(monkeypatch)

    def missing() -> object:
        raise ImportError("missing")

    monkeypatch.setattr(activation, "_load_control", missing)
    assert not activation.ensure_aimdo_devices((0,))


def test_activation_never_trusts_lib_and_verifies_get_devctx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_activation(monkeypatch)
    control = _Control(init_ok=False)
    control.lib = object()
    monkeypatch.setattr(activation, "_load_control", lambda: control)
    assert not activation.ensure_aimdo_devices((0,))
    assert control.init_calls == [(0,)]


def test_activation_accepts_existing_context_without_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_activation(monkeypatch)
    control = _Control(ready={2})
    monkeypatch.setattr(activation, "_load_control", lambda: control)
    assert activation.ensure_aimdo_devices((2,))
    assert control.init_calls == []


def test_activation_rejects_a_null_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_activation(monkeypatch)
    control = _Control(init_ok=False)

    def null_context(_device_id: int) -> object | None:
        return None

    monkeypatch.setattr(control, "get_devctx", null_context)
    monkeypatch.setattr(activation, "_load_control", lambda: control)
    assert not activation.ensure_aimdo_devices((0,))
    assert control.init_calls == [(0,)]


def test_activation_permits_one_verified_init_and_refuses_later_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_activation(monkeypatch)
    control = _Control()
    monkeypatch.setattr(activation, "_load_control", lambda: control)
    assert activation.ensure_aimdo_devices((0,))
    assert control.init_calls == [(0,)]
    assert not activation.ensure_aimdo_devices((1,))
    assert control.init_calls == [(0,)]


def test_activation_forwards_headroom_and_deduplicates_preserving_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_activation(monkeypatch)
    control = _Control()
    monkeypatch.setattr(activation, "_load_control", lambda: control)
    assert activation.ensure_aimdo_devices(((2, 64 * 1024**2), 1, (2, 64 * 1024**2), 1))
    assert control.init_calls == [((2, 64 * 1024**2), 1)]


def test_activation_rejects_negative_or_conflicting_headroom() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        activation.ensure_aimdo_devices(((0, -1),))
    with pytest.raises(ValueError, match="conflicting headroom"):
        activation.ensure_aimdo_devices((0, (0, 1)))


class TestVisibleDeviceAdmission:
    @pytest.fixture
    def control(self, monkeypatch: pytest.MonkeyPatch) -> _Control:
        _reset_activation(monkeypatch)
        control = _Control()
        monkeypatch.setattr(activation, "_load_control", lambda: control)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

        def no_memory_query(*_args: object) -> object:
            pytest.fail("admission must not query free memory")

        def properties(index: int) -> object:
            return SimpleNamespace(total_memory=(24, 16, 8)[index] * 1024**3)

        monkeypatch.setattr(torch.cuda, "mem_get_info", no_memory_query)
        monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
        return control

    def test_admits_visible_devices_not_budget_allowlist(self, control: _Control) -> None:
        gib = 1024**3
        policy = MemoryPolicy(hard_budgets={"cuda:0": 20 * gib, "cuda:1": 18 * gib})
        assert activation.ensure_visible_aimdo_devices(policy)
        assert control.init_calls == [((0, 4 * gib), (1, 0), (2, 0))]
        assert activation.ensure_visible_aimdo_devices()
        assert activation.ensure_aimdo_devices((2,))
        assert len(control.init_calls) == 1

    def test_later_policy_calls_do_not_requery_capacity(
        self, control: _Control, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = MemoryPolicy(hard_budgets={"cuda:0": 20 * 1024**3})
        assert activation.ensure_visible_aimdo_devices(policy)

        def fail(_index: int) -> object:
            pytest.fail("successful admission must not query capacity again")

        monkeypatch.setattr(torch.cuda, "get_device_properties", fail)
        assert activation.ensure_visible_aimdo_devices(policy)
        assert activation.ensure_visible_aimdo_devices(policy)
        assert len(control.init_calls) == 1

    def test_empty_namespace_does_not_attempt_init(
        self, control: _Control, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert not activation.ensure_visible_aimdo_devices()
        assert control.init_calls == []

    def test_properties_failure_never_discards_budget(
        self, control: _Control, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error = RuntimeError("secondary device lost")

        def fail(_index: int) -> object:
            raise error

        monkeypatch.setattr(torch.cuda, "get_device_properties", fail)
        with pytest.raises(RuntimeError) as caught:
            activation.ensure_visible_aimdo_devices(MemoryPolicy(hard_budgets={"cuda:1": 1}))
        assert caught.value is error
        assert control.init_calls == []

    def test_concurrent_first_users_share_one_admission(
        self, control: _Control, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        barrier = threading.Barrier(3)
        queries: list[int] = []
        policy = MemoryPolicy(hard_budgets={"cuda:0": 20 * 1024**3})

        def properties(index: int) -> object:
            queries.append(index)
            return SimpleNamespace(total_memory=24 * 1024**3)

        monkeypatch.setattr(torch.cuda, "get_device_properties", properties)

        def admit(index: int) -> bool:
            barrier.wait(timeout=5)
            return activation.ensure_visible_aimdo_devices(
                policy
            ) and activation.ensure_aimdo_devices(
                (index,),
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            assert all(executor.map(admit, (2, 0, 1)))
        assert control.init_calls == [((0, 4 * 1024**3), (1, 0), (2, 0))]
        assert queries == [0]

    @pytest.mark.parametrize("failure", ["false", "exception", "cancellation"])
    def test_failed_init_is_never_promoted_to_ready_or_retried(
        self, control: _Control, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        calls: list[object] = []

        def fail(entries: object) -> bool:
            calls.append(entries)
            control.ready.update((0, 1, 2))
            if failure == "exception":
                raise RuntimeError("native failure")
            if failure == "cancellation":
                raise KeyboardInterrupt
            return False

        monkeypatch.setattr(control, "init_devices", fail)
        if failure == "cancellation":
            with pytest.raises(KeyboardInterrupt):
                activation.ensure_visible_aimdo_devices()
        else:
            assert not activation.ensure_visible_aimdo_devices()
        assert not activation.ensure_visible_aimdo_devices()
        assert not activation.ensure_aimdo_devices((2,))
        assert calls == [(0, 1, 2)]

    def test_external_partial_contexts_are_not_reinitialized(self, control: _Control) -> None:
        control.ready.add(0)
        assert not activation.ensure_visible_aimdo_devices()
        assert not activation.ensure_aimdo_devices((0,))
        assert control.init_calls == []

    def test_direct_construction_initializes_all_devices_before_later_users(
        self, control: _Control, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(aimdo_mod, "ComfyAimdoBackend", FakeVbarBackend)
        for device, dtype in ((2, torch.float32), (0, torch.float16), (1, torch.bfloat16)):
            mechanism = AimdoWeights(
                {"weight": torch.ones(1, dtype=dtype)},
                load_device=f"cuda:{device}",
                offload_device=CPU,
            )
            mechanism.unload()
        assert control.init_calls == [(0, 1, 2)]


def test_simple_headroom_validates_before_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(activation, "_load_control", lambda: pytest.fail())
    with pytest.raises(ValueError, match="non-negative"):
        activation.set_simple_vram_headroom(-1)


def test_simple_headroom_returns_false_when_unimportable_or_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_activation(monkeypatch)

    def missing() -> object:
        raise ImportError("missing dinkster-aimdo")

    monkeypatch.setattr(activation, "_native_ready", True)
    monkeypatch.setattr(activation, "_load_control", missing)
    assert not activation.set_simple_vram_headroom(1)

    monkeypatch.setattr(activation, "_native_ready", False)
    control = _Control(ready={0})
    monkeypatch.setattr(activation, "_load_control", lambda: control)
    assert not activation.set_simple_vram_headroom(1)
    assert cast("_NativeControl", control.lib).headrooms == []


def test_simple_headroom_forwards_after_verified_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_activation(monkeypatch)
    control = _Control()
    monkeypatch.setattr(activation, "_load_control", lambda: control)
    assert activation.ensure_aimdo_devices((0,))
    assert activation.set_simple_vram_headroom(256 * 1024**2)
    assert cast("_NativeControl", control.lib).headrooms == [256 * 1024**2]


def test_aimdo_resident_bytes_returns_zero_for_non_cuda() -> None:
    assert activation.aimdo_resident_bytes(CPU) == 0


def test_aimdo_resident_bytes_returns_zero_when_aimdo_unimportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _Control(ready={0})
    monkeypatch.setattr(activation, "_load_control", lambda: control)

    def missing_model_vbar(_name: str) -> object:
        raise ImportError("missing dinkster-aimdo")

    monkeypatch.setattr(activation.importlib, "import_module", missing_model_vbar)
    assert activation.aimdo_resident_bytes(CUDA0) == 0


def test_aimdo_resident_bytes_returns_zero_for_uninitialized_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(activation, "_load_control", lambda: _Control())
    assert activation.aimdo_resident_bytes(CUDA0) == 0


def test_aimdo_memory_status_returns_zero_when_unavailable_or_uninitialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert activation.aimdo_memory_status(CPU) == AimdoMemoryStatus(0, 0)
    monkeypatch.setattr(activation, "_load_control", lambda: _Control())
    assert activation.aimdo_memory_status(CUDA0) == AimdoMemoryStatus(0, 0)

    def missing() -> object:
        raise ImportError("missing dinkster-aimdo")

    monkeypatch.setattr(activation, "_load_control", missing)
    assert activation.aimdo_memory_status(CUDA0) == AimdoMemoryStatus(0, 0)


def test_dynamic_cuda_memory_counts_only_evictable_pages_and_caps_total() -> None:
    raw = CudaMemorySnapshot(
        total_bytes=100,
        driver_free_bytes=60,
        allocator_reclaimable_bytes=30,
    )
    status = AimdoMemoryStatus(evictable_bytes=20, pinned_bytes=40)

    snapshot = activation.dynamic_cuda_memory_snapshot(
        CUDA0,
        cuda_memory=lambda _device: raw,
        aimdo_memory=lambda _device: status,
    )

    assert snapshot == CudaMemorySnapshot(
        total_bytes=100,
        driver_free_bytes=60,
        allocator_reclaimable_bytes=30,
        dynamic_evictable_bytes=20,
        dynamic_pinned_bytes=40,
    )
    assert snapshot.free_bytes == 100
    assert activation.dynamic_free_memory(
        CUDA0,
        cuda_memory=lambda _device: raw,
        aimdo_memory=lambda _device: status,
    ) == DeviceMemory(free_total=100, free_torch=30)


def test_dynamic_free_memory_preserves_non_cuda_projection() -> None:
    expected = DeviceMemory(free_total=123, free_torch=45)

    def raw_memory(_device: torch.device) -> DeviceMemory:
        return expected

    assert activation.dynamic_free_memory(CPU, free_memory=raw_memory) == expected


def test_dynamic_cuda_memory_propagates_original_classification_error() -> None:
    error = RuntimeError("event query failed")
    raw = CudaMemorySnapshot(
        total_bytes=100,
        driver_free_bytes=60,
        allocator_reclaimable_bytes=30,
    )

    def fail(_device: torch.device) -> AimdoMemoryStatus:
        raise error

    with pytest.raises(RuntimeError) as caught:
        activation.dynamic_cuda_memory_snapshot(
            CUDA0,
            cuda_memory=lambda _device: raw,
            aimdo_memory=fail,
        )
    assert caught.value is error


def test_constructor_turns_failed_activation_into_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> bool:
        return False

    monkeypatch.setattr(aimdo_mod, "ensure_visible_aimdo_devices", unavailable)
    with pytest.raises(AimdoUnavailableError, match="device 0"):
        AimdoWeights(
            {"weight": torch.ones(1)},
            load_device=CUDA0,
            offload_device=CPU,
        )


def test_production_backend_reports_missing_numpy_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_numpy(name: str) -> object:
        assert name == "numpy"
        raise ImportError("missing numpy")

    monkeypatch.setattr(aimdo_mod.importlib, "import_module", missing_numpy)
    with pytest.raises(AimdoUnavailableError, match="require numpy"):
        ComfyAimdoBackend()


def test_manager_force_full_load_propagates_aimdo_guard() -> None:
    mechanism, _ = _aimdo({"weight": torch.ones(2, 2)})
    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            min_weight_memory_ratio=0.0,
        ),
        free_memory=lambda _device: DeviceMemory(1 << 30, 0),
        empty_cache=lambda _device: None,
    )
    with pytest.raises(AimdoForceFullLoadError, match="cannot force a full load"):
        manager.load([mechanism], force_full_load=True)
    assert mechanism not in manager.registered()


class _TrackingFactory:
    def __init__(self) -> None:
        self.instances: list[ResidentWeights] = []

    def __call__(
        self,
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> EnrolledResidency:
        mechanism = ResidentWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
        )
        self.instances.append(mechanism)
        return mechanism


def test_enrollment_default_is_unchanged_and_factory_is_injected() -> None:
    default_module = INITLESS.linear(2, 2)
    default = enroll_component(default_module, load_device=CPU, offload_device=CPU)
    assert isinstance(default, ResidentWeights)

    factory = _TrackingFactory()
    injected_module = INITLESS.linear(2, 2)
    injected = enroll_component(
        injected_module,
        load_device=CPU,
        offload_device=CPU,
        mechanism_factory=factory,
    )
    assert injected is factory.instances[0]

    assembled = AssembledSD(
        family=SD15,
        diffusion=cast("UNetModel", INITLESS.linear(2, 2)),
        clip_l=cast("ClipTextModel", INITLESS.linear(2, 2)),
        clip_g=None,
        vae=cast("AutoencoderKL", INITLESS.linear(2, 2)),
    )
    enrolled = enroll_assembled(
        assembled,
        load_device=CPU,
        offload_device=CPU,
        mechanism_factory=factory,
    )
    assert set(enrolled) == {"diffusion", "clip_l", "vae"}
    assert len(factory.instances) == 4


@pytest.mark.parametrize("target", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [4, 5000], ids=["eager", "paged"])
def test_storage_dtype_policy_aimdo_fake_offload_reload_is_byte_exact(
    width: int,
    target: torch.dtype,
) -> None:
    backend = FakeVbarBackend()
    components = tuple(CastOperations(target).linear(width, 1) for _ in range(3))
    for component in components:
        component.load_state_dict(
            {
                "weight": torch.randn(1, width, dtype=torch.float32),
                "bias": torch.randn(1, dtype=torch.float32),
            },
            strict=True,
            assign=True,
        )
    assembled = AssembledSD(
        family=SD15,
        diffusion=cast("UNetModel", components[0]),
        clip_l=cast("ClipTextModel", components[1]),
        clip_g=None,
        vae=cast("AutoencoderKL", components[2]),
        _storage_dtype_follows_compute=True,
        _component_compute_dtypes={
            "diffusion": target,
            "clip_l": target,
            "vae": target,
        },
    )

    seen_patch_sets: list[PatchSet[torch.Tensor] | None] = []

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> EnrolledResidency:
        seen_patch_sets.append(patch_set)
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            backend=backend,
            stream_count=0,
        )

    original = components[0].weight.detach().clone()
    delta = torch.full_like(original, 0.125)
    patch_set = PatchSet({"weight": (PatchEntry(DiffPatch(delta)),)})
    enrolled = enroll_assembled(
        assembled,
        load_device=CUDA0,
        offload_device=CPU,
        patch_sets={"diffusion": patch_set},
        mechanism_factory=factory,
    )
    assert set(enrolled.storage_dtype_report.outcomes.values()) == {"converted"}
    assert seen_patch_sets == [None, None, None]
    assert torch.equal(components[0].weight, (original + delta).to(target))
    expected = tuple(
        {
            name: parameter.detach().view(torch.uint8).clone()
            for name, parameter in component.named_parameters()
        }
        for component in components
    )

    for _ in range(2):
        for mechanism in enrolled.values():
            mechanism.partially_load(0)
            mechanism.unload()
        for component, component_expected in zip(components, expected, strict=True):
            for name, parameter in component.named_parameters():
                assert parameter.dtype is target
                assert torch.equal(parameter.view(torch.uint8), component_expected[name])


def test_storage_dtype_policy_aimdo_eager_tie_survives_forced_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeVbarBackend()
    first = CastOperations(torch.float16).linear(4, 4)
    second = CastOperations(torch.float16).linear(4, 4)
    second.weight = first.weight
    tied = torch.nn.Sequential(first, second)
    siblings = tuple(CastOperations(torch.float16).linear(4, 4) for _ in range(2))
    assembled = AssembledSD(
        family=SD15,
        diffusion=cast("UNetModel", tied),
        clip_l=cast("ClipTextModel", siblings[0]),
        clip_g=None,
        vae=cast("AutoencoderKL", siblings[1]),
        _storage_dtype_follows_compute=True,
        _component_compute_dtypes={
            "diffusion": torch.float16,
            "clip_l": torch.float16,
            "vae": torch.float16,
        },
    )
    moves = 0

    def cloning_move(
        stored: StoredWeight, _device: torch.device, *, non_blocking: bool = False
    ) -> StoredWeight:
        nonlocal moves
        moves += 1
        assert isinstance(stored, torch.Tensor)
        return stored.detach().clone()

    monkeypatch.setattr(aimdo_mod, "move_stored", cloning_move)

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> EnrolledResidency:
        return AimdoWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
            backend=backend,
            stream_count=0,
        )

    enrolled = enroll_assembled(
        assembled,
        load_device=CUDA0,
        offload_device=CPU,
        mechanism_factory=factory,
    )
    for mechanism in enrolled.values():
        mechanism.partially_load(0)
    assert first.weight is second.weight
    for mechanism in enrolled.values():
        mechanism.unload()
    assert first.weight is second.weight
    assert moves > 0
