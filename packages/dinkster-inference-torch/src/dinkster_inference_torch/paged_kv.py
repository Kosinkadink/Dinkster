"""Paged key/value storage for transactional generation sessions."""

from __future__ import annotations

import heapq
import math
import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Self, cast

import torch
from dinkster_memory import ConsumerItem, PageMap, PressureSignal


class PagedKVError(RuntimeError):
    """Base error for paged KV allocation and session state."""


class PagedKVCapacityError(PagedKVError):
    """The requested blocks cannot fit without evicting active state."""


class PagedKVSessionBusyError(PagedKVError):
    """The session already has an active lease."""


class PagedKVSessionMissingError(PagedKVError):
    """The requested session does not exist."""


class PagedKVShapeError(PagedKVError):
    """A KV tensor does not match the cache geometry."""


@dataclass(frozen=True, slots=True)
class PagedKVGeometry:
    block_tokens: int
    layer_count: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        for name in ("block_tokens", "layer_count", "kv_heads", "head_dim"):
            value = cast("object", getattr(self, name))
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if type(cast("object", self.dtype)) is not torch.dtype:
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise ValueError("paged KV storage requires a floating-point dtype")

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        return (self.layer_count, self.block_tokens, self.kv_heads, self.head_dim)

    @property
    def block_nbytes(self) -> int:
        element_size = torch.empty((), dtype=self.dtype).element_size()
        return 2 * math.prod(self.tensor_shape) * element_size


@dataclass(frozen=True, slots=True)
class PagedKVBlockState:
    block_id: int
    tier: Literal["device", "host"]
    reference_count: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class PagedKVSessionState:
    session_id: str
    token_count: int
    block_ids: tuple[int, ...]
    device_block_count: int
    host_block_count: int
    active: bool


@dataclass(frozen=True, slots=True)
class PagedKVBlockView:
    """Read-only borrowed tensors that remain valid until the lease closes."""

    block_id: int
    token_count: int
    key: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class _Block:
    block_id: int
    key: torch.Tensor
    value: torch.Tensor
    tier: Literal["device", "host"]
    reference_count: int = 1

    @property
    def nbytes(self) -> int:
        return self.key.nbytes + self.value.nbytes


@dataclass(slots=True)
class _Session:
    session_id: str
    block_ids: list[int]
    token_count: int
    last_used: int


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    block_ids: tuple[int, ...]
    token_count: int


class PagedKVSessionLease:
    """One active session transaction; close rolls back unless committed.

    Block views borrow live cache storage and must not be mutated or retained
    after this lease closes.
    """

    def __init__(self, cache: PagedKVCache, session_id: str) -> None:
        self._cache = cache
        self.session_id = session_id
        self._closed = False

    @property
    def token_count(self) -> int:
        return self._cache._lease_token_count(self)  # pyright: ignore[reportPrivateUsage]

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        try:
            self._cache._append(self, key, value)  # pyright: ignore[reportPrivateUsage]
        except BaseException:
            self.close()
            raise

    def block_views(self, layer: int) -> tuple[PagedKVBlockView, ...]:
        return self._cache._block_views(self, layer)  # pyright: ignore[reportPrivateUsage]

    def commit(self) -> None:
        if self._closed:
            return
        self._cache._finish(self, commit=True)  # pyright: ignore[reportPrivateUsage]
        self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._cache._finish(self, commit=False)  # pyright: ignore[reportPrivateUsage]
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class PagedKVCache:
    """Fixed-block KV storage shared by explicit generation sessions.

    One instance owns the KV geometry for one model placement on one explicit
    device. Sessions share immutable prefix blocks by reference. An active
    lease checkpoints those references, so appending to a partial block uses
    copy-on-write and rollback never corrupts a committed prefix.
    """

    demand_paged = True

    def __init__(
        self,
        cache_id: str,
        model_identity: str,
        geometry: PagedKVGeometry,
        *,
        load_device: torch.device | str,
        max_device_blocks: int,
        max_host_blocks: int = 0,
        session_evictor: Callable[[str], None] | None = None,
    ) -> None:
        if type(cast("object", cache_id)) is not str or not cache_id:
            raise ValueError("cache_id must be a non-empty string")
        if type(cast("object", model_identity)) is not str or not model_identity:
            raise ValueError("model_identity must be a non-empty string")
        if type(cast("object", geometry)) is not PagedKVGeometry:
            raise TypeError("geometry must be PagedKVGeometry")
        for name, value in (
            ("max_device_blocks", max_device_blocks),
            ("max_host_blocks", max_host_blocks),
        ):
            if type(cast("object", value)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if max_device_blocks <= 0:
            raise ValueError("max_device_blocks must be positive")
        if max_host_blocks < 0:
            raise ValueError("max_host_blocks must be non-negative")
        device = torch.device(load_device)
        if device.type == "cuda" and ":" not in str(device):
            raise ValueError("CUDA KV placement requires an explicit device index")
        if max_host_blocks and device.type == "cpu":
            raise ValueError("a CPU cache cannot offload to the same host tier")
        if session_evictor is not None and not callable(session_evictor):
            raise TypeError("session_evictor must be callable or None")

        self.cache_id = cache_id
        self.model_identity = model_identity
        self.geometry = geometry
        self.load_device = device
        self.max_device_blocks = max_device_blocks
        self.max_host_blocks = max_host_blocks
        self._session_evictor = session_evictor
        self._device_lane = _device_lane(device)
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}
        self._blocks: dict[int, _Block] = {}
        self._active: dict[str, tuple[PagedKVSessionLease, _Checkpoint]] = {}
        self._device_blocks = 0
        self._host_blocks = 0
        self._clock = 0
        self._next_block_id = 0
        self._free_block_ids: list[int] = []

    @property
    def block_nbytes(self) -> int:
        return self.geometry.block_nbytes

    def create_session(self, session_id: str) -> None:
        _validate_session_id(session_id)
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"paged KV session {session_id!r} already exists")
            self._sessions[session_id] = _Session(session_id, [], 0, self._tick())

    def fork_session(
        self,
        source_session_id: str,
        target_session_id: str,
        *,
        token_count: int | None = None,
    ) -> None:
        _validate_session_id(target_session_id)
        with self._lock:
            source = self._session(source_session_id)
            if source_session_id in self._active:
                raise PagedKVSessionBusyError(
                    f"paged KV session {source_session_id!r} has an active lease"
                )
            if target_session_id in self._sessions:
                raise ValueError(f"paged KV session {target_session_id!r} already exists")
            target_tokens = source.token_count if token_count is None else token_count
            if type(cast("object", target_tokens)) is not int:
                raise TypeError("fork token_count must be an exact integer or None")
            if not 0 <= target_tokens <= source.token_count:
                raise ValueError("fork token_count must be within the committed source prefix")
            block_count = (
                target_tokens + self.geometry.block_tokens - 1
            ) // self.geometry.block_tokens
            block_ids = source.block_ids[:block_count]
            self._retain(block_ids)
            self._sessions[target_session_id] = _Session(
                target_session_id,
                list(block_ids),
                target_tokens,
                self._tick(),
            )

    def pin(self, session_id: str) -> PagedKVSessionLease:
        with self._lock:
            session = self._session(session_id)
            if session_id in self._active:
                raise PagedKVSessionBusyError(
                    f"paged KV session {session_id!r} has an active lease"
                )
            self._restore_session(session)
            checkpoint = _Checkpoint(tuple(session.block_ids), session.token_count)
            self._retain(checkpoint.block_ids)
            lease = PagedKVSessionLease(self, session_id)
            self._active[session_id] = (lease, checkpoint)
            session.last_used = self._tick()
            return lease

    def close_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._active:
                raise PagedKVSessionBusyError(
                    f"paged KV session {session_id!r} has an active lease"
                )
            self._drop_session(session_id)

    def session_states(self) -> tuple[PagedKVSessionState, ...]:
        with self._lock:
            states: list[PagedKVSessionState] = []
            for session_id in sorted(self._sessions):
                session = self._sessions[session_id]
                device_blocks = sum(
                    self._blocks[block_id].tier == "device" for block_id in session.block_ids
                )
                states.append(
                    PagedKVSessionState(
                        session_id,
                        session.token_count,
                        tuple(session.block_ids),
                        device_blocks,
                        len(session.block_ids) - device_blocks,
                        session_id in self._active,
                    )
                )
            return tuple(states)

    def block_states(self) -> tuple[PagedKVBlockState, ...]:
        with self._lock:
            return tuple(
                PagedKVBlockState(
                    block.block_id,
                    block.tier,
                    block.reference_count,
                    block.nbytes,
                )
                for block in sorted(self._blocks.values(), key=lambda item: item.block_id)
            )

    def total_bytes(self) -> int:
        with self._lock:
            return len(self._blocks) * self.block_nbytes

    def loaded_bytes(self) -> int:
        with self._lock:
            return self._device_blocks * self.block_nbytes

    def automatically_reclaimable_bytes(self) -> int:
        return 0

    def offloaded_bytes(self) -> int:
        with self._lock:
            return self._host_blocks * self.block_nbytes

    def working_set_reservation_bytes(self) -> int:
        return 0

    def reserve_working_set(self) -> AbstractContextManager[None]:
        return nullcontext()

    def partially_load(self, extra_memory: int | None) -> int:
        with self._lock:
            if extra_memory is not None and extra_memory < 0:
                return -self.partially_unload(-extra_memory)
            before = self.loaded_bytes()
            target_blocks = (
                None
                if extra_memory is None
                else max(0, (extra_memory + self.block_nbytes - 1) // self.block_nbytes)
            )
            moved = 0
            sessions = sorted(
                self._sessions.values(),
                key=lambda session: (-session.last_used, session.session_id),
            )
            for session in sessions:
                if session.session_id not in self._sessions:
                    continue
                for block_id in session.block_ids:
                    block = self._blocks[block_id]
                    if block.tier != "host":
                        continue
                    if target_blocks is not None and moved >= target_blocks:
                        return self.loaded_bytes() - before
                    try:
                        self._ensure_device_slots(1, exclude={session.session_id})
                    except PagedKVCapacityError:
                        return self.loaded_bytes() - before
                    self._move_to_device(block)
                    moved += 1
            return self.loaded_bytes() - before

    def partially_unload(self, memory_to_free: int) -> int:
        with self._lock:
            if memory_to_free <= 0:
                return 0
            if self._active:
                active = ", ".join(sorted(self._active))
                raise PagedKVSessionBusyError(
                    f"cannot partially unload paged KV cache with active sessions: {active}"
                )
            return self._reclaim_device(memory_to_free)

    def partial_unload_capacity(self) -> int:
        """Device bytes reclaimable while every session is idle."""
        with self._lock:
            if self._active:
                return 0
            blocks = sum(block.tier == "device" for block in self._blocks.values())
            return blocks * self.block_nbytes

    def can_fully_offload(self) -> bool:
        """Whether every device block can move to host without dropping sessions."""
        with self._lock:
            available = self.max_host_blocks - self._host_block_count()
            return not self._active and available >= self._device_block_count()

    def unload(self) -> None:
        with self._lock:
            if self._active:
                active = ", ".join(sorted(self._active))
                raise PagedKVSessionBusyError(
                    f"cannot unload paged KV cache with active sessions: {active}"
                )
            for session_id in list(self._sessions):
                self._drop_session(session_id)

    def release_working_buffers(self) -> bool:
        return False

    def footprint(self, device: str) -> int:
        if device == self._device_lane:
            return self.loaded_bytes()
        if device == "ram":
            return self.offloaded_bytes()
        return 0

    async def shed(self, pressure: PressureSignal) -> int:
        with self._lock:
            if pressure.items is not None and self.cache_id not in pressure.items:
                return 0
            if pressure.device == self._device_lane:
                return self._reclaim_device(pressure.bytes_needed)
            if pressure.device != "ram":
                return 0
            before = self.offloaded_bytes()
            for session in self._idle_sessions():
                if (
                    self.offloaded_bytes() == 0
                    or before - self.offloaded_bytes() >= pressure.bytes_needed
                ):
                    break
                self._evict_session(session.session_id)
            return before - self.offloaded_bytes()

    def details(self) -> Sequence[ConsumerItem]:
        with self._lock:
            pinned = self._active_block_ids()
            flags = tuple(
                2 if block.block_id in pinned else 1 if block.tier == "device" else 0
                for block in sorted(self._blocks.values(), key=lambda item: item.block_id)
            )
            bytes_by_residency = {self._device_lane: self.loaded_bytes()}
            bytes_by_residency["ram"] = bytes_by_residency.get("ram", 0) + self.offloaded_bytes()
            return (
                ConsumerItem(
                    item_id=self.cache_id,
                    display_name=self.model_identity,
                    bytes_by_residency=bytes_by_residency,
                    pages=PageMap(page_bytes=self.block_nbytes, flags=flags),
                ),
            )

    def _append(
        self,
        lease: PagedKVSessionLease,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        with self._lock:
            session = self._active_session(lease)
            self._validate_append(key, value)
            incoming_offset = 0
            incoming_tokens = key.shape[1]
            block_offset = session.token_count % self.geometry.block_tokens
            if block_offset:
                block = self._blocks[session.block_ids[-1]]
                # The transaction checkpoint owns one reference but appends only
                # after the committed prefix. Other session references require COW.
                if block.reference_count > 2:
                    replacement = self._allocate_block(exclude={session.session_id})
                    try:
                        with torch.no_grad():
                            replacement.key[:, :block_offset].copy_(block.key[:, :block_offset])
                            replacement.value[:, :block_offset].copy_(block.value[:, :block_offset])
                    except BaseException:
                        self._release((replacement.block_id,))
                        raise
                    self._release((block.block_id,))
                    session.block_ids[-1] = replacement.block_id
                    block = replacement
            else:
                block = self._allocate_block(exclude={session.session_id})
                session.block_ids.append(block.block_id)

            while incoming_offset < incoming_tokens:
                available = self.geometry.block_tokens - block_offset
                count = min(available, incoming_tokens - incoming_offset)
                source = slice(incoming_offset, incoming_offset + count)
                target = slice(block_offset, block_offset + count)
                with torch.no_grad():
                    block.key[:, target].copy_(key[:, source])
                    block.value[:, target].copy_(value[:, source])
                incoming_offset += count
                session.token_count += count
                block_offset += count
                if incoming_offset < incoming_tokens:
                    block = self._allocate_block(exclude={session.session_id})
                    session.block_ids.append(block.block_id)
                    block_offset = 0

    def _block_views(
        self,
        lease: PagedKVSessionLease,
        layer: int,
    ) -> tuple[PagedKVBlockView, ...]:
        with self._lock:
            if type(cast("object", layer)) is not int:
                raise TypeError("layer must be an exact integer")
            if not 0 <= layer < self.geometry.layer_count:
                raise ValueError("layer is outside the paged KV geometry")
            session = self._active_session(lease)
            remaining = session.token_count
            views: list[PagedKVBlockView] = []
            for block_id in session.block_ids:
                count = min(remaining, self.geometry.block_tokens)
                block = self._blocks[block_id]
                if block.tier != "device":
                    raise AssertionError("active session contains a host-resident block")
                views.append(
                    PagedKVBlockView(
                        block_id,
                        count,
                        block.key[layer, :count],
                        block.value[layer, :count],
                    )
                )
                remaining -= count
            return tuple(views)

    def _finish(self, lease: PagedKVSessionLease, *, commit: bool) -> None:
        with self._lock:
            session = self._active_session(lease)
            _, checkpoint = self._active.pop(session.session_id)
            if commit:
                self._release(checkpoint.block_ids)
            else:
                self._release(tuple(session.block_ids))
                session.block_ids = list(checkpoint.block_ids)
                session.token_count = checkpoint.token_count
            session.last_used = self._tick()

    def _lease_token_count(self, lease: PagedKVSessionLease) -> int:
        with self._lock:
            return self._active_session(lease).token_count

    def _active_session(self, lease: PagedKVSessionLease) -> _Session:
        active = self._active.get(lease.session_id)
        if active is None or active[0] is not lease:
            raise PagedKVSessionBusyError("paged KV lease is closed or no longer active")
        return self._sessions[lease.session_id]

    def _validate_append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if not isinstance(key, torch.Tensor) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            value, torch.Tensor
        ):
            raise TypeError("paged KV append values must be tensors")
        if key.shape != value.shape:
            raise PagedKVShapeError("key and value shapes must match")
        expected = (
            self.geometry.layer_count,
            key.shape[1] if key.ndim == 4 else -1,
            self.geometry.kv_heads,
            self.geometry.head_dim,
        )
        if key.ndim != 4 or key.shape != expected or key.shape[1] <= 0:
            raise PagedKVShapeError(
                "KV tensors must have shape [layers, positive tokens, kv_heads, head_dim]"
            )
        if key.dtype != self.geometry.dtype or value.dtype != self.geometry.dtype:
            raise PagedKVShapeError("KV tensor dtype does not match the cache geometry")
        if key.device != self.load_device or value.device != self.load_device:
            raise PagedKVShapeError("KV tensor device does not match the cache placement")

    def _session(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise PagedKVSessionMissingError(f"paged KV session {session_id!r} does not exist")
        return session

    def _allocate_block(self, *, exclude: set[str]) -> _Block:
        self._ensure_device_slots(1, exclude=exclude)
        shape = self.geometry.tensor_shape
        key = torch.empty(shape, dtype=self.geometry.dtype, device=self.load_device)
        value = torch.empty(shape, dtype=self.geometry.dtype, device=self.load_device)
        if self._free_block_ids:
            block_id = heapq.heappop(self._free_block_ids)
        else:
            block_id = self._next_block_id
            self._next_block_id += 1
        block = _Block(
            block_id,
            key,
            value,
            "device",
        )
        if block.nbytes != self.block_nbytes:
            raise AssertionError("allocated KV block bytes do not match the geometry")
        self._blocks[block.block_id] = block
        self._device_blocks += 1
        return block

    def _ensure_device_slots(
        self,
        count: int,
        *,
        exclude: set[str],
        protect: set[int] | None = None,
    ) -> None:
        missing = count - (self.max_device_blocks - self._device_block_count())
        if missing <= 0:
            return
        self._reclaim_device(
            missing * self.block_nbytes,
            exclude=exclude,
            protect=protect,
        )
        if self.max_device_blocks - self._device_block_count() < count:
            raise PagedKVCapacityError(
                f"paged KV cache needs {count} device blocks but active state prevents reclamation"
            )

    def _restore_session(self, session: _Session) -> None:
        host_ids = tuple(
            block_id for block_id in session.block_ids if self._blocks[block_id].tier == "host"
        )
        unique_host_ids = tuple(dict.fromkeys(host_ids))
        self._ensure_device_slots(
            len(unique_host_ids),
            exclude={session.session_id},
            protect=set(session.block_ids),
        )
        for block_id in unique_host_ids:
            self._move_to_device(self._blocks[block_id])

    def _move_to_device(self, block: _Block) -> None:
        if block.tier == "device":
            return
        key = block.key.to(self.load_device)
        value = block.value.to(self.load_device)
        block.key = key
        block.value = value
        block.tier = "device"
        self._host_blocks -= 1
        self._device_blocks += 1

    def _move_to_host(self, block: _Block) -> None:
        if block.tier == "host":
            return
        if self.max_host_blocks <= self._host_block_count():
            raise PagedKVCapacityError("paged KV host tier is full")
        key = block.key.to("cpu")
        value = block.value.to("cpu")
        block.key = key
        block.value = value
        block.tier = "host"
        self._device_blocks -= 1
        self._host_blocks += 1

    def _reclaim_device(
        self,
        bytes_needed: int,
        *,
        exclude: set[str] | None = None,
        protect: set[int] | None = None,
    ) -> int:
        if bytes_needed <= 0:
            return 0
        excluded = set() if exclude is None else exclude
        before = self.loaded_bytes()
        pinned = self._active_block_ids()
        if protect is not None:
            pinned.update(protect)
        for session in self._idle_sessions(exclude=excluded):
            if before - self.loaded_bytes() >= bytes_needed:
                break
            if self.max_host_blocks:
                for block_id in tuple(session.block_ids):
                    block = self._blocks[block_id]
                    if block.tier != "device" or block_id in pinned:
                        continue
                    try:
                        self._move_to_host(block)
                    except PagedKVCapacityError:
                        self._evict_session(session.session_id)
                        break
                    if before - self.loaded_bytes() >= bytes_needed:
                        break
            else:
                self._evict_session(session.session_id)
        return before - self.loaded_bytes()

    def _idle_sessions(
        self,
        *,
        exclude: set[str] | None = None,
    ) -> list[_Session]:
        excluded = set() if exclude is None else exclude
        return sorted(
            (
                session
                for session in self._sessions.values()
                if session.session_id not in self._active and session.session_id not in excluded
            ),
            key=lambda session: (session.last_used, session.session_id),
        )

    def _active_block_ids(self) -> set[int]:
        block_ids: set[int] = set()
        for session_id, (_lease, checkpoint) in self._active.items():
            block_ids.update(checkpoint.block_ids)
            block_ids.update(self._sessions[session_id].block_ids)
        return block_ids

    def _drop_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            self._release(tuple(session.block_ids))

    def _evict_session(self, session_id: str) -> None:
        if self._session_evictor is None:
            self._drop_session(session_id)
        else:
            self._session_evictor(session_id)

    def _retain(self, block_ids: Sequence[int]) -> None:
        for block_id in block_ids:
            self._blocks[block_id].reference_count += 1

    def _release(self, block_ids: Sequence[int]) -> None:
        for block_id in block_ids:
            block = self._blocks[block_id]
            block.reference_count -= 1
            if block.reference_count < 0:
                raise AssertionError("paged KV block reference count became negative")
            if block.reference_count == 0:
                if block.tier == "device":
                    self._device_blocks -= 1
                else:
                    self._host_blocks -= 1
                del self._blocks[block_id]
                heapq.heappush(self._free_block_ids, block_id)

    def _device_block_count(self) -> int:
        return self._device_blocks

    def _host_block_count(self) -> int:
        return self._host_blocks

    def _tick(self) -> int:
        self._clock += 1
        return self._clock


def _validate_session_id(session_id: str) -> None:
    if type(cast("object", session_id)) is not str or not session_id:
        raise ValueError("session_id must be a non-empty string")


def _device_lane(device: torch.device) -> str:
    if device.type == "cpu":
        return "ram"
    return f"vram:{device}"


__all__ = [
    "PagedKVBlockState",
    "PagedKVBlockView",
    "PagedKVCache",
    "PagedKVCapacityError",
    "PagedKVError",
    "PagedKVGeometry",
    "PagedKVSessionBusyError",
    "PagedKVSessionLease",
    "PagedKVSessionMissingError",
    "PagedKVSessionState",
    "PagedKVShapeError",
]
