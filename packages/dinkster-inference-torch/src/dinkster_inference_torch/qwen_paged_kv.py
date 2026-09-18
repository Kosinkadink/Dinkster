"""Layer-local paged KV storage for explicitly placed Qwen models."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from types import TracebackType
from typing import Self

import torch

from .paged_kv import (
    PagedKVBlockState,
    PagedKVBlockView,
    PagedKVCache,
    PagedKVGeometry,
    PagedKVSessionBusyError,
    PagedKVSessionLease,
    PagedKVSessionState,
)
from .qwen_layer_placement import QwenLayerPlacement, QwenLayerRange


class QwenPagedKVLease:
    """One transaction spanning every device-local Qwen cache."""

    def __init__(
        self,
        cache: QwenPagedKVCache,
        leases: tuple[PagedKVSessionLease, ...],
    ) -> None:
        self._cache = cache
        self._leases = leases
        self._closed = False

    @property
    def token_count(self) -> int:
        with self._cache._lock:  # pyright: ignore[reportPrivateUsage]
            counts = {lease.token_count for lease in self._leases}
            if len(counts) != 1:
                raise RuntimeError("Qwen device-local KV lengths disagree")
            return counts.pop()

    def append_layers(
        self,
        key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        if len(key_values) != self._cache.layer_count:
            raise ValueError("Qwen KV append must contain one pair per model layer")
        with self._cache._lock:  # pyright: ignore[reportPrivateUsage]
            try:
                for item, lease in zip(self._cache._items, self._leases, strict=True):  # pyright: ignore[reportPrivateUsage]
                    selected = key_values[item.layer_range.start : item.layer_range.stop]
                    key = torch.stack([layer_key[0].transpose(0, 1) for layer_key, _ in selected])
                    value = torch.stack(
                        [layer_value[0].transpose(0, 1) for _, layer_value in selected]
                    )
                    lease.append(key, value)
            except BaseException:
                self.close()
                raise

    def block_views(self, layer: int) -> tuple[PagedKVBlockView, ...]:
        if type(layer) is not int or not 0 <= layer < self._cache.layer_count:
            raise ValueError("layer is outside the Qwen paged KV placement")
        with self._cache._lock:  # pyright: ignore[reportPrivateUsage]
            for item, lease in zip(self._cache._items, self._leases, strict=True):  # pyright: ignore[reportPrivateUsage]
                if item.layer_range.start <= layer < item.layer_range.stop:
                    return lease.block_views(layer - item.layer_range.start)
        raise AssertionError("Qwen paged KV placement does not cover the requested layer")

    def commit(self) -> None:
        if self._closed:
            return
        with self._cache._lock:  # pyright: ignore[reportPrivateUsage]
            for lease in self._leases:
                lease.commit()
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        with self._cache._lock:  # pyright: ignore[reportPrivateUsage]
            for lease in reversed(self._leases):
                lease.close()
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


class _QwenPagedKVResidency:
    """Manager-facing residency for one device-local cache partition."""

    demand_paged = True

    def __init__(self, owner: QwenPagedKVCache, cache: PagedKVCache) -> None:
        self._owner = owner
        self._cache = cache
        self.load_device = cache.load_device

    def total_bytes(self) -> int:
        with self._owner._lock:  # pyright: ignore[reportPrivateUsage]
            return self._cache.total_bytes()

    def loaded_bytes(self) -> int:
        with self._owner._lock:  # pyright: ignore[reportPrivateUsage]
            return self._cache.loaded_bytes()

    def automatically_reclaimable_bytes(self) -> int:
        return 0

    def offloaded_bytes(self) -> int:
        with self._owner._lock:  # pyright: ignore[reportPrivateUsage]
            return self._cache.offloaded_bytes()

    def working_set_reservation_bytes(self) -> int:
        return 0

    def reserve_working_set(self) -> AbstractContextManager[None]:
        return nullcontext()

    def partially_load(self, extra_memory: int | None) -> int:
        if extra_memory is not None and extra_memory < 0:
            return -self.partially_unload(-extra_memory)
        with self._owner._lock:  # pyright: ignore[reportPrivateUsage]
            return self._cache.partially_load(extra_memory)

    def can_fully_offload(self) -> bool:
        with self._owner._lock:  # pyright: ignore[reportPrivateUsage]
            return self._cache.can_fully_offload()

    def partial_unload_capacity(self) -> int:
        with self._owner._lock:  # pyright: ignore[reportPrivateUsage]
            return self._cache.partial_unload_capacity()

    def partially_unload(self, memory_to_free: int) -> int:
        if memory_to_free <= 0:
            return 0
        with self._owner._lock:  # pyright: ignore[reportPrivateUsage]
            before = self._cache.loaded_bytes()
            host_blocks = sum(state.tier == "host" for state in self._cache.block_states())
            available = max(0, self._cache.max_host_blocks - host_blocks)
            if available:
                self._cache.partially_unload(
                    min(memory_to_free, available * self._cache.block_nbytes)
                )
            for state in self._owner.session_states():
                if before - self._cache.loaded_bytes() >= memory_to_free:
                    break
                if not state.active:
                    self._owner.close_session(state.session_id)
            return before - self._cache.loaded_bytes()

    def unload(self) -> None:
        with self._owner._lock:  # pyright: ignore[reportPrivateUsage]
            self._owner.unload()

    def release_working_buffers(self) -> bool:
        return False


class _CacheItem:
    def __init__(self, layer_range: QwenLayerRange, cache: PagedKVCache) -> None:
        self.layer_range = layer_range
        self.cache = cache


class QwenPagedKVCache:
    """Transactional Qwen KV cache partitioned by contiguous layer range."""

    def __init__(
        self,
        cache_id: str,
        model_identity: str,
        placement: QwenLayerPlacement,
        *,
        block_tokens: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        max_device_blocks: int,
        max_host_blocks: int = 0,
    ) -> None:
        if type(placement) is not QwenLayerPlacement:
            raise TypeError("Qwen paged KV placement must be QwenLayerPlacement")
        self.cache_id = cache_id
        self.model_identity = model_identity
        self.layer_count = placement.layer_count
        self._lock = threading.RLock()
        items: list[_CacheItem] = []
        for index, layer_range in enumerate(placement.ranges):
            cache = PagedKVCache(
                f"{cache_id}:range-{index}",
                model_identity,
                PagedKVGeometry(
                    block_tokens,
                    layer_range.stop - layer_range.start,
                    kv_heads,
                    head_dim,
                    dtype,
                ),
                load_device=layer_range.device,
                max_device_blocks=max_device_blocks,
                max_host_blocks=max_host_blocks,
                session_evictor=self._evict_session,
            )
            items.append(_CacheItem(layer_range, cache))
        self._items = tuple(items)
        self._mechanisms = tuple(_QwenPagedKVResidency(self, item.cache) for item in self._items)

    @property
    def block_nbytes(self) -> int:
        return sum(item.cache.block_nbytes for item in self._items)

    @property
    def residency_mechanisms(self) -> tuple[_QwenPagedKVResidency, ...]:
        return self._mechanisms

    def create_session(self, session_id: str) -> None:
        with self._lock:
            created: list[PagedKVCache] = []
            try:
                for item in self._items:
                    item.cache.create_session(session_id)
                    created.append(item.cache)
            except BaseException:
                for cache in reversed(created):
                    cache.close_session(session_id)
                raise

    def fork_session(
        self,
        source_session_id: str,
        target_session_id: str,
        *,
        token_count: int | None = None,
    ) -> None:
        with self._lock:
            created: list[PagedKVCache] = []
            try:
                for item in self._items:
                    item.cache.fork_session(
                        source_session_id,
                        target_session_id,
                        token_count=token_count,
                    )
                    created.append(item.cache)
            except BaseException:
                for cache in reversed(created):
                    cache.close_session(target_session_id)
                raise

    def pin(self, session_id: str) -> QwenPagedKVLease:
        with self._lock:
            leases: list[PagedKVSessionLease] = []
            try:
                for item in self._items:
                    leases.append(item.cache.pin(session_id))
            except BaseException:
                for lease in reversed(leases):
                    lease.close()
                raise
            return QwenPagedKVLease(self, tuple(leases))

    def close_session(self, session_id: str) -> None:
        with self._lock:
            states = self.session_states()
            matching = next((state for state in states if state.session_id == session_id), None)
            if matching is not None and matching.active:
                raise PagedKVSessionBusyError(
                    f"paged KV session {session_id!r} has an active lease"
                )
            for item in self._items:
                item.cache.close_session(session_id)

    def _evict_session(self, session_id: str) -> None:
        with self._lock:
            for item in self._items:
                item.cache._drop_session(session_id)  # pyright: ignore[reportPrivateUsage]

    def session_states(self) -> tuple[PagedKVSessionState, ...]:
        with self._lock:
            collections = tuple(item.cache.session_states() for item in self._items)
            first = collections[0]
            semantics = tuple(
                tuple((state.session_id, state.token_count, state.active) for state in states)
                for states in collections
            )
            if any(current != semantics[0] for current in semantics[1:]):
                raise RuntimeError("Qwen device-local KV session states disagree")
            by_range = tuple(
                {state.session_id: state for state in states} for states in collections
            )
            block_maps: list[dict[int, int]] = []
            offset = 0
            for item in self._items:
                states = item.cache.block_states()
                block_maps.append(
                    {state.block_id: offset + index for index, state in enumerate(states)}
                )
                offset += len(states)
            return tuple(
                PagedKVSessionState(
                    state.session_id,
                    state.token_count,
                    tuple(
                        block_maps[index][block_id]
                        for index, states in enumerate(by_range)
                        for block_id in states[state.session_id].block_ids
                    ),
                    sum(states[state.session_id].device_block_count for states in by_range),
                    sum(states[state.session_id].host_block_count for states in by_range),
                    state.active,
                )
                for state in first
            )

    def block_states(self) -> tuple[PagedKVBlockState, ...]:
        with self._lock:
            if len(self._items) == 1:
                return self._items[0].cache.block_states()
            states: list[PagedKVBlockState] = []
            block_id = 0
            for item in self._items:
                for state in item.cache.block_states():
                    states.append(
                        PagedKVBlockState(
                            block_id,
                            state.tier,
                            state.reference_count,
                            state.nbytes,
                        )
                    )
                    block_id += 1
            return tuple(states)

    def total_bytes(self) -> int:
        with self._lock:
            return sum(item.cache.total_bytes() for item in self._items)

    def loaded_bytes(self) -> int:
        with self._lock:
            return sum(item.cache.loaded_bytes() for item in self._items)

    def offloaded_bytes(self) -> int:
        with self._lock:
            return sum(item.cache.offloaded_bytes() for item in self._items)

    def unload(self) -> None:
        with self._lock:
            active = tuple(state.session_id for state in self.session_states() if state.active)
            if active:
                raise PagedKVSessionBusyError(
                    f"cannot unload Qwen paged KV cache with active sessions: {', '.join(active)}"
                )
            for item in self._items:
                item.cache.unload()


__all__ = ["QwenPagedKVCache", "QwenPagedKVLease"]
