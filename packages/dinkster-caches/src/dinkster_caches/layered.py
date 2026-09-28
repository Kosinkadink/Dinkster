"""LayeredCache: cache stores composed fastest-first (DESIGN 3.4).

memory -> disk -> peer, or any subset: get() promotes slower hits. A memory
layer immediately followed by disk defers large costed entries until
eviction. Small and uncosted entries retain write-through persistence.
Other layers retain write-through behavior. Disk refuses unpersistable entries.

Invalidation fans out: drop_referencing reaches every layer that supports
it, because a stale resident stub served from *any* layer is the dangling
reference the release gate exists to prevent. clear() likewise reaches every
mutable layer so a code reload cannot leave stale persistent entries behind.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextvars import ContextVar
from itertools import pairwise
from typing import cast
from weakref import WeakValueDictionary

from dinkster_protocol import CacheKey, CacheStore
from dinkster_values import MEBIBYTE, Value

from .disk import DiskCacheStore
from .memory import MemoryLRUCache, entry_cost

DEFAULT_DISK_SPILL_MIN_BYTES = MEBIBYTE


class LayeredCache:
    def __init__(
        self, *layers: CacheStore, disk_spill_min_bytes: int = DEFAULT_DISK_SPILL_MIN_BYTES
    ) -> None:
        if not layers:
            raise ValueError("LayeredCache requires at least one layer")
        if disk_spill_min_bytes < 0:
            raise ValueError("disk_spill_min_bytes must be >= 0")
        self._layers = layers
        self._disk_spill_min_bytes = disk_spill_min_bytes
        self._spill_disks: dict[MemoryLRUCache, DiskCacheStore] = {}
        for faster, slower in pairwise(layers):
            if isinstance(faster, MemoryLRUCache) and isinstance(slower, DiskCacheStore):
                self._spill_disks[faster] = slower

                async def persist(
                    key: CacheKey,
                    outputs: Mapping[str, Value],
                    evicted: bool,
                    disk: DiskCacheStore = slower,
                ) -> bool:
                    return await self._write_disk(disk, key, outputs, evicted=evicted)

                faster.set_persistence_handler(persist)
        self._key_locks: WeakValueDictionary[CacheKey, asyncio.Lock] = WeakValueDictionary()
        self._generation = 0
        self.hits = 0
        self.misses = 0
        self.hits_by_layer: dict[str, int] = {}
        self._hit_layer: ContextVar[str | None] = ContextVar(
            "dinkster_layered_cache_hit_layer", default=None
        )

    @property
    def layers(self) -> tuple[CacheStore, ...]:
        return self._layers

    async def _write_disk(
        self,
        disk: DiskCacheStore,
        key: CacheKey,
        outputs: Mapping[str, Value],
        *,
        evicted: bool,
    ) -> bool:
        """Write through small outputs and flush deferred outputs on eviction."""
        if not evicted:
            disk.discard(key)
        cost = entry_cost(outputs, "ram")
        if cost > 0 and cost >= self._disk_spill_min_bytes and not evicted:
            return False
        await disk.put(key, outputs)
        return True

    async def get(self, key: CacheKey) -> Mapping[str, Value] | None:
        generation = self._generation
        async with self._key_locks.setdefault(key, asyncio.Lock()):
            for index, layer in enumerate(self._layers):
                if generation != self._generation:
                    break
                hit = await layer.get(key)
                if generation != self._generation:
                    break
                if hit is not None:
                    # Persist a peer hit before admitting its clean RAM copy.
                    for faster in reversed(self._layers[:index]):
                        if generation != self._generation:
                            break
                        if isinstance(faster, MemoryLRUCache) and faster in self._spill_disks:
                            await faster.promote(key, hit)
                        else:
                            await faster.put(key, hit)
                    if generation != self._generation:
                        break
                    layer_name = getattr(layer, "cache_layer", f"layer-{index}")
                    self.hits += 1
                    self.hits_by_layer[layer_name] = self.hits_by_layer.get(layer_name, 0) + 1
                    self._hit_layer.set(layer_name)
                    return hit
        self.misses += 1
        self._hit_layer.set(None)
        return None

    async def put(self, key: CacheKey, outputs: Mapping[str, Value]) -> None:
        generation = self._generation
        async with self._key_locks.setdefault(key, asyncio.Lock()):
            if generation != self._generation:
                return
            for layer in self._layers:
                if generation != self._generation:
                    return
                if layer not in self._spill_disks.values():
                    await layer.put(key, outputs)

    def drop_referencing(self, resource_id: str) -> int:
        self._generation += 1
        dropped = 0
        for layer in self._layers:
            dropper = getattr(layer, "drop_referencing", None)
            if callable(dropper):
                dropped += cast(int, dropper(resource_id))
        return dropped

    def clear(self) -> int:
        self._generation += 1
        dropped = 0
        for layer in self._layers:
            clearer = getattr(layer, "clear", None)
            if callable(clearer):
                dropped += cast(int, clearer())
        self._hit_layer.set(None)
        return dropped

    def take_hit_layer(self) -> str | None:
        """Return the source of the latest hit for engine telemetry."""
        layer = self._hit_layer.get()
        self._hit_layer.set(None)
        return layer
