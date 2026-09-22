"""In-memory LRU cache store."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import cast

from dinkster_memory import PressureSignal, system_memory_snapshot
from dinkster_protocol import CacheKey
from dinkster_values import (
    COST_META_KEY,
    RESOURCE_ID_META_KEY,
    Value,
    iter_value_tree,
    value_resource_ids,
)


def _values_cost(values: Iterable[Value], device: str) -> int:
    """Declared bytes retained per payload identity on a device.

    Costs ride the envelope (COST_META_KEY: residency class -> bytes);
    values without declared cost count as zero - honest accounting of
    what was declared, never a guess. Values carrying RESOURCE_ID_META_KEY
    are *references* to owner-resolved state (resident stubs, resource
    handles): their cost is the owner's footprint, not this cache's, and
    counting it here would show the governor the same gigabytes twice.
    Aliases across keys and list positions count once. Separate payload
    objects count separately even when their encoded bytes are identical.
    Conflicting declarations for one payload use the largest declared cost.
    """
    costs: dict[int, int] = {}
    for top in values:
        for value in iter_value_tree(top):
            if value.meta.get(RESOURCE_ID_META_KEY) is not None:
                continue
            cost = value.meta.get(COST_META_KEY)
            if isinstance(cost, Mapping):
                declared = cast("Mapping[str, object]", cost).get(device, 0)
                if type(declared) is int and declared > 0:
                    identity = id(value.payload)
                    costs[identity] = max(costs.get(identity, 0), declared)
    return sum(costs.values())


def entry_cost(entry: Mapping[str, Value], device: str) -> int:
    return _values_cost(entry.values(), device)


def _references(entry: Mapping[str, Value], resource_id: str) -> bool:
    return any(resource_id in value_resource_ids(value) for value in entry.values())


class MemoryLRUCache:
    cache_layer = "memory"

    def __init__(self, max_entries: int = 1024, *, max_bytes: int | None = None) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_bytes is None:
            max_bytes = system_memory_snapshot().effective_total_bytes // 8
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[CacheKey, Mapping[str, Value]] = OrderedDict()
        self._clean: set[CacheKey] = set()
        self._persist: Callable[[CacheKey, Mapping[str, Value], bool], Awaitable[bool]] | None = (
            None
        )
        self._mutation_lock = asyncio.Lock()
        self._generation = 0
        self.hits = 0
        self.misses = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def set_persistence_handler(
        self, handler: Callable[[CacheKey, Mapping[str, Value], bool], Awaitable[bool]]
    ) -> None:
        """Serialize admission/eviction writes with RAM mutation.

        The handler receives whether this is eviction and returns whether it
        wrote eagerly. Exceptions retain the previous entry or dirty victim.
        """
        if self._persist is not None:
            raise ValueError("memory cache already belongs to a spill policy")
        self._persist = handler

    async def get(self, key: CacheKey) -> Mapping[str, Value] | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return dict(entry)

    async def put(self, key: CacheKey, outputs: Mapping[str, Value]) -> None:
        await self._put(key, outputs, persisted=False)

    async def promote(self, key: CacheKey, outputs: Mapping[str, Value]) -> None:
        """Admit an already-persisted copy without writing it back on eviction."""
        await self._put(key, outputs, persisted=True)

    async def _put(self, key: CacheKey, outputs: Mapping[str, Value], *, persisted: bool) -> None:
        generation = self._generation
        async with self._mutation_lock:
            if generation != self._generation:
                return
            if not persisted and self._persist is not None:
                self._clean.discard(key)
                persisted = await self._persist(key, outputs, False)
                if generation != self._generation:
                    return
            previous = self._entries.get(key)
            entry = dict(outputs)
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._clean.discard(key)
            if persisted:
                self._clean.add(key)
            try:
                while self.footprint("ram") > self._max_bytes:
                    victim = next(k for k, v in self._entries.items() if entry_cost(v, "ram") > 0)
                    await self._evict(victim)
                while len(self._entries) > self._max_entries:
                    await self._evict(next(iter(self._entries)))
            except BaseException:
                # A failed admission leaves the dirty victim available for retry;
                # the caller still owns the rejected outputs.
                if self._entries.get(key) is entry:
                    self._remove(key)
                    if previous is not None and generation == self._generation:
                        self._entries[key] = previous
                raise

    def _remove(self, key: CacheKey) -> None:
        del self._entries[key]
        self._clean.discard(key)

    async def _evict(self, key: CacheKey, device: str = "ram") -> int:
        entry = self._entries[key]
        if key not in self._clean and self._persist is not None:
            await self._persist(key, entry, True)
        if self._entries.get(key) is not entry:
            return 0  # explicit invalidation already removed it
        before = self.footprint(device)
        self._remove(key)
        return before - self.footprint(device)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> int:
        """Drop every entry, returning how many were held. Cache keys
        fingerprint a node's schema signature and inputs, never its
        implementation (hazard H4) - so a dev-mode pack reload whose code
        changed behind an unchanged signature would stale-hit forever.
        Reload clears the cache instead (the HotReloadHack precedent)."""
        count = len(self._entries)
        self._generation += 1
        self._entries.clear()
        self._clean.clear()
        return count

    def drop_referencing(self, resource_id: str) -> int:
        """Drop every entry holding a value that references the resource
        (by its envelope RESOURCE_ID_META_KEY). This is the invalidation
        half of invalidate-then-release: after it returns, no stub for
        that resource can come out of this cache, so the owner may safely
        drop the resource itself. Returns entries dropped. The next
        request for a dropped entry misses and recomputes its producer -
        which reloads the resource through the ordinary path.
        """
        keys = [key for key, entry in self._entries.items() if _references(entry, resource_id)]
        self._generation += 1
        for key in keys:
            self._remove(key)
        return len(keys)

    # -- governed consumer (dinkster_memory.Shedder, structurally) ---------

    def footprint(self, device: str) -> int:
        return _values_cost(
            (value for entry in self._entries.values() for value in entry.values()), device
        )

    async def shed(self, pressure: PressureSignal) -> int:
        """Evict LRU-first until the pressure is met or nothing costed
        remains. Only the governor calls this (hazard H13)."""
        freed = 0
        async with self._mutation_lock:
            # Oldest first; zero-cost entries cannot relieve this device.
            for key in list(self._entries):
                if freed >= pressure.bytes_needed:
                    break
                entry = self._entries.get(key)
                if entry is None:
                    continue
                cost = entry_cost(entry, pressure.device)
                if cost > 0:
                    freed += await self._evict(key, pressure.device)
        return freed
