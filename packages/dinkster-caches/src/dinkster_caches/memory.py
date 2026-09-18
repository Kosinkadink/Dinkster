"""In-memory LRU cache store."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import cast

from dinkster_memory import PressureSignal
from dinkster_protocol import CacheKey
from dinkster_values import (
    COST_META_KEY,
    RESOURCE_ID_META_KEY,
    Value,
    iter_value_tree,
    value_resource_ids,
)


def _entry_cost(entry: Mapping[str, Value], device: str) -> int:
    """Bytes an entry holds on a device, read from value cost metadata.

    Costs ride the envelope (COST_META_KEY: residency class -> bytes);
    values without declared cost count as zero - honest accounting of
    what was declared, never a guess. Values carrying RESOURCE_ID_META_KEY
    are *references* to owner-resolved state (resident stubs, resource
    handles): their cost is the owner's footprint, not this cache's, and
    counting it here would show the governor the same gigabytes twice.
    Both rules apply per node of the value tree, so a list's children are
    accounted (and stubs among them excluded) individually (DESIGN 3.13).
    """
    total = 0
    for top in entry.values():
        for value in iter_value_tree(top):
            if value.meta.get(RESOURCE_ID_META_KEY) is not None:
                continue
            cost = value.meta.get(COST_META_KEY)
            if isinstance(cost, Mapping):
                declared = cast("Mapping[str, object]", cost).get(device, 0)
                if isinstance(declared, int):
                    total += declared
    return total


def _references(entry: Mapping[str, Value], resource_id: str) -> bool:
    return any(resource_id in value_resource_ids(value) for value in entry.values())


class MemoryLRUCache:
    cache_layer = "memory"

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._entries: OrderedDict[CacheKey, Mapping[str, Value]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    async def get(self, key: CacheKey) -> Mapping[str, Value] | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return dict(entry)

    async def put(self, key: CacheKey, outputs: Mapping[str, Value]) -> None:
        self._entries[key] = dict(outputs)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> int:
        """Drop every entry, returning how many were held. Cache keys
        fingerprint a node's schema signature and inputs, never its
        implementation (hazard H4) - so a dev-mode pack reload whose code
        changed behind an unchanged signature would stale-hit forever.
        Reload clears the cache instead (the HotReloadHack precedent)."""
        count = len(self._entries)
        self._entries.clear()
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
        for key in keys:
            del self._entries[key]
        return len(keys)

    # -- governed consumer (dinkster_memory.Shedder, structurally) ---------

    def footprint(self, device: str) -> int:
        return sum(_entry_cost(entry, device) for entry in self._entries.values())

    async def shed(self, pressure: PressureSignal) -> int:
        """Evict LRU-first until the pressure is met or nothing costed
        remains. Only the governor calls this (hazard H13)."""
        freed = 0
        # Oldest first; skip entries that hold nothing on this device -
        # evicting them frees no pressure and costs a recompute.
        for key in list(self._entries):
            if freed >= pressure.bytes_needed:
                break
            cost = _entry_cost(self._entries[key], pressure.device)
            if cost <= 0:
                continue
            del self._entries[key]
            freed += cost
        return freed
