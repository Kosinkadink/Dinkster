"""LayeredCache: cache stores composed fastest-first (DESIGN 3.4).

memory -> disk -> peer, or any subset: get() walks the layers in order and
*promotes* a hit into every faster layer, so a value fetched once from a
peer lands on local disk and in local memory - the next request never
crosses the network. put() writes through every layer; layers that refuse
an entry (a disk store seeing resource stubs, a read-only peer store)
simply keep less, which is always safe: a layer's contract is "may serve
what it accepted", never "must hold everything".

Invalidation fans out: drop_referencing reaches every layer that supports
it, because a stale resident stub served from *any* layer is the dangling
reference the release gate exists to prevent. clear() likewise reaches every
mutable layer so a code reload cannot leave stale persistent entries behind.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from typing import cast

from dinkster_protocol import CacheKey, CacheStore
from dinkster_values import Value


class LayeredCache:
    def __init__(self, *layers: CacheStore) -> None:
        if not layers:
            raise ValueError("LayeredCache requires at least one layer")
        self._layers = layers
        self.hits = 0
        self.misses = 0
        self.hits_by_layer: dict[str, int] = {}
        self._hit_layer: ContextVar[str | None] = ContextVar(
            "dinkster_layered_cache_hit_layer", default=None
        )

    @property
    def layers(self) -> tuple[CacheStore, ...]:
        return self._layers

    async def get(self, key: CacheKey) -> Mapping[str, Value] | None:
        for index, layer in enumerate(self._layers):
            hit = await layer.get(key)
            if hit is not None:
                for faster in self._layers[:index]:
                    await faster.put(key, hit)
                layer_name = getattr(layer, "cache_layer", f"layer-{index}")
                self.hits += 1
                self.hits_by_layer[layer_name] = self.hits_by_layer.get(layer_name, 0) + 1
                self._hit_layer.set(layer_name)
                return hit
        self.misses += 1
        self._hit_layer.set(None)
        return None

    async def put(self, key: CacheKey, outputs: Mapping[str, Value]) -> None:
        for layer in self._layers:
            await layer.put(key, outputs)

    def drop_referencing(self, resource_id: str) -> int:
        dropped = 0
        for layer in self._layers:
            dropper = getattr(layer, "drop_referencing", None)
            if callable(dropper):
                dropped += cast(int, dropper(resource_id))
        return dropped

    def clear(self) -> int:
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
