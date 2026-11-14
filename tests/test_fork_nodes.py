from __future__ import annotations

import pytest
from dinkster_schema import TypeExpr
from dinkster_values import ResidentLookupError, TypeRegistry

from tests.async_scenarios import run_scenario

CONDITIONING = TypeExpr.concrete("CONDITIONING")
INT = TypeExpr.concrete("INT")


def test_hook_conditioning_release_invalidates_only_dependent_values() -> None:
    from dinkster_caches import MemoryLRUCache
    from dinkster_inference_wire.conditioning_wire import register_conditioning_type
    from dinkster_memory.resident import ResidentPool
    from dinkster_protocol import CacheKey
    from dinkster_values import RESOURCE_REFS_META_KEY, ResidentCodec, resident_resource_id

    async def scenario() -> None:
        pool = ResidentPool(terminal_release=lambda _: True)
        registry = TypeRegistry()
        spec = register_conditioning_type(registry, resident_table=pool)
        cache = MemoryLRUCache(max_bytes=1024)
        pool.register_invalidator(cache.drop_referencing)
        first_hook, replacement_hook = object(), object()
        embedding = object()
        first = [[embedding, {"hooks": first_hook, "strength": 0.3}]]
        replacement = [[embedding, {"hooks": replacement_hook, "strength": 1.7}]]
        plain = [[embedding, {"strength": 0.6}]]
        first_value = registry.wrap("CONDITIONING", first)
        replacement_value = registry.wrap("CONDITIONING", replacement)
        plain_value = registry.wrap("CONDITIONING", plain)
        assert first_value.fingerprint == registry.wrap("CONDITIONING", first).fingerprint
        assert first_value.fingerprint != replacement_value.fingerprint
        assert first_value.meta.get(RESOURCE_REFS_META_KEY) == (
            resident_resource_id(pool.rid_for(first_hook)),
        )
        first_bytes = spec.encode(first_value.resolve())
        replacement_bytes = spec.encode(replacement_value.resolve())
        assert registry.input_object("CONDITIONING", spec.decode(first_bytes)) is first
        assert registry.input_object("CONDITIONING", spec.decode(replacement_bytes)) is replacement
        assert plain_value.resolve() is plain
        assert spec.encode(plain) == b"DMFR\x01" + ResidentCodec(pool).encode(plain)
        assert spec.decode(spec.encode(plain)) is plain
        first_key, replacement_key = CacheKey("first"), CacheKey("replacement")
        await cache.put(first_key, {"conditioning": first_value})
        await cache.put(replacement_key, {"conditioning": replacement_value})
        assert await cache.get(first_key) is not None
        assert pool.release_resident(first_hook)
        assert await cache.get(first_key) is None
        assert await cache.get(replacement_key) is not None
        with pytest.raises(ResidentLookupError):
            spec.decode(first_bytes)
        assert registry.input_object("CONDITIONING", spec.decode(replacement_bytes)) is replacement
        foreign = TypeRegistry()
        foreign_spec = register_conditioning_type(foreign, resident_table=ResidentPool())
        with pytest.raises(ResidentLookupError):
            foreign_spec.decode(replacement_bytes)
        assert first[0][1] == {"hooks": first_hook, "strength": 0.3}
        assert replacement[0][1] == {"hooks": replacement_hook, "strength": 1.7}
        assert pool.release_resident(replacement_hook)
        assert await cache.get(replacement_key) is None
        with pytest.raises(ResidentLookupError):
            spec.decode(replacement_bytes)

    run_scenario(scenario())
