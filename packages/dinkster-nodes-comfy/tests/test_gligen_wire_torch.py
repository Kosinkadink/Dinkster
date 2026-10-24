from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    TypeRegistry,
    initialize_inference_process,
    register_inference_types,
)
from dinkster_caches import MemoryLRUCache
from dinkster_memory.resident import ResidentPool
from dinkster_values import (
    RESOURCE_REFS_META_KEY,
    ResidentLookupError,
    resident_owners,
    resident_resource_id,
)


def _gligen() -> Any:
    from dinkster_inference.gligen import Gligen
    from dinkster_inference.model_patcher import CoreModelPatcher

    return CoreModelPatcher(
        Gligen([], torch.nn.Identity(), key_dim=2), torch.device("cpu"), torch.device("cpu")
    )


def _conditioning(model: object, pooled: torch.Tensor) -> list[object]:
    return [[pooled, {"gligen": ("position", model, [(pooled, 5, 7, 2, 1)]), "source": "kept"}]]


@pytest.mark.parametrize("release_target", ("owner", "reference"))
def test_gligen_conditioning_retains_real_model_owners_and_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch, release_target: str
) -> None:
    initialize_inference_process()

    async def scenario() -> None:
        pool = ResidentPool(terminal_release=lambda _: True)
        monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
        registry = TypeRegistry()
        register_inference_types(registry)
        model = _gligen()
        owner, reference = object(), object()
        model._dinkster_resident_owner = owner
        model._dinkster_resident_refs = (reference,)
        pooled = torch.tensor([[1.0, 2.0]])
        raw = _conditioning(model, pooled)
        value = registry.wrap(CONDITIONING_TYPE, raw)
        carrier = value.resolve()
        assert carrier._dinkster_resident_payload is raw
        assert raw[0][1]["source"] == "kept"
        assert resident_owners(carrier) == (raw, model, owner, reference)
        assert value.meta.get(RESOURCE_REFS_META_KEY) == (
            resident_resource_id(pool.rid_for(model)),
            resident_resource_id(pool.rid_for(owner)),
            resident_resource_id(pool.rid_for(reference)),
        )

        encoded = registry.spec(CONDITIONING_TYPE).encode(carrier)
        cache = MemoryLRUCache()
        pool.register_invalidator(cache.drop_referencing)
        await cache.put("conditioning", {"value": value})
        assert await cache.get("conditioning") is not None

        target = {"owner": owner, "reference": reference}[release_target]
        assert pool.release_resident(target)
        assert await cache.get("conditioning") is None
        with pytest.raises(ResidentLookupError):
            registry.spec(CONDITIONING_TYPE).decode(encoded)

    asyncio.run(scenario())


def test_gligen_conditioning_model_release_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()

    async def scenario() -> None:
        pool = ResidentPool(terminal_release=lambda _: True)
        monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
        registry = TypeRegistry()
        register_inference_types(registry)
        model = _gligen()
        pooled = torch.tensor([[1.0, 2.0]])
        value = registry.wrap(CONDITIONING_TYPE, _conditioning(model, pooled))
        encoded = registry.spec(CONDITIONING_TYPE).encode(value.resolve())
        cache = MemoryLRUCache()
        pool.register_invalidator(cache.drop_referencing)
        await cache.put("conditioning", {"value": value})
        assert pool.release_resident(model)
        assert await cache.get("conditioning") is None
        with pytest.raises(ResidentLookupError):
            registry.spec(CONDITIONING_TYPE).decode(encoded)

    asyncio.run(scenario())


def test_gligen_conditioning_rebuilds_owners_without_stale_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()

    async def scenario() -> None:
        pool = ResidentPool(terminal_release=lambda _: True)
        monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
        registry = TypeRegistry()
        register_inference_types(registry)
        pooled = torch.tensor([[1.0, 2.0]])
        first, replacement, unrelated = _gligen(), _gligen(), _gligen()
        raw = _conditioning(first, pooled)
        first_value = registry.wrap(CONDITIONING_TYPE, raw)
        assert resident_owners(first_value.resolve()) == (raw, first)

        raw[0][1]["gligen"] = ("position", replacement, [(pooled, 5, 7, 2, 1)])
        raw.append(_conditioning(unrelated, pooled)[0])
        replacement_value = registry.wrap(CONDITIONING_TYPE, raw)
        replacement_carrier = replacement_value.resolve()
        assert resident_owners(replacement_carrier) == (raw, replacement, unrelated)
        assert resident_owners(replacement_carrier) == (raw, replacement, unrelated)

        cache = MemoryLRUCache()
        pool.register_invalidator(cache.drop_referencing)
        await cache.put("replacement", {"value": replacement_value})
        assert await cache.get("replacement") is not None
        assert pool.release_resident(first)
        assert await cache.get("replacement") is not None
        assert pool.release_resident(replacement)
        assert await cache.get("replacement") is None
        assert pool.get(pool.rid_for(unrelated)) is unrelated

        foreign = TypeRegistry()
        foreign_pool = ResidentPool()
        from dinkster_inference_wire.conditioning_wire import register_conditioning_type

        foreign_spec = register_conditioning_type(foreign, resident_table=foreign_pool)
        encoded = registry.spec(CONDITIONING_TYPE).encode(replacement_carrier)
        with pytest.raises(
            ResidentLookupError, match="another worker or an earlier worker lifetime"
        ):
            foreign_spec.decode(encoded)

    asyncio.run(scenario())
