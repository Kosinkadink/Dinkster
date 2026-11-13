from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    initialize_inference_process,
    register_inference_types,
)
from dinkster_caches import MemoryLRUCache
from dinkster_memory.resident import ResidentPool
from dinkster_values import (
    ResidentLookupError,
    resident_owners,
)


def _gligen() -> Any:
    from dinkster_inference.gligen import Gligen
    from dinkster_inference.model_patcher import CoreModelPatcher

    return CoreModelPatcher(
        Gligen([], torch.nn.Identity(), key_dim=2), torch.device("cpu"), torch.device("cpu")
    )


def _conditioning(model: object, pooled: torch.Tensor) -> list[object]:
    return [[pooled, {"gligen": ("position", model, [(pooled, 5, 7, 2, 1)]), "source": "kept"}]]


class GligenConditioningSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="gligen-fixture.source",
            outputs=(OutputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        model = _gligen()
        model._dinkster_resident_owner = object()
        model._dinkster_resident_refs = (object(),)
        pooled = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        return cls.outputs(conditioning=_conditioning(model, pooled))


class GligenConditioningInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="gligen-fixture.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: list[list[Any]]) -> dict[str, object]:
        from dinkster_inference.gligen import Gligen
        from dinkster_inference.model_patcher import CoreModelPatcher

        assert len(conditioning) == 1
        pooled, metadata = conditioning[0]
        position, model, areas = metadata["gligen"]
        assert position == "position"
        assert type(model) is CoreModelPatcher
        assert type(model.model) is Gligen
        assert len(resident_owners(model)) == 2
        assert torch.equal(pooled, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert len(areas) == 1
        area_pooled, width, height, x, y = areas[0]
        assert area_pooled is pooled
        assert (width, height, x, y) == (5, 7, 2, 1)
        assert metadata["source"] == "kept"
        assert metadata["strength"] == 0.25
        return cls.outputs(summary="gligen conditioning kept")


GLIGEN_FIXTURES = (GligenConditioningSource, GligenConditioningInspect)


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
