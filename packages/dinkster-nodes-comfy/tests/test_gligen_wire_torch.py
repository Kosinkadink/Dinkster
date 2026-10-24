from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    GLIGEN_TYPE,
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
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory.resident import ResidentPool
from dinkster_values import (
    RESOURCE_REFS_META_KEY,
    ResidentLookupError,
    resident_owners,
    resident_resource_id,
)

from dinkster.compose import PackSpec, ServingComposer


def _gligen() -> Any:
    from dinkster_inference.gligen import Gligen
    from dinkster_inference.model_patcher import CoreModelPatcher

    return CoreModelPatcher(
        Gligen([], torch.nn.Identity(), key_dim=2), torch.device("cpu"), torch.device("cpu")
    )


def _conditioning(model: object, pooled: torch.Tensor) -> list[object]:
    return [[pooled, {"gligen": ("position", model, [(pooled, 5, 7, 2, 1)]), "source": "kept"}]]


def test_gligen_native_and_shared_registration_retain_first_spec() -> None:
    from dinkster_memory.resident import default_pool
    from dinkster_native.native import register_native_types

    for registrations in (
        (register_native_types, register_inference_types),
        (register_inference_types, register_native_types),
    ):
        registry = TypeRegistry()
        registrations[0](registry)
        spec = registry.spec(GLIGEN_TYPE)
        registrations[1](registry)
        assert registry.spec(GLIGEN_TYPE) is spec
        assert default_pool() in registry.resident_tables


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


@pytest.mark.parametrize("in_process", (True, False))
def test_gligen_conditioning_reaches_native_consumer_through_complete_manifest(
    tmp_path: Path, in_process: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = root / "dinkster-pack.toml"
        source_root = str(root / "src")
        monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")
        monkeypatch.syspath_prepend(source_root)
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            "[pack]\n"
            'name = "gligen-fixture"\n'
            'publisher = "dinkster"\n'
            'namespaces = ["gligen-fixture"]\n'
            "\n[pack.entry]\n"
            'nodes = "gligen_fixture:GLIGEN_FIXTURES"\n'
            'types = "gligen_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        (tmp_path / "gligen_fixture.py").write_text(
            "from test_gligen_wire_torch import GLIGEN_FIXTURES\n"
            "from dinkster_nodes_comfy import register_inference_types\n",
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), source_root)
                ),
            }
        )
        try:
            if in_process:
                await composer.add_pack(PackSpec(manifest=manifest, in_process=True))
                await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=True))
            else:
                group_manifests = (manifest, fixture_manifest)
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        worker_group="gligen-fixture",
                        group_manifests=group_manifests,
                    )
                )
                await composer.add_pack(
                    PackSpec(
                        manifest=fixture_manifest,
                        worker_group="gligen-fixture",
                        group_manifests=group_manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("gligen-fixture.source", {}),
                    "native": GraphNode(
                        "ConditioningSetAreaStrength",
                        {"conditioning": Link("source", "conditioning"), "strength": 0.25},
                    ),
                    "inspect": GraphNode(
                        "gligen-fixture.inspect",
                        {"conditioning": Link("native", "CONDITIONING")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "gligen conditioning kept"
                if attempt:
                    assert set(result.cached) == {"source", "native", "inspect"}
                else:
                    assert not result.cached
        finally:
            await composer.close()

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
