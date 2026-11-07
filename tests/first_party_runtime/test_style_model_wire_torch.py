from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_UPSCALE_MODEL_TYPE,
    MODEL_PATCH_TYPE,
    STYLE_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    initialize_inference_process,
    materialize_clip_vision_output,
    register_inference_types,
)
from dinkster_caches import MemoryLRUCache
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory.resident import ResidentPool, default_pool
from dinkster_values import (
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    RESOURCE_REFS_META_KEY,
    ResidentLookupError,
    process_instance_token,
    register_resident_type,
    resident_resource_id,
)

from dinkster.compose import PackSpec, ServingComposer, default_pack_spec


def _model() -> Any:
    initialize_inference_process()
    from dinkster_inference.sd import StyleModel

    projection = torch.nn.Linear(3, 2)
    projection.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0]]))
    projection.bias = torch.nn.Parameter(torch.tensor([0.75, -0.5]))
    return StyleModel(projection)


@pytest.mark.parametrize("release_target", ("self", "owner", "reference"))
def test_actual_style_model_identity_cache_and_release(release_target: str) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_inference_types(registry)
        spec = registry.spec(STYLE_MODEL_TYPE)
        source = _model()
        owner, reference = object(), object()
        if release_target != "self":
            source._dinkster_resident_owner = owner
            source._dinkster_resident_refs = (reference,)
        assert type(source) is importlib.import_module("dinkster_inference.sd").StyleModel
        encoded = spec.encode(source)
        assert spec.decode(encoded) is source
        assert spec.meta is not None
        metadata = spec.meta(source)
        assert spec.fingerprint is not None
        assert spec.fingerprint(source) == resident_resource_id(default_pool().rid_for(source))
        if release_target == "self":
            assert spec.fingerprint(source) == metadata[RESOURCE_ID_META_KEY]
        else:
            assert metadata[RESOURCE_ID_META_KEY] == resident_resource_id(
                default_pool().rid_for(owner)
            )
            assert metadata[RESOURCE_REFS_META_KEY] == (
                resident_resource_id(default_pool().rid_for(reference)),
            )
        assert metadata[RESOURCE_OWNER_META_KEY] == process_instance_token()
        assert set(json.loads(encoded)) == {"residentId"}
        foreign = register_resident_type(TypeRegistry(), STYLE_MODEL_TYPE, table=ResidentPool())
        with pytest.raises(
            ResidentLookupError, match="another worker or an earlier worker lifetime"
        ):
            foreign.decode(encoded)
        cache = MemoryLRUCache()
        default_pool().register_invalidator(cache.drop_referencing)
        await cache.put("style", {"style": registry.wrap(STYLE_MODEL_TYPE, source)})
        assert await cache.get("style") is not None
        released = {"self": source, "owner": owner, "reference": reference}[release_target]
        assert default_pool().release_resident(released)
        assert await cache.get("style") is None
        with pytest.raises(
            ResidentLookupError, match="another worker or an earlier worker lifetime"
        ):
            spec.decode(encoded)

    asyncio.run(scenario())


class StyleSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="style-fixture.source",
            outputs=(OutputSpec("style", TypeExpr.concrete(STYLE_MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(style=_model())


class StyleConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="style-fixture.consume",
            inputs=(InputSpec("style", TypeExpr.concrete(STYLE_MODEL_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, style: Any) -> dict[str, object]:
        assert type(style) is importlib.import_module("dinkster_inference.sd").StyleModel
        features = torch.tensor([[[2.0, 3.0, -1.0], [-4.0, 0.5, 5.0]]])
        before = features.clone()
        weights = style.model.weight.clone()
        vision = materialize_clip_vision_output({"last_hidden_state": features})
        output = style.get_cond(vision)
        assert torch.equal(features, before)
        assert torch.equal(style.model.weight, weights)
        return cls.outputs(summary=json.dumps(output.tolist()))


@pytest.mark.parametrize("in_process", (True,))
def test_style_model_ordinary_manifest_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={"PYTHONPATH": str(Path(__file__).parent), "CUDA_VISIBLE_DEVICES": ""}
        )
        try:
            (tmp_path / "style_fixture.py").write_text(
                "from test_style_model_wire_torch import StyleSource, StyleConsumer\n"
                "from dinkster_api.v1 import register_inference_types\n"
                "NODES = [StyleSource, StyleConsumer]\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                '[pack]\nname = "style-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["style-fixture"]\n'
                '[pack.entry]\nnodes = "style_fixture:NODES"\n'
                'types = "style_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("style-fixture.source", {}),
                    "consumer": GraphNode(
                        "style-fixture.consume", {"style": Link("source", "style")}
                    ),
                }
            )
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert json.loads(result.outputs["consumer"]["summary"].resolve()) == [
                    [[-3.75, -3.75], [-1.75, 13.625]]
                ]
                assert set(result.cached) == ({"source", "consumer"} if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())


def _model_patch() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    projection = torch.nn.Linear(2, 1)
    projection.weight = torch.nn.Parameter(torch.tensor([[2.0, -1.0]]))
    projection.bias = torch.nn.Parameter(torch.tensor([0.5]))
    return ModelPatcher(projection, torch.device("cpu"), torch.device("cpu"))


@pytest.mark.parametrize("value_type", (MODEL_PATCH_TYPE, LATENT_UPSCALE_MODEL_TYPE))
@pytest.mark.parametrize("release_target", ("self", "owner", "reference"))
def test_actual_model_patch_identity_cache_and_independent_release(
    value_type: str, release_target: str
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_inference_types(registry)
        spec = registry.spec(value_type)
        source = _model_patch()
        unrelated = _model_patch()
        owner, reference = object(), object()
        if release_target != "self":
            source._dinkster_resident_owner = owner
            source._dinkster_resident_refs = (reference,)
        assert (
            type(source) is importlib.import_module("dinkster_inference.model_patcher").ModelPatcher
        )
        encoded = spec.encode(source)
        assert spec.decode(encoded) is source
        assert spec.meta is not None
        metadata = spec.meta(source)
        assert spec.fingerprint is not None
        assert spec.fingerprint(source) == resident_resource_id(default_pool().rid_for(source))
        if release_target == "self":
            assert metadata[RESOURCE_ID_META_KEY] == spec.fingerprint(source)
        else:
            assert metadata[RESOURCE_ID_META_KEY] == resident_resource_id(
                default_pool().rid_for(owner)
            )
            assert metadata[RESOURCE_REFS_META_KEY] == (
                resident_resource_id(default_pool().rid_for(reference)),
            )
        assert metadata[RESOURCE_OWNER_META_KEY] == process_instance_token()
        assert set(json.loads(encoded)) == {"residentId"}
        foreign = register_resident_type(TypeRegistry(), value_type, table=ResidentPool())
        with pytest.raises(
            ResidentLookupError, match="another worker or an earlier worker lifetime"
        ):
            foreign.decode(encoded)
        cache = MemoryLRUCache()
        default_pool().register_invalidator(cache.drop_referencing)
        await cache.put("patch", {"patch": registry.wrap(value_type, source)})
        await cache.put("unrelated", {"patch": registry.wrap(value_type, unrelated)})
        assert await cache.get("patch") is not None
        assert await cache.get("unrelated") is not None
        released = {"self": source, "owner": owner, "reference": reference}[release_target]
        assert default_pool().release_resident(released)
        assert await cache.get("patch") is None
        assert await cache.get("unrelated") is not None
        with pytest.raises(
            ResidentLookupError, match="another worker or an earlier worker lifetime"
        ):
            spec.decode(encoded)
        unrelated_encoded = spec.encode(unrelated)
        assert default_pool().release_resident(unrelated)
        assert await cache.get("unrelated") is None
        with pytest.raises(
            ResidentLookupError, match="another worker or an earlier worker lifetime"
        ):
            spec.decode(unrelated_encoded)

    asyncio.run(scenario())


class ModelPatchSource(Node):
    value_type = MODEL_PATCH_TYPE

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="model-patch-fixture.source",
            outputs=(OutputSpec("model_patch", TypeExpr.concrete(cls.value_type)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model_patch=_model_patch())


class ModelPatchConsumer(Node):
    value_type = MODEL_PATCH_TYPE

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="model-patch-fixture.consume",
            inputs=(InputSpec("model_patch", TypeExpr.concrete(cls.value_type)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model_patch: Any) -> dict[str, object]:
        assert (
            type(model_patch)
            is importlib.import_module("dinkster_inference.model_patcher").ModelPatcher
        )
        return cls.outputs(
            summary=json.dumps(model_patch.model(torch.tensor([[2.0, 3.0]])).tolist())
        )


@pytest.mark.parametrize(
    "value_type,in_process",
    (
        (MODEL_PATCH_TYPE, True),
        (LATENT_UPSCALE_MODEL_TYPE, True),
        (LATENT_UPSCALE_MODEL_TYPE, False),
    ),
)
def test_model_patch_full_native_manifest_and_fixture_hosts(
    tmp_path: Path, in_process: bool, value_type: str
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={"PYTHONPATH": str(Path(__file__).parent), "CUDA_VISIBLE_DEVICES": ""}
        )
        try:
            module = "model_patch_fixture_" + str(in_process) + "_" + value_type.lower()
            (tmp_path / f"{module}.py").write_text(
                "from test_style_model_wire_torch import ModelPatchConsumer, ModelPatchSource\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "class Source(ModelPatchSource):\n"
                f"    value_type = {value_type!r}\n"
                "class Consumer(ModelPatchConsumer):\n"
                f"    value_type = {value_type!r}\n"
                "NODES = [Source, Consumer]\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "model-patch-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["model-patch-fixture"]\n'
                f'[pack.entry]\nnodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            packages = Path(__file__).resolve().parents[2]
            generation = default_pack_spec("dinkster-nodes-generation")
            native_manifest = packages / "dinkster-native" / "dinkster-pack.toml"
            if in_process:
                await composer.add_pack(generation)
                await composer.add_pack(PackSpec(manifest=native_manifest, trust_reserved=True))
                await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=True))
            else:
                group_manifests = (Path(generation.manifest), native_manifest, fixture_manifest)
                await composer.add_pack(
                    PackSpec(
                        manifest=generation.manifest,
                        packs=generation.packs,
                        trust_reserved=True,
                        worker_group="model-patch-fixture",
                        group_manifests=group_manifests,
                    )
                )
                await composer.add_pack(
                    PackSpec(
                        manifest=native_manifest,
                        trust_reserved=True,
                        worker_group="model-patch-fixture",
                        group_manifests=group_manifests,
                    )
                )
                await composer.add_pack(
                    PackSpec(
                        manifest=fixture_manifest,
                        worker_group="model-patch-fixture",
                        group_manifests=group_manifests,
                    )
                )
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("model-patch-fixture.source", {}),
                    "consumer": GraphNode(
                        "model-patch-fixture.consume",
                        {"model_patch": Link("source", "model_patch")},
                    ),
                }
            )
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert json.loads(result.outputs["consumer"]["summary"].resolve()) == [[1.5]]
                assert set(result.cached) == ({"source", "consumer"} if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
