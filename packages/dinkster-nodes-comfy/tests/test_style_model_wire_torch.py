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

from dinkster.compose import PackSpec, ServingComposer


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


@pytest.mark.parametrize("in_process", (True, False))
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
