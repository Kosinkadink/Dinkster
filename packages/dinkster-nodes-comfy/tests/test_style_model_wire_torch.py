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
    MODEL_PATCH_TYPE,
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


def _model_patch() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    projection = torch.nn.Linear(2, 1)
    projection.weight = torch.nn.Parameter(torch.tensor([[2.0, -1.0]]))
    projection.bias = torch.nn.Parameter(torch.tensor([0.5]))
    return ModelPatcher(projection, torch.device("cpu"), torch.device("cpu"))


def test_actual_model_patch_identity_cache_and_independent_release() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_inference_types(registry)
        spec = registry.spec(MODEL_PATCH_TYPE)
        source = _model_patch()
        unrelated = _model_patch()
        owner, reference = object(), object()
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
        assert metadata[RESOURCE_ID_META_KEY] == resident_resource_id(default_pool().rid_for(owner))
        assert metadata[RESOURCE_REFS_META_KEY] == (
            resident_resource_id(default_pool().rid_for(reference)),
        )
        assert metadata[RESOURCE_OWNER_META_KEY] == process_instance_token()
        assert set(json.loads(encoded)) == {"residentId"}
        foreign = register_resident_type(TypeRegistry(), MODEL_PATCH_TYPE, table=ResidentPool())
        with pytest.raises(
            ResidentLookupError, match="another worker or an earlier worker lifetime"
        ):
            foreign.decode(encoded)
        cache = MemoryLRUCache()
        default_pool().register_invalidator(cache.drop_referencing)
        await cache.put("patch", {"patch": registry.wrap(MODEL_PATCH_TYPE, source)})
        await cache.put("unrelated", {"patch": registry.wrap(MODEL_PATCH_TYPE, unrelated)})
        assert await cache.get("patch") is not None
        assert await cache.get("unrelated") is not None
        assert default_pool().release_resident(reference)
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
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="model-patch-source.source",
            outputs=(OutputSpec("model_patch", TypeExpr.concrete(MODEL_PATCH_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model_patch=_model_patch())


class ModelPatchConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="model-patch-consumer.consume",
            inputs=(InputSpec("model_patch", TypeExpr.concrete(MODEL_PATCH_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model_patch: Any) -> dict[str, object]:
        assert (
            type(model_patch)
            is importlib.import_module("dinkster_inference.model_patcher").ModelPatcher
        )
        output = model_patch.model(torch.tensor([[2.0, 3.0]]))
        return cls.outputs(summary=json.dumps(output.tolist()))


@pytest.mark.parametrize("in_process", (True, False))
def test_model_patch_ordinary_manifest_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={"PYTHONPATH": str(Path(__file__).parent), "CUDA_VISIBLE_DEVICES": ""}
        )
        try:
            module = "model_patch_fixture_" + str(in_process)
            (tmp_path / f"{module}.py").write_text(
                "from test_style_model_wire_torch import ModelPatchConsumer, ModelPatchSource\n"
                "from dinkster_api.v1 import register_inference_types\n"
                "NODES = [ModelPatchSource, ModelPatchConsumer]\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "model-patch-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["model-patch-source", "model-patch-consumer"]\n'
                f'[pack.entry]\nnodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("model-patch-source.source", {}),
                    "consumer": GraphNode(
                        "model-patch-consumer.consume",
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
