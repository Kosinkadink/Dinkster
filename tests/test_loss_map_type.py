"""LOSS_MAP uses the shared structural framing without numeric normalization."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    LOSS_MAP_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import encode_latent

from dinkster.compose import PackSpec, ServingComposer

LOSS_MAP = TypeExpr.concrete(LOSS_MAP_TYPE)
STRING = TypeExpr.concrete(CORE_STRING)


class LossMapSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="loss-map-source.produce",
            inputs=(),
            outputs=(OutputSpec("loss_map", LOSS_MAP),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(loss_map={"loss": [3.25, -1.5, 0.0], "epoch": 7, "tag": "L1"})


class LossMapConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="loss-map-consumer.consume",
            inputs=(InputSpec("loss_map", LOSS_MAP),),
            outputs=(OutputSpec("summary", STRING),),
        )

    @classmethod
    def execute(cls, *, loss_map: object) -> Mapping[str, object]:
        value = loss_map
        if type(value) is not dict:
            raise TypeError("LOSS_MAP consumer requires a mapping")
        mapping = value
        return cls.outputs(summary=f"{mapping['loss']!r};{mapping['epoch']!r};{mapping['tag']!r}")


def test_loss_map_preserves_ordered_losses_and_structural_extras() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LOSS_MAP_TYPE)
    value = {"loss": [3.25, -1.5, 0.0], "epoch": 7, "details": {"name": "L1"}}

    encoded = spec.encode(value)
    assert spec.decode(encoded) == value
    assert spec.fingerprint is not None
    latent_fingerprint = registry.spec(LATENT_TYPE).fingerprint
    assert latent_fingerprint is not None
    assert spec.fingerprint(value) == spec.fingerprint(spec.decode(encoded))
    assert spec.fingerprint(value) != latent_fingerprint({"samples": value})


def test_loss_map_accepts_empty_losses_and_rejects_invalid_frames() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LOSS_MAP_TYPE)

    assert spec.decode(spec.encode({"loss": [], "tag": ["kept"]})) == {"loss": [], "tag": ["kept"]}
    with pytest.raises(ValueError, match="ordered list"):
        spec.decode(encode_latent({"samples": {"loss": (1.0,)}}))
    with pytest.raises(ValueError, match="only samples"):
        spec.decode(encode_latent({"samples": {"loss": []}, "other": {}}))


def test_loss_map_crosses_distinct_isolated_workers_and_caches(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            module = "loss_map_fixture"
            (tmp_path / f"{module}.py").write_text(
                "from test_loss_map_type import LossMapConsumer, LossMapSource\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "SOURCE_NODES = [LossMapSource]\n"
                "CONSUMER_NODES = [LossMapConsumer]\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            consumer_manifest = tmp_path / "consumer-dinkster-pack.toml"
            source_manifest.write_text(
                '[pack]\nname = "loss-map-source"\npublisher = "dinkster"\n'
                'namespaces = ["loss-map-source"]\n[pack.entry]\n'
                f'nodes = "{module}:SOURCE_NODES"\ntypes = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            consumer_manifest.write_text(
                '[pack]\nname = "loss-map-consumer"\npublisher = "dinkster"\n'
                'namespaces = ["loss-map-consumer"]\n[pack.entry]\n'
                f'nodes = "{module}:CONSUMER_NODES"\ntypes = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=False))
            await composer.add_pack(PackSpec(manifest=consumer_manifest, in_process=False))
            graph = Graph(
                nodes={
                    "source": GraphNode("loss-map-source.produce", {}),
                    "consumer": GraphNode(
                        "loss-map-consumer.consume", {"loss_map": Link("source", "loss_map")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "[3.25, -1.5, 0.0];7;'L1'"
                if attempt:
                    assert set(result.cached) == {"source", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
