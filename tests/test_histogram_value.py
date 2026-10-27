"""Cold structural HISTOGRAM values retain their exact integer bins."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    HISTOGRAM_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import (
    TypeRegistry,
    register_core_types,
    register_histogram_type,
)
from dinkster_values.builtin_types import HISTOGRAM_TYPE as CANONICAL_HISTOGRAM_TYPE

from dinkster.compose import PackSpec, ServingComposer

HISTOGRAM = [0, -7, 2**63, -(2**63), 13]


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def test_histogram_cold_builtin_identity_and_first_authority() -> None:
    assert HISTOGRAM_TYPE == "HISTOGRAM"
    assert HISTOGRAM_TYPE is CANONICAL_HISTOGRAM_TYPE
    registry = _registry()
    spec = registry.spec(HISTOGRAM_TYPE)
    assert register_histogram_type(registry) is spec

    first = TypeRegistry()
    authority = first.register(
        HISTOGRAM_TYPE, encode=lambda _value: b"authority", decode=lambda _data: []
    )
    assert register_histogram_type(first) is authority


@pytest.mark.parametrize("value", ([], [0], HISTOGRAM))
def test_histogram_roundtrip_preserves_order_and_integer_contents(value: list[int]) -> None:
    registry = _registry()
    spec = registry.spec(HISTOGRAM_TYPE)
    payload = spec.encode(registry.wrap(HISTOGRAM_TYPE, value).resolve())

    assert spec.decode(payload) == value


@pytest.mark.parametrize(
    "value",
    ((), [True], [1.0], ["1"], {"bins": [1]}, [[1]], [None]),
)
def test_histogram_rejects_non_integer_list_shapes(value: object) -> None:
    registry = _registry()
    with pytest.raises((TypeError, ValueError)):
        registry.wrap(HISTOGRAM_TYPE, value)


@pytest.mark.parametrize("payload", (b"json:true", b"json:[true]", b"json:[1.5]", b"pickle:bins"))
def test_histogram_rejects_malformed_encoded_shapes(payload: bytes) -> None:
    spec = _registry().spec(HISTOGRAM_TYPE)
    assert spec.validate_encoded is not None
    with pytest.raises((TypeError, ValueError)):
        spec.validate_encoded(payload, {})


def test_histogram_identity_is_type_separated() -> None:
    registry = _registry()
    histogram = registry.wrap(HISTOGRAM_TYPE, [1, 2])
    array = registry.wrap("ARRAY", [1, 2])

    assert histogram.fingerprint != array.fingerprint


class HistogramProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="histogram-value-fixture.producer",
            outputs=(OutputSpec("histogram", TypeExpr.concrete(HISTOGRAM_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(histogram=HISTOGRAM)


class HistogramConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="histogram-value-transport.consumer",
            inputs=(InputSpec("histogram", TypeExpr.concrete(HISTOGRAM_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, histogram: object) -> Mapping[str, object]:
        assert histogram == HISTOGRAM
        return cls.outputs(result="histogram")


def test_histogram_crosses_distinct_isolated_workers_with_cached_replay(tmp_path: Path) -> None:
    async def scenario() -> None:
        test_directory = Path(__file__).parent
        modules = {
            "histogram_producer": ("HistogramProducer", "histogram-value-fixture"),
            "histogram_consumer": ("HistogramConsumer", "histogram-value-transport"),
        }
        manifests: dict[str, Path] = {}
        for module, (node, namespace) in modules.items():
            (tmp_path / f"{module}.py").write_text(
                f"from test_histogram_value import {node}\n"
                "from dinkster_values import register_histogram_type\n"
                f"NODES = ({node},)\n",
                encoding="ascii",
            )
            manifest = tmp_path / f"{module}-pack.toml"
            manifest.write_text(
                f'[pack]\nname="{module}"\npublisher="dinkster"\n'
                f'namespaces=["{namespace}"]\n[pack.entry]\n'
                f'nodes="{module}:NODES"\ntypes="{module}:register_histogram_type"\n',
                encoding="ascii",
            )
            manifests[module] = manifest
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(test_directory))),
            },
        )
        try:
            for module in modules:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifests[module],
                        worker_group=module,
                        group_manifests=(manifests[module],),
                    )
                )
            graph = Graph(
                nodes={
                    "producer": GraphNode("histogram-value-fixture.producer"),
                    "consumer": GraphNode(
                        "histogram-value-transport.consumer",
                        {"histogram": Link("producer", "histogram")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["consumer"])
            assert first.outputs["consumer"]["result"].resolve() == "histogram"
            assert set(first.executed) == {"producer", "consumer"}
            replay = await engine.run(graph, ["consumer"])
            assert replay.outputs["consumer"]["result"].resolve() == "histogram"
            assert set(replay.cached) == {"producer", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
