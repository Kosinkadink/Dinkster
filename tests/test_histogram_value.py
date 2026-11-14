"""Cold structural HISTOGRAM values retain their exact integer bins."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

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

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario

# Hand-derived synthetic contract input; not a ComfyUI source fixture.
HISTOGRAM = [0, -7, 2**63, -(2**63), 13]


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

    run_scenario(scenario())
