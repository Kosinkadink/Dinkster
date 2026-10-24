"""Torch-free schemas and manifest claims for custom-sampling guiders."""

import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import pytest
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    GUIDER_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.cfg_guiders import CFG_GUIDER_NODES
from dinkster_workers import load_manifest


class FixtureGuiderSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cfg-guiders-fixture.source",
            outputs=(OutputSpec("guider", TypeExpr.concrete(GUIDER_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(guider=SimpleNamespace(cfg=3.5))


class _Guider(Protocol):
    cfg: float


class FixtureGuiderConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cfg-guiders-fixture.consumer",
            inputs=(InputSpec("guider", TypeExpr.concrete(GUIDER_TYPE)),),
            outputs=(OutputSpec("cfg", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, guider: object) -> Mapping[str, object]:
        return cls.outputs(cfg=cast(_Guider, guider).cfg)


def test_cfg_guider_schemas_are_cold_and_registered_in_source_order() -> None:
    assert "torch" not in sys.modules
    expected = (
        ("CFGGuider", ("model", "positive", "negative", "cfg")),
        ("BasicGuider", ("model", "conditioning")),
        (
            "DualCFGGuider",
            ("model", "cond1", "cond2", "negative", "cfg_conds", "cfg_cond2_negative", "style"),
        ),
        ("DualModelGuider", ("model", "model_negative", "positive", "cfg", "negative")),
        ("CFGOverride", ("model", "cfg", "start_percent", "end_percent")),
    )
    assert (
        tuple(
            (node.schema().node_type, tuple(port.id for port in node.schema().inputs))
            for node in CFG_GUIDER_NODES
        )
        == expected
    )
    assert all(node in COMFY_NODES for node in CFG_GUIDER_NODES)
    for node in CFG_GUIDER_NODES[:-1]:
        assert node.schema().inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert node.schema().outputs[0].type == TypeExpr.concrete(GUIDER_TYPE)
    assert CFG_GUIDER_NODES[-1].schema().outputs[0].type == TypeExpr.concrete(MODEL_TYPE)
    assert CFG_GUIDER_NODES[0].schema().inputs[1].type == TypeExpr.concrete(CONDITIONING_TYPE)


def test_cfg_guider_manifest_claims_match_the_complete_registry() -> None:
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert tuple(node.schema().node_type for node in COMFY_NODES) == manifest.namespaces


@pytest.mark.parametrize("in_process", (True, False))
def test_complete_manifest_transports_guider_to_fixture_consumer_with_cache(
    tmp_path: Path, in_process: bool
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        manifest = Path("packages/dinkster-nodes-comfy/dinkster-pack.toml")
        (tmp_path / "cfg_guiders_fixture.py").write_text(
            "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
            "from test_comfy_cfg_guiders_nodes import FixtureGuiderConsumer, FixtureGuiderSource\n"
            "NODES = (*COMFY_NODES, FixtureGuiderSource, FixtureGuiderConsumer)\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            manifest.read_text(encoding="ascii")
            .replace("namespaces = [", 'namespaces = ["cfg-guiders-fixture",')
            .replace("dinkster_nodes_comfy:COMFY_NODES", "cfg_guiders_fixture:NODES")
            .replace(
                "dinkster_nodes_comfy:register_inference_types",
                "cfg_guiders_fixture:register_inference_types",
            ),
            encoding="ascii",
        )
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("cfg-guiders-fixture.source"),
                    "consumer": GraphNode(
                        "cfg-guiders-fixture.consumer", {"guider": Link("source", "guider")}
                    ),
                }
            )
            for run in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["cfg"].resolve() == 3.5
                if run:
                    assert {"source", "consumer"} <= set(result.cached)
        finally:
            await composer.close()

    asyncio.run(scenario())
