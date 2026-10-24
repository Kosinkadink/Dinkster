"""Dataset text behavior and transport through the ordinary native pack."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import CORE_STRING, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.dataset_text import (
    AddTextPrefix,
    AddTextSuffix,
    ReplaceText,
    StripWhitespace,
    TextToLowercase,
    TextToUppercase,
    TruncateText,
)

from dinkster.compose import PackSpec, ServingComposer

CASES: tuple[tuple[type[Node], dict[str, Any], str], ...] = (
    (TextToLowercase, {"texts": " A\u0130\u00dfB\n"}, " ai\u0307\u00dfb\n"),
    (TextToUppercase, {"texts": " a\u00dfb\t"}, " ASSB\t"),
    (TruncateText, {"texts": "A\U0001f600e\u0301tail", "max_length": 3}, "A\U0001f600e"),
    (AddTextPrefix, {"texts": "body", "prefix": "pre:"}, "pre:body"),
    (AddTextSuffix, {"texts": "body", "suffix": ":post"}, "body:post"),
    (ReplaceText, {"texts": "aAa a", "find": "a", "replace": "xy"}, "xyAxy xy"),
    (StripWhitespace, {"texts": " \tinner \nspace\r\n"}, "inner \nspace"),
)


@pytest.mark.parametrize(("node", "inputs", "expected"), CASES)
def test_dataset_text_preserves_reference_wrapping(
    node: type[Node], inputs: dict[str, Any], expected: str
) -> None:
    assert node.execute(**inputs) == {"texts": [expected]}


def test_dataset_text_empty_and_slice_boundaries() -> None:
    assert TruncateText.execute(texts="abc", max_length=3) == {"texts": ["abc"]}
    assert TruncateText.execute(texts="abc", max_length=9) == {"texts": ["abc"]}
    assert ReplaceText.execute(texts="ab", find="", replace="-") == {"texts": ["-a-b-"]}
    assert StripWhitespace.execute(texts="\t\n ") == {"texts": [""]}
    assert AddTextPrefix.execute(texts="", prefix="pre") == {"texts": ["pre"]}


def test_dataset_text_unwraps_singleton_parameters() -> None:
    assert TruncateText.execute(texts="abcd", max_length=[2]) == {"texts": ["ab"]}
    assert AddTextPrefix.execute(texts="body", prefix=["pre:"]) == {"texts": ["pre:body"]}
    assert AddTextSuffix.execute(texts="body", suffix=[":post"]) == {"texts": ["body:post"]}
    assert ReplaceText.execute(texts="ababa", find=["a"], replace=["X"]) == {"texts": ["XbXbX"]}
    with pytest.raises(TypeError):
        AddTextPrefix.execute(texts="body", prefix=["one", "two"])


class ObserveTextPayload(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        string = TypeExpr.concrete(CORE_STRING)
        return NodeSchema(
            node_type="dataset-text-fixture.observe",
            inputs=(InputSpec("value", string),),
            outputs=(OutputSpec("json", string),),
        )

    @classmethod
    def execute(cls, *, value: object) -> Mapping[str, object]:
        return cls.outputs(json=json.dumps(value))


@pytest.mark.parametrize("in_process", (True, False))
def test_dataset_text_ordinary_manifest_payloads(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = (
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
            )
            (tmp_path / "dataset_text_fixture.py").write_text(
                "from test_comfy_dataset_text_nodes import ObserveTextPayload\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, ObserveTextPayload)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["dataset-text-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "dataset_text_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "dataset_text_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            nodes: dict[str, GraphNode] = {}
            targets = []
            for node, inputs, _expected in CASES:
                name = node.schema().node_type
                nodes[name] = GraphNode(name, inputs)
                target = name + "-observed"
                nodes[target] = GraphNode(
                    "dataset-text-fixture.observe", {"value": Link(name, "texts")}
                )
                targets.append(target)
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(Graph(nodes=nodes), targets)
                for node, _inputs, expected in CASES:
                    observed = result.outputs[node.schema().node_type + "-observed"]["json"]
                    encoded = observed.resolve()
                    assert isinstance(encoded, str)
                    assert json.loads(encoded) == [expected]
        finally:
            await composer.close()

    asyncio.run(scenario())
