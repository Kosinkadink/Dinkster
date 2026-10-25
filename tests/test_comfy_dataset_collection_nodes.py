"""Native dataset collection schemas and ordinary pack transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    schema_to_wire,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.dataset_collections import DATASET_COLLECTION_NODES


def test_dataset_collection_schemas_match_the_fixed_reference() -> None:
    expected = (
        ("ShuffleDataset", "Shuffle Images List", ("images", "seed"), ("images",)),
        (
            "ShuffleImageTextDataset",
            "Shuffle Pairs of Image-Text",
            ("images", "texts", "seed"),
            ("images", "texts"),
        ),
        (
            "ImageDeduplication",
            "Deduplicate Images",
            ("images", "similarity_threshold"),
            ("images",),
        ),
        (
            "ImageGrid",
            "Make Image Grid",
            ("images", "columns", "cell_width", "cell_height", "padding"),
            ("images",),
        ),
        ("MergeTextLists", "Merge Text Lists (DEPRECATED)", ("texts",), ("texts",)),
    )
    actual = tuple(
        (
            node.schema().node_type,
            node.schema().display_name,
            tuple(input_spec.id for input_spec in node.schema().inputs),
            tuple(output_spec.id for output_spec in node.schema().outputs),
        )
        for node in DATASET_COLLECTION_NODES
    )
    assert actual == expected

    expected_input_kinds = {
        "ShuffleDataset": {"images": "list", "seed": "concrete"},
        "ShuffleImageTextDataset": {"images": "list", "texts": "list", "seed": "concrete"},
        "ImageDeduplication": {"images": "list", "similarity_threshold": "concrete"},
        "ImageGrid": {
            "images": "list",
            "columns": "concrete",
            "cell_width": "concrete",
            "cell_height": "concrete",
            "padding": "concrete",
        },
        "MergeTextLists": {"texts": "list"},
    }
    for node in DATASET_COLLECTION_NODES:
        wire = cast(dict[str, Any], schema_to_wire(node.schema()))
        input_kinds = {
            item["id"]: item["type"]["kind"]
            for item in wire["interface"]
            if item["role"] == "input"
        }
        assert input_kinds == expected_input_kinds[node.schema().node_type]
    shuffle = cast(dict[str, Any], schema_to_wire(DATASET_COLLECTION_NODES[0].schema()))
    seed = next(item for item in shuffle["interface"] if item["id"] == "seed")
    assert seed["default"] == 0
    assert seed["widget"] == {"type": "NUMBER", "min": 0, "max": str(0xFFFFFFFFFFFFFFFF)}
    grid = cast(dict[str, Any], schema_to_wire(DATASET_COLLECTION_NODES[3].schema()))
    assert next(item for item in grid["interface"] if item["id"] == "columns")["default"] == 4
    assert next(item for item in grid["interface"] if item["id"] == "cell_width")["default"] == 256
    assert next(item for item in grid["interface"] if item["id"] == "cell_height")["default"] == 256
    assert next(item for item in grid["interface"] if item["id"] == "padding")["default"] == 4


class ObserveTextPayload(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        string = TypeExpr.concrete(CORE_STRING)
        return NodeSchema(
            node_type="dataset-collections-fixture.observe",
            inputs=(InputSpec("value", string),),
            outputs=(OutputSpec("json", string),),
        )

    @classmethod
    def execute(cls, *, value: object) -> Mapping[str, object]:
        return cls.outputs(json=json.dumps(value))


def test_merge_text_lists_runs_repeatedly_through_both_ordinary_pack_hosts(
    tmp_path: Path,
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        manifest = (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        )
        (tmp_path / "dataset_collections_fixture.py").write_text(
            "from test_comfy_dataset_collection_nodes import ObserveTextPayload\n"
            "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
            "NODES = (*COMFY_NODES, ObserveTextPayload)\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            manifest.read_text(encoding="ascii")
            .replace("namespaces = [", 'namespaces = ["dataset-collections-fixture",')
            .replace("dinkster_nodes_comfy:COMFY_NODES", "dataset_collections_fixture:NODES")
            .replace(
                "dinkster_nodes_comfy:register_inference_types",
                "dataset_collections_fixture:register_inference_types",
            ),
            encoding="ascii",
        )
        for in_process in (True, False):
            composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
            try:
                await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
                engine = composer.composition.make_engine(lambda _event: None)
                graph = Graph(
                    nodes={
                        "merge": GraphNode("MergeTextLists", {"texts": ["one", "two"]}),
                        "observe": GraphNode(
                            "dataset-collections-fixture.observe", {"value": Link("merge", "texts")}
                        ),
                    }
                )
                for _ in range(2):
                    result = await engine.run(graph, ["observe"])
                    observed = result.outputs["observe"]["json"].resolve()
                    assert isinstance(observed, str)
                    assert json.loads(observed) == [["one", "two"]]
            finally:
                await composer.close()

    asyncio.run(scenario())
