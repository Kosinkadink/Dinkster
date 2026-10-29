"""Public Gaussian splat inspection contracts without Torch execution."""

import tomllib
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    SPLAT_TYPE,
    InputSpec,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_engine import ParentGraphBudget, admit_parent_graph
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire, validate
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.gaussian_splat import GetSplatCount, MergeSplat, TransformSplat
from dinkster_schema import elaborate


def test_splat_count_schema_and_production_claim_are_canonical() -> None:
    schema = GetSplatCount.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema.node_type == "GetSplatCount" and schema.category == "3d/splat"
    assert [(port.id, port.type.types, port.required) for port in schema.inputs] == [
        ("splat", (SPLAT_TYPE,), True)
    ]
    assert [(port.id, port.type.types) for port in schema.outputs] == [
        ("splat", (SPLAT_TYPE,)),
        ("count", (CORE_INT,)),
    ]
    assert sum(node is GetSplatCount for node in COMFY_NODES) == 1
    manifest = (
        Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    )
    assert tomllib.loads(manifest.read_text())["pack"]["namespaces"].count("GetSplatCount") == 1


def test_transform_splat_public_schema_and_production_claim() -> None:
    schema = TransformSplat.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema.node_type == "TransformSplat" and schema.category == "3d/splat"
    assert schema.inputs[0].type.types == (SPLAT_TYPE,) and schema.inputs[0].required
    assert [(port.id, port.default, port.type.types) for port in schema.inputs[1:]] == [
        (f"{operation}_{axis}", default, (CORE_FLOAT,))
        for operation, default in (("translate", 0.0), ("rotate", 0.0), ("scale", 1.0))
        for axis in "xyz"
    ]
    assert [(port.id, port.type.types) for port in schema.outputs] == [("splat", (SPLAT_TYPE,))]
    assert sum(node is TransformSplat for node in COMFY_NODES) == 1
    manifest = (
        Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    )
    assert tomllib.loads(manifest.read_text())["pack"]["namespaces"].count("TransformSplat") == 1


@pytest.mark.parametrize("members", [("splat0", "splat1"), ("splat10", "splat2")])
def test_merge_splat_required_initial_slots_and_optional_authored_tail(
    members: tuple[str, ...],
) -> None:
    schema = schema_from_wire(schema_to_wire(MergeSplat.schema()))
    family = schema.input_families[0]
    assert (family.id, family.min_members, family.max_members, family.member_prefix) == (
        "splats",
        2,
        32,
        "splat",
    )
    assert isinstance(family.template[0], InputSpec)
    assert family.required and family.template[0].required and family.optional_after_minimum
    assert family.template[0].type == TypeExpr.concrete(SPLAT_TYPE)
    assert [(port.id, port.type.types) for port in schema.outputs] == [("splat", (SPLAT_TYPE,))]
    assert sum(node is MergeSplat for node in COMFY_NODES) == 1
    manifest = (
        Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    )
    assert tomllib.loads(manifest.read_text())["pack"]["namespaces"].count("MergeSplat") == 1
    effective = elaborate(schema, tuple(f"splats.{member}" for member in members))
    assert [(port.id, port.required) for port in effective.inputs] == [
        (f"splats.{member}", member in ("splat0", "splat1"))
        for member in sorted({"splat0", "splat1", *members}, key=lambda name: int(name[5:]))
    ]
    source = NodeSchema(
        "test.source", outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),)
    )
    inputs = {f"splats.splat{index}": Link("source", "splat") for index in range(2)}
    graph = Graph({"source": GraphNode(source.node_type), "merge": GraphNode("MergeSplat", inputs)})
    schemas = {source.node_type: source, schema.node_type: effective}
    assert validate(graph, schemas, ("merge",)) == []
    admitted = admit_parent_graph(
        graph,
        graph_to_wire(Graph({})),
        namespace="test.splats",
        origins=(),
        virtual_joins=(),
        budget=ParentGraphBudget(),
        schemas=schemas,
    )
    assert graph_to_wire(admitted.graph) == graph_to_wire(graph)
    for missing in ("splats.splat0", "splats.splat1"):
        incomplete = Graph(
            {
                **graph.nodes,
                "merge": GraphNode(
                    "MergeSplat", {key: value for key, value in inputs.items() if key != missing}
                ),
            }
        )
        assert [
            (item.code, item.input_id) for item in validate(incomplete, schemas, ("merge",))
        ] == [("missing-input", missing)]
