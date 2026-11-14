"""Graph wire format: the JSON document shape frontends submit (DESIGN 3.5).
Links use the reserved $link marker; everything else round-trips untouched."""

from __future__ import annotations

import pytest
from dinkster_graph import (
    Graph,
    GraphNode,
    GraphWireError,
    Link,
    TypedLiteral,
    graph_from_wire,
    graph_to_wire,
)


def sample_graph() -> Graph:
    return Graph(
        nodes={
            "loader": GraphNode("test.load", {"name": "sd15", "device": "cuda:0"}),
            "sampler": GraphNode(
                "test.sample",
                {"model": Link("loader", "model"), "steps": 20, "cfg": 7.5},
            ),
            "batch": GraphNode(
                "test.batch",
                {"item[0]": Link("sampler", "image"), "item[1]": "literal"},
                output_members={"out": ("a", "b")},
            ),
        }
    )


def test_round_trip() -> None:
    graph = sample_graph()
    wire = graph_to_wire(graph)
    back = graph_from_wire(wire)
    assert back.nodes.keys() == graph.nodes.keys()
    for node_id, node in graph.nodes.items():
        got = back.nodes[node_id]
        assert isinstance(node, GraphNode) and isinstance(got, GraphNode)
        assert got.node_type == node.node_type
        assert got.inputs == node.inputs
        assert got.output_members == node.output_members


def test_typed_literal_round_trip_and_wire_shape() -> None:
    graph = Graph(
        nodes={
            "n": GraphNode(
                "t",
                {
                    "scalar": TypedLiteral("INT", 7),
                    "listy": TypedLiteral("list<FLOAT>", [1.0, 2.0]),
                    "nully": TypedLiteral("STRING", None),
                },
            )
        }
    )
    wire = graph_to_wire(graph)
    inputs = wire["nodes"]["n"]["inputs"]
    assert inputs["scalar"] == {"$typed": {"type": "INT", "value": 7}}
    assert inputs["listy"] == {"$typed": {"type": "list<FLOAT>", "value": [1.0, 2.0]}}
    # Explicit null is a present value: it must survive the round trip.
    assert inputs["nully"] == {"$typed": {"type": "STRING", "value": None}}
    back = graph_from_wire(wire)
    node = back.nodes["n"]
    assert isinstance(node, GraphNode)
    assert node.inputs == graph.nodes["n"].inputs


def test_unsafe_integer_literal_round_trip_and_wire_shape() -> None:
    uint64_max = 0xFFFFFFFFFFFFFFFF
    graph = Graph(nodes={"n": GraphNode("t", {"seed": uint64_max, "safe": 2**53 - 1})})
    wire = graph_to_wire(graph)
    inputs = wire["nodes"]["n"]["inputs"]
    assert inputs["seed"] == {"$int": str(uint64_max)}
    assert inputs["safe"] == 2**53 - 1
    assert graph_from_wire(wire) == graph


@pytest.mark.parametrize(
    "literal",
    ([0xFFFFFFFFFFFFFFFF], {"nested": 2**53}, TypedLiteral("INT", 2**53)),
)
def test_nested_unsafe_integer_literal_rejected_on_encode(literal: object) -> None:
    graph = Graph(nodes={"n": GraphNode("t", {"x": literal})})
    with pytest.raises(GraphWireError, match="nested inside a literal"):
        graph_to_wire(graph)


@pytest.mark.parametrize(
    "literal",
    (
        [2**53],
        {"nested": 2**53},
        {"$typed": {"type": "INT", "value": 2**53}},
    ),
)
def test_nested_unsafe_integer_literal_rejected_on_decode(literal: object) -> None:
    wire = {"nodes": {"n": {"nodeType": "t", "inputs": {"x": literal}}}}
    with pytest.raises(GraphWireError, match="nested inside a literal"):
        graph_from_wire(wire)


def test_malformed_document_shapes_rejected() -> None:
    with pytest.raises(GraphWireError):
        graph_from_wire(None)
    with pytest.raises(GraphWireError):
        graph_from_wire([])
    with pytest.raises(GraphWireError):
        graph_from_wire({"nodes": []})
    with pytest.raises(GraphWireError):  # missing nodeType
        graph_from_wire({"nodes": {"n": {"inputs": {}}}})
    with pytest.raises(GraphWireError):  # empty node id
        graph_from_wire({"nodes": {"": {"nodeType": "t"}}})
    with pytest.raises(GraphWireError):  # inputs must be an object
        graph_from_wire({"nodes": {"n": {"nodeType": "t", "inputs": []}}})
    with pytest.raises(GraphWireError):  # members must be a list of strings
        graph_from_wire({"nodes": {"n": {"nodeType": "t", "outputMembers": {"out": [1, 2]}}}})
