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


def test_wire_shape_is_stable_json() -> None:
    wire = graph_to_wire(sample_graph())
    sampler = wire["nodes"]["sampler"]
    assert sampler["nodeType"] == "test.sample"
    assert sampler["inputs"]["model"] == {"$link": {"node": "loader", "output": "model"}}
    assert sampler["inputs"]["steps"] == 20
    assert "outputMembers" not in sampler  # omitted when empty
    assert wire["nodes"]["batch"]["outputMembers"] == {"out": ["a", "b"]}


def test_literal_dict_with_reserved_key_rejected_on_encode() -> None:
    for key in ("$link", "$typed", "$int"):
        graph = Graph(nodes={"n": GraphNode("t", {"x": {key: "not a marker"}})})
        with pytest.raises(GraphWireError, match="reserved"):
            graph_to_wire(graph)


def test_typed_literal_round_trip_and_wire_shape() -> None:
    graph = Graph(
        nodes={
            "n": GraphNode(
                "t",
                {
                    "scalar": TypedLiteral("core.int", 7),
                    "listy": TypedLiteral("list<core.float>", [1.0, 2.0]),
                    "nully": TypedLiteral("core.string", None),
                },
            )
        }
    )
    wire = graph_to_wire(graph)
    inputs = wire["nodes"]["n"]["inputs"]
    assert inputs["scalar"] == {"$typed": {"type": "core.int", "value": 7}}
    assert inputs["listy"] == {"$typed": {"type": "list<core.float>", "value": [1.0, 2.0]}}
    # Explicit null is a present value: it must survive the round trip.
    assert inputs["nully"] == {"$typed": {"type": "core.string", "value": None}}
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


def test_malformed_decimal_integer_literal_rejected_on_decode() -> None:
    malformed = (
        9007199254740992,
        "9007199254740991",
        "+9007199254740992",
        "09007199254740992",
        "1e16",
        "18446744073709551616",
        "-9223372036854775809",
    )
    for decimal in malformed:
        wire = {"nodes": {"n": {"nodeType": "t", "inputs": {"x": {"$int": decimal}}}}}
        with pytest.raises(GraphWireError, match="decimal integer literal"):
            graph_from_wire(wire)

    graph = Graph(nodes={"n": GraphNode("t", {"x": 0x10000000000000000})})
    with pytest.raises(GraphWireError, match="outside the supported range"):
        graph_to_wire(graph)

    raw_unsafe = {"nodes": {"n": {"nodeType": "t", "inputs": {"x": 2**53}}}}
    with pytest.raises(GraphWireError, match="must use the decimal integer marker"):
        graph_from_wire(raw_unsafe)


@pytest.mark.parametrize(
    "literal",
    ([0xFFFFFFFFFFFFFFFF], {"nested": 2**53}, TypedLiteral("core.int", 2**53)),
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
        {"$typed": {"type": "core.int", "value": 2**53}},
    ),
)
def test_nested_unsafe_integer_literal_rejected_on_decode(literal: object) -> None:
    wire = {"nodes": {"n": {"nodeType": "t", "inputs": {"x": literal}}}}
    with pytest.raises(GraphWireError, match="nested inside a literal"):
        graph_from_wire(wire)


def test_safe_nested_integer_literals_still_round_trip() -> None:
    safe = 2**53 - 1
    graph = Graph(
        nodes={
            "n": GraphNode(
                "t",
                {"list": [safe], "typed": TypedLiteral("core.int", safe)},
            )
        }
    )
    assert graph_from_wire(graph_to_wire(graph)) == graph


def test_typed_literal_round_trip_inside_region_body() -> None:
    wire = {
        "nodes": {
            "r": {
                "region": {
                    "kind": "map",
                    "body": {
                        "nodes": {
                            "n": {
                                "nodeType": "t",
                                "inputs": {"x": {"$typed": {"type": "core.int", "value": 3}}},
                            }
                        }
                    },
                    "inputs": {"p": {"$typed": {"type": "core.int", "value": 9}}},
                    "ports": {"p": {"kind": "concrete", "types": ["core.int"]}},
                    "outputs": {},
                }
            }
        }
    }
    graph = graph_from_wire(wire)
    region = graph.nodes["r"]
    from dinkster_graph import RegionNode

    assert isinstance(region, RegionNode)
    assert region.inputs["p"] == TypedLiteral("core.int", 9)
    body_node = region.body.nodes["n"]
    assert isinstance(body_node, GraphNode)
    assert body_node.inputs["x"] == TypedLiteral("core.int", 3)
    assert graph_to_wire(graph)["nodes"]["r"]["region"]["inputs"]["p"] == {
        "$typed": {"type": "core.int", "value": 9}
    }


def test_malformed_typed_literal_rejected_on_decode() -> None:
    for bad_typed in (
        "nope",  # payload not an object
        {"value": 7},  # missing type
        {"type": 1, "value": 7},  # non-string type
        {"type": "", "value": 7},  # empty type
        {"type": "core.int"},  # missing value (explicit null is fine)
        ["core.int", 7],  # payload not an object
    ):
        wire = {"nodes": {"n": {"nodeType": "t", "inputs": {"x": {"$typed": bad_typed}}}}}
        with pytest.raises(GraphWireError, match="malformed typed literal"):
            graph_from_wire(wire)


def test_marker_inputs_must_be_exactly_the_marker() -> None:
    # Two markers in one input object, or plain keys riding alongside a
    # marker, decode as errors rather than guesses (the encoder can never
    # produce either shape).
    both = {
        "$link": {"node": "a", "output": "o"},
        "$typed": {"type": "core.int", "value": 7},
    }
    extra_by_link = {"$link": {"node": "a", "output": "o"}, "x": 1}
    extra_by_typed = {"$typed": {"type": "core.int", "value": 7}, "x": 1}
    extra_by_int = {"$int": "9007199254740992", "x": 1}
    for bad_input in (both, extra_by_link, extra_by_typed, extra_by_int):
        wire = {"nodes": {"n": {"nodeType": "t", "inputs": {"x": bad_input}}}}
        with pytest.raises(GraphWireError, match="exactly one reserved key"):
            graph_from_wire(wire)


def test_malformed_link_rejected_on_decode() -> None:
    for bad_link in ("nope", {"node": "a"}, {"node": 1, "output": "o"}, ["a", "o"]):
        wire = {"nodes": {"n": {"nodeType": "t", "inputs": {"x": {"$link": bad_link}}}}}
        with pytest.raises(GraphWireError, match="malformed link"):
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


def test_node_ids_with_path_characters_rejected_on_decode() -> None:
    # '/', '[' and ']' are banned in node ids at every nesting level so that
    # runtime iteration ids (r[3]/node) and diagnostic paths (outer/inner)
    # parse back to document node ids mechanically.
    for bad_id in ("a/b", "a[0", "a]b", "r[3]"):
        with pytest.raises(GraphWireError, match="may not contain"):
            graph_from_wire({"nodes": {bad_id: {"nodeType": "t"}}})
        # Same ban inside region bodies (decode recurses through the same path).
        with pytest.raises(GraphWireError, match="may not contain"):
            graph_from_wire(
                {
                    "nodes": {
                        "r": {
                            "region": {
                                "kind": "map",
                                "body": {"nodes": {bad_id: {"nodeType": "t"}}},
                            }
                        }
                    }
                }
            )


def test_inputs_and_members_optional_on_decode() -> None:
    graph = graph_from_wire({"nodes": {"n": {"nodeType": "t"}}})
    node = graph.nodes["n"]
    assert isinstance(node, GraphNode)
    assert node.node_type == "t"
    assert node.inputs == {}
    assert node.output_members == {}
