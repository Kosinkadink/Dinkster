"""Declared records have directional named-field connections."""

from dataclasses import replace

import pytest
from dinkster_api.v1 import Input, NodeSchema, Output, RecordField, TypeExpr
from dinkster_graph import Graph, GraphNode, Link, validate
from dinkster_schema import schema_from_wire, schema_to_wire

INTEGER = TypeExpr.concrete("INT")
TEXT = TypeExpr.concrete("STRING")


def _connection(source: TypeExpr, destination: TypeExpr):
    producer = schema_from_wire(
        schema_to_wire(NodeSchema(node_type="source.make", body=(Output("record", source),)))
    )
    consumer = schema_from_wire(
        schema_to_wire(NodeSchema(node_type="target.read", body=(Input("record", destination),)))
    )
    graph = Graph(
        nodes={
            "make": GraphNode(producer.node_type),
            "read": GraphNode(consumer.node_type, {"record": Link("make", "record")}),
        }
    )
    return validate(graph, {producer.node_type: producer, consumer.node_type: consumer}, ["read"])


def test_record_connection_matches_names_across_packs_and_keeps_extra_fields() -> None:
    source = TypeExpr.record_of(
        "source.settings",
        RecordField("source-text", "prompt", TEXT),
        RecordField("source-count", "seed", INTEGER),
    )
    destination = TypeExpr.record_of(
        "target.settings", RecordField("target-count", "seed", INTEGER)
    )
    assert _connection(source, destination) == []
    problems = _connection(destination, source)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert problems[0].node_id == "read"
    assert problems[0].input_id == "record"
    assert "prompt" in problems[0].message

    wrong_type = replace(destination, fields=(RecordField("target-count", "seed", TEXT),))
    problems = _connection(source, wrong_type)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert all(word in problems[0].message for word in ("seed", "INT", "STRING"))
    problems = _connection(source, INTEGER)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "INT" in problems[0].message
    assert "record" in problems[0].message


def test_record_rename_and_reorder_follow_only_explicit_identity() -> None:
    original = TypeExpr.record_of(
        "source.settings",
        RecordField("count", "seed", INTEGER),
        RecordField("text", "prompt", TEXT),
    )
    renamed = replace(
        original,
        fields=(original.fields[1], replace(original.fields[0], name="noise")),
    )
    assert _connection(renamed, original) == []
    assert _connection(original, renamed) == []
    cross_pack = replace(original, types=("target.settings",))
    problems = _connection(renamed, cross_pack)
    assert len(problems) == 1
    assert "seed" in problems[0].message
    replaced_identity = replace(
        original, fields=(replace(original.fields[0], id="new-count"), original.fields[1])
    )
    problems = _connection(replaced_identity, original)
    assert len(problems) == 1
    assert "seed" in problems[0].message


def test_nested_record_connection_names_the_missing_inner_field() -> None:
    source = TypeExpr.record_of(
        "source.outer",
        RecordField(
            "inner",
            "settings",
            TypeExpr.record_of("source.inner", RecordField("s", "seed", INTEGER)),
        ),
    )
    destination = TypeExpr.record_of(
        "target.outer",
        RecordField(
            "i", "settings", TypeExpr.record_of("target.inner", RecordField("p", "prompt", TEXT))
        ),
    )
    problems = _connection(source, destination)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "settings" in problems[0].message
    assert "prompt" in problems[0].message


@pytest.mark.parametrize("source", [INTEGER, TypeExpr.wildcard()])
def test_record_input_never_guesses_fields_from_an_undeclared_type(source: TypeExpr) -> None:
    destination = TypeExpr.record_of("target.settings", RecordField("count", "seed", INTEGER))
    problems = _connection(source, destination)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "Expected record" in problems[0].message
