"""Pre-validation graph migrations declared by node schemas."""

from __future__ import annotations

from dinkster_graph import Graph, GraphNode, RegionNode, migrate_pure_node_type_replacements
from dinkster_schema import (
    InputSpec,
    MappingSource,
    NodeSchema,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
)

OLD = "legacy.node"
NEW = "current.node"
STRING = TypeExpr.concrete("core.string")


def renamed_schema() -> NodeSchema:
    return NodeSchema(
        node_type=NEW,
        inputs=(InputSpec("value", STRING),),
        outputs=(OutputSpec("value", STRING),),
        aliases=(OLD,),
        replacements=(
            ReplacementRule(
                from_type=OLD,
                cases=(
                    ReplacementCase.build(
                        NEW,
                        inputs={"value": MappingSource.copy("value")},
                        outputs={"value": "value"},
                    ),
                ),
            ),
        ),
    )


def test_pure_node_type_replacement_recurses_without_mutating_document_state() -> None:
    legacy = Graph(
        {
            "top": GraphNode(OLD, {"value": "top"}),
            "region": RegionNode(
                kind="map",
                body=Graph({"nested": GraphNode(OLD, {"value": "nested"})}),
            ),
        }
    )

    migrated = migrate_pure_node_type_replacements(legacy, {NEW: renamed_schema()})

    top = migrated.nodes["top"]
    region = migrated.nodes["region"]
    assert isinstance(top, GraphNode)
    assert top == GraphNode(NEW, {"value": "top"})
    assert isinstance(region, RegionNode)
    assert region.body.nodes["nested"] == GraphNode(NEW, {"value": "nested"})
    assert legacy.nodes["top"] == GraphNode(OLD, {"value": "top"})


def test_aliases_and_live_predecessors_do_not_trigger_document_migration() -> None:
    legacy = Graph({"node": GraphNode(OLD, {"value": "kept"})})
    alias_only = NodeSchema(node_type=NEW, aliases=(OLD,))
    assert migrate_pure_node_type_replacements(legacy, {NEW: alias_only}) is legacy

    schemas = {
        OLD: NodeSchema(node_type=OLD),
        NEW: renamed_schema(),
    }
    assert migrate_pure_node_type_replacements(legacy, schemas) is legacy
