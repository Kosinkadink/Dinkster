"""Deterministic migrations for retired graph node types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from dinkster_schema import MappingSource, NodeSchema

from .model import Graph, GraphNode, RegionNode


def _pure_node_type_replacements(schemas: Mapping[str, NodeSchema]) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for target_type, schema in schemas.items():
        if schema.input_families or schema.output_families or schema.combos or schema.slots:
            continue
        expected_inputs = {spec.id: MappingSource.copy(spec.id) for spec in schema.inputs}
        expected_outputs = {spec.id: spec.id for spec in schema.outputs}
        for rule in schema.replacements:
            if rule.from_type in schemas or len(rule.cases) != 1 or rule.migration is not None:
                continue
            case = rule.cases[0]
            if (
                case.to != target_type
                or not case.unconditional
                or case.nodes is not None
                or case.slot_variants
                or dict(case.inputs) != expected_inputs
                or case.input_families
                or case.output_families
                or case.links
                or dict(case.outputs) != expected_outputs
            ):
                continue
            previous = replacements.setdefault(rule.from_type, target_type)
            if previous != target_type:
                raise ValueError(
                    f"retired node type {rule.from_type!r} has conflicting pure replacements"
                )
    return replacements


def migrate_pure_node_type_replacements(graph: Graph, schemas: Mapping[str, NodeSchema]) -> Graph:
    """Apply declared pure renames for node types absent from ``schemas``.

    Rich replacement rules remain frontend-owned. This narrow subset is safe
    to apply during backend admission because it preserves every document
    field and changes only a retired node type to its declared successor.
    """

    replacements = _pure_node_type_replacements(schemas)
    if not replacements:
        return graph

    def migrate(current: Graph) -> Graph:
        changed = False
        nodes: dict[str, GraphNode | RegionNode] = {}
        for node_id, node in current.nodes.items():
            if isinstance(node, RegionNode):
                body = migrate(node.body)
                migrated = node if body is node.body else replace(node, body=body)
            else:
                target = replacements.get(node.node_type)
                migrated = node if target is None else replace(node, node_type=target)
            nodes[node_id] = migrated
            changed |= migrated is not node
        return Graph(nodes) if changed else current

    return migrate(graph)
