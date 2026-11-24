"""One-time substitutions of retired graph node ids using schema-declared rules."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from dinkster_schema import (
    MappingSource,
    NodeSchema,
    ReplacementCase,
    ReplacementPredicate,
    ReplacementRule,
    elaborate,
    schema_signature,
)
from dinkster_schema.body import Group, InputGroupView, input_group_view, scope_rows

from .model import Graph, GraphNode, Link, RegionNode, TypedLiteral, is_link, literal_value

_log = logging.getLogger(__name__)


def _input_family(schema: NodeSchema, family_id: str) -> InputGroupView | None:
    """The target's group of inputs with this id."""
    return input_group_view(schema.body, family_id)


def _stores_members(schema: NodeSchema, family_id: str) -> bool:
    """Whether a node stores this group's members: a group grown by a button
    has no wires or count to derive them from."""
    return any(
        isinstance(row, Group) and row.id == family_id and row.grows == "button"
        for row in scope_rows(schema.body)
    )


def _equal_value(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        left_list, right_list = cast("list[object]", left), cast("list[object]", right)
        return len(left_list) == len(right_list) and all(
            _equal_value(a, b) for a, b in zip(left_list, right_list, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        left_dict = cast("dict[str, object]", left)
        right_dict = cast("dict[str, object]", right)
        return list(left_dict) == list(right_dict) and all(
            _equal_value(value, right_dict[key]) for key, value in left_dict.items()
        )
    return type(cast("object", left)) is type(right) and left == right


_UNION_STAMPS: dict[type, str] = {bool: "BOOLEAN", int: "INT", float: "FLOAT", str: "STRING"}


def _stored_value(node: GraphNode, input_id: str) -> tuple[bool, object]:
    value = node.inputs.get(input_id)
    if input_id in node.inputs and not isinstance(value, Link):
        return True, value.value if isinstance(value, TypedLiteral) else value
    return input_id in node.slot_variants, node.slot_variants.get(input_id)


def _matches(predicate: ReplacementPredicate | None, node: GraphNode) -> bool:
    if predicate is None or predicate.kind == "always":
        return True
    if predicate.kind == "not":
        return not _matches(predicate.of[0], node)
    if predicate.kind == "all":
        return all(_matches(child, node) for child in predicate.of)
    if predicate.kind == "any":
        return any(_matches(child, node) for child in predicate.of)
    if predicate.kind == "inputConnected":
        return isinstance(node.inputs.get(predicate.input), Link)
    present, value = _stored_value(node, predicate.input)
    if predicate.kind == "valuePresent":
        return present
    return present and _equal_value(value, predicate.value)


def _mapped_value(mapping: MappingSource, node: GraphNode) -> tuple[bool, object]:
    if mapping.kind == "constant":
        return True, mapping.value
    connected = node.inputs.get(mapping.input)
    if mapping.kind == "link":
        return isinstance(connected, Link), connected
    if mapping.kind == "copy" and mapping.input in node.inputs:
        return True, connected
    present, value = _stored_value(node, mapping.input)
    if not present or mapping.transform is None:
        return present, value
    transform = mapping.transform
    if transform.kind == "enumRename":
        names = dict(transform.map)
        if not isinstance(value, str) or value not in names:
            raise ValueError(f"replacement value {mapping.input!r} has no enum mapping")
        return True, names[value]
    if type(value) not in (int, float):
        raise ValueError(f"replacement value {mapping.input!r} must be numeric to scale")
    assert isinstance(value, int | float)
    result = value * transform.factor + transform.offset
    if not math.isfinite(result):
        raise ValueError(f"replacement value {mapping.input!r} scaled outside finite numbers")
    return True, result


def _apply_case(
    graph: Graph,
    node_id: str,
    source: GraphNode,
    case: ReplacementCase,
    schemas: Mapping[str, NodeSchema],
) -> tuple[Graph, dict[str, Link]]:
    targets = {
        "": (node_id, case.to),
        **{
            local_id: (f"{node_id}:{local_id}", helper.type)
            for local_id, helper in case.nodes or ()
        },
    }
    for local_id, (target_id, target_type) in targets.items():
        if local_id and target_id in graph.nodes:
            raise ValueError(f"replacement helper id {target_id!r} already exists")
        if target_type not in schemas:
            raise ValueError(f"replacement target {target_type!r} is unavailable")

    def address(raw: str) -> tuple[str, str]:
        local_id, separator, port = raw.partition(":")
        if not separator:
            local_id, port = "", local_id
        if local_id not in targets:
            raise ValueError(f"replacement address {raw!r} names no helper")
        return local_id, port

    inputs: dict[str, dict[str, object]] = {
        local_id: {
            spec.id: spec.default
            for spec in schemas[target_type].inputs
            if spec.default is not None
        }
        for local_id, (_, target_type) in targets.items()
    }
    for local_id, helper in case.nodes or ():
        inputs[local_id].update(dict(helper.values))
    consumed_connections: set[str] = set()

    def consume(input_id: str, mapping: MappingSource) -> None:
        if mapping.kind not in ("copy", "link") or not isinstance(
            source.inputs.get(input_id), Link
        ):
            return
        if input_id in consumed_connections:
            raise ValueError(f"replacement maps connection {input_id!r} twice")
        consumed_connections.add(input_id)

    for raw, mapping in case.inputs:
        local_id, port = address(raw)
        present, value = _mapped_value(mapping, source)
        if present:
            inputs[local_id][port] = value
            consume(mapping.input, mapping)
    input_membership: dict[str, dict[str, tuple[str, ...]]] = {local_id: {} for local_id in targets}
    for raw, mapping in case.input_families:
        local_id, family_id = address(raw)
        family = _input_family(schemas[targets[local_id][1]], family_id)
        if family is None:
            raise ValueError(f"replacement input family {raw!r} is unavailable")
        if family.rows is None:
            raise ValueError(f"replacement input family {raw!r} needs a static template")
        template = family.rows
        input_members: list[tuple[str, tuple[tuple[str, MappingSource], ...], GraphNode]]
        spread_items: dict[str, object] = {}
        if mapping.kind == "spread":
            present, stored = _stored_value(source, mapping.source_input)
            if present and not isinstance(stored, list):
                raise ValueError(
                    f"replace: stored value of '{mapping.source_input}' is not a list; "
                    "cannot spread"
                )
            items = cast("list[object]", stored) if present else []
            if len(items) > family.limit:
                dropped = ", ".join(
                    item if isinstance(item, str) else json.dumps(item, separators=(",", ":"))
                    for item in items[family.limit :]
                )
                _log.warning(
                    "replace: kept the first %d of %d items of '%s'; dropped %s",
                    family.limit,
                    len(items),
                    mapping.source_input,
                    dropped,
                )
                items = items[: family.limit]
            if family.position_key is None:
                raise ValueError("replacement target family has no positional naming")
            suffixes = [
                family.position_key(index) for index in range(max(len(items), family.min_members))
            ]
            input_members = [(suffix, (), source) for suffix in suffixes]
            spread_items = dict(zip(suffixes, items, strict=False))
        elif mapping.kind == "members":
            input_members = [(member.suffix, member.inputs, source) for member in mapping.members]
            for member in mapping.members:
                for _, value_mapping in member.inputs:
                    consume(value_mapping.input, value_mapping)
        else:
            if not mapping.source_template:
                raise ValueError("retired input family requires its sourceTemplate")
            prefix = mapping.source_family + "."
            suffixes = dict.fromkeys(
                key[len(prefix) :].split(".", 1)[0]
                for key in dict.fromkeys((*source.inputs, *source.slot_variants))
                if key.startswith(prefix)
            )
            input_members = []
            for index, suffix in enumerate(suffixes):
                target_suffix = suffix
                if mapping.member_ids == "targetPositions":
                    if family.position_key is None:
                        raise ValueError("replacement target family has no positional naming")
                    try:
                        target_suffix = family.position_key(index)
                    except IndexError:
                        raise ValueError(
                            "replacement input family exceeds target positions"
                        ) from None
                source_inputs: dict[str, object] = {}
                source_variants: dict[str, str] = {}
                for input_id in mapping.source_template:
                    path = (
                        prefix
                        + suffix
                        + ("" if len(mapping.source_template) == 1 else "." + input_id)
                    )
                    if path in source.inputs:
                        source_inputs[input_id] = source.inputs[path]
                    if path in source.slot_variants:
                        source_variants[input_id] = source.slot_variants[path]
                for _, value_mapping in mapping.inputs:
                    original_path = (
                        prefix
                        + suffix
                        + ("" if len(mapping.source_template) == 1 else "." + value_mapping.input)
                    )
                    consume(original_path, value_mapping)
                input_members.append(
                    (
                        target_suffix,
                        mapping.inputs,
                        replace(source, inputs=source_inputs, slot_variants=source_variants),
                    )
                )
        if _stores_members(schemas[targets[local_id][1]], family_id):
            input_membership[local_id][family_id] = tuple(suffix for suffix, _, _ in input_members)
        for suffix, mappings, member_source in input_members:
            member_path = f"{family_id}.{suffix}"
            for spec in template:
                port = f"{member_path}.{spec.id}" if family.records else member_path
                if spec.default is not None:
                    inputs[local_id][port] = spec.default
            for input_id, value_mapping in mappings:
                port = f"{member_path}.{input_id}" if family.records else member_path
                present, value = _mapped_value(value_mapping, member_source)
                if present:
                    inputs[local_id][port] = value
            if suffix in spread_items:
                port = f"{member_path}.{mapping.member_input}" if family.records else member_path
                inputs[local_id][port] = spread_items[suffix]
    for input_id, value in source.inputs.items():
        if isinstance(value, Link) and input_id not in consumed_connections:
            raise ValueError(f"replacement would drop incoming connection {input_id!r}")
    variants: dict[str, dict[str, str]] = {local_id: {} for local_id in targets}
    for raw, choice in case.slot_variants:
        local_id, port = address(raw)
        variants[local_id][port] = choice
    for internal in case.links:
        from_local, output = address(internal.from_address)
        to_local, port = address(internal.to)
        inputs[to_local][port] = Link(targets[from_local][0], output)

    outputs: dict[str, Link] = {}
    for target, old_output in case.outputs:
        local_id, port = address(target)
        outputs[old_output] = Link(targets[local_id][0], port)

    output_members: dict[str, tuple[str, ...]] = {}
    for family_id, mapping in case.output_families:
        if mapping.kind == "copy":
            if mapping.source_family not in source.output_members:
                raise ValueError("replacement source output family has no stored membership")
            members = source.output_members[mapping.source_family]
            source_outputs = [f"{mapping.source_family}.{suffix}" for suffix in members]
        else:
            members = tuple(member.suffix for member in mapping.members)
            source_outputs = [member.output for member in mapping.members]
        output_members[family_id] = members
        for suffix, old_output in zip(members, source_outputs, strict=True):
            if old_output in outputs:
                raise ValueError(f"replacement output {old_output!r} is mapped twice")
            outputs[old_output] = Link(node_id, f"{family_id}.{suffix}")

    def mapped_inputs(original: Mapping[str, object]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in original.items():
            if isinstance(value, Link) and value.node_id == node_id:
                if value.output_id not in outputs:
                    raise ValueError(
                        f"replacement would drop outgoing connection {value.output_id!r}"
                    )
                value = outputs[value.output_id]
            result[key] = value
        return result

    nodes = dict(graph.nodes)
    for local_id, (target_id, target_type) in targets.items():
        membership = {**input_membership[local_id], **(output_members if not local_id else {})}
        effective = elaborate(
            schemas[target_type],
            inputs[local_id],
            membership,
            variants[local_id],
            linked=is_link,
            literal=literal_value,
        )
        for spec in effective.inputs:
            if spec.default is not None:
                inputs[local_id].setdefault(spec.id, spec.default)
            value = inputs[local_id].get(spec.id)
            stamp = _UNION_STAMPS.get(type(value)) if spec.type.kind == "union" else None
            # A plain literal needs a concrete input; on a union it carries its member type.
            if stamp is not None and stamp in spec.type.types:
                inputs[local_id][spec.id] = TypedLiteral(stamp, value)
        nodes[target_id] = GraphNode(
            target_type,
            inputs[local_id],
            output_members=membership,
            slot_variants=variants[local_id],
            schema_signature=schema_signature(schemas[target_type]),
        )
    target_ids = {target_id for target_id, _ in targets.values()}
    for key, node in tuple(nodes.items()):
        if key not in target_ids:
            nodes[key] = replace(node, inputs=mapped_inputs(node.inputs))
    return Graph(nodes), outputs


def migrate_node_replacements(graph: Graph, schemas: Mapping[str, NodeSchema]) -> Graph:
    """Replace absent ids, never reinterpret a currently served node id.

    The same pack rules drive frontend document opening and backend admission.
    Graph literals, links, predicates, transforms, families and helper nodes
    use those declarations directly. Historical foreign widget streams are
    decoded by the document loader before graph compilation.
    """
    rules: dict[str, list[ReplacementRule]] = {}
    for schema in schemas.values():
        for rule in schema.replacements:
            if not rule.from_version:
                rules.setdefault(rule.from_type, []).append(rule)
    return _migrate(graph, schemas, rules)[0]


def _migrate(
    graph: Graph,
    schemas: Mapping[str, NodeSchema],
    rules: Mapping[str, list[ReplacementRule]],
) -> tuple[Graph, dict[str, dict[str, Link]]]:
    current = graph
    output_moves: dict[str, dict[str, Link]] = {}
    for node_id in graph.nodes:
        original = current.nodes[node_id]
        if isinstance(original, RegionNode):
            body, body_moves = _migrate(original.body, schemas, rules)

            def moved(link: Link, moves: Mapping[str, dict[str, Link]] = body_moves) -> Link:
                ports = moves.get(link.node_id)
                if ports is None:
                    return link
                if link.output_id not in ports:
                    raise ValueError(f"replacement removes a region output {link.output_id!r}")
                return ports[link.output_id]

            if body is not original.body:
                current = Graph(
                    {
                        **current.nodes,
                        node_id: replace(
                            original,
                            body=body,
                            outputs={
                                key: replace(output, source=moved(output.source))
                                for key, output in original.outputs.items()
                            },
                            continue_source=None
                            if original.continue_source is None
                            else moved(original.continue_source),
                        ),
                    }
                )
            continue
        node = original
        seen: set[str] = set()
        while node.node_type not in schemas:
            if node.node_type in seen:
                raise ValueError(f"replacement cycle for {node.node_type!r}")
            seen.add(node.node_type)
            candidates = [
                rule
                for rule in rules.get(node.node_type, ())
                if not rule.from_signature or rule.from_signature == node.schema_signature
            ]
            if not candidates:
                break
            failures: list[str] = []
            for rule in candidates:
                case = next(case for case in rule.cases if _matches(case.when, node))
                try:
                    migrated, ports = _apply_case(current, node_id, node, case, schemas)
                except ValueError as exc:
                    failures.append(str(exc))
                    continue
                current = migrated
                break
            else:
                raise ValueError(f"no replacement for {node.node_type!r}: {'; '.join(failures)}")
            output_moves[node_id] = ports
            replaced = current.nodes[node_id]
            assert isinstance(replaced, GraphNode)
            node = replaced
    return current, output_moves
