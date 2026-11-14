"""One-time substitutions of retired graph node ids using schema-declared rules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from dinkster_schema import (
    InputSpec,
    MappingSource,
    NodeSchema,
    ReplacementCase,
    ReplacementPredicate,
    ReplacementRule,
    elaborate,
    schema_signature,
)

from .model import Graph, GraphNode, Link, RegionNode, TypedLiteral


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


def _matches(predicate: ReplacementPredicate | None, node: GraphNode) -> bool:
    if predicate is None or predicate.kind == "always":
        return True
    if predicate.kind == "not":
        return not _matches(predicate.of[0], node)
    if predicate.kind == "all":
        return all(_matches(child, node) for child in predicate.of)
    if predicate.kind == "any":
        return any(_matches(child, node) for child in predicate.of)
    value = node.inputs.get(predicate.input)
    if predicate.kind == "inputConnected":
        return isinstance(value, Link)
    present = predicate.input in node.inputs and not isinstance(value, Link)
    if predicate.kind == "valuePresent":
        return present
    if isinstance(value, TypedLiteral):
        value = value.value
    return present and _equal_value(value, predicate.value)


def _mapped_value(mapping: MappingSource, node: GraphNode) -> tuple[bool, object]:
    if mapping.kind == "constant":
        return True, mapping.value
    if mapping.input not in node.inputs:
        return False, None
    value = node.inputs[mapping.input]
    if mapping.kind == "link":
        return isinstance(value, Link), value
    if mapping.kind == "copy":
        return True, value
    if isinstance(value, Link):
        return False, None
    if mapping.transform is None:
        return True, value
    if isinstance(value, TypedLiteral):
        value = value.value
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
    for raw, mapping in case.inputs:
        local_id, port = address(raw)
        present, value = _mapped_value(mapping, source)
        if present:
            inputs[local_id][port] = value
    for raw, mapping in case.input_families:
        local_id, family_id = address(raw)
        family = next(
            (
                family
                for family in schemas[targets[local_id][1]].input_families
                if family.id == family_id
            ),
            None,
        )
        if family is None:
            raise ValueError(f"replacement input family {raw!r} is unavailable")
        template = family.template
        if not all(isinstance(spec, InputSpec) for spec in template):
            raise ValueError(f"replacement input family {raw!r} needs a static template")
        input_members: list[tuple[str, tuple[tuple[str, MappingSource], ...], GraphNode]]
        if mapping.kind == "members":
            input_members = [(member.suffix, member.inputs, source) for member in mapping.members]
        else:
            if not mapping.source_template:
                raise ValueError("retired input family requires its sourceTemplate")
            prefix = mapping.source_family + "."
            suffixes = dict.fromkeys(
                key[len(prefix) :].split(".", 1)[0]
                for key in source.inputs
                if key.startswith(prefix)
            )
            input_members = []
            for index, suffix in enumerate(suffixes):
                target_suffix = suffix
                if mapping.member_ids == "targetPositions":
                    if family.member_names is not None:
                        if index >= len(family.member_names):
                            raise ValueError("replacement input family exceeds target positions")
                        target_suffix = family.member_names[index]
                    elif family.member_prefix is not None:
                        target_suffix = f"{family.member_prefix}{index}"
                    else:
                        raise ValueError("replacement target family has no positional naming")
                source_inputs = {
                    input_id: source.inputs[path]
                    for input_id in mapping.source_template
                    if (
                        path := prefix
                        + suffix
                        + ("" if len(mapping.source_template) == 1 else "." + input_id)
                    )
                    in source.inputs
                }
                input_members.append(
                    (target_suffix, mapping.inputs, replace(source, inputs=source_inputs))
                )
        for suffix, mappings, member_source in input_members:
            member_path = f"{family_id}.{suffix}"
            for spec in template:
                assert isinstance(spec, InputSpec)
                port = member_path if len(template) == 1 else f"{member_path}.{spec.id}"
                if spec.default is not None:
                    inputs[local_id][port] = spec.default
            for input_id, value_mapping in mappings:
                port = member_path if len(template) == 1 else f"{member_path}.{input_id}"
                present, value = _mapped_value(value_mapping, member_source)
                if present:
                    inputs[local_id][port] = value
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
                    continue
                value = outputs[value.output_id]
            result[key] = value
        return result

    nodes = dict(graph.nodes)
    for local_id, (target_id, target_type) in targets.items():
        membership = output_members if not local_id else {}
        effective = elaborate(
            schemas[target_type], inputs[local_id], membership, variants[local_id]
        )
        for spec in effective.inputs:
            if spec.default is not None:
                inputs[local_id].setdefault(spec.id, spec.default)
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
            if rule.migration is None and not rule.from_version:
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
            if len(candidates) != 1:
                raise ValueError(f"ambiguous replacements for {node.node_type!r}")
            rule = candidates[0]
            case = next(case for case in rule.cases if _matches(case.when, node))
            current, ports = _apply_case(current, node_id, node, case, schemas)
            output_moves[node_id] = ports
            replaced = current.nodes[node_id]
            assert isinstance(replaced, GraphNode)
            node = replaced
    return current, output_moves
