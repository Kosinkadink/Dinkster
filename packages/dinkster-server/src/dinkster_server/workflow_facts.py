"""Read the editor's workflow documents in the engine: facts of a saved document,
and the numbers a session write changes. The library never interprets graphs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from dinkster_assets import AssetError, AssetRef, LocalAssetRef, is_digest
from dinkster_assets.model import NamedAssetResolver
from dinkster_schema import (
    AssetWidget,
    Choice,
    Group,
    Input,
    NodeSchema,
    NumberWidget,
    Row,
    Section,
    TypeChoice,
    Use,
    WidgetRepresentations,
)
from dinkster_schema.flatten import apply_use
from dinkster_values.builtin_types import CORE_INT

if TYPE_CHECKING:
    from .library import ServerLibrary


def workflow_facts(
    data: bytes,
    digest: str,
    schemas: Mapping[str, NodeSchema],
    library: ServerLibrary,
    *,
    node_packs: Mapping[str, str] | None = None,
    include_models: bool = True,
) -> dict[str, object] | None:
    try:
        value: object = json.loads(data)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(value, dict):
        return None
    document = cast("dict[str, object]", value)
    if document.get("format") != "dinkster-workflow" or document.get("formatVersion") != 1:
        return None
    graphs = document.get("graphs")
    environment = document.get("environment", {})
    if not isinstance(graphs, dict) or not isinstance(environment, dict):
        return None
    stamps = cast("dict[str, object]", environment).get("packs", {})
    stamped_nodes = cast("dict[str, object]", environment).get("nodes", {})
    if not isinstance(stamps, dict) or not isinstance(stamped_nodes, dict):
        return None
    used_packs: set[str] = set()
    models: list[dict[str, object]] = []
    for graph_id, raw_graph in cast("dict[str, object]", graphs).items():
        if not isinstance(raw_graph, dict):
            return None
        nodes = cast("dict[str, object]", raw_graph).get("nodes")
        if not isinstance(nodes, dict):
            return None
        for node_id, raw_node in cast("dict[str, object]", nodes).items():
            if not isinstance(raw_node, dict):
                return None
            node = cast("dict[str, object]", raw_node)
            node_type, values = node.get("type"), node.get("values", {})
            schema = schemas.get(node_type) if isinstance(node_type, str) else None
            if not isinstance(node_type, str) or not isinstance(values, dict):
                return None
            if node_type.startswith("#") or node.get("virtual") is True:
                continue
            if schema is None and include_models:
                return None
            raw_stamp = cast("dict[str, object]", stamped_nodes).get(node_type, {})
            if not isinstance(raw_stamp, dict):
                return None
            pack = cast("dict[str, object]", raw_stamp).get("pack")
            if pack is None and node_packs is not None and schema is not None:
                pack = node_packs.get(node_type, "core")
            if not isinstance(pack, str) or not pack:
                return None
            used_packs.add(pack)
            if not include_models:
                continue
            assert schema is not None
            inputs = cast("dict[str, object]", values)
            for spec in schema.inputs:
                widget = spec.widget
                if isinstance(widget, WidgetRepresentations):
                    widget = widget.representations[0].widget
                if not isinstance(widget, AssetWidget) or not widget.kind.startswith("model/"):
                    continue
                selected = inputs.get(spec.id)
                selections: list[tuple[object, list[str | int]]] = [
                    (selected, [graph_id, node_id, spec.id])
                ]
                while selections:
                    selected, location = selections.pop()
                    if isinstance(selected, list):
                        selections.extend(
                            (item, [*location, index])
                            for index, item in reversed(
                                list(enumerate(cast("list[object]", selected)))
                            )
                        )
                        continue
                    if isinstance(selected, dict):
                        wire = cast("dict[str, object]", selected)
                        name, identity = wire.get("name"), wire.get("digest")
                        identity = (
                            identity if isinstance(identity, str) and is_digest(identity) else None
                        )
                    elif isinstance(selected, str):
                        name, identity = selected, None
                    else:
                        continue
                    if not isinstance(name, str) or not name:
                        continue
                    if identity is None and isinstance(library.resolver, NamedAssetResolver):
                        try:
                            ref = library.resolver.bind_name(name, widget.kind)
                            verified = ref.verified() if isinstance(ref, LocalAssetRef) else ref
                            identity = verified.digest if isinstance(verified, AssetRef) else None
                        except (AssetError, OSError):
                            pass
                    models.append(
                        {
                            "location": json.dumps(location),
                            "name": name,
                            "kind": widget.kind,
                            "digest": identity,
                        }
                    )
    packs: list[dict[str, object]] = []
    for pack in sorted(used_packs - {"core"}):
        raw = cast("dict[str, object]", stamps).get(pack, {})
        if not isinstance(raw, dict):
            return None
        stamp = cast("dict[str, object]", raw)
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in (
                stamp.get(key) for key in ("version", "artifactDigest", "source", "publisher")
            )
        ):
            return None
        packs.append(
            {
                "pack": pack,
                **{
                    key: stamp.get(key)
                    for key in ("version", "artifactDigest", "source", "publisher")
                },
            }
        )
    pending: list[object] = [document]
    machine_paths = False
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(cast("dict[str, object]", item).values())
        elif isinstance(item, list):
            pending.extend(cast("list[object]", item))
        elif isinstance(item, str) and re.match(r"^(?:/|[A-Za-z]:[\\/]|\\\\|local:)", item):
            machine_paths = True
    return {"digest": digest, "packs": packs, "models": models, "machinePaths": machine_paths}


def is_node_number(path: Sequence[str | int], value: object) -> bool:
    """Whether ``path`` is a node's field in a workflow and ``value`` holds numbers."""
    return (
        len(path) == 6
        and path[0] == "graphs"
        and path[2] == "nodes"
        and path[4] == "values"
        and bool(_numbers(value))
    )


def number_write_refusal(
    before: object,
    after: object,
    schemas: Mapping[str, NodeSchema],
    held: Callable[[tuple[str | int, ...], object], bool],
) -> dict[str, object] | None:
    """Refuse a write that leaves a changed number outside its field's limits.

    A number is refused below ``min``, above ``max``, or, in a whole-number
    field, off ``step`` counted from ``min`` (or 0 without one). Only values
    the write changes are judged, so a stored number still waiting for its
    fix never blocks other edits, and a value the same field held earlier,
    as ``held`` reports, is accepted back, so undo works. A key that more
    than one row could own is refused only when every row refuses it."""
    problems: list[dict[str, object]] = []
    old_nodes = {(graph_id, node_id): node for graph_id, node_id, node in _workflow_nodes(before)}
    for graph_id, node_id, node in _workflow_nodes(after):
        old_node = old_nodes.get((graph_id, node_id))
        schema = schemas.get(cast("str", node.get("type")))
        values = node.get("values")
        # A patch copies only the containers it changes, so most nodes are the same object.
        if old_node is node or schema is None or not isinstance(values, dict):
            continue
        old_values = old_node.get("values") if old_node is not None else None
        if not isinstance(old_values, dict):
            old_values = {}
        for key, value in cast("dict[str, object]", values).items():
            old = cast("dict[str, object]", old_values).get(key, _ABSENT)
            if type(old) is type(value) and old == value:
                continue
            if held(("graphs", graph_id, "nodes", node_id, "values", key), value):
                continue
            numbers = _numbers(value)
            fields = _number_fields(schema, key)
            if not numbers or not fields:
                continue
            for number in numbers:
                refused = [_limit_problem(widget, whole, number) for widget, whole in fields]
                if all(problem is not None for problem in refused):
                    widget, reason = cast("tuple[NumberWidget, str]", refused[0])
                    name = schema.display_name or schema.node_type
                    problems.append(
                        {
                            "graph": graph_id,
                            "node": node_id,
                            "field": key,
                            "value": number,
                            "min": widget.min,
                            "max": widget.max,
                            "step": widget.step,
                            "message": f"{name}: {key} = {number} refused. {reason}",
                        }
                    )
    if not problems:
        return None
    return {
        "error": "value-outside-limits",
        "message": " ".join(str(problem["message"]) for problem in problems) + " Nothing stored.",
        "problems": problems,
    }


_ABSENT = object()


def _workflow_nodes(document: object) -> list[tuple[str, str, dict[str, object]]]:
    graphs = (
        cast("dict[str, object]", document).get("graphs") if isinstance(document, dict) else None
    )
    found: list[tuple[str, str, dict[str, object]]] = []
    if not isinstance(graphs, dict):
        return found
    for graph_id, graph in cast("dict[str, object]", graphs).items():
        nodes = cast("dict[str, object]", graph).get("nodes") if isinstance(graph, dict) else None
        if isinstance(nodes, dict):
            for node_id, node in cast("dict[str, object]", nodes).items():
                if isinstance(node, dict):
                    found.append((graph_id, node_id, cast("dict[str, object]", node)))
    return found


def _numbers(value: object) -> list[int | float]:
    items = cast("list[object]", value) if isinstance(value, list) else [value]
    numbers: list[int | float] = []
    for item in items:
        if isinstance(item, dict) and set(cast("dict[str, object]", item)) == {"$int"}:
            item = cast("dict[str, object]", item)["$int"]
            if isinstance(item, str) and re.fullmatch(r"-?[0-9]+", item):
                numbers.append(int(item))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            numbers.append(item)
    return numbers


def _limit_problem(
    widget: NumberWidget, whole: bool, number: int | float
) -> tuple[NumberWidget, str] | None:
    low, high = widget.min, widget.max
    if (low is not None and number < low) or (high is not None and number > high):
        if low is not None and high is not None:
            allowed = f"{low} to {high}"
        else:
            allowed = f"at least {low}" if low is not None else f"at most {high}"
        return widget, f"Allowed: {allowed}."
    step = widget.step
    if whole and isinstance(number, int) and isinstance(step, int):
        start = low if isinstance(low, int) else 0
        if (number - start) % step:
            return widget, f"Not on the step of {step} from {start}."
    return None


def _number_fields(schema: NodeSchema, key: str) -> list[tuple[NumberWidget, bool]]:
    """The number widgets of every row whose flattened id may be ``key``."""
    if not schema.body:
        rows: list[Input] = [spec for spec in schema.inputs if spec.id == key]
    else:
        rows = []
        _find_inputs(schema.body, "", key, {}, rows)
    fields: list[tuple[NumberWidget, bool]] = []
    for row in rows:
        widgets = (
            [option.widget for option in row.widget.representations]
            if isinstance(row.widget, WidgetRepresentations)
            else [row.widget]
        )
        element = row.type.element if row.type.kind == "list" else row.type
        whole = element is not None and element.runtime_type_id() == CORE_INT
        fields.extend((widget, whole) for widget in widgets if isinstance(widget, NumberWidget))
    return fields


def _find_inputs(
    rows: Sequence[object], prefix: str, key: str, shared: Mapping[str, object], found: list[Input]
) -> None:
    for row in rows:
        if isinstance(row, (Section, Row)):
            _find_inputs(row.rows, prefix, key, shared, found)
            continue
        if isinstance(row, Use):
            row = apply_use(cast("Any", shared[row.id]), row)
        path = prefix + cast("Any", row).id
        if isinstance(row, Input):
            if path == key:
                found.append(row)
        elif not key.startswith(path + "."):
            continue
        elif isinstance(row, Group):
            member, _, inner = key[len(path) + 1 :].partition(".")
            if isinstance(row.value, Input) and not inner:
                found.append(row.value)
            elif row.member is not None:
                _find_inputs(row.member, f"{path}.{member}.", key, {}, found)
        elif isinstance(row, Choice):
            inside = {cast("Any", item).id: item for item in row.shared}
            for option in row.options:
                _find_inputs(option.rows, path + ".", key, inside, found)
        elif isinstance(row, TypeChoice):
            variants = row.variants or ()
            for rows_inside in (row.rows, *(variant.rows for variant in variants)):
                _find_inputs(rows_inside, path + ".", key, {}, found)
