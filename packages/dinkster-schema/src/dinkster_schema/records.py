"""Declared record structure at engine and worker value boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

from dinkster_values import (
    ListPayload,
    RecordPayload,
    TuplePayload,
    TypeRegistry,
    Value,
    is_absent,
    make_list_value,
    make_record_value,
    make_tuple_value,
    value_children,
)

from .model import TypeExpr
from .wire import type_expr_from_wire


def record_selection(
    record: TypeExpr, stored: object
) -> tuple[tuple[str, ...], TypeExpr, str | None]:
    """Resolve stable IDs, retaining the selected type when a saved field stops fitting."""
    if not isinstance(stored, str):
        raise ValueError("record field selector must be a stored literal field path")
    try:
        selection = json.loads(stored)
    except ValueError as exc:
        raise ValueError("record field selector must be a JSON field path") from exc
    saved_type = None
    name = ""
    if isinstance(selection, dict):
        saved = cast(dict[str, object], selection)
        raw_type = saved.get("type")
        if not isinstance(raw_type, dict) or not isinstance(saved.get("name"), str):
            raise ValueError("saved record selection requires a name and declared type")
        saved_type = type_expr_from_wire(cast(dict[str, object], raw_type))
        if saved_type.runtime_type_id() is None or saved_type.requires_elaboration:
            raise ValueError("saved record selection requires a resolved type")
        name = cast(str, saved["name"])
        selection = saved.get("path")
    if selection == [] and saved_type is None and record.kind == "record":
        if not record.fields:
            return (), TypeExpr.wildcard(), None
        selection = [min(record.fields, key=lambda field: field.id).id]
    if (
        not isinstance(selection, list)
        or not selection
        or any(not isinstance(key, str) or not key for key in cast(list[object], selection))
    ):
        raise ValueError("record field selector requires a nonempty array of field IDs")
    path = tuple(cast(list[str], selection))
    current = record
    for key in path:
        field = next((field for field in current.fields if field.id == key), None)
        if current.kind != "record" or field is None:
            if saved_type is not None:
                return path, saved_type, f"Record field '{name}' was removed."
            raise ValueError(f"record field {key!r} is not declared")
        current = field.type

    def fits(wanted: TypeExpr, supplied: TypeExpr) -> bool:
        if wanted.kind == "record":
            return wanted.record_connection_problem(supplied) is None
        if wanted.kind != supplied.kind:
            return False
        if wanted.element is not None and supplied.element is not None:
            return fits(wanted.element, supplied.element)
        if wanted.kind == "tuple":
            return len(wanted.elements) == len(supplied.elements) and all(
                fits(item, found)
                for item, found in zip(wanted.elements, supplied.elements, strict=True)
            )
        return wanted == supplied

    if saved_type is not None and not fits(saved_type, current):
        detail = saved_type.record_connection_problem(current)
        return (
            path,
            saved_type,
            f"Record field '{name}' changed type from {saved_type.runtime_type_id()} "
            f"to {current.runtime_type_id()}." + (f" {detail}" if detail is not None else ""),
        )
    return path, current, None


def record_field_bindings(wanted: TypeExpr, source: TypeExpr) -> tuple[object, ...]:
    """Identify declaration-selected fields, including absent optional children."""
    if wanted.kind == source.kind == "record":
        same_record = wanted.types == source.types
        supplied = {(item.id if same_record else item.name): item for item in source.fields}
        return tuple(
            (field.id, found.id, record_field_bindings(field.type, found.type))
            for field in sorted(wanted.fields, key=lambda item: item.id)
            if (found := supplied.get(field.id if same_record else field.name)) is not None
        )
    if wanted.element is not None and source.element is not None:
        return record_field_bindings(wanted.element, source.element)
    return tuple(
        record_field_bindings(item, supplied)
        for item, supplied in zip(wanted.elements, source.elements, strict=False)
    )


def contains_records(expression: TypeExpr) -> bool:
    return (
        expression.kind == "record"
        or (expression.element is not None and contains_records(expression.element))
        or any(contains_records(item) for item in expression.elements)
    )


def record_value_problem(expression: TypeExpr, value: Value) -> str | None:
    """Enforce field shape from envelopes, never by resolving their data."""
    if is_absent(value):
        return None
    payload = value.payload
    if expression.kind == "record":
        if not isinstance(payload, RecordPayload):
            return f"Expected record {expression.types[0]}, got {value.type_id}."
        same_record = expression.types[0] == value.type_id
        supplied = {
            identity if same_record else name: child
            for (identity, name), child in zip(payload.fields, payload.children, strict=True)
        }
        for field in expression.fields:
            child = supplied.get(field.id if same_record else field.name)
            if child is None or is_absent(child):
                if field.optional:
                    continue
                return f"Missing required record field: {field.name}."
            problem = record_value_problem(field.type, child)
            if problem is not None:
                return f"Record field '{field.name}': {problem}"
            if not contains_records(field.type) and not field.type.accepts_concrete(child.type_id):
                return (
                    f"Record field '{field.name}' expects {field.type.runtime_type_id()}, "
                    f"got {child.type_id}."
                )
        return None
    if isinstance(payload, RecordPayload) and expression.kind not in (
        "wildcard",
        "variable",
        "union",
    ):
        if expression.kind == "concrete" and expression.accepts_concrete(value.type_id):
            return None
        expected = expression.runtime_type_id() or expression.kind
        return f"Expected {expected}, got record {value.type_id}."
    if not contains_records(expression):
        return None
    children = value_children(value)
    if expression.kind == "list" and not isinstance(payload, ListPayload):
        return f"Expected {expression.runtime_type_id()}, got {value.type_id}."
    if expression.kind == "tuple" and (
        not isinstance(payload, TuplePayload) or len(payload.children) != len(expression.elements)
    ):
        return f"Expected {expression.runtime_type_id()}, got {value.type_id}."
    if children is not None:
        if expression.kind == "list" and expression.element is not None:
            for child in children:
                if (problem := record_value_problem(expression.element, child)) is not None:
                    return problem
        elif expression.kind == "tuple":
            for item, child in zip(expression.elements, children, strict=False):
                if (problem := record_value_problem(item, child)) is not None:
                    return problem
                if not contains_records(item) and not item.accepts_concrete(child.type_id):
                    return f"Expected tuple position {item.runtime_type_id()}, got {child.type_id}."
    return None


def refresh_record_names(expression: TypeExpr, value: Value) -> Value:
    """Rebind cached field names to the declaration without loading values."""
    payload = value.payload
    if expression.kind == "record" and isinstance(payload, RecordPayload):
        declared = {field.id: field for field in expression.fields}
        same_record = expression.types[0] == value.type_id
        fields = tuple(
            (identity, declared[identity].name if same_record and identity in declared else name)
            for identity, name in payload.fields
        )
        children = tuple(
            refresh_record_names(declared[identity].type, child)
            if same_record and identity in declared
            else child
            for (identity, _), child in zip(payload.fields, payload.children, strict=True)
        )
        return replace(value, payload=RecordPayload(fields, children))
    if (
        isinstance(payload, ListPayload)
        and expression.kind == "list"
        and expression.element is not None
    ):
        return replace(
            value,
            payload=replace(
                payload,
                children=tuple(
                    refresh_record_names(expression.element, child) for child in payload.children
                ),
            ),
        )
    if isinstance(payload, TuplePayload) and expression.kind == "tuple":
        return replace(
            value,
            payload=replace(
                payload,
                children=tuple(
                    refresh_record_names(item, child)
                    for item, child in zip(expression.elements, payload.children, strict=True)
                ),
            ),
        )
    return value


def wrap_declared_value(expression: TypeExpr, obj: object, registry: TypeRegistry) -> Value:
    """Wrap raw fields by their declaration without hiding child envelopes."""
    if not contains_records(expression):
        runtime = expression.runtime_type_id()
        if runtime is None:
            if not isinstance(obj, Value):
                raise ValueError("value type could not be resolved")
            runtime = obj.type_id
        return registry.wrap(runtime, obj)
    if isinstance(obj, Value):
        if (problem := record_value_problem(expression, obj)) is not None:
            raise TypeError(problem)
        return obj
    if expression.kind == "record":
        if not isinstance(obj, Mapping):
            raise TypeError(f"record {expression.types[0]} expects named fields")
        supplied = cast(Mapping[str, object], obj)
        unknown = set(supplied) - {field.name for field in expression.fields}
        if unknown:
            raise ValueError(f"Undeclared record fields: {', '.join(sorted(unknown))}.")
        fields: dict[str, tuple[str, Value]] = {}
        for field in expression.fields:
            if field.name not in supplied:
                if field.optional:
                    continue
                raise ValueError(f"Missing required record field: {field.name}.")
            raw = supplied[field.name]
            if isinstance(raw, Value) and is_absent(raw) and field.optional:
                continue
            child = wrap_declared_value(field.type, raw, registry)
            if is_absent(child) and not field.optional:
                raise ValueError(f"Missing required record field: {field.name}.")
            fields[field.id] = (field.name, child)
        return make_record_value(expression.types[0], fields)
    runtime = expression.runtime_type_id()
    if runtime is None:
        raise ValueError("value type could not be resolved")
    if (
        expression.kind == "list"
        and expression.element is not None
        and expression.element.kind != "wildcard"
    ):
        if not isinstance(obj, (list, tuple)):
            raise TypeError(f"{runtime} expects a sequence")
        return make_list_value(
            cast(str, expression.element.runtime_type_id()),
            [
                wrap_declared_value(expression.element, item, registry)
                for item in cast(Sequence[object], obj)
            ],
        )
    if expression.kind == "tuple":
        if not isinstance(obj, tuple) or len(cast(tuple[object, ...], obj)) != len(
            expression.elements
        ):
            raise TypeError(f"{runtime} expects a tuple of {len(expression.elements)} positions")
        return make_tuple_value(
            [
                wrap_declared_value(item, raw, registry)
                for item, raw in zip(
                    expression.elements, cast(tuple[object, ...], obj), strict=True
                )
            ]
        )
    return registry.wrap(runtime, obj)


def project_record_value(expression: TypeExpr, value: Value) -> Value:
    """Bind fields to the destination's identities for kwargs and cache keys."""
    if not contains_records(expression) or is_absent(value):
        return value
    if (problem := record_value_problem(expression, value)) is not None:
        raise TypeError(problem)
    payload = value.payload
    if expression.kind == "record" and isinstance(payload, RecordPayload):
        same_record = expression.types[0] == value.type_id
        supplied = {
            identity if same_record else name: child
            for (identity, name), child in zip(payload.fields, payload.children, strict=True)
        }
        return make_record_value(
            expression.types[0],
            {
                field.id: (field.name, project_record_value(field.type, supplied[key]))
                for field in expression.fields
                if (key := field.id if same_record else field.name) in supplied
                and not is_absent(supplied[key])
            },
        )
    children = value_children(value)
    if children is not None:
        if expression.kind == "list" and expression.element is not None:
            projected = tuple(project_record_value(expression.element, child) for child in children)
            return make_list_value(cast(str, expression.element.runtime_type_id()), projected)
        if expression.kind == "tuple":
            return make_tuple_value(
                tuple(
                    project_record_value(item, child)
                    for item, child in zip(expression.elements, children, strict=True)
                )
            )
    return value
