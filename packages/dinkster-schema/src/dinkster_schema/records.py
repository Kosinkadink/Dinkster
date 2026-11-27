"""Declared record structure at engine and worker value boundaries."""

from __future__ import annotations

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
