"""Elaboration: a node's declared body becomes its flat interface for one run.

``elaborate(schema, stored inputs, group members, chosen keys)`` returns an
ordinary NodeSchema marked ``effective``: flat inputs and outputs, the chosen
keys, and no body. It reads only document state (stored values and link
state, group members and chosen options), never values produced at run time,
so it runs before validation, planning, caching and invocation. Flattening
orders each scope's rows by id, so the flat inputs need not follow the order
the body declares them in. Malformed or oversized document
state raises ElaborationError; elaboration never repairs it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import cast

from .body import Choice, Described, Group, Rule, State, TypeChoice, scope_rows
from .flatten import FlattenError, flatten_body, scope_member_type
from .model import (
    Input,
    NodeSchema,
    Output,
    TypeExpr,
)

MAX_INTERFACE_ITEMS = 1024
"""Hard cap on total effective interface items (inputs + outputs)."""


class ElaborationError(ValueError):
    """Document state cannot elaborate against this schema (deterministic)."""


def elaborate(
    schema: NodeSchema,
    stored_inputs: Mapping[str, object] | Iterable[str],
    output_members: Mapping[str, Sequence[str]] | None = None,
    slot_variants: Mapping[str, str] | None = None,
    *,
    linked: Callable[[object], bool] | None = None,
    literal: Callable[[object], object] | None = None,
    formulas: bool = True,
) -> NodeSchema:
    """The effective concrete interface of one node.

    stored_inputs is either the graph node's input mapping or, for callers
    that only need document-driven interfaces, its input keys; count formulas
    and described rows need the mapping. output_members is each group's
    member keys in the run's order, by flattened path; slot_variants is each
    choice's and type choice's chosen key, by flattened path. A missing key
    means nothing is chosen, which only an optional choice allows.
    ``linked`` tells a wired input's stored entry from a value and
    ``literal`` unwraps a value that carries its own type; count and leader
    formulas read both. ``formulas=False`` says the mapping holds only
    some values (a worker re-checking a run): count and leader formulas are
    unknown and the run's recorded members and choices stand.
    """
    if schema.body:
        return _elaborate_body(
            schema,
            stored_inputs,
            output_members or {},
            slot_variants or {},
            linked,
            literal,
            formulas,
        )
    if output_members:
        raise ElaborationError(f"{schema.node_type}: output members stored for a static schema")
    if slot_variants:
        raise ElaborationError(f"{schema.node_type}: slot variants stored for a static schema")
    if schema.additional_inputs is None:
        return schema
    stored_ids = tuple(stored_inputs)
    inputs = [
        *schema.inputs,
        *_additional_inputs(schema, schema.additional_inputs, schema.inputs, (), stored_ids),
    ]
    return replace(schema, inputs=tuple(inputs), additional_inputs=None, effective=True)


def _resolve_applies(
    schema: NodeSchema,
    inputs: list[Input],
    outputs: list[Output],
    chosen: Mapping[str, str],
) -> tuple[list[Input], list[Output], dict[str, object]]:
    """Resolve every ``applies`` scope against the chosen options.

    ``applies`` keys name required top-level choices, so ``chosen`` holds
    each. Covered metadata loses its resolved scope; uncovered metadata is
    not a sound promise and is dropped."""

    def covered(applies: Mapping[str, tuple[str, ...]]) -> bool:
        return all(chosen[choice_id] in keys for choice_id, keys in applies.items())

    if schema.chunk_safe_applies is not None and not covered(schema.chunk_safe_applies):
        assert schema.chunk_safe is not None
        inputs = [
            replace(spec, accepts_stream=False) if spec.id in schema.chunk_safe[0] else spec
            for spec in inputs
        ]
    mirror = schema.mirror
    if mirror is not None and mirror.applies is not None:
        mirror = replace(mirror, applies=None) if covered(mirror.applies) else None
    outputs = [
        replace(
            output,
            represents=replace(output.represents, applies=None)
            if covered(output.represents.applies)
            else None,
        )
        if output.represents is not None and output.represents.applies is not None
        else output
        for output in outputs
    ]
    chunk_safe = (
        (
            tuple(name for name in schema.chunk_safe[0] if name in {spec.id for spec in inputs}),
            schema.chunk_safe[1],
        )
        if schema.chunk_safe is not None
        and (schema.chunk_safe_applies is None or covered(schema.chunk_safe_applies))
        else None
    )
    return inputs, outputs, {"mirror": mirror, "chunk_safe": chunk_safe, "chunk_safe_applies": None}


def failed_rules(
    schema: NodeSchema,
    stored_inputs: Mapping[str, object],
    output_members: Mapping[str, Sequence[str]] | None = None,
    slot_variants: Mapping[str, str] | None = None,
    *,
    linked: Callable[[object], bool] | None = None,
    literal: Callable[[object], object] | None = None,
) -> list[Rule]:
    """The node's rules whose check is false for these stored inputs.

    A check that reads a wired input, or whose formula fails, has no answer
    before the run and is not returned. Arguments are as for ``elaborate``;
    document state that does not elaborate has no failed rules."""
    if not schema.validate:
        return []
    try:
        flat = flatten_body(
            schema.body,
            stored_inputs,
            output_members or {},
            slot_variants or {},
            tables=schema.tables,
            derived=schema.derived,
            linked=linked,
            literal=literal,
            rules=[rule.check for rule in schema.validate],
        )
    except FlattenError:
        return []
    return [
        rule for rule, outcome in zip(schema.validate, flat.rules, strict=True) if outcome is False
    ]


def _additional_inputs(
    schema: NodeSchema,
    additional: TypeExpr,
    declared: Sequence[Input],
    reserved: Sequence[str],
    stored_ids: Iterable[str],
) -> list[Input]:
    """An optional input for each stored id the schema never declared."""
    seen = {spec.id for spec in declared}
    added: list[Input] = []
    for input_id in stored_ids:
        if input_id in seen:
            continue
        if not input_id or "\0" in input_id:
            raise ElaborationError("additional input names must be non-empty strings without NUL")
        if any(input_id == root or input_id.startswith(root + ".") for root in reserved):
            raise ElaborationError(
                f"{schema.node_type}: additional input '{input_id}' belongs to "
                "a declared dynamic construct"
            )
        added.append(Input(input_id, scope_member_type(additional, input_id), required=False))
        seen.add(input_id)
    return added


def _elaborate_body(
    schema: NodeSchema,
    stored_inputs: Mapping[str, object] | Iterable[str],
    output_members: Mapping[str, Sequence[str]],
    choices: Mapping[str, str],
    linked: Callable[[object], bool] | None,
    literal: Callable[[object], object] | None,
    formulas: bool,
) -> NodeSchema:
    stored = (
        cast("Mapping[str, object]", stored_inputs)
        if isinstance(stored_inputs, Mapping)
        else tuple(stored_inputs)
    )
    try:
        flat = flatten_body(
            schema.body,
            stored,
            output_members,
            choices,
            tables=schema.tables,
            derived=schema.derived,
            linked=linked,
            literal=literal,
            formulas=formulas,
        )
    except FlattenError as error:
        raise ElaborationError(f"{schema.node_type}: {error}") from error
    if schema.additional_inputs is not None:
        # A box or a state of the top scope owns its id and the ids under it.
        reserved = tuple(
            row.id
            for row in scope_rows(schema.body)
            if isinstance(row, (State, Group, Choice, TypeChoice, Described))
        )
        flat.inputs.extend(
            _additional_inputs(
                schema,
                schema.additional_inputs,
                flat.inputs,
                reserved,
                stored if isinstance(stored, tuple) else tuple(stored),
            )
        )
    _validate_projected_namespace(schema, flat.inputs)
    values = stored if isinstance(stored, Mapping) else None
    outputs = [
        replace(output, type=_project_output_type(output.type, flat.inputs, values))
        if output.type.requires_elaboration
        else output
        for output in flat.outputs
    ]
    inputs, outputs, scoped = _resolve_applies(
        schema, list(flat.inputs), outputs, dict(flat.choices)
    )
    total = len(inputs) + len(outputs)
    if total > MAX_INTERFACE_ITEMS:
        raise ElaborationError(
            f"{schema.node_type}: effective interface has {total} items, "
            f"budget is {MAX_INTERFACE_ITEMS}"
        )
    return replace(
        schema,
        body=(),
        tables={},
        derived={},
        validate=(),
        title_suffix="",
        menu=(),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        slot_choices=tuple(flat.choices),
        effective=True,
        **scoped,
    )


def _project_output_type(
    expr: TypeExpr,
    inputs: Sequence[Input],
    stored: Mapping[str, object] | None,
) -> TypeExpr:
    if expr.kind == "record_field":
        source = next(spec.type for spec in inputs if spec.id == expr.input_id)
        selector = next(spec for spec in inputs if spec.id == expr.index)
        value = selector.default if stored is None else stored.get(selector.id, selector.default)
        if not isinstance(value, str):
            raise ElaborationError("record field selector must be a stored literal field path")
        try:
            path = json.loads(value)
        except ValueError as exc:
            raise ElaborationError("record field selector must be a JSON field path") from exc
        if (
            not isinstance(path, list)
            or not path
            or any(not isinstance(key, str) for key in cast("list[object]", path))
        ):
            raise ElaborationError("record field selector requires a nonempty array of field IDs")
        for key in cast("list[str]", path):
            field = next((field for field in source.fields if field.id == key), None)
            if source.kind != "record" or field is None:
                raise ElaborationError(f"record field {key!r} is not declared")
            source = field.type
        return source
    if expr.kind == "tuple_element":
        element = (
            next(spec.type for spec in inputs if spec.id == expr.input_id)
            if expr.input_id
            else expr.element
        )
        assert element is not None
        index = expr.index
        if isinstance(index, str):
            if stored is None:
                raise ElaborationError("tuple index requires stored literal input values")
            selector = next(spec for spec in inputs if spec.id == index)
            value = stored.get(index, selector.default)
            if type(value) is not int:
                raise ElaborationError(
                    "tuple index must be a constant INT; use a list for same-typed items"
                )
            index = value
        return TypeExpr(kind="tuple_element", element=element, index=index)
    return replace(
        expr,
        element=None
        if expr.element is None
        else _project_output_type(expr.element, inputs, stored),
        elements=tuple(_project_output_type(item, inputs, stored) for item in expr.elements),
    )


def _validate_projected_namespace(schema: NodeSchema, inputs: Sequence[Input]) -> None:
    ids = [spec.id for spec in inputs]
    if len(set(ids)) != len(ids):
        duplicates = sorted({input_id for input_id in ids if ids.count(input_id) > 1})
        raise ElaborationError(
            f"{schema.node_type}: projected input namespace collision: {', '.join(duplicates)}"
        )
