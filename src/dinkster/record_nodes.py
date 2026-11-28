"""Host-owned implementations of helpers for declared record types."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import replace

from dinkster_schema import (
    ABSENT,
    ComboOption,
    ComboWidget,
    Input,
    Node,
    NodeSchema,
    Output,
    TypeExpr,
    type_expr_to_wire,
)
from dinkster_schema.body import all_rows
from dinkster_schema.records import record_selection
from dinkster_values import CORE_COMBO, RecordPayload, Value, is_absent, make_record_value


def record_helpers(schemas: Iterable[NodeSchema]) -> dict[str, type[Node]]:
    records: dict[str, TypeExpr] = {}

    def visit(expression: TypeExpr) -> None:
        if expression.kind == "record":
            identity = expression.types[0]
            previous = records.setdefault(identity, expression)
            known = {field.id: field for field in previous.fields}
            if any(
                field.id in known
                and (field.name, field.optional, field.default)
                != (known[field.id].name, known[field.id].optional, known[field.id].default)
                for field in expression.fields
            ):
                raise ValueError(f"Record {identity!r} has conflicting declarations")
            if expression.record_connection_problem(previous) is not None:
                if previous.record_connection_problem(expression) is not None:
                    raise ValueError(f"Record {identity!r} has conflicting declarations")
                records[identity] = expression
            for field in expression.fields:
                visit(field.type)
        if expression.element is not None:
            visit(expression.element)
        for item in expression.elements:
            visit(item)

    for schema in schemas:
        for row in (*schema.inputs, *schema.outputs, *all_rows(schema.body)):
            if isinstance(row, (Input, Output)):
                visit(row.type)
    return {
        **{f"{identity}.make": _maker(expression) for identity, expression in records.items()},
        **{f"{identity}.read": _reader(expression) for identity, expression in records.items()},
        **{f"{identity}.change": _changer(expression) for identity, expression in records.items()},
    }


def _maker(expression: TypeExpr) -> type[Node]:
    # Hex encoding is injective and turns every permitted field identity into
    # a body identifier, including identities with hyphens or leading digits.
    inputs = {field.id: f"field_{field.id.encode('ascii').hex()}" for field in expression.fields}

    class MakeRecord(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type=f"{expression.types[0]}.make",
                display_name=f"Make {expression.types[0]}",
                category="Records",
                body=(
                    *(
                        Input(
                            inputs[field.id],
                            field.type,
                            required=not field.optional,
                            default=field.default,
                            label=field.name,
                        )
                        for field in expression.fields
                    ),
                    Output("record", expression),
                ),
                input_values=True,
            )

        @classmethod
        def execute(cls, *, input_values: Mapping[str, Value]) -> Mapping[str, object]:
            return cls.outputs(
                record=make_record_value(
                    expression.types[0],
                    {
                        field.id: (field.name, value)
                        for field in expression.fields
                        if (value := input_values.get(inputs[field.id])) is not None
                        and not is_absent(value)
                    },
                )
            )

    return MakeRecord


def _reader(expression: TypeExpr) -> type[Node]:
    options: list[ComboOption] = []

    def visit(record: TypeExpr, path: tuple[str, ...], names: tuple[str, ...]) -> None:
        for field in record.fields:
            field_path = (*path, field.id)
            field_names = (*names, field.name)
            options.append(
                ComboOption(
                    json.dumps(
                        {
                            "path": field_path,
                            "name": " / ".join(field_names),
                            "type": type_expr_to_wire(field.type),
                        }
                    ),
                    label=" / ".join(field_names),
                )
            )
            if field.type.kind == "record":
                visit(field.type, field_path, field_names)

    visit(expression, (), ())
    default = (
        json.dumps([min(expression.fields, key=lambda field: field.id).id])
        if expression.fields
        else "[]"
    )

    class ReadRecord(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type=f"{expression.types[0]}.read",
                display_name=f"Read {expression.types[0]}",
                category="Records",
                editor_role="record-read",
                body=(
                    Input("record", expression),
                    Input(
                        "field",
                        TypeExpr.concrete(CORE_COMBO),
                        default=default,
                        widget=ComboWidget(options=tuple(options)),
                        pin=False,
                    ),
                    Output("value", TypeExpr.record_field("record", "field"), optional=True),
                ),
                input_values=True,
            )

        @classmethod
        def execute(cls, *, field: str, input_values: Mapping[str, Value]) -> Mapping[str, object]:
            current = input_values["record"]
            declaration = expression
            path, _, problem = record_selection(expression, field)
            if not path or problem is not None:
                return cls.outputs(value=ABSENT)
            for identity in path:
                selected = next(item for item in declaration.fields if item.id == identity)
                payload = current.payload
                if not isinstance(payload, RecordPayload):
                    raise ValueError("record input does not contain field envelopes")
                child = next(
                    (
                        child
                        for (key, name), child in zip(payload.fields, payload.children, strict=True)
                        if (
                            key == identity
                            if current.type_id == declaration.types[0]
                            else name == selected.name
                        )
                    ),
                    None,
                )
                if child is None or is_absent(child):
                    return cls.outputs(value=ABSENT)
                current = child
                declaration = selected.type
            return cls.outputs(value=current)

    return ReadRecord


def _changer(expression: TypeExpr) -> type[Node]:
    reader = _reader(expression).define_schema()

    class ChangeRecord(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type=f"{expression.types[0]}.change",
                display_name=f"Change {expression.types[0]}",
                category="Records",
                editor_role="record-change",
                body=(
                    Input("record", replace(expression, template_id="Record")),
                    reader.inputs[1],
                    Input("value", TypeExpr.record_field("record", "field")),
                    Output("record", TypeExpr.variable("Record")),
                ),
                input_values=True,
                input_types=True,
            )

        @classmethod
        def execute(
            cls,
            *,
            field: str,
            input_values: Mapping[str, Value],
            input_types: Mapping[str, TypeExpr],
        ) -> Mapping[str, object]:
            def change(
                current: Value, wanted: TypeExpr, source: TypeExpr, path: tuple[str, ...]
            ) -> Value:
                selected = next(item for item in wanted.fields if item.id == path[0])
                supplied = next(
                    item
                    for item in source.fields
                    if (
                        item.id == selected.id
                        if wanted.types == source.types
                        else item.name == selected.name
                    )
                )
                payload = current.payload
                if not isinstance(payload, RecordPayload):
                    raise ValueError("record input does not contain field envelopes")
                fields = {
                    identity: (name, child)
                    for (identity, name), child in zip(
                        payload.fields, payload.children, strict=True
                    )
                }
                if len(path) == 1:
                    replacement = input_values["value"]
                else:
                    if supplied.id not in fields:
                        raise ValueError(
                            f"Cannot change a field inside absent record: {selected.name}."
                        )
                    replacement = change(
                        fields[supplied.id][1], selected.type, supplied.type, path[1:]
                    )
                fields[supplied.id] = (supplied.name, replacement)
                return replace(make_record_value(current.type_id, fields), meta=current.meta)

            current = input_values["record"]
            path, _, problem = record_selection(expression, field)
            if not path or problem is not None:
                return cls.outputs(record=current)
            source = input_types.get("record")
            if source is None:
                raise ValueError("Change requires the linked record declaration")
            return cls.outputs(record=change(current, expression, source, path))

    return ChangeRecord
