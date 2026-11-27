"""Host-owned implementations of helpers for declared record types."""

from collections.abc import Iterable, Mapping

from dinkster_schema import Input, Node, NodeSchema, Output, TypeExpr
from dinkster_schema.body import all_rows
from dinkster_values import Value, is_absent, make_record_value


def record_makers(schemas: Iterable[NodeSchema]) -> dict[str, type[Node]]:
    records: dict[str, TypeExpr] = {}

    def visit(expression: TypeExpr) -> None:
        if expression.kind == "record":
            identity = expression.types[0]
            previous = records.setdefault(identity, expression)
            known = {field.id: field for field in previous.fields}
            if any(
                field.id in known
                and (field.name, field.optional) != (known[field.id].name, known[field.id].optional)
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
    return {f"{identity}.make": _maker(expression) for identity, expression in records.items()}


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
