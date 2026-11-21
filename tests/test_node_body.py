import pytest
from dinkster_assets.model import AssetRef
from dinkster_schema.body import ASSET_REF_CLASS


def test_lookup_asset_parameter_names_the_class_execute_receives() -> None:
    # The schema matches the annotation by module and name so it needs no
    # dependency on dinkster-assets; this keeps the string on the real class.
    assert (AssetRef.__module__, AssetRef.__qualname__) == ASSET_REF_CLASS


def test_each_declared_field_moves_the_signatures_its_class_names() -> None:
    from tools.probe_declaration_signatures import main

    lines: list[str] = []
    assert main(lines.append) == 0, "\n".join(line for line in lines if "DIFFERS" in line)


def test_a_body_schema_takes_a_replaced_body_and_derives_its_inputs_again() -> None:
    from dataclasses import replace

    from dinkster_schema.body import Row, Section, replace_rows
    from dinkster_schema.model import InputSpec, NodeSchema, TypeExpr

    int_type = TypeExpr.concrete("INT")
    schema = NodeSchema(
        "test.provider",
        body=(
            InputSpec("a", int_type, default=0),
            Section(
                "advanced",
                rows=(
                    Row(
                        rows=(
                            InputSpec("provider", int_type, default=1),
                            InputSpec("b", int_type, default=0),
                        )
                    ),
                ),
            ),
        ),
    )
    shown = replace(
        schema,
        body=replace_rows(schema.body, {"provider": InputSpec("provider", int_type, default=7)}),
    )
    provider = shown.input("provider")
    assert provider is not None and provider.default == 7
    assert [spec.id for spec in shown.inputs] == ["a", "provider", "b"]
    with pytest.raises(ValueError, match="not both"):
        NodeSchema("test.both", body=schema.body, inputs=(InputSpec("z", int_type),))
