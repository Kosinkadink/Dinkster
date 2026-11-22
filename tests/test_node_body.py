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
    from dinkster_schema.model import Input, NodeSchema, TypeExpr

    int_type = TypeExpr.concrete("INT")
    schema = NodeSchema(
        "test.provider",
        body=(
            Input("a", int_type, default=0),
            Section(
                "advanced",
                rows=(
                    Row(
                        rows=(
                            Input("provider", int_type, default=1),
                            Input("b", int_type, default=0),
                        )
                    ),
                ),
            ),
        ),
    )
    shown = replace(
        schema,
        body=replace_rows(schema.body, {"provider": Input("provider", int_type, default=7)}),
    )
    provider = shown.input("provider")
    assert provider is not None and provider.default == 7
    assert [spec.id for spec in shown.inputs] == ["a", "provider", "b"]
    with pytest.raises(ValueError, match="not both"):
        NodeSchema("test.both", body=schema.body, inputs=(Input("z", int_type),))


def test_identity_replacement_migrates_into_a_body_with_groups_and_choices() -> None:
    from dinkster_graph import Graph, GraphNode
    from dinkster_graph.migrate import migrate_node_replacements
    from dinkster_schema.body import Choice, Group, Named, Numbered, Option, Section, Use
    from dinkster_schema.identity import with_identity_replacement
    from dinkster_schema.model import Input, NodeSchema, Output, TypeExpr
    from dinkster_schema.replace import validate_replacement_references

    text = TypeExpr.concrete("STRING")
    number = TypeExpr.concrete("INT")
    audio = Choice(
        "audio",
        options=(Option("none"), Option("opus", rows=(Input("bitrate", number, default=96),))),
    )
    schema = with_identity_replacement(
        NodeSchema(
            "new.save",
            body=(
                Section("advanced", rows=(Input("seed", number, default=0),)),
                Group("images", value=Input("image", text), keys=Numbered("image", start=0)),
                Group(
                    "loras",
                    member=(Input("name", text), Input("strength", number, default=1)),
                    keys=Named(("first", "second")),
                ),
                Choice(
                    "format",
                    options=(
                        Option("mp4", rows=(Use("crf"),)),
                        Option("webm", rows=(Use("crf", default=31), audio)),
                    ),
                    shared=(Input("crf", number, default=23),),
                ),
                Output("file", text),
            ),
        ),
        "Old.Save",
    )
    old = GraphNode(
        "Old.Save",
        {
            "seed": 3,
            "images.image2": "b",
            "images.image0": "a",
            "loras.second.name": "x",
            "loras.second.strength": 2,
            "format": "webm",
            "format.crf": 30,
            "format.audio": "opus",
            "format.audio.bitrate": 128,
        },
    )
    assert validate_replacement_references({"new.save": schema}) == ()
    migrated = migrate_node_replacements(Graph({"n": old}), {"new.save": schema}).nodes["n"]
    assert isinstance(migrated, GraphNode)
    assert migrated.node_type == "new.save"
    assert dict(migrated.slot_variants) == {"format": "webm", "format.audio": "opus"}
    # The chosen keys move to the stored choices; every other value keeps its id.
    values = {key: value for key, value in old.inputs.items() if key not in migrated.slot_variants}
    assert {key: migrated.inputs[key] for key in values} == values
    # Member order survives: image2 was stored before image0.
    assert [key for key in migrated.inputs if key.startswith("images.")] == [
        "images.image2",
        "images.image0",
    ]
