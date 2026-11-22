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


def test_identity_replacement_offers_only_the_options_a_use_names() -> None:
    from dinkster_schema.body import Choice, Option, Use
    from dinkster_schema.identity import with_identity_replacement
    from dinkster_schema.model import NodeSchema

    codec = Choice("codec", options=(Option("h264"), Option("vp9"), Option("av1")))
    mp4 = Option("mp4", rows=(Use("codec", only=("h264", "av1")),))
    webm = Option("webm", rows=(Use("codec", only=("vp9",)),))
    schema = with_identity_replacement(
        NodeSchema("new.save", body=(Choice("format", (mp4, webm), shared=(codec,)),)),
        "Old.Save",
    )
    offered = {case.slot_variants for case in schema.replacements[0].cases}
    assert offered == {
        (("format", "mp4"), ("format.codec", "h264")),
        (("format", "mp4"), ("format.codec", "av1")),
        (("format", "webm"), ("format.codec", "vp9")),
    }


def test_a_fields_change_applies_only_steps_that_set_fields_and_may_read_previous() -> None:
    from dinkster_schema.body import Button, Choice, Formula, Option, Reset, Set, Toggle
    from dinkster_schema.model import Input, NodeSchema, TypeExpr
    from dinkster_schema.wire import schema_from_wire, schema_signature, schema_to_wire

    float_type = TypeExpr.concrete("FLOAT")
    x = Input("x", float_type, default=0.0)
    convert = (Set("x", Formula('x / 100 if previous(unit) == "percent" else x * 100')),)
    unit = Choice("unit", (Option("pixels"), Option("percent")), on_edit=convert)
    schema = NodeSchema("test.crop", body=(x, unit, Input("y", float_type, on_edit=(Reset("x"),))))
    back = schema_from_wire(schema_to_wire(schema))
    assert back.body == schema.body
    assert schema_signature(schema) == schema_signature(
        NodeSchema("test.crop", body=(x, Choice("unit", unit.options), Input("y", float_type)))
    )
    with pytest.raises(ValueError, match="unit on_edit: Toggle is not allowed"):
        NodeSchema("test.toggle", body=(x, Choice("unit", unit.options, on_edit=(Toggle("x"),))))
    with pytest.raises(ValueError, match="previous"):
        NodeSchema("test.button", body=(x, Button("convert", "Convert", steps=convert)))


def test_a_state_outside_the_run_is_kept_but_never_read_by_the_engine() -> None:
    from dinkster_schema.body import (
        Choice,
        Count,
        Formula,
        Group,
        Note,
        Numbered,
        Option,
        Rule,
        State,
    )
    from dinkster_schema.model import Input, NodeSchema, TypeExpr
    from dinkster_schema.wire import schema_from_wire, schema_signature, schema_to_wire

    int_type = TypeExpr.concrete("INT")
    last = State("last_width", int_type, default=0, part_of_run=False)
    x = Input("x", int_type, default=0)
    shown = Note("was", Formula('"was " + str(last_width)'))
    schema = NodeSchema("test.crop", body=(x, last, shown))
    assert [spec.id for spec in schema.inputs] == ["x"]
    assert schema_from_wire(schema_to_wire(schema)).body == schema.body
    assert schema_signature(schema) == schema_signature(NodeSchema("test.crop", body=(x,)))
    # A derived formula carries the read into the engine-worked formula that uses it.
    derived = {"wide": Formula("last_width > 100")}
    refused = {
        "choice mode leader": (
            Choice("mode", (Option("a"), Option("b")), leader=Formula('"a" if wide else "b"')),
        ),
        "group items count": (
            Group(
                "items",
                value=Input("v", int_type),
                keys=Numbered(""),
                grows=Count(Formula("last_width")),
            ),
        ),
    }
    for where, rows in refused.items():
        with pytest.raises(ValueError, match=f"{where}: reads 'last_width'"):
            NodeSchema("test.refused", body=(x, last, *rows), derived=derived)
    with pytest.raises(ValueError, match="validate rule: reads 'last_width'"):
        NodeSchema(
            "test.rule",
            body=(x, last),
            validate=(Rule(Formula("x <= last_width"), "too wide"),),
        )


def test_a_combo_reads_a_named_list_from_that_lists_route() -> None:
    from dinkster_schema.model import ComboWidget, NamedList

    named = ComboWidget(options=NamedList("devices.gpu"), refresh_button=True)
    assert named == ComboWidget(remote_route="/api/choices/devices.gpu", refresh_button=True)
    with pytest.raises(ValueError, match="not both"):
        ComboWidget(options=NamedList("devices.gpu"), remote_route="/api/choices/devices.cpu")


def test_a_whole_float_in_a_table_stays_a_float_on_the_wire() -> None:
    import json

    from dinkster_schema.model import Input, NodeSchema, TypeExpr
    from dinkster_schema.wire import schema_from_wire, schema_to_wire

    tables = {"scale": {"half": 0.5, "same": 1.0, "double": 2}}
    schema = NodeSchema("test.table", body=(Input("x", TypeExpr.concrete("INT")),), tables=tables)
    wire = json.loads(json.dumps(schema_to_wire(schema)))
    # A JavaScript reader parses 1.0 as 1, so a whole float is marked.
    assert wire["tables"]["scale"] == {
        "table": [["half", 0.5], ["same", {"float": 1.0}], ["double", 2]]
    }
    back = schema_from_wire(wire).tables["scale"]
    assert back == tables["scale"]
    assert [type(back[key]) for key in ("half", "same", "double")] == [float, float, int]


def test_an_optional_choice_cannot_declare_a_default() -> None:
    from dinkster_schema.body import Choice, Option

    with pytest.raises(ValueError, match="an optional choice cannot declare a default"):
        Choice("audio", options=(Option("opus"),), required=False, default="opus")
