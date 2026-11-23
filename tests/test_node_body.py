from typing import Any, cast

import pytest
from dinkster_assets.model import AssetRef
from dinkster_schema.body import ASSET_REF_CLASS, CallContext, lookup, pack_function


@lookup
def loras_in(folder: str, strength: float) -> list[str]:
    return [folder] * int(strength)


@lookup
def loras_by_name(folder: str) -> dict[str, str]:
    return {folder: folder}


@lookup
def loras_named(folder: str) -> list[str]:
    return [folder]


@lookup
def widest(folder: str) -> int | None:
    return len(folder) or None


@pack_function
def bake_note(count: int) -> dict[str, object]:
    return {"note": str(count)}


def unmarked(count: int) -> dict[str, object]:
    return {"note": str(count)}


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


def test_input_and_output_labels_and_visible_take_formulas_outside_the_signature() -> None:
    from dinkster_schema.body import Formula
    from dinkster_schema.model import Input, NodeSchema, Output, TypeExpr
    from dinkster_schema.wire import schema_from_wire, schema_signature, schema_to_wire

    int_type = TypeExpr.concrete("INT")
    mode = Input("mode", TypeExpr.concrete("BOOLEAN"), default=False)
    plain = (mode, Input("x", int_type), Output("y", int_type))
    schema = NodeSchema(
        "test.shown",
        body=(
            mode,
            Input("x", int_type, label=Formula('"Width" if mode else "Size"')),
            Output("y", int_type, label=Formula('"W" if mode else "S"'), visible=Formula("mode")),
        ),
    )
    wire = schema_to_wire(schema)
    rows = cast("list[dict[str, object]]", wire["body"])
    output = next(row for row in rows if row.get("id") == "y")
    assert output["label"] == {"formula": '"W" if mode else "S"'}
    assert output["visible"] == {"formula": "mode"}
    assert schema_from_wire(wire).body == schema.body
    assert schema_signature(schema) == schema_signature(NodeSchema("test.shown", body=plain))
    with pytest.raises(ValueError, match="missing"):
        NodeSchema("test.unknown", body=(mode, Output("y", int_type, visible=Formula("missing"))))
    with pytest.raises(ValueError, match="text or a Formula"):
        Output("y", int_type, label=cast("Any", 3))


def test_an_editor_names_the_rows_it_writes_and_reads_where_it_stands() -> None:
    from dinkster_schema.body import Choice, Editor, Group, Numbered, Option, Row, State
    from dinkster_schema.model import CompositorWidget, Input, NodeSchema, Output, TypeExpr
    from dinkster_schema.wire import schema_from_wire, schema_signature, schema_to_wire
    from dinkster_values import CustomWidgetDescriptor

    int_type = TypeExpr.concrete("INT")
    state = State("viewport_state", TypeExpr.concrete("STRING"), part_of_run=False)
    width = Input("width", int_type, default=512)
    mode = Choice("mode", (Option("a", rows=(Input("inner", int_type),)), Option("b")))
    viewport = Editor(
        "viewport",
        CustomWidgetDescriptor(
            "LOAD_3D",
            {"zoom": 2},
            writes={"state": "viewport_state", "size": "mode.inner"},
            reads={"width": "width", "mode": "mode"},
        ),
    )
    compositor = Editor("canvas", CompositorWidget(writes={"value": "width"}))
    body = (width, state, mode, viewport, compositor, Output("out", int_type))
    schema = NodeSchema("test.editor", body=body)
    wire = schema_to_wire(schema)
    rows = cast("list[dict[str, object]]", wire["body"])
    by_id = {row.get("id"): row for row in rows}
    assert by_id["viewport"]["widget"] == {
        "type": "LOAD_3D",
        "zoom": 2,
        "writes": {"state": "viewport_state", "size": "mode.inner"},
        "reads": {"width": "width", "mode": "mode"},
    }
    assert by_id["canvas"]["widget"] == {"type": "COMPOSITOR", "writes": {"value": "width"}}
    assert schema_from_wire(wire).body == schema.body
    # Layout: no stored interface, no signature.
    bare = NodeSchema("test.editor", body=(width, state, mode, Output("out", int_type)))
    assert schema.inputs == bare.inputs
    assert schema_signature(schema) == schema_signature(bare)
    # A member's editor names the member's rows.
    in_member = Editor("e", CompositorWidget(reads={"w": "width"}))
    NodeSchema("test.member", body=(Group("items", member=(width, in_member), keys=Numbered("")),))

    refused = {
        "writes 'mode', which is no Input or State": Editor(
            "e", CustomWidgetDescriptor("LOAD_3D", writes={"state": "mode"})
        ),
        "names no row here: 'missing'": Editor(
            "e", CustomWidgetDescriptor("LOAD_3D", reads={"model": "missing"})
        ),
        "names no row here: 'inner'": Editor(
            "e", CustomWidgetDescriptor("LOAD_3D", reads={"model": "inner"})
        ),
    }
    for message, editor in refused.items():
        with pytest.raises(ValueError, match=message):
            NodeSchema("test.refused", body=(width, state, mode, editor))
    with pytest.raises(ValueError, match="editor e is declared twice"):
        NodeSchema("test.twice", body=(width, in_member, in_member))
    with pytest.raises(ValueError, match="names no row in writes or reads"):
        Editor("e", CompositorWidget())
    with pytest.raises(ValueError, match="in both writes and reads"):
        CompositorWidget(writes={"value": "width"}, reads={"size": "width"})
    with pytest.raises(ValueError, match="Row holds Input, Button and Note rows only"):
        Row(cast("Any", (width, compositor)))
    own = CustomWidgetDescriptor("RESOLUTION", reads={"width": "width"})
    with pytest.raises(ValueError, match="names its own input"):
        NodeSchema("test.own", body=(Input("width", int_type, widget=own),))
    NodeSchema(
        "test.other",
        body=(width, Input("height", int_type, widget=own)),
    )


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


def test_a_field_not_sent_is_kept_but_never_read_by_the_engine() -> None:
    from dinkster_schema.body import Choice, Formula, Option, Rule
    from dinkster_schema.model import Input, NodeSchema, TypeExpr
    from dinkster_schema.wire import schema_from_wire, schema_signature, schema_to_wire

    int_type = TypeExpr.concrete("INT")
    x = Input("x", int_type, default=0)
    last = Input("last_width", int_type, default=0, pin=False, sent=False)
    schema = NodeSchema("test.unsent", body=(x, last))
    assert [spec.id for spec in schema.inputs] == ["x"]
    wire = schema_to_wire(schema)
    assert {"id": "last_width", "sent": False}.items() <= cast("Any", wire["body"])[1].items()
    assert schema_from_wire(wire).body == schema.body
    assert schema_signature(schema) == schema_signature(NodeSchema("test.unsent", body=(x,)))
    with pytest.raises(ValueError, match="not sent cannot be wired"):
        Input("last_width", int_type, default=0, sent=False)
    leader = Choice(
        "mode", (Option("a"), Option("b")), leader=Formula('"a" if last_width else "b"')
    )
    with pytest.raises(ValueError, match="choice mode leader: reads 'last_width'.*sent=False"):
        NodeSchema("test.refused", body=(x, last, leader))
    with pytest.raises(ValueError, match="validate rule: reads 'last_width'"):
        NodeSchema(
            "test.rule",
            body=(x, last),
            validate=(Rule(Formula("x < last_width"), "no"),),
        )


def test_options_completions_and_limits_take_a_formula_or_a_lookup() -> None:
    from dinkster_schema.body import Formula, Lookup, State
    from dinkster_schema.model import (
        ComboWidget,
        Input,
        NodeSchema,
        NumberWidget,
        StringWidget,
        TypeExpr,
    )
    from dinkster_schema.wire import schema_from_wire, schema_to_wire

    string = TypeExpr.concrete("STRING")
    int_type = TypeExpr.concrete("INT")

    def node(lora: object = None, width: object = None, prompt: object = None) -> NodeSchema:
        return NodeSchema(
            "test.looked",
            body=(
                Input("folder", string, default="sd"),
                Input("seen", string, default="", pin=False, sent=False),
                State("count", int_type, default=0, part_of_run=False),
                Input("lora", TypeExpr.concrete("COMBO"), widget=cast("Any", lora)),
                Input("width", int_type, default=8, widget=cast("Any", width)),
                Input("prompt", string, default="", widget=cast("Any", prompt)),
            ),
        )

    # Formulas and lookups here may read what a run never receives.
    loaded = node(
        ComboWidget(options=Lookup(loras_named, args=("seen",))),
        NumberWidget(min=Formula("count"), max=Lookup(widest, args=("folder",)), step=2),
        StringWidget(completions=Formula('split(seen, ",")')),
    )
    wire = schema_to_wire(loaded)
    widgets = [row.get("widget") for row in cast("list[dict[str, Any]]", wire["body"])]
    assert widgets[3:] == [
        {"type": "COMBO", "options": {"lookup": f"{__name__}.loras_named", "args": ["seen"]}},
        {
            "type": "NUMBER",
            "min": {"formula": "count"},
            "max": {"lookup": f"{__name__}.widest", "args": ["folder"]},
            "step": 2,
        },
        {"type": "STRING", "multiline": True, "completions": {"formula": 'split(seen, ",")'}},
    ]
    assert schema_to_wire(schema_from_wire(wire)) == wire

    with pytest.raises(ValueError, match="loras_in: takes 2 to 2 arguments, given 1"):
        node(ComboWidget(options=Lookup(loras_in, args=("folder",))))
    with pytest.raises(ValueError, match="answers .*str.* where a list of options is used"):
        node(ComboWidget(options=Lookup(loras_by_name, args=("folder",))))
    with pytest.raises(ValueError, match="where a number or None for max is used"):
        node(width=NumberWidget(max=Lookup(loras_named, args=("folder",))))
    with pytest.raises(ValueError, match="where a list of completions is used"):
        node(prompt=StringWidget(completions=Lookup(widest, args=("folder",))))
    with pytest.raises(ValueError, match="height"):
        node(width=NumberWidget(max=Formula("height")))
    with pytest.raises(ValueError, match="no other option source"):
        ComboWidget(options=Lookup(loras_named, args=("folder",)), refresh_button=True)


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
    assert isinstance(back, dict)
    assert back == tables["scale"]
    assert [type(back[key]) for key in ("half", "same", "double")] == [float, float, int]


def test_an_optional_choice_cannot_declare_a_default() -> None:
    from dinkster_schema.body import Choice, Option

    with pytest.raises(ValueError, match="an optional choice cannot declare a default"):
        Choice("audio", options=(Option("opus"),), required=False, default="opus")


def test_applies_on_a_body_node_names_a_required_choice_and_resolves_by_its_option() -> None:
    from dinkster_schema.body import Choice, Option
    from dinkster_schema.elaborate import elaborate
    from dinkster_schema.model import Input, MirrorSpec, MirrorTolerance, NodeSchema, TypeExpr

    mirror = MirrorSpec(
        kind="glsl",
        precision="bounded",
        tolerance=MirrorTolerance(per_channel=1.0 / 255.0),
        source="void main() {}",
        applies={"operation": ("blur", "sharpen")},
    )
    operation = Choice("operation", options=(Option("blur"), Option("sharpen"), Option("noise")))
    body = (Input("a", TypeExpr.concrete("INT")), operation)
    schema = NodeSchema("test.scoped", body=body, mirror=mirror)
    covered = elaborate(schema, ("a",), slot_variants={"operation": "sharpen"}).mirror
    assert covered is not None and covered.applies is None
    assert elaborate(schema, ("a",), slot_variants={"operation": "noise"}).mirror is None
    optional = (body[0], Choice("operation", options=operation.options, required=False))
    with pytest.raises(ValueError, match="must name a required choice"):
        NodeSchema("test.optional", body=optional, mirror=mirror)


def test_lookups_and_pack_functions_are_checked_against_the_fields_they_are_handed() -> None:
    from dinkster_schema.body import Button, Call, Lookup, Pick
    from dinkster_schema.model import Input, NodeSchema, TypeExpr
    from dinkster_schema.wire import schema_from_wire, schema_to_wire

    def node(*steps: object) -> NodeSchema:
        return NodeSchema(
            "test.lookups",
            body=(
                Input("folder", TypeExpr.concrete("STRING"), default="loras"),
                Input("count", TypeExpr.concrete("INT"), default=1),
                Input("scale", TypeExpr.concrete("FLOAT"), default=0.5),
                Input("note", TypeExpr.concrete("STRING"), default=""),
                Button("go", "Go", steps=cast("Any", steps)),
            ),
        )

    # A whole-number field fits a float parameter; the names survive the wire.
    loaded = node(
        Pick("lora", Lookup(loras_in, args=("folder", "count"))),
        Call(bake_note, args=("count",), writes=("note",)),
    )
    wire = schema_to_wire(loaded)
    assert f"{__name__}.loras_in" in str(wire) and f"{__name__}.bake_note" in str(wire)
    assert schema_to_wire(schema_from_wire(wire)) == wire

    # A field holding a decimal number does not fit an int parameter.
    with pytest.raises(ValueError, match="field 'scale' does not fit parameter count"):
        node(Call(bake_note, args=("scale",), writes=("note",)))
    with pytest.raises(ValueError, match="takes 2 to 2 arguments, given 1"):
        node(Pick("lora", Lookup(loras_in, args=("folder",))))
    with pytest.raises(ValueError, match="a list of options"):
        node(Pick("lora", Lookup(loras_by_name, args=("folder",))))
    with pytest.raises(ValueError, match="unmarked is used without @pack_function"):
        Call(unmarked, args=("count",), writes=("note",))
    with pytest.raises(ValueError, match="cannot be a lambda"):
        lookup(lambda folder: [folder])

    def nested(folder: str) -> list[str]:
        return [folder]

    with pytest.raises(ValueError, match="not inside a function or a class"):
        pack_function(nested)


class Secret(str):
    """Stands for a secret annotation, which pack functions cannot take yet."""


@pack_function
def with_context(context: CallContext, count: int) -> dict[str, object]:
    return {"note": str(count)}


@pack_function
def with_secret(count: int, key: Secret) -> dict[str, object]:
    return {"note": str(count)}


@pack_function
def context_second(count: int, context: CallContext) -> dict[str, object]:
    return {"note": str(count)}


@pack_function
def takes_asset(image: AssetRef) -> dict[str, object]:
    return {"note": str(image)}


@lookup
def looks_with_context(context: CallContext, folder: str) -> list[str]:
    return [folder]


def test_button_steps_are_checked_where_they_run() -> None:
    from dinkster_schema.body import (
        AddNode,
        Button,
        Call,
        Disconnect,
        Formula,
        Group,
        Lookup,
        Numbered,
        Pick,
        Rule,
        SetEach,
    )
    from dinkster_schema.model import Input, NodeSchema, TypeExpr

    def node(*steps: object) -> NodeSchema:
        return NodeSchema(
            "test.steps",
            body=(
                Input("folder", TypeExpr.concrete("STRING"), default="loras"),
                Input("count", TypeExpr.concrete("INT"), default=1),
                Input("note", TypeExpr.concrete("STRING"), default=""),
                Input("image", TypeExpr.concrete("IMAGE"), required=False),
                Group(
                    "loras",
                    keys=Numbered("lora_"),
                    grows="button",
                    member=(Input("strength", TypeExpr.concrete("FLOAT"), default=1.0),),
                ),
                Button("go", "Go", steps=cast("Any", steps)),
            ),
        )

    # The host fills a first CallContext; the button hands only the rest.
    node(Call(with_context, args=("count",), writes=("note",)))
    with pytest.raises(ValueError, match="takes 1 to 1 arguments, given 2"):
        node(Call(with_context, args=("count", "count"), writes=("note",)))
    with pytest.raises(ValueError, match="Secrets for pack functions are not built yet"):
        node(Call(with_secret, args=("count", "folder"), writes=("note",)))
    with pytest.raises(ValueError, match="the CallContext parameter is the first one"):
        node(Call(context_second, args=("count",), writes=("note",)))
    with pytest.raises(ValueError, match="a lookup gets no CallContext"):
        node(Pick("lora", Lookup(looks_with_context, args=("folder",))))
    # A group is never a Call's write.
    with pytest.raises(ValueError, match="'loras' names no field"):
        node(Call(bake_note, args=("count",), writes=("loras",)))

    # SetEach works its value out in each member's scope, where the member's rows are names.
    node(SetEach("loras", "strength", Formula("strength / 2")))
    node(SetEach("loras", "strength", Formula("loras.strength[0]")))
    with pytest.raises(ValueError, match="unknown name 'strength'"):
        node(Call(bake_note, args=(Formula("int(strength)"),), writes=("note",)))

    # Disconnect and AddNode name an input of this node.
    node(Disconnect("image"), AddNode("dev.image.gradient", to="image"))
    with pytest.raises(ValueError, match="'loras' names no input of this node"):
        node(AddNode("dev.image.gradient", to="loras"))
    with pytest.raises(ValueError, match="'missing' names no field"):
        node(Disconnect("missing"))

    # An asset parameter, a Call in a rule's fix, and an input with no pin are refused.
    with pytest.raises(ValueError, match="parameter image is an AssetRef"):
        node(Call(takes_asset, args=("image",), writes=("note",)))
    with pytest.raises(ValueError, match="a Rule's fix cannot Call"):
        Rule(Formula("count > 0"), "Needs a count", fix=(Call(bake_note, ("count",), ("note",)),))
    with pytest.raises(ValueError, match="input 'seen' has no pin"):
        NodeSchema(
            "test.hidden",
            body=(
                Input("seen", TypeExpr.concrete("STRING"), default="", pin=False, sent=False),
                Button("go", "Go", steps=(Disconnect("seen"),)),
            ),
        )

    # The node's menu sees the body's fields and groups as a body button does.
    NodeSchema(
        "test.menu",
        body=node(Disconnect("image")).body[:-1],
        menu=(
            Button(
                "spread",
                "Spread",
                steps=cast("Any", (SetEach("loras", "strength", Formula("strength / 2")),)),
            ),
        ),
    )


def test_randomize_takes_a_number_a_yes_or_no_field_or_written_options() -> None:
    from dinkster_schema.body import Button, Lookup, Randomize
    from dinkster_schema.model import ComboWidget, Input, NamedList, NodeSchema, TypeExpr

    combo = TypeExpr.concrete("COMBO")

    def node(*steps: object) -> NodeSchema:
        return NodeSchema(
            "test.random",
            body=(
                Input("seed", TypeExpr.concrete("INT"), default=1),
                Input("cfg", TypeExpr.concrete("FLOAT"), default=1.0),
                Input("flag", TypeExpr.concrete("BOOLEAN"), default=False),
                Input("note", TypeExpr.concrete("STRING"), default=""),
                Input("seen", TypeExpr.concrete("STRING"), default="", pin=False, sent=False),
                Input("size", combo, default="m", widget=ComboWidget(options=("s", "m", "l"))),
                Input(
                    "lora", combo, widget=ComboWidget(options=Lookup(loras_named, args=("seen",)))
                ),
                Input("gpu", combo, widget=ComboWidget(options=NamedList("devices.gpu"))),
                Button("go", "Go", steps=cast("Any", steps)),
            ),
        )

    node(Randomize("seed", 1, 9), Randomize("cfg"), Randomize("flag"), Randomize("size"))
    with pytest.raises(ValueError, match="'note' is a text field"):
        node(Randomize("note"))
    for listed in ("lora", "gpu"):
        with pytest.raises(
            ValueError, match=f"'{listed}' is a choice whose options are not written"
        ):
            node(Randomize(listed))
    with pytest.raises(ValueError, match="'flag' is a yes or no field; low and high"):
        node(Randomize("flag", 0))
    with pytest.raises(ValueError, match="'size' is a choice; low and high"):
        node(Randomize("size", high="l"))
    # A menu button names the same fields.
    NodeSchema(
        "test.random",
        body=node(Randomize("seed")).body[:-1],
        menu=(Button("roll", "Roll", steps=(Randomize("size"),)),),
    )


def test_identity_replacement_starts_a_choice_with_nothing_stored_where_the_editor_does() -> None:
    from dinkster_schema.body import Choice, Option
    from dinkster_schema.identity import with_identity_replacement
    from dinkster_schema.model import Input, NodeSchema, TypeExpr

    int_type = TypeExpr.concrete("INT")
    options = (
        Option("a", rows=(Input("x", int_type, default=1),)),
        Option("b", rows=(Input("y", int_type, default=2),)),
    )

    def fallback(choice: Choice) -> tuple[list[str], dict[str, str]]:
        schema = with_identity_replacement(NodeSchema("new.node", body=(choice,)), "Old")
        (case,) = [case for case in schema.replacements[0].cases if case.when is None]
        return sorted(dict(case.inputs)), dict(case.slot_variants)

    # A required choice starts on its default; an optional one on none.
    assert fallback(Choice("c", options, default="b")) == (["c.y"], {"c": "b"})
    assert fallback(Choice("c", options)) == (["c.x"], {"c": "a"})
    assert fallback(Choice("c", options, required=False)) == ([], {})


def test_a_schema_with_no_rows_keeps_its_body_fields_on_the_wire() -> None:
    from dinkster_schema.body import Button, Formula, Queue
    from dinkster_schema.model import NodeSchema
    from dinkster_schema.wire import schema_from_wire, schema_to_wire

    schema = NodeSchema(
        "test.title", title_suffix=Formula('"draft"'), menu=(Button("go", "Go", (Queue(),)),)
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema


@pytest.mark.parametrize(
    ("row", "field", "value"),
    [(0, "adjacentTo", "a"), (0, "advanced", True), (1, "advanced", True)],
)
def test_the_wire_refuses_the_layout_flags_sections_and_rows_replaced(
    row: int, field: str, value: object
) -> None:
    from dinkster_schema.model import Input, NodeSchema, Output, TypeExpr
    from dinkster_schema.wire import schema_from_wire, schema_to_wire

    int_type = TypeExpr.concrete("INT")
    wire = schema_to_wire(
        NodeSchema("test.old", body=(Input("a", int_type), Output("o", int_type)))
    )
    cast("list[dict[str, object]]", wire["body"])[row][field] = value
    with pytest.raises(ValueError, match=f"unknown fields: \\['{field}'\\]"):
        schema_from_wire(wire)
