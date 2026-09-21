import dataclasses
import json
from typing import Any, cast

import pytest
from dinkster_schema import (
    OPTION_KEY_PATTERN,
    AssetWidget,
    BooleanWidget,
    ColorWidget,
    ComboWidget,
    Deprecation,
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    MirrorSpec,
    MirrorTolerance,
    MultiComboWidget,
    Node,
    NodeOutputError,
    NodeSchema,
    NumberWidget,
    OutputFamilySpec,
    OutputKnownValue,
    OutputRepresents,
    OutputSpec,
    SaveTargetWidget,
    SlotVariant,
    StringWidget,
    TextCompletionItem,
    TextCompletions,
    TypeExpr,
    WidgetRepresentation,
    WidgetRepresentations,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)

SCHEMA = NodeSchema(
    node_type="test.node",
    display_name="Test Node",
    category="test",
    inputs=(
        InputSpec("a", TypeExpr.concrete("core.int")),
        InputSpec("b", TypeExpr.union("core.int", "core.float"), required=False, default=1),
        InputSpec("anything", TypeExpr.wildcard(), required=False),
    ),
    outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
)


def _represented_schema(
    *,
    represents: OutputRepresents | None = None,
    combos: tuple[DynamicComboSpec, ...] = (),
) -> NodeSchema:
    image = TypeExpr.concrete("dinkster.image")
    return NodeSchema(
        node_type="test.represented-output",
        inputs=(
            InputSpec(
                "image",
                TypeExpr.asset_of(image),
                widget=AssetWidget(kind="media/image"),
            ),
        ),
        outputs=(
            OutputSpec(
                "image",
                image,
                represents=represents or OutputRepresents("image", "decoded-image"),
            ),
        ),
        combos=combos,
    )


EXPRESSION_MIRROR = MirrorSpec(
    kind="expression",
    precision="bounded",
    tolerance=MirrorTolerance(relative=1e-12),
    grammar_version=1,
)


GLSL_MIRROR = MirrorSpec(
    kind="glsl",
    precision="bounded",
    tolerance=MirrorTolerance(per_channel=1.0 / 255.0),
    source="void main() {}",
)


SCOPED_GLSL_MIRROR = dataclasses.replace(
    GLSL_MIRROR, applies={"operation": ("gaussian_blur", "sharpen")}
)


OPERATION_COMBO = DynamicComboSpec(
    "operation",
    options=(
        DynamicComboOption("gaussian_blur"),
        DynamicComboOption("sharpen"),
        DynamicComboOption("noise"),
    ),
    default="gaussian_blur",
)


SCOPED_SCHEMA = dataclasses.replace(SCHEMA, combos=(OPERATION_COMBO,), mirror=SCOPED_GLSL_MIRROR)


class TwoOutputs(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.two_outputs",
            outputs=(
                OutputSpec("first", TypeExpr.concrete("core.int")),
                OutputSpec("second", TypeExpr.concrete("core.int")),
            ),
        )


def test_deprecation_and_visibility_ride_the_wire() -> None:
    """Lifecycle metadata is additive wire data: deprecation {message,
    since?, replacement?} and searchVisibility, both omitted at their
    defaults (omission means "not deprecated"/"normal", never null)."""
    schema = dataclasses.replace(
        SCHEMA,
        deprecation=Deprecation(
            message="use test.better: same math, honest name",
            since="2.0.0",
            replacement="test.better",
        ),
        search_visibility="hidden",
    )
    wire = schema_to_wire(schema)
    assert wire["deprecation"] == {
        "message": "use test.better: same math, honest name",
        "since": "2.0.0",
        "replacement": "test.better",
    }
    assert wire["searchVisibility"] == "hidden"
    assert schema_from_wire(wire) == schema

    # Defaults never appear on the wire; optional Deprecation fields either.
    plain = schema_to_wire(SCHEMA)
    assert "deprecation" not in plain
    assert "searchVisibility" not in plain
    minimal = dataclasses.replace(SCHEMA, deprecation=Deprecation(message="bye"))
    assert schema_to_wire(minimal)["deprecation"] == {"message": "bye"}


def test_presentation_never_joins_the_signature() -> None:
    """Hazard H15: presentation is pixels, never identity. Renaming a node's
    badge, moving it to another menu, or rewording its tooltip/port docs must
    never invalidate caches or migrate workflows."""
    renamed = dataclasses.replace(
        SCHEMA,
        display_name="Completely Different Name",
        category="other/menu",
        description="reworded prose",
        inputs=tuple(dataclasses.replace(spec, doc="new tooltip") for spec in SCHEMA.inputs),
        outputs=tuple(dataclasses.replace(spec, doc="new tooltip") for spec in SCHEMA.outputs),
    )
    assert schema_signature(renamed) == schema_signature(SCHEMA)


def test_output_preview_authoring_requires_an_actual_bool() -> None:
    for malformed in (0, 1, "true", None, [], {}):
        with pytest.raises(ValueError, match="preview must be a bool"):
            OutputSpec(
                "value",
                TypeExpr.concrete("core.string"),
                preview=malformed,  # type: ignore[arg-type]
            )


def test_output_family_preview_authoring_requires_an_actual_bool() -> None:
    for malformed in (0, 1, "true", None, [], {}):
        with pytest.raises(ValueError, match="preview must be a bool"):
            OutputFamilySpec(
                "values",
                TypeExpr.concrete("core.string"),
                preview=malformed,  # type: ignore[arg-type]
            )


def test_output_preview_signature_stripping_never_reaches_input_defaults() -> None:
    empty_default = NodeSchema(
        node_type="test.preview-default",
        inputs=(InputSpec("value", TypeExpr.wildcard(), default={}),),
    )
    preview_key_default = dataclasses.replace(
        empty_default,
        inputs=(dataclasses.replace(empty_default.inputs[0], default={"preview": True}),),
    )
    assert schema_signature(empty_default) != schema_signature(preview_key_default)


def test_input_family_vocabulary_validation_and_wire_shape() -> None:
    prefix = InputFamilySpec(
        "items",
        TypeExpr.concrete("core.string"),
        min_members=1,
        max_members=3,
        member_prefix="items",
    )
    names = InputFamilySpec(
        "named",
        TypeExpr.concrete("core.string"),
        member_names=(),
    )
    wire = schema_to_wire(
        NodeSchema(node_type="test.family-vocabulary", input_families=(prefix, names))
    )
    entries = cast("list[dict[str, Any]]", wire["interface"])
    assert entries[0]["memberPrefix"] == "items"
    assert entries[0]["template"][0]["role"] == "input"
    assert "type" not in entries[0]
    assert entries[1]["memberNames"] == []
    assert "maxMembers" not in entries[1]
    assert schema_from_wire(wire).input_families == (prefix, names)

    with pytest.raises(ValueError, match="mutually exclusive"):
        InputFamilySpec("bad", TypeExpr.wildcard(), member_prefix="x", member_names=("x",))
    with pytest.raises(ValueError, match="duplicate member_names"):
        InputFamilySpec("bad", TypeExpr.wildcard(), member_names=("x", "x"))
    with pytest.raises(ValueError, match="capacity"):
        InputFamilySpec("bad", TypeExpr.wildcard(), min_members=1, member_names=())
    with pytest.raises(ValueError, match="template must not be empty"):
        InputFamilySpec("bad", ())
    assert (
        InputFamilySpec(
            "large", TypeExpr.wildcard(), max_members=1000, member_prefix="item_"
        ).max_members
        == 1000
    )
    with pytest.raises(ValueError, match=r"\[1, 1000\]"):
        InputFamilySpec("too_large", TypeExpr.wildcard(), max_members=1001, member_prefix="item_")


def test_recursive_dynamic_entries_round_trip_at_every_depth() -> None:
    variable = TypeExpr.variable("T", ("core.string", "core.float"))
    nested = DynamicComboSpec(
        "mode",
        options=(
            DynamicComboOption(
                "batch",
                inputs=(
                    DynamicComboSpec(
                        "subcombo",
                        options=(
                            DynamicComboOption(
                                "one",
                                inputs=(InputSpec("value", variable),),
                            ),
                        ),
                    ),
                    InputFamilySpec(
                        "frames",
                        (InputSpec("frame", TypeExpr.list_of(variable)),),
                        member_prefix="image",
                    ),
                    DynamicSlotSpec(
                        "source",
                        variants=(
                            SlotVariant(
                                "text",
                                TypeExpr.concrete("core.string"),
                                inputs=(InputSpec("variant_dep", variable),),
                            ),
                        ),
                        inputs=(InputSpec("shared", variable),),
                    ),
                    DynamicSlotSpec(
                        "open",
                        slot_type=TypeExpr.concrete("core.string"),
                        inputs=(InputSpec("dependent", variable),),
                        force_input=True,
                    ),
                ),
            ),
        ),
        default="batch",
    )
    schema = NodeSchema(node_type="test.recursive", combos=(nested,))
    wire = schema_to_wire(schema)
    assert schema_from_wire(wire) == schema


def test_recursive_wire_decode_is_fail_closed() -> None:
    family = InputFamilySpec(
        "items",
        (InputSpec("value", TypeExpr.concrete("core.string")),),
    )
    base = schema_to_wire(NodeSchema(node_type="test.fail-closed", input_families=(family,)))
    entry = cast("list[dict[str, Any]]", base["interface"])[0]

    malformed = [
        {**entry, "type": {"kind": "wildcard"}},
        {**entry, "template": []},
        {**entry, "template": [{"role": "future", "id": "x"}]},
        {
            **entry,
            "template": [
                {"role": "input", "id": "x", "type": {"kind": "wildcard"}},
                {"role": "input", "id": "x", "type": {"kind": "wildcard"}},
            ],
        },
        {
            **entry,
            "template": [{"role": "input", "id": "bad.id", "type": {"kind": "wildcard"}}],
        },
        {
            **entry,
            "template": [{"role": "input", "id": "x", "type": {"kind": "future"}}],
        },
        {
            **entry,
            "template": [
                {
                    "role": "input",
                    "id": "x",
                    "type": {"kind": "wildcard"},
                    "required": True,
                    "onAbsent": "omit",
                }
            ],
        },
    ]
    for bad_entry in malformed:
        bad = {**base, "interface": [bad_entry]}
        with pytest.raises(ValueError):
            schema_from_wire(cast("dict[str, Any]", bad))


def test_dynamic_slot_forms_and_combo_default_validation() -> None:
    concrete = TypeExpr.concrete("core.string")
    with pytest.raises(ValueError, match="exactly one"):
        DynamicSlotSpec("neither")
    with pytest.raises(ValueError, match="exactly one"):
        DynamicSlotSpec("both", variants=(SlotVariant("x", concrete),), slot_type=concrete)
    with pytest.raises(ValueError, match="must not be a variable"):
        DynamicSlotSpec("variable", slot_type=TypeExpr.variable("T"))
    with pytest.raises(ValueError, match="must be optional"):
        DynamicSlotSpec("required_open", slot_type=concrete, required=True)
    with pytest.raises(ValueError, match="default"):
        DynamicComboSpec("combo", options=(DynamicComboOption("x"),), default="missing")
    with pytest.raises(ValueError, match="only input dynamic entries"):
        InputFamilySpec("outputs", (OutputSpec("bad", concrete),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key",
    ["Flux.2 [pro]", "Flux.2 [max]", "scale dimensions", "center"],
)
def test_dynamic_combo_option_key_grammar_accepts_printable_ascii_tokens(
    key: str,
) -> None:
    assert OPTION_KEY_PATTERN.fullmatch(key)
    assert DynamicComboOption(key).key == key


@pytest.mark.parametrize(
    "key",
    ["", " x", "x ", "a  b", " ", "non-ascii-\u00e9", "Flux.2 [pro]\n"],
)
def test_dynamic_combo_option_key_grammar_rejects_other_shapes(key: str) -> None:
    with pytest.raises(ValueError, match="dynamic combo option key must match"):
        DynamicComboOption(key)


def test_dynamic_combo_option_keys_keep_exact_identity_and_default_membership() -> None:
    spaced = DynamicComboOption(
        "Flux.2 [pro]",
        (InputSpec("width", TypeExpr.concrete("core.string")),),
    )
    other = DynamicComboOption("Flux.2 [max]")
    combo = DynamicComboSpec(
        "mode",
        (spaced, other),
        default="Flux.2 [pro]",
    )
    schema = NodeSchema(node_type="test.space-option-wire", combos=(combo,))

    assert combo.option("Flux.2 [pro]") is spaced
    assert combo.option("Flux.2 [max]") is other
    assert schema_from_wire(schema_to_wire(schema)) == schema
    with pytest.raises(ValueError, match="duplicate option keys"):
        DynamicComboSpec(
            "mode",
            (DynamicComboOption("scale dimensions"), DynamicComboOption("scale dimensions")),
        )
    with pytest.raises(ValueError, match="dynamic combo option key must match"):
        DynamicComboSpec(
            "mode",
            (DynamicComboOption("scale dimensions "),),
            default="scale dimensions ",
        )


@pytest.mark.parametrize(
    "keys",
    [("a [x", "a [x] y"), ("a [x] y", "a [x")],
)
def test_dynamic_combo_option_keys_reject_frontend_branch_aliases(
    keys: tuple[str, str],
) -> None:
    with pytest.raises(ValueError, match="alias frontend branch paths") as excinfo:
        DynamicComboSpec(
            "mode",
            tuple(DynamicComboOption(key) for key in keys),
        )
    assert all(repr(key) in str(excinfo.value) for key in keys)


def test_dynamic_combo_option_keys_allow_non_aliasing_bfl_siblings() -> None:
    combo = DynamicComboSpec(
        "mode",
        (DynamicComboOption("Flux.2 [pro]"), DynamicComboOption("Flux.2 [max]")),
    )
    assert tuple(option.key for option in combo.options) == (
        "Flux.2 [pro]",
        "Flux.2 [max]",
    )


def test_dynamic_combo_option_key_grammar_does_not_widen_other_identifiers() -> None:
    concrete = TypeExpr.concrete("core.string")
    with pytest.raises(ValueError, match="dynamic combo id must match"):
        DynamicComboSpec("combo.id", (DynamicComboOption("valid"),))
    with pytest.raises(ValueError, match="input family id must match"):
        InputFamilySpec("family.id", concrete)
    with pytest.raises(ValueError, match="dynamic slot id must match"):
        DynamicSlotSpec("slot.id", slot_type=concrete, required=False)
    with pytest.raises(ValueError, match="entry id must match"):
        DynamicComboOption(
            "valid",
            (InputSpec("dependent.input", concrete),),
        )
    with pytest.raises(ValueError, match="slot variant key must match"):
        SlotVariant("variant.key", concrete)
    with pytest.raises(ValueError, match="input family member_prefix must match"):
        InputFamilySpec("family", concrete, member_prefix="member.prefix")
    with pytest.raises(ValueError, match="input family member name must match"):
        InputFamilySpec(
            "family",
            concrete,
            member_names=("member.name",),
        )


def test_anchored_grammars_reject_trailing_newlines() -> None:
    with pytest.raises(ValueError, match="dynamic combo id must match"):
        DynamicComboSpec("combo\n", (DynamicComboOption("valid"),))
    with pytest.raises(ValueError, match="entry id must match"):
        DynamicComboOption(
            "valid",
            (InputSpec("dependent\n", TypeExpr.concrete("core.string")),),
        )
    with pytest.raises(ValueError, match="slot variant key must match"):
        SlotVariant("variant\n", TypeExpr.concrete("core.string"))
    with pytest.raises(ValueError, match="asset widget kind"):
        AssetWidget(kind="model/lora\n")


def test_asset_widget_wire_roundtrip_and_signature_exclusion() -> None:
    """Widget data is a typed frontend presentation contract, but pixels
    cannot change execution identity or invalidate cached computation."""
    accept = ("image/png", "image/jpeg", "image/webp")
    schema = dataclasses.replace(
        SCHEMA,
        inputs=(
            InputSpec("asset", TypeExpr.concrete("dinkster.asset"), widget=AssetWidget(accept)),
        )
        + SCHEMA.inputs[1:],
    )
    wire = schema_to_wire(schema)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert interface[0]["widget"] == {"type": "ASSET", "accept": list(accept)}
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) == schema_signature(
        dataclasses.replace(
            schema,
            inputs=(dataclasses.replace(schema.inputs[0], widget=None),) + schema.inputs[1:],
        )
    )


def test_save_target_widget_wire_roundtrip_and_signature_exclusion() -> None:
    """SAVE_TARGET is a closed widget kind: a
    save-destination picker whose value is a structured dinkster.save_target
    mapping, never a raw path string. Same rule as ASSET: presentation
    metadata rides the wire but never joins execution identity."""
    schema = dataclasses.replace(
        SCHEMA,
        inputs=(
            InputSpec(
                "target",
                TypeExpr.concrete("dinkster.save_target"),
                widget=SaveTargetWidget(".png"),
            ),
        )
        + SCHEMA.inputs[1:],
    )
    wire = schema_to_wire(schema)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert interface[0]["widget"] == {"type": "SAVE_TARGET", "suffix": ".png"}
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) == schema_signature(
        dataclasses.replace(
            schema,
            inputs=(dataclasses.replace(schema.inputs[0], widget=None),) + schema.inputs[1:],
        )
    )

    # Empty suffix is omitted from the wire (omission, never null).
    bare = dataclasses.replace(
        SCHEMA,
        inputs=(
            InputSpec(
                "target",
                TypeExpr.concrete("dinkster.save_target"),
                widget=SaveTargetWidget(),
            ),
        )
        + SCHEMA.inputs[1:],
    )
    bare_wire = schema_to_wire(bare)
    bare_interface = cast("list[dict[str, Any]]", bare_wire["interface"])
    assert bare_interface[0]["widget"] == {"type": "SAVE_TARGET"}
    assert schema_from_wire(bare_wire) == bare


def test_combo_widget_wire_roundtrip_and_signature_exclusion() -> None:
    """COMBO is a closed widget kind bound to core.combo. Static options
    render immediately; a remote route names a
    server-enumerated list (registered samplers and kin) that REPLACES
    them once fetched. Same rule as ASSET/SAVE_TARGET: presentation rides
    the wire but never joins execution identity."""
    combo = ComboWidget(
        options=("euler", "dpmpp_2m"),
        remote_route="/api/choices/test.samplers",
        refresh_button=True,
    )
    schema = dataclasses.replace(
        SCHEMA,
        inputs=(InputSpec("choice", TypeExpr.concrete("core.combo"), widget=combo),)
        + SCHEMA.inputs[1:],
    )
    wire = schema_to_wire(schema)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert interface[0]["widget"] == {
        "type": "COMBO",
        "options": ["euler", "dpmpp_2m"],
        "remote": {"route": "/api/choices/test.samplers", "refreshButton": True},
    }
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) == schema_signature(
        dataclasses.replace(
            schema,
            inputs=(dataclasses.replace(schema.inputs[0], widget=None),) + schema.inputs[1:],
        )
    )

    # Static-only: remote omitted. Remote-only: options omitted and
    # refreshButton omitted when false (omission, never null).
    static_only = dataclasses.replace(
        SCHEMA,
        inputs=(
            dataclasses.replace(
                InputSpec("choice", TypeExpr.concrete("core.combo")),
                widget=ComboWidget(options=("a", "b")),
            ),
        )
        + SCHEMA.inputs[1:],
    )
    static_wire = schema_to_wire(static_only)
    static_interface = cast("list[dict[str, Any]]", static_wire["interface"])
    assert static_interface[0]["widget"] == {"type": "COMBO", "options": ["a", "b"]}
    assert schema_from_wire(static_wire) == static_only

    remote_only = dataclasses.replace(
        SCHEMA,
        inputs=(
            dataclasses.replace(
                InputSpec("choice", TypeExpr.concrete("core.combo")),
                widget=ComboWidget(remote_route="/api/choices/test.remote"),
            ),
        )
        + SCHEMA.inputs[1:],
    )
    remote_wire = schema_to_wire(remote_only)
    remote_interface = cast("list[dict[str, Any]]", remote_wire["interface"])
    assert remote_interface[0]["widget"] == {
        "type": "COMBO",
        "remote": {"route": "/api/choices/test.remote"},
    }
    assert schema_from_wire(remote_wire) == remote_only


def test_multicombo_widget_model_and_wire_are_strict() -> None:
    combo_list = TypeExpr.list_of(TypeExpr.concrete("core.combo"))
    widget = MultiComboWidget(options=("a", "b"))
    assert InputSpec("values", combo_list, widget=widget).widget is widget

    for wrong_type in (
        TypeExpr.concrete("core.combo"),
        TypeExpr.list_of(TypeExpr.concrete("core.string")),
        TypeExpr.list_of(combo_list),
        TypeExpr.wildcard(),
    ):
        with pytest.raises(ValueError, match="requires a list<core.combo> input"):
            InputSpec("values", wrong_type, widget=widget)

    for bad_default in ("a", ["a", 1], [True], [["a"]]):
        with pytest.raises(ValueError, match="default must be an array of strings"):
            InputSpec("values", combo_list, default=bad_default, widget=widget)

    for factory, message in (
        (lambda: MultiComboWidget(), "needs static options"),
        (lambda: MultiComboWidget(options=("",)), "non-empty strings"),
        (lambda: MultiComboWidget(options=(1,)), "non-empty strings"),  # type: ignore[arg-type]
        (lambda: MultiComboWidget(options=("a",), placeholder=1), "placeholder must be a string"),  # type: ignore[arg-type]
        (lambda: MultiComboWidget(options=("a",), chip=1), "chip must be a bool"),  # type: ignore[arg-type]
        (lambda: MultiComboWidget(options=("a",), refresh_button=True), "requires a remote route"),
        (lambda: MultiComboWidget(options=("a",), remote_timeout_ms=1), "requires a remote route"),
    ):
        with pytest.raises(ValueError, match=message):
            factory()

    base = schema_to_wire(
        NodeSchema(
            node_type="test.multicombo-strict",
            inputs=(InputSpec("values", combo_list, default=[], widget=widget),),
        )
    )
    entry = cast("list[dict[str, Any]]", base["interface"])[0]
    malformed_widgets = (
        {"type": "MULTI_COMBO", "options": ["a"], "unknown": True},
        {"type": "MULTI_COMBO", "options": [1]},
        {"type": "MULTI_COMBO", "options": ["a"], "placeholder": 1},
        {"type": "MULTI_COMBO", "options": ["a"], "chip": 1},
        {"type": "MULTI_COMBO"},
        {"type": "MULTI_COMBO", "options": ["a"], "remote": {"route": "/bad"}},
    )
    for malformed in malformed_widgets:
        candidate = json.loads(json.dumps(base))
        cast("list[dict[str, Any]]", candidate["interface"])[0]["widget"] = malformed
        with pytest.raises(ValueError):
            schema_from_wire(candidate)

    for malformed_default in ("a", ["a", 1]):
        candidate = json.loads(json.dumps(base))
        cast("list[dict[str, Any]]", candidate["interface"])[0]["default"] = malformed_default
        with pytest.raises(ValueError, match="default must be an array of strings"):
            schema_from_wire(candidate)

    entry["type"] = {"kind": "concrete", "types": ["core.combo"]}
    with pytest.raises(ValueError, match="requires a list<core.combo> input"):
        schema_from_wire(base)


def test_combo_remote_policy_validation_is_strict() -> None:
    for field, bad, message in (
        ("control_after_refresh", "middle", "control_after_refresh"),
        ("remote_timeout_ms", True, "remote_timeout_ms"),
        ("remote_timeout_ms", 0, "remote_timeout_ms"),
        ("remote_timeout_ms", 60_001, "remote_timeout_ms"),
        ("remote_max_retries", False, "remote_max_retries"),
        ("remote_max_retries", -1, "remote_max_retries"),
        ("remote_max_retries", 6, "remote_max_retries"),
        ("remote_refresh_ms", True, "remote_refresh_ms"),
        ("remote_refresh_ms", -1, "remote_refresh_ms"),
        ("remote_refresh_ms", 86_400_001, "remote_refresh_ms"),
    ):
        with pytest.raises(ValueError, match=message):
            ComboWidget(
                remote_route="/api/choices/test.remote",
                refresh_button=True,
                **{field: bad},  # type: ignore[arg-type]
            )

    for field, value in (
        ("control_after_refresh", "first"),
        ("remote_timeout_ms", 1),
        ("remote_max_retries", 0),
        ("remote_refresh_ms", 0),
    ):
        with pytest.raises(ValueError, match="requires a remote route"):
            ComboWidget(options=("a",), **{field: value})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a refresh button"):
        ComboWidget(
            remote_route="/api/choices/test.remote",
            control_after_refresh="first",
        )


def test_boolean_widget_wire_roundtrip_and_signature_exclusion() -> None:
    """BOOLEAN is a closed widget kind: custom toggle
    labels on a core.boolean input (ComfyUI's label_on/label_off). A
    label-less boolean carries no widget - the type alone means "render
    a toggle" - and empty labels are omitted from the wire, never null.
    Same rule as every widget: presentation, never execution identity."""
    both = dataclasses.replace(
        SCHEMA,
        inputs=(
            dataclasses.replace(
                InputSpec("flag", TypeExpr.concrete("core.boolean")),
                widget=BooleanWidget(label_on="enable", label_off="disable"),
            ),
        )
        + SCHEMA.inputs[1:],
    )
    wire = schema_to_wire(both)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert interface[0]["widget"] == {
        "type": "BOOLEAN",
        "labelOn": "enable",
        "labelOff": "disable",
    }
    assert schema_from_wire(wire) == both
    assert schema_signature(both) == schema_signature(
        dataclasses.replace(
            both,
            inputs=(dataclasses.replace(both.inputs[0], widget=None),) + both.inputs[1:],
        )
    )

    one_label = dataclasses.replace(
        SCHEMA,
        inputs=(
            dataclasses.replace(
                InputSpec("flag", TypeExpr.concrete("core.boolean")),
                widget=BooleanWidget(label_on="mute"),
            ),
        )
        + SCHEMA.inputs[1:],
    )
    one_wire = schema_to_wire(one_label)
    one_interface = cast("list[dict[str, Any]]", one_wire["interface"])
    assert one_interface[0]["widget"] == {"type": "BOOLEAN", "labelOn": "mute"}
    assert schema_from_wire(one_wire) == one_label


def test_boolean_widget_validated() -> None:
    with pytest.raises(ValueError, match="at least one custom label"):
        BooleanWidget()


def test_number_widget_wire_roundtrip_and_signature_exclusion() -> None:
    """NUMBER is a closed widget kind: min/max/step
    constraints and the control-after-generate controller on numeric
    inputs. Absent fields are omitted from the wire (absence = unbounded /
    no controller), a constraint-less numeric input carries no widget at
    all, and like every widget it is presentation, never identity."""
    full = dataclasses.replace(
        SCHEMA,
        inputs=(
            dataclasses.replace(
                SCHEMA.inputs[0],
                widget=NumberWidget(
                    min=0,
                    max=100,
                    step=5,
                    control_after_generate="randomize",
                    display="knob",
                ),
            ),
        )
        + SCHEMA.inputs[1:],
    )
    wire = schema_to_wire(full)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert interface[0]["widget"] == {
        "type": "NUMBER",
        "min": 0,
        "max": 100,
        "step": 5,
        "controlAfterGenerate": "randomize",
        "display": "knob",
    }
    assert schema_from_wire(wire) == full
    assert schema_signature(full) == schema_signature(SCHEMA)

    controller_only = dataclasses.replace(
        SCHEMA,
        inputs=(
            dataclasses.replace(
                SCHEMA.inputs[0],
                widget=NumberWidget(control_after_generate="fixed"),
            ),
        )
        + SCHEMA.inputs[1:],
    )
    controller_wire = schema_to_wire(controller_only)
    controller_interface = cast("list[dict[str, Any]]", controller_wire["interface"])
    assert controller_interface[0]["widget"] == {
        "type": "NUMBER",
        "controlAfterGenerate": "fixed",
    }
    assert schema_from_wire(controller_wire) == controller_only
    assert schema_signature(controller_only) == schema_signature(full)


def test_number_widget_validated() -> None:
    with pytest.raises(ValueError, match="at least one constraint"):
        NumberWidget()
    with pytest.raises(ValueError, match="must not exceed max"):
        NumberWidget(min=2, max=1)
    with pytest.raises(ValueError, match="step must be positive"):
        NumberWidget(step=0)
    with pytest.raises(ValueError, match="step must be positive"):
        NumberWidget(step=-1)
    with pytest.raises(ValueError, match="must be finite"):
        NumberWidget(min=float("nan"))
    with pytest.raises(ValueError, match="must be finite"):
        NumberWidget(max=float("inf"))
    with pytest.raises(ValueError, match="control_after_generate"):
        NumberWidget(control_after_generate="rand")  # type: ignore[arg-type]


def test_string_widget_validated() -> None:
    for malformed in (0, 1, "false", []):
        with pytest.raises(ValueError, match="multiline must be a bool"):
            StringWidget(multiline=malformed)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one presentation field"):
        StringWidget(multiline=None)


def test_text_completion_authoring_is_closed() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TextCompletions()
    with pytest.raises(ValueError, match="immutable tuple"):
        TextCompletions(items=cast("Any", [TextCompletionItem("sin")]))
    with pytest.raises(ValueError, match="unique"):
        TextCompletions(input_families=("values", "values"))
    with pytest.raises(ValueError, match="non-empty"):
        TextCompletionItem("")
    with pytest.raises(ValueError, match="kind"):
        TextCompletionItem("sin", kind=cast("Any", "future"))


def test_widget_representations_authoring_is_closed_and_domain_preserving() -> None:
    single = WidgetRepresentation("single-line", StringWidget(multiline=False))
    multiline = WidgetRepresentation("multiline", StringWidget(multiline=True))
    with pytest.raises(ValueError, match="immutable tuple"):
        WidgetRepresentations(cast("Any", [single]), default="single-line", user_switchable=True)
    with pytest.raises(ValueError, match="must not be empty"):
        WidgetRepresentations((), default="multiline", user_switchable=True)
    with pytest.raises(ValueError, match="must be unique"):
        WidgetRepresentations((single, single), default="single-line", user_switchable=True)
    with pytest.raises(ValueError, match="default must be a string"):
        WidgetRepresentations((single,), default=1, user_switchable=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must name a declared"):
        WidgetRepresentations((single,), default="missing", user_switchable=True)
    with pytest.raises(ValueError, match="user_switchable must be a bool"):
        WidgetRepresentations((single,), default="single-line", user_switchable=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one canonical value domain"):
        WidgetRepresentations(
            (single, WidgetRepresentation("asset", AssetWidget())),
            default="single-line",
            user_switchable=True,
        )
    with pytest.raises(ValueError, match="share one asset kind"):
        WidgetRepresentations(
            (
                WidgetRepresentation("image", AssetWidget(kind="media/image")),
                WidgetRepresentation("model", AssetWidget(kind="model/checkpoint")),
            ),
            default="image",
            user_switchable=True,
        )
    for malformed_id in ("", "two words", "../escape"):
        with pytest.raises(ValueError, match="widget representation id"):
            WidgetRepresentation(malformed_id, StringWidget())
    with pytest.raises(ValueError, match="display_name must be a string"):
        WidgetRepresentation("line", StringWidget(), display_name=1)  # type: ignore[arg-type]

    representations = WidgetRepresentations(
        (single, multiline), default="multiline", user_switchable=True
    )
    with pytest.raises(ValueError, match="string widget requires a concrete core.string"):
        InputSpec("value", TypeExpr.concrete("core.int"), widget=representations)


def test_widget_socket_binding_enforced() -> None:
    """NUMBER/STRING descriptors bind to the socket type and constraints
    interpret in the socket's domain: NUMBER
    only on concrete core.int/core.float with integral constraints on int
    sockets, STRING only on concrete core.string. The backend refuses to
    emit what the frontend would warn-and-drop."""
    # NUMBER on a non-numeric concrete socket.
    with pytest.raises(ValueError, match="number widget requires a concrete"):
        InputSpec("s", TypeExpr.concrete("core.string"), widget=NumberWidget(min=0))
    # NUMBER on non-concrete sockets (union / wildcard).
    with pytest.raises(ValueError, match="number widget requires a concrete"):
        InputSpec(
            "u",
            TypeExpr.union("core.int", "core.float"),
            widget=NumberWidget(min=0),
        )
    with pytest.raises(ValueError, match="number widget requires a concrete"):
        InputSpec("w", TypeExpr.wildcard(), widget=NumberWidget(min=0))
    # Fractional constraints are malformed in an int socket's domain.
    with pytest.raises(ValueError, match="step must be integral"):
        InputSpec("i", TypeExpr.concrete("core.int"), widget=NumberWidget(step=0.1))
    with pytest.raises(ValueError, match="min must be integral"):
        InputSpec("i", TypeExpr.concrete("core.int"), widget=NumberWidget(min=0.5))
    with pytest.raises(ValueError, match="max must be integral"):
        InputSpec("i", TypeExpr.concrete("core.int"), widget=NumberWidget(max=9.5))
    # Float sockets take integral or fractional constraints alike.
    InputSpec(
        "f",
        TypeExpr.concrete("core.float"),
        widget=NumberWidget(min=0, max=1.0, step=0.05),
    )
    # Integral-valued floats are integral (JSON has one number type).
    InputSpec(
        "i",
        TypeExpr.concrete("core.int"),
        widget=NumberWidget(min=0.0, max=10.0, step=1.0),
    )
    # STRING only on concrete core.string.
    with pytest.raises(ValueError, match="string widget requires a concrete"):
        InputSpec("i", TypeExpr.concrete("core.int"), widget=StringWidget())
    with pytest.raises(ValueError, match="string widget requires a concrete"):
        InputSpec("w", TypeExpr.wildcard(), widget=StringWidget())

    InputSpec("color", TypeExpr.concrete("core.string"), widget=ColorWidget())
    with pytest.raises(ValueError, match="color widget requires a concrete core.string"):
        InputSpec("color", TypeExpr.concrete("core.combo"), widget=ColorWidget())

    # COMBO only on concrete core.combo - not string, wildcard, union, or a
    # list whose element is combo. The descriptor describes one combo socket.
    InputSpec(
        "combo",
        TypeExpr.concrete("core.combo"),
        widget=ComboWidget(options=("a", "b")),
    )
    for wrong in (
        TypeExpr.concrete("core.string"),
        TypeExpr.wildcard(),
        TypeExpr.union("core.combo", "core.string"),
        TypeExpr.list_of(TypeExpr.concrete("core.combo")),
    ):
        with pytest.raises(ValueError, match="combo widget requires a concrete"):
            InputSpec("combo", wrong, widget=ComboWidget(options=("a", "b")))


def test_combo_widget_binding_is_enforced_during_wire_decode() -> None:
    schema = dataclasses.replace(
        SCHEMA,
        inputs=(InputSpec("choice", TypeExpr.concrete("core.combo")),) + SCHEMA.inputs[1:],
    )
    wire = schema_to_wire(schema)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    interface[0]["type"] = {"kind": "concrete", "types": ["core.string"]}
    interface[0]["widget"] = {"type": "COMBO", "options": ["a", "b"]}
    with pytest.raises(ValueError, match="combo widget requires a concrete core.combo"):
        schema_from_wire(wire)


def test_number_widget_rejects_bool_constraints() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        NumberWidget(min=True)
    with pytest.raises(ValueError, match="must be a number"):
        NumberWidget(step=False)


def test_display_name_rides_the_wire_but_never_the_signature() -> None:
    """Per-input displayName is a presentation label: empty
    means "derive from the id" and is omitted from the wire, and like
    doc/widget it never joins the schema signature (hazard H15)."""
    labeled = dataclasses.replace(
        SCHEMA,
        inputs=(dataclasses.replace(SCHEMA.inputs[0], display_name="Fancy Label"),)
        + SCHEMA.inputs[1:],
    )
    wire = schema_to_wire(labeled)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert interface[0]["displayName"] == "Fancy Label"
    assert "displayName" not in interface[1]
    assert schema_from_wire(wire) == labeled
    assert schema_signature(labeled) == schema_signature(SCHEMA)


def test_combo_source_is_retired_from_model_and_wire() -> None:
    """core.combo identity replaces the retired suggestion marker.
    No OutputSpec field or comboSource spelling remains."""
    combo_schema = dataclasses.replace(
        SCHEMA,
        outputs=(
            OutputSpec("choice", TypeExpr.concrete("core.combo")),
            OutputSpec("text", TypeExpr.concrete("core.string")),
        ),
    )
    wire = schema_to_wire(combo_schema)
    entries = {
        entry["id"]: entry
        for entry in cast("list[dict[str, Any]]", wire["interface"])
        if entry["role"] == "output"
    }
    assert all("comboSource" not in entry for entry in entries.values())
    assert "combo_source" not in {field.name for field in dataclasses.fields(OutputSpec)}
    assert schema_from_wire(wire) == combo_schema


def test_combo_widget_validated() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        ComboWidget(options=("a", ""))
    with pytest.raises(ValueError, match="canonical /api/choices"):
        ComboWidget(remote_route="api/x")
    with pytest.raises(ValueError, match="static options, a remote route"):
        ComboWidget()
    with pytest.raises(ValueError, match="refresh button requires"):
        ComboWidget(options=("a",), refresh_button=True)
    with pytest.raises(ValueError, match="control_after_generate"):
        ComboWidget(options=("a",), control_after_generate="cycle")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "route",
    (
        "/api/choices/x/y",
        "/api/choices/x?query=1",
        "/api/choices/x#fragment",
        "/api/choices/x%2fy",
        "/api/choices/x\\y",
        "/api/choices/.",
        "/api/choices/..",
        "/api/choices//x",
        "//host/api/choices/x",
        "https://host/api/choices/x",
        "/api/choices/X",
    ),
)
def test_combo_widget_remote_route_is_exact_canonical_choice_route(route: str) -> None:
    with pytest.raises(ValueError, match="canonical /api/choices"):
        ComboWidget(remote_route=route)


def test_strict_wire_decode_rejects_noncanonical_combo_route() -> None:
    wire = schema_to_wire(
        NodeSchema(
            node_type="test.remote-combo",
            inputs=(
                InputSpec(
                    "choice",
                    TypeExpr.concrete("core.combo"),
                    widget=ComboWidget(remote_route="/api/choices/test.valid"),
                ),
            ),
        )
    )
    interface = cast("list[dict[str, Any]]", wire["interface"])
    remote = cast("dict[str, Any]", interface[0]["widget"])["remote"]
    cast("dict[str, Any]", remote)["route"] = "/api/choices/test.valid?elsewhere=1"
    with pytest.raises(ValueError, match="canonical /api/choices"):
        schema_from_wire(wire)


def test_save_target_widget_suffix_validated() -> None:
    for bad in ("png", ".p/ng", ".p\\ng", "/abs"):
        with pytest.raises(ValueError, match="bare extension"):
            SaveTargetWidget(suffix=bad)


def test_malformed_widget_wire_rejected() -> None:
    """Structurally invalid built-in widget descriptors refuse loudly."""
    for bad in (
        "ASSET",
        {"type": "ASSET", "accept": "image/png"},
        {"type": "ASSET", "accept": [1]},
        {"type": "SAVE_TARGET", "suffix": 7},
        {"type": "COMBO"},
        {"type": "COMBO", "options": "euler"},
        {"type": "COMBO", "options": [1]},
        {"type": "COMBO", "remote": "route"},
        {"type": "COMBO", "remote": {}},
        {"type": "COMBO", "remote": {"route": "/x", "refreshButton": "yes"}},
        {"type": "NUMBER"},
        {"type": "NUMBER", "min": "0"},
        {"type": "NUMBER", "min": True},
        {"type": "NUMBER", "max": [1]},
        {"type": "NUMBER", "step": 0},
        {"type": "NUMBER", "step": -1},
        {"type": "NUMBER", "min": 2, "max": 1},
        {"type": "NUMBER", "controlAfterGenerate": "rand"},
        {"type": "NUMBER", "controlAfterGenerate": True},
        {"type": "NUMBER", "display": "dial"},
        {"type": "NUMBER", "display": True},
        {"type": "NUMBER", "display": None},
        {"type": "NUMBER", "round": 0},
        {"type": "NUMBER", "round": None},
        {"type": "NUMBER", "round": True},
        {"type": "NUMBER", "display": "slider", "future": True},
        {"type": "STRING"},
        {"type": "STRING", "multiline": None, "placeholder": "x"},
        {"type": "STRING", "multiline": "yes"},
        {"type": "STRING", "placeholder": 1},
        {"type": "STRING", "placeholder": None},
        {"type": "STRING", "dynamicPrompts": 1},
        {"type": "STRING", "dynamicPrompts": None},
        {"type": "COMBO", "options": ["a"], "controlAfterGenerate": "cycle"},
        {"type": "COMBO", "options": ["a"], "controlAfterGenerate": None},
        {"type": "COLOR", "future": True},
    ):
        wire = schema_to_wire(SCHEMA)
        interface = cast("list[dict[str, Any]]", wire["interface"])
        interface[0]["widget"] = bad
        with pytest.raises(ValueError):
            schema_from_wire(wire)


def test_lifecycle_metadata_never_joins_the_signature() -> None:
    """Deprecating or hiding a node changes how search lists it, never what
    it computes - caches must survive (same rationale as occupies/ioBound)."""
    deprecated = dataclasses.replace(
        SCHEMA,
        deprecation=Deprecation(message="bye", replacement="test.better"),
        search_visibility="deprecated",
    )
    assert schema_signature(deprecated) == schema_signature(SCHEMA)


def test_lifecycle_metadata_validated() -> None:
    with pytest.raises(ValueError, match="requires a message"):
        Deprecation(message="")
    with pytest.raises(ValueError, match="cannot replace itself"):
        dataclasses.replace(
            SCHEMA, deprecation=Deprecation(message="m", replacement=SCHEMA.node_type)
        )
    with pytest.raises(ValueError, match="unknown search_visibility"):
        dataclasses.replace(SCHEMA, search_visibility="invisible")  # pyright: ignore[reportArgumentType]


def test_aliases_and_output_node_ride_the_wire() -> None:
    """Resolution/targeting metadata is additive wire data: aliases (names
    submission adapters resolve, e.g. a v1 class_type) and outputNode (the
    default-target hint), both omitted at their defaults."""
    schema = dataclasses.replace(SCHEMA, aliases=("OldName", "OlderName"), output_node=True)
    wire = schema_to_wire(schema)
    assert wire["aliases"] == ["OldName", "OlderName"]
    assert wire["outputNode"] is True
    assert schema_from_wire(wire) == schema

    plain = schema_to_wire(SCHEMA)
    assert "aliases" not in plain
    assert "outputNode" not in plain


def test_aliases_and_output_node_never_join_the_signature() -> None:
    """What names resolve to a node and whether prompts target it by default
    never change what it computes - caches must survive."""
    renamed = dataclasses.replace(SCHEMA, aliases=("OldName",), output_node=True)
    assert schema_signature(renamed) == schema_signature(SCHEMA)


def test_emits_previews_never_joins_the_signature() -> None:
    """Whether a node ships live previews while running never changes what
    it computes - declaring the capability must not invalidate caches."""
    flagged = dataclasses.replace(SCHEMA, emits_previews=True)
    assert schema_signature(flagged) == schema_signature(SCHEMA)


def test_output_represents_is_strict_frozen_presentation_metadata() -> None:
    mutable_applies: Any = {"mode": ["covered"]}
    represents = OutputRepresents(
        "image",
        "future-rendition",
        applies=mutable_applies,
    )
    assert represents.applies == {"mode": ("covered",)}
    with pytest.raises(dataclasses.FrozenInstanceError):
        represents.rendition = "decoded-image"  # type: ignore[misc]
    with pytest.raises(TypeError):
        cast("Any", represents.applies)["mode"] = ("other",)
    with pytest.raises(ValueError, match="input must be a non-empty string"):
        OutputRepresents("", "decoded-image")
    with pytest.raises(ValueError, match="rendition must be a non-empty string"):
        OutputRepresents("image", "")
    with pytest.raises(ValueError, match="applies must not be empty"):
        OutputRepresents("image", "decoded-image", applies={})
    with pytest.raises(ValueError, match="represents must be an OutputRepresents"):
        OutputSpec("out", TypeExpr.concrete("dinkster.image"), represents=cast("Any", {}))


def test_output_represents_validates_input_and_combo_scope() -> None:
    represents = OutputRepresents("image", "decoded-image")
    schema = _represented_schema(represents=represents)
    with pytest.raises(ValueError, match="is not a declared top-level input"):
        dataclasses.replace(
            schema,
            outputs=(
                dataclasses.replace(
                    schema.outputs[0],
                    represents=dataclasses.replace(represents, input="missing"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="must use an asset widget"):
        dataclasses.replace(
            schema,
            inputs=(InputSpec("image", TypeExpr.concrete("core.string")),),
        )

    combo = DynamicComboSpec(
        "mode",
        options=(DynamicComboOption("covered"), DynamicComboOption("uncovered")),
        default="covered",
    )
    scoped = _represented_schema(
        represents=OutputRepresents(
            "image",
            "decoded-image",
            applies={"mode": ("covered",)},
        ),
        combos=(combo,),
    )
    with pytest.raises(ValueError, match="does not name a declared dynamic combo"):
        dataclasses.replace(scoped, combos=())
    with pytest.raises(ValueError, match=r"covers undeclared option\(s\): missing"):
        dataclasses.replace(
            scoped,
            outputs=(
                dataclasses.replace(
                    scoped.outputs[0],
                    represents=OutputRepresents(
                        "image",
                        "decoded-image",
                        applies={"mode": ("missing",)},
                    ),
                ),
            ),
        )

    covered = elaborate(scoped, ["image"], slot_variants={"mode": "covered"})
    assert covered.outputs[0].represents == OutputRepresents("image", "decoded-image")
    uncovered = elaborate(scoped, ["image"], slot_variants={"mode": "uncovered"})
    assert uncovered.outputs[0].represents is None


def test_output_represents_never_joins_schema_signature() -> None:
    represented = _represented_schema()
    without_representation = dataclasses.replace(
        represented,
        outputs=(dataclasses.replace(represented.outputs[0], represents=None),),
    )
    assert schema_signature(represented) == schema_signature(without_representation)


def test_output_known_value_wire_decode_rejects_unknown_fields_and_bad_shapes() -> None:
    integer = TypeExpr.concrete("core.int")
    schema = NodeSchema(
        node_type="test.known-output-wire",
        inputs=(InputSpec("value", integer),),
        outputs=(OutputSpec("value", integer, known_value=OutputKnownValue("value")),),
    )

    def malformed(known_value: object) -> dict[str, object]:
        wire = schema_to_wire(schema)
        cast("list[dict[str, Any]]", wire["interface"])[1]["knownValue"] = known_value
        return wire

    with pytest.raises(ValueError, match="output.knownValue must be an object"):
        schema_from_wire(malformed(1))
    with pytest.raises(ValueError, match="output.knownValue has unknown fields"):
        schema_from_wire(malformed({"input": "value", "surprise": True}))
    with pytest.raises(ValueError, match="output.knownValue.input must be a string"):
        schema_from_wire(malformed({"input": 1}))


def test_mirror_never_joins_the_signature() -> None:
    """A mirror is a presentation-only estimate renderer - declaring,
    changing, or removing one never changes what the node computes, so
    caches must survive."""
    mirrored = dataclasses.replace(SCHEMA, mirror=EXPRESSION_MIRROR)
    assert schema_signature(mirrored) == schema_signature(SCHEMA)
    retuned = dataclasses.replace(SCHEMA, mirror=GLSL_MIRROR)
    assert schema_signature(retuned) == schema_signature(SCHEMA)


def test_mirror_wire_decode_rejects_unknown_fields_and_bad_shapes() -> None:
    """The v29 mirror decoder is closed over its declared fields, like every
    current-version structured wire shape: a typoed or foreign key must fail
    loudly instead of decoding to a spec that silently drops it."""

    def malformed_mirror(**changes: object) -> dict[str, object]:
        wire = schema_to_wire(dataclasses.replace(SCHEMA, mirror=EXPRESSION_MIRROR))
        cast("dict[str, Any]", wire["mirror"]).update(changes)
        return wire

    with pytest.raises(ValueError, match="mirror has unknown fields"):
        schema_from_wire(malformed_mirror(surprise=1))
    with pytest.raises(ValueError, match="mirror.tolerance has unknown fields"):
        schema_from_wire(malformed_mirror(tolerance={"relative": 1e-12, "surprise": 1}))
    with pytest.raises(ValueError, match="mirror.tolerance must be an object"):
        schema_from_wire(malformed_mirror(tolerance=1))
    with pytest.raises(ValueError, match="mirror.tolerance.relative"):
        schema_from_wire(malformed_mirror(tolerance={"relative": "tight"}))


def test_mirror_applies_never_joins_the_signature() -> None:
    """Scoping a mirror is as presentation-only as declaring one."""
    assert schema_signature(SCOPED_SCHEMA) == schema_signature(
        dataclasses.replace(SCOPED_SCHEMA, mirror=None)
    )


def test_mirror_applies_must_name_a_declared_required_combo_option() -> None:
    """The scope is only as trustworthy as its vocabulary: a key that names
    no combo, an optional combo (whose absent choice elaborates to nothing),
    or an option the combo never declared could silently disable estimates
    forever or survive an option rename as a stale gate."""
    with pytest.raises(ValueError, match="does not name a declared dynamic combo"):
        dataclasses.replace(SCHEMA, mirror=SCOPED_GLSL_MIRROR)
    with pytest.raises(ValueError, match="must name a required combo"):
        dataclasses.replace(
            SCOPED_SCHEMA,
            combos=(dataclasses.replace(OPERATION_COMBO, required=False, default=None),),
        )
    with pytest.raises(ValueError, match=r"covers undeclared option\(s\): melt"):
        dataclasses.replace(
            SCOPED_SCHEMA,
            mirror=dataclasses.replace(
                GLSL_MIRROR, applies={"operation": ("gaussian_blur", "melt")}
            ),
        )
    # Elaboration resolves the scope (dropping or stripping it), so a schema
    # whose combo is gone but whose applies key remains is always malformed -
    # even when the consumed choice survives in slot_choices.
    with pytest.raises(ValueError, match="does not name a declared dynamic combo"):
        dataclasses.replace(
            SCOPED_SCHEMA, combos=(), slot_choices=(("operation", "gaussian_blur"),)
        )


def test_mirror_applies_resolved_by_elaboration() -> None:
    """Elaboration consumes the combo, so it resolves the scope too: a covered
    choice keeps the mirror with nothing left to gate on, an uncovered choice
    has no sound estimate. Elaborated schemas never carry ``applies``."""
    covered = elaborate(SCOPED_SCHEMA, ["a"], slot_variants={"operation": "sharpen"})
    assert covered.mirror == dataclasses.replace(SCOPED_GLSL_MIRROR, applies=None)
    assert covered.slot_choices == (("operation", "sharpen"),)

    uncovered = elaborate(SCOPED_SCHEMA, ["a"], slot_variants={"operation": "noise"})
    assert uncovered.mirror is None

    unscoped = dataclasses.replace(SCOPED_SCHEMA, mirror=GLSL_MIRROR)
    assert elaborate(unscoped, ["a"], slot_variants={"operation": "noise"}).mirror is GLSL_MIRROR


def test_mirror_applies_validated() -> None:
    def scoped(applies: object) -> MirrorSpec:
        return dataclasses.replace(GLSL_MIRROR, applies=cast("Any", applies))

    with pytest.raises(ValueError, match="applies must be a mapping"):
        scoped([("operation", ("a",))])
    with pytest.raises(ValueError, match="must not be empty"):
        scoped({})
    with pytest.raises(ValueError, match="non-empty combo ids"):
        scoped({"": ("a",)})
    with pytest.raises(ValueError, match="non-empty combo ids"):
        scoped({1: ("a",)})
    with pytest.raises(ValueError, match="sequence of options"):
        scoped({"operation": "solo"})
    with pytest.raises(ValueError, match="at least one option"):
        scoped({"operation": ()})
    with pytest.raises(ValueError, match="non-empty strings"):
        scoped({"operation": ("a", "")})
    with pytest.raises(ValueError, match="non-empty strings"):
        scoped({"operation": ("a", 2)})
    with pytest.raises(ValueError, match="duplicate options"):
        scoped({"operation": ("a", "a")})
    # The accepted mapping freezes with tuple values.
    mirror = scoped({"operation": ["a", "b"]})
    assert mirror.applies == {"operation": ("a", "b")}
    with pytest.raises(TypeError):
        cast("Any", mirror.applies)["operation"] = ("c",)


def test_mirror_applies_decode_rejects_bad_shapes() -> None:
    def malformed(applies: object) -> dict[str, object]:
        wire = schema_to_wire(SCOPED_SCHEMA)
        cast("dict[str, Any]", wire["mirror"])["applies"] = applies
        return wire

    with pytest.raises(ValueError, match="mirror.applies must be an object"):
        schema_from_wire(malformed(1))
    with pytest.raises(ValueError, match=r"mirror\.applies\['operation'\] must be an array"):
        schema_from_wire(malformed({"operation": "solo"}))
    with pytest.raises(ValueError, match="array of strings"):
        schema_from_wire(malformed({"operation": ["a", 2]}))
    with pytest.raises(ValueError, match="mirror.applies key must be a string"):
        schema_from_wire(malformed({1: ["a"]}))
    with pytest.raises(ValueError, match="at least one option"):
        schema_from_wire(malformed({"operation": []}))


def test_aliases_validated() -> None:
    with pytest.raises(ValueError, match="duplicate aliases"):
        dataclasses.replace(SCHEMA, aliases=("A", "A"))
    with pytest.raises(ValueError, match="non-empty"):
        dataclasses.replace(SCHEMA, aliases=("",))


def test_search_terms_ride_the_wire() -> None:
    """Discovery keywords (ComfyUI's search-alias concept) are additive
    wire data for node search only - never resolution - and are omitted
    when empty."""
    schema = dataclasses.replace(SCHEMA, search_terms=("OldName", "blur"))
    wire = schema_to_wire(schema)
    assert wire["searchTerms"] == ["OldName", "blur"]
    assert schema_from_wire(wire) == schema
    assert "searchTerms" not in schema_to_wire(SCHEMA)


def test_search_terms_never_join_the_signature() -> None:
    """How a node is FOUND never changes what it computes - adding search
    keywords must not invalidate caches."""
    keyworded = dataclasses.replace(SCHEMA, search_terms=("blur", "gaussian"))
    assert schema_signature(keyworded) == schema_signature(SCHEMA)


def test_search_terms_validated() -> None:
    with pytest.raises(ValueError, match="duplicate search terms"):
        dataclasses.replace(SCHEMA, search_terms=("A", "A"))
    with pytest.raises(ValueError, match="non-empty"):
        dataclasses.replace(SCHEMA, search_terms=("",))


def test_type_expr_closed_model_validated() -> None:
    with pytest.raises(ValueError):
        TypeExpr(kind="concrete", types=())
    with pytest.raises(ValueError):
        TypeExpr(kind="union", types=("only-one",))
    with pytest.raises(ValueError):
        TypeExpr(kind="variable")


def test_accepts_concrete() -> None:
    assert TypeExpr.wildcard().accepts_concrete("anything")
    assert TypeExpr.union("a", "b").accepts_concrete("a")
    assert not TypeExpr.union("a", "b").accepts_concrete("c")
    assert TypeExpr.variable("T").accepts_concrete("anything")
    assert not TypeExpr.variable("T", allowed=("a",)).accepts_concrete("b")


def test_runtime_type_atom() -> None:
    # The inverse of runtime_type_id: peel structured layers to the atom.
    assert TypeExpr.runtime_type_atom("core.int") == "core.int"
    assert TypeExpr.runtime_type_atom("list<core.int>") == "core.int"
    assert TypeExpr.runtime_type_atom("list<list<core.int>>") == "core.int"
    assert TypeExpr.runtime_type_atom("stream<core.int>") == "core.int"
    assert TypeExpr.runtime_type_atom("asset<stream<core.int>>") == "core.int"
    # Malformed grammar is None: empty, unclosed, empty constructor, or
    # angle brackets inside what should be an atom.
    for bad in (
        "",
        "list<>",
        "list<core.int",
        "stream<>",
        "stream<core.int",
        "core<int>",
        "list<list<>>",
        "a>b",
    ):
        assert TypeExpr.runtime_type_atom(bad) is None, bad


def test_outputs_helper_accepts_exact_ids() -> None:
    assert TwoOutputs.outputs(first=1, second=2) == {"first": 1, "second": 2}


def test_outputs_helper_rejects_typos_at_return_site() -> None:
    with pytest.raises(NodeOutputError) as excinfo:
        TwoOutputs.outputs(first=1, secondd=2)
    message = str(excinfo.value)
    assert "test.two_outputs" in message
    assert "missing outputs: second" in message
    assert "undeclared outputs: secondd" in message


def test_outputs_helper_rejects_missing() -> None:
    with pytest.raises(NodeOutputError, match="missing outputs: second"):
        TwoOutputs.outputs(first=1)


def test_schema_cached_per_subclass() -> None:
    assert TwoOutputs.schema() is TwoOutputs.schema()

    class Child(TwoOutputs):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(node_type="test.child")

    # Subclass gets its own schema, not the parent's cached one.
    assert Child.schema().node_type == "test.child"
    assert TwoOutputs.schema().node_type == "test.two_outputs"


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValueError):
        NodeSchema(
            node_type="dup",
            inputs=(
                InputSpec("x", TypeExpr.concrete("core.int")),
                InputSpec("x", TypeExpr.concrete("core.int")),
            ),
        )
