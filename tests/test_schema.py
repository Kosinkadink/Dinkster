import dataclasses
import json
from typing import Any, cast

import pytest
from dinkster_schema import (
    OPTION_KEY_PATTERN,
    SCHEMA_WIRE_VERSION,
    AssetWidget,
    BooleanWidget,
    ColorWidget,
    ComboOption,
    ComboWidget,
    CompositorWidget,
    CurveWidget,
    Deprecation,
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputFamilyOptionSource,
    InputFamilySpec,
    InputSpec,
    MirrorSpec,
    MirrorTolerance,
    MultiComboWidget,
    Node,
    NodeOutputError,
    NodeSchema,
    NumberWidget,
    OutputCountSpec,
    OutputFamilySpec,
    OutputKnownValue,
    OutputRepresents,
    OutputSpec,
    SaveTargetWidget,
    SchemaWireVersionRequirement,
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
    type_expr_from_wire,
    type_expr_to_wire,
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


def test_wire_roundtrip() -> None:
    wire = schema_to_wire(SCHEMA)
    assert wire["schemaVersion"] == 40
    assert SCHEMA_WIRE_VERSION == 40
    roundtripped = schema_from_wire(wire)
    assert roundtripped == SCHEMA


def test_dispatch_affinity_uses_wire45_without_changing_schema_identity() -> None:
    marked = dataclasses.replace(SCHEMA, dispatch_affinity="native")
    assert schema_signature(marked) == schema_signature(SCHEMA)

    for version in range(40, 45):
        downlevel = schema_to_wire(marked, wire_version=version)
        assert "dispatchAffinity" not in downlevel
        assert schema_from_wire(downlevel) == SCHEMA

    wire45 = schema_to_wire(marked, wire_version=45)
    assert wire45["dispatchAffinity"] == "native"
    assert schema_from_wire(wire45) == marked


@pytest.mark.parametrize("value", [False, True, 0, 1, "compat", [], {}])
def test_dispatch_affinity_rejects_values_other_than_native(value: Any) -> None:
    with pytest.raises(ValueError, match="unknown dispatch_affinity"):
        dataclasses.replace(SCHEMA, dispatch_affinity=value)  # type: ignore[arg-type]

    wire = schema_to_wire(SCHEMA, wire_version=45)
    wire["dispatchAffinity"] = value
    match = "dispatchAffinity must be a string" if not isinstance(value, str) else "unsupported"
    with pytest.raises(ValueError, match=match):
        schema_from_wire(wire)


def test_dispatch_affinity_is_rejected_on_older_wire_labels() -> None:
    for version in range(40, 45):
        wire = schema_to_wire(SCHEMA, wire_version=version)
        wire["dispatchAffinity"] = "native"
        with pytest.raises(ValueError, match="dispatchAffinity requires schema wire 45"):
            schema_from_wire(wire)


def test_curve_widget_is_an_exact_wire35_socket_bound_descriptor() -> None:
    schema = NodeSchema(
        node_type="test.curve-widget",
        inputs=(InputSpec("curve", TypeExpr.concrete("dinkster.curve"), widget=CurveWidget()),),
    )
    wire = schema_to_wire(schema)
    entry = cast("list[dict[str, object]]", wire["interface"])[0]
    assert entry["widget"] == {"type": "CURVE"}
    assert schema_from_wire(wire) == schema
    without_widget = dataclasses.replace(
        schema,
        inputs=(dataclasses.replace(schema.inputs[0], widget=None),),
    )
    assert schema_signature(schema) == schema_signature(without_widget)
    represented = dataclasses.replace(
        schema,
        inputs=(
            dataclasses.replace(
                schema.inputs[0],
                widget=WidgetRepresentations(
                    (
                        WidgetRepresentation("editor", CurveWidget()),
                        WidgetRepresentation("alternate", CurveWidget()),
                    ),
                    default="editor",
                    user_switchable=True,
                ),
            ),
        ),
    )
    assert schema_from_wire(schema_to_wire(represented)) == represented
    assert schema_signature(represented) == schema_signature(without_widget)
    with pytest.raises(SchemaWireVersionRequirement, match="CURVE widget requires schema wire 35"):
        schema_to_wire(schema, wire_version=34)
    malformed = schema_to_wire(schema)
    cast("dict[str, object]", cast("list[dict[str, object]]", malformed["interface"])[0]["widget"])[
        "extra"
    ] = True
    with pytest.raises(ValueError, match="CURVE widget has unknown fields"):
        schema_from_wire(malformed)
    downlevel = schema_to_wire(SCHEMA, wire_version=34)
    cast("list[dict[str, object]]", downlevel["interface"])[0]["widget"] = {"type": "CURVE"}
    with pytest.raises(ValueError, match="CURVE widget requires schema wire 35"):
        schema_from_wire(downlevel)
    with pytest.raises(ValueError, match="curve widget requires a concrete dinkster.curve"):
        InputSpec("wrong", TypeExpr.concrete("core.string"), widget=CurveWidget())

    def nested_schema(widget: CurveWidget | None) -> NodeSchema:
        return NodeSchema(
            node_type="test.nested-curve-widget",
            input_families=(
                InputFamilySpec(
                    "families",
                    (
                        DynamicComboSpec(
                            "mode",
                            (
                                DynamicComboOption(
                                    "curve",
                                    (
                                        DynamicSlotSpec(
                                            "slot",
                                            variants=(
                                                SlotVariant(
                                                    "curve",
                                                    TypeExpr.concrete("dinkster.curve"),
                                                    inputs=(
                                                        InputSpec(
                                                            "value",
                                                            TypeExpr.concrete("dinkster.curve"),
                                                            widget=widget,
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    member_names=("primary",),
                ),
            ),
        )

    assert schema_signature(nested_schema(CurveWidget())) == schema_signature(nested_schema(None))


def test_compositor_widget_is_an_exact_wire37_socket_bound_descriptor() -> None:
    schema = NodeSchema(
        node_type="test.compositor-widget",
        inputs=(
            InputSpec(
                "compositor",
                TypeExpr.concrete("dinkster.compositor"),
                widget=CompositorWidget(),
            ),
        ),
    )
    wire = schema_to_wire(schema)
    entry = cast("list[dict[str, object]]", wire["interface"])[0]
    assert entry["widget"] == {"type": "COMPOSITOR"}
    assert schema_from_wire(wire) == schema
    without_widget = dataclasses.replace(
        schema,
        inputs=(dataclasses.replace(schema.inputs[0], widget=None),),
    )
    assert schema_signature(schema) == schema_signature(without_widget)

    with pytest.raises(
        SchemaWireVersionRequirement,
        match="COMPOSITOR widget requires schema wire 37",
    ):
        schema_to_wire(schema, wire_version=36)
    malformed = schema_to_wire(schema)
    cast("dict[str, object]", cast("list[dict[str, object]]", malformed["interface"])[0]["widget"])[
        "extra"
    ] = True
    with pytest.raises(ValueError, match="COMPOSITOR widget has unknown fields"):
        schema_from_wire(malformed)
    downlevel = schema_to_wire(SCHEMA, wire_version=36)
    cast("list[dict[str, object]]", downlevel["interface"])[0]["widget"] = {"type": "COMPOSITOR"}
    with pytest.raises(ValueError, match="COMPOSITOR widget requires schema wire 37"):
        schema_from_wire(downlevel)
    with pytest.raises(
        ValueError, match="compositor widget requires a concrete dinkster.compositor"
    ):
        InputSpec("wrong", TypeExpr.concrete("core.string"), widget=CompositorWidget())


def test_output_family_count_wire_and_reference_contract() -> None:
    count = InputSpec("count", TypeExpr.concrete("core.int"))
    schema = NodeSchema(
        node_type="test.counted-output",
        inputs=(count,),
        output_families=(
            OutputFamilySpec(
                "items", TypeExpr.concrete("core.string"), count=OutputCountSpec("count")
            ),
        ),
    )
    wire = schema_to_wire(schema)
    assert cast("list[dict[str, object]]", wire["interface"])[1]["count"] == {
        "input": "count",
        "suffix": "index",
    }
    assert schema_from_wire(wire) == schema
    with pytest.raises(SchemaWireVersionRequirement) as required:
        schema_to_wire(schema, wire_version=25)
    assert required.value.required_version == 26

    for bad_count in (
        InputSpec("count", TypeExpr.concrete("core.float")),
        InputSpec("count", TypeExpr.concrete("core.int"), required=False),
    ):
        with pytest.raises(ValueError, match="required top-level core.int"):
            dataclasses.replace(schema, inputs=(bad_count,))
    shared_count = dataclasses.replace(
        schema,
        output_families=(
            *schema.output_families,
            OutputFamilySpec(
                "other", TypeExpr.concrete("core.string"), count=OutputCountSpec("count")
            ),
        ),
    )
    assert len(shared_count.output_families) == 2

    unbound = dataclasses.replace(
        schema,
        output_families=(dataclasses.replace(schema.output_families[0], count=None),),
    )
    both_inputs = dataclasses.replace(
        schema,
        inputs=(*schema.inputs, InputSpec("other_count", TypeExpr.concrete("core.int"))),
    )
    other_reference = dataclasses.replace(
        both_inputs,
        output_families=(
            dataclasses.replace(
                both_inputs.output_families[0], count=OutputCountSpec("other_count")
            ),
        ),
    )
    assert schema_signature(schema) != schema_signature(unbound)
    assert schema_signature(both_inputs) != schema_signature(other_reference)

    for malformed in (
        None,
        {"input": "count"},
        {"input": "count", "suffix": "index", "extra": True},
        {"input": "count", "suffix": "number"},
    ):
        bad_wire = schema_to_wire(schema)
        cast("list[dict[str, object]]", bad_wire["interface"])[1]["count"] = malformed
        with pytest.raises(ValueError):
            schema_from_wire(bad_wire)
    old_wire = schema_to_wire(schema)
    old_wire["schemaVersion"] = 25
    with pytest.raises(ValueError, match="requires schema wire 26"):
        schema_from_wire(old_wire)


def test_signature_stable_and_content_sensitive() -> None:
    assert schema_signature(SCHEMA) == "381dd2d2c8ba1f23b112948e2b1c3f3e9a331b59"
    changed = NodeSchema(
        node_type=SCHEMA.node_type,
        version=2,
        display_name=SCHEMA.display_name,
        category=SCHEMA.category,
        inputs=SCHEMA.inputs,
        outputs=SCHEMA.outputs,
    )
    assert schema_signature(changed) != schema_signature(SCHEMA)


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


def test_unsupported_wire_versions_are_rejected() -> None:
    """Only frozen wires 21/22 and current wire 23 decode."""
    for stale in (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
        22.0,
        23.0,
        None,
        "15",
    ):
        wire = schema_to_wire(SCHEMA)
        wire["schemaVersion"] = stale
        with pytest.raises(ValueError, match="unsupported schemaVersion"):
            schema_from_wire(wire)
    with pytest.raises(ValueError, match="unsupported schemaVersion"):
        schema_to_wire(SCHEMA, wire_version=18.0)  # type: ignore[arg-type]


def test_lazy_is_true_only_on_wire_16_and_excluded_from_wire_15() -> None:
    plain = NodeSchema(
        node_type="test.lazy-wire",
        inputs=(InputSpec("value", TypeExpr.concrete("core.int")),),
    )
    lazy = NodeSchema(
        node_type="test.lazy-wire",
        inputs=(InputSpec("value", TypeExpr.concrete("core.int"), lazy=True),),
    )
    plain_entry = cast("list[dict[str, Any]]", schema_to_wire(plain)["interface"])[0]
    lazy_entry = cast("list[dict[str, Any]]", schema_to_wire(lazy)["interface"])[0]
    assert "lazy" not in plain_entry
    assert lazy_entry["lazy"] is True
    assert schema_from_wire(schema_to_wire(lazy)) == lazy
    assert schema_signature(lazy) != schema_signature(plain)
    with pytest.raises(ValueError, match="cannot be represented"):
        schema_to_wire(lazy, wire_version=15)
    wire15 = schema_to_wire(plain, wire_version=15)
    with pytest.raises(ValueError, match="unsupported schemaVersion"):
        schema_from_wire(wire15)
    cast("list[dict[str, Any]]", wire15["interface"])[0]["lazy"] = False
    with pytest.raises(ValueError, match="unsupported schemaVersion"):
        schema_from_wire(wire15)
    cast("list[dict[str, Any]]", wire15["interface"])[0]["lazy"] = True
    with pytest.raises(ValueError, match="unsupported schemaVersion"):
        schema_from_wire(wire15)


def test_output_preview_is_true_only_wire16_presentation_metadata() -> None:
    plain = NodeSchema(
        node_type="test.output-preview",
        outputs=(OutputSpec("value", TypeExpr.concrete("core.string")),),
    )
    marked = dataclasses.replace(
        plain,
        outputs=(
            dataclasses.replace(plain.outputs[0], preview=True),
            OutputSpec(
                "optional-preview",
                TypeExpr.concrete("core.string"),
                optional=True,
                preview=True,
            ),
        ),
    )
    plain_entry = cast("list[dict[str, Any]]", schema_to_wire(plain)["interface"])[0]
    marked_entries = cast("list[dict[str, Any]]", schema_to_wire(marked)["interface"])
    marked_entry = marked_entries[0]
    assert "preview" not in plain_entry
    assert marked_entry["preview"] is True
    assert marked_entries[1]["preview"] is True
    assert marked_entries[1]["optional"] is True
    assert schema_from_wire(schema_to_wire(marked)) == marked
    single_marked = dataclasses.replace(
        plain, outputs=(dataclasses.replace(plain.outputs[0], preview=True),)
    )
    assert schema_signature(plain) == schema_signature(single_marked)

    wire15 = schema_to_wire(single_marked, wire_version=15)
    entry15 = cast("list[dict[str, Any]]", wire15["interface"])[0]
    assert "preview" not in entry15
    entry15["preview"] = True
    with pytest.raises(ValueError, match="unsupported schemaVersion"):
        schema_from_wire(wire15)

    unknown = schema_to_wire(plain)
    cast("list[dict[str, Any]]", unknown["interface"])[0]["futureMetadata"] = {"ignored": True}
    assert schema_from_wire(unknown) == plain

    for malformed in (False, 0, 1, "true", None, {}):
        wire16 = schema_to_wire(plain)
        cast("list[dict[str, Any]]", wire16["interface"])[0]["preview"] = malformed
        if malformed is False:
            assert schema_from_wire(wire16) == plain
        else:
            with pytest.raises(ValueError, match="output.preview must be a boolean"):
                schema_from_wire(wire16)


def test_output_preview_authoring_requires_an_actual_bool() -> None:
    for malformed in (0, 1, "true", None, [], {}):
        with pytest.raises(ValueError, match="preview must be a bool"):
            OutputSpec(
                "value",
                TypeExpr.concrete("core.string"),
                preview=malformed,  # type: ignore[arg-type]
            )


def test_output_family_preview_has_wire_and_signature_parity() -> None:
    plain = NodeSchema(
        node_type="test.output-family-preview",
        output_families=(OutputFamilySpec("values", TypeExpr.concrete("core.string")),),
    )
    marked = dataclasses.replace(
        plain,
        output_families=(dataclasses.replace(plain.output_families[0], preview=True),),
    )
    plain_entry = cast("list[dict[str, Any]]", schema_to_wire(plain)["interface"])[0]
    marked_entry = cast("list[dict[str, Any]]", schema_to_wire(marked)["interface"])[0]
    assert "preview" not in plain_entry
    assert marked_entry["preview"] is True
    assert schema_from_wire(schema_to_wire(marked)) == marked
    assert schema_signature(plain) == schema_signature(marked)

    false_wire = schema_to_wire(plain)
    cast("list[dict[str, Any]]", false_wire["interface"])[0]["preview"] = False
    assert schema_from_wire(false_wire) == plain

    wire15 = schema_to_wire(marked, wire_version=15)
    entry15 = cast("list[dict[str, Any]]", wire15["interface"])[0]
    assert "preview" not in entry15
    for ignored in (False, True, 0, "true", None, {}):
        entry15["preview"] = ignored
        with pytest.raises(ValueError, match="unsupported schemaVersion"):
            schema_from_wire(wire15)

    for malformed in (0, 1, "true", None, {}):
        wire16 = schema_to_wire(plain)
        cast("list[dict[str, Any]]", wire16["interface"])[0]["preview"] = malformed
        with pytest.raises(ValueError, match="outputFamily.preview must be a boolean"):
            schema_from_wire(wire16)


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


def test_input_presentation_flags_are_emit_only() -> None:
    plain = NodeSchema(
        node_type="test.presentation-flags",
        inputs=(InputSpec("value", TypeExpr.concrete("core.string")),),
    )
    flagged = dataclasses.replace(
        plain,
        inputs=(
            dataclasses.replace(
                plain.inputs[0],
                force_input=True,
                advanced=True,
                hidden=True,
            ),
        ),
    )
    plain_entry = cast("list[dict[str, Any]]", schema_to_wire(plain)["interface"])[0]
    flagged_entry = cast("list[dict[str, Any]]", schema_to_wire(flagged)["interface"])[0]
    assert not ({"forceInput", "advanced", "hidden"} & plain_entry.keys())
    assert flagged_entry["forceInput"] is True
    assert flagged_entry["advanced"] is True
    assert flagged_entry["hidden"] is True
    assert schema_from_wire(schema_to_wire(flagged)) == flagged
    wire37 = schema_to_wire(flagged, wire_version=37)
    assert "hidden" not in cast("list[dict[str, Any]]", wire37["interface"])[0]
    assert schema_from_wire(wire37) == dataclasses.replace(
        flagged,
        inputs=(dataclasses.replace(flagged.inputs[0], hidden=False),),
    )
    cast("list[dict[str, Any]]", wire37["interface"])[0]["hidden"] = True
    with pytest.raises(ValueError, match="input.hidden requires schema wire 38"):
        schema_from_wire(wire37)
    assert schema_signature(plain) == schema_signature(flagged)


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


def test_recursive_lazy_decode_is_versioned_at_nested_input_depth() -> None:
    family = InputFamilySpec(
        "items",
        (InputSpec("value", TypeExpr.concrete("core.string")),),
        member_prefix="item",
    )
    schema = NodeSchema(node_type="test.recursive-lazy", input_families=(family,))

    wire16 = schema_to_wire(schema)
    family_entry = cast("list[dict[str, Any]]", wire16["interface"])[0]
    nested = cast("list[dict[str, Any]]", family_entry["template"])[0]
    nested["lazy"] = True
    decoded = schema_from_wire(wire16)
    assert cast(InputSpec, decoded.input_families[0].template[0]).lazy

    nested["lazy"] = "yes"
    with pytest.raises(ValueError, match="input.lazy must be a boolean"):
        schema_from_wire(wire16)

    wire15 = schema_to_wire(schema, wire_version=15)
    family_entry = cast("list[dict[str, Any]]", wire15["interface"])[0]
    nested = cast("list[dict[str, Any]]", family_entry["template"])[0]
    nested["lazy"] = True
    with pytest.raises(ValueError, match="unsupported schemaVersion"):
        schema_from_wire(wire15)


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
    """SAVE_TARGET is the second closed widget kind (wire v5): a
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
    """COMBO is the third closed widget kind (wire v9), bound to core.combo
    since wire v14. Static options render immediately; a remote route names a
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


def test_combo_input_family_option_source_wire43_and_elaboration() -> None:
    source = InputFamilyOptionSource("values")
    schema = NodeSchema(
        node_type="test.family-options",
        inputs=(
            InputSpec(
                "choice",
                TypeExpr.concrete("core.combo"),
                widget=ComboWidget(option_source=source),
            ),
        ),
        input_families=(InputFamilySpec("values", (InputSpec("value", TypeExpr.variable("T")),)),),
        outputs=(OutputSpec("value", TypeExpr.variable("T")),),
    )
    wire = schema_to_wire(schema, wire_version=43)
    choice = cast("list[dict[str, Any]]", wire["interface"])[0]
    assert choice["widget"] == {
        "type": "COMBO",
        "optionSource": {"inputFamily": "values"},
    }
    assert schema_from_wire(wire) == schema
    with pytest.raises(SchemaWireVersionRequirement, match="43"):
        schema_to_wire(schema, wire_version=42)
    mislabeled_wire42 = json.loads(json.dumps(wire))
    mislabeled_wire42["schemaVersion"] = 42
    with pytest.raises(ValueError, match="unknown fields.*optionSource"):
        schema_from_wire(mislabeled_wire42)

    effective = elaborate(
        schema,
        {
            "choice": "m2",
            "values.m7": 7,
            "values.m2": 2,
        },
    )
    effective_widget = effective.inputs[0].widget
    assert isinstance(effective_widget, ComboWidget)
    assert effective_widget.option_source is None
    assert effective_widget.options == ("m7", "m2")
    effective_wire = schema_to_wire(effective, wire_version=42)
    effective_choice = cast("list[dict[str, Any]]", effective_wire["interface"])[0]
    assert "optionSource" not in cast("dict[str, Any]", effective_choice["widget"])
    reordered = elaborate(
        schema,
        {
            "choice": "m9",
            "values.m2": 2,
            "values.m9": 9,
        },
    )
    reordered_widget = reordered.inputs[0].widget
    assert isinstance(reordered_widget, ComboWidget)
    assert reordered_widget.options == ("m2", "m9")

    static_options = dataclasses.replace(
        schema,
        inputs=(
            dataclasses.replace(
                schema.inputs[0],
                widget=ComboWidget(options=("m7", "m2")),
            ),
        ),
    )
    assert schema_signature(static_options) == schema_signature(schema)

    with pytest.raises(ValueError, match="unknown input family"):
        dataclasses.replace(
            schema,
            inputs=(
                dataclasses.replace(
                    schema.inputs[0],
                    widget=ComboWidget(option_source=InputFamilyOptionSource("missing")),
                ),
            ),
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        ComboWidget(option_source=source, options=("fallback",))
    with pytest.raises(ValueError, match="only legal on top-level inputs"):
        NodeSchema(
            node_type="test.nested-family-options",
            input_families=(
                InputFamilySpec(
                    "values",
                    (
                        InputSpec(
                            "value",
                            TypeExpr.concrete("core.combo"),
                            widget=ComboWidget(option_source=source),
                        ),
                    ),
                ),
            ),
        )


def test_combo_option_wire23_roundtrip_downlevel_and_validation() -> None:
    class MutableComboOption(ComboOption):
        pass

    option = ComboOption(
        value="dinkster.dpmpp_sde",
        label="dpmpp_sde",
        info="DPM++ stochastic differential equation sampler",
        folder="DPM++/SDE",
    )
    schema = NodeSchema(
        node_type="test.structured-combo",
        inputs=(
            InputSpec(
                "sampler",
                TypeExpr.concrete("core.combo"),
                default=option.value,
                widget=ComboWidget(options=("dinkster.euler", option)),
            ),
        ),
    )
    current = schema_to_wire(schema)
    current_widget = cast(
        "dict[str, Any]", cast("list[dict[str, Any]]", current["interface"])[0]["widget"]
    )
    assert current_widget["options"] == [
        "dinkster.euler",
        {
            "value": "dinkster.dpmpp_sde",
            "label": "dpmpp_sde",
            "info": "DPM++ stochastic differential equation sampler",
            "folder": "DPM++/SDE",
        },
    ]
    assert schema_from_wire(current) == schema

    wire22 = schema_to_wire(schema, wire_version=22)
    wire22_widget = cast(
        "dict[str, Any]", cast("list[dict[str, Any]]", wire22["interface"])[0]["widget"]
    )
    assert wire22_widget["options"] == ["dinkster.euler", "dinkster.dpmpp_sde"]
    assert schema_from_wire(wire22) == dataclasses.replace(
        schema,
        inputs=(
            dataclasses.replace(
                schema.inputs[0],
                widget=ComboWidget(options=("dinkster.euler", "dinkster.dpmpp_sde")),
            ),
        ),
    )

    for kwargs, message in (
        ({"value": ""}, "value must be a non-empty string"),
        ({"value": "x", "label": ""}, "label must be a non-empty string"),
        ({"value": "x", "info": ""}, "info must be a non-empty string"),
        ({"value": "x", "folder": ""}, "relative /-separated path"),
        ({"value": "x", "folder": "/root"}, "non-empty segments"),
        ({"value": "x", "folder": "root/"}, "non-empty segments"),
        ({"value": "x", "folder": "root//child"}, "non-empty segments"),
        ({"value": "x", "folder": "root\\child"}, "relative /-separated path"),
        ({"value": "x", "folder": "root/../child"}, "other than '.' or '..'"),
    ):
        with pytest.raises(ValueError, match=message):
            ComboOption(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ComboOption values"):
        ComboWidget(options=(MutableComboOption(value="x"),))

    malformed_options = (
        {"value": "x", "unknown": True},
        {"value": ""},
        {"value": "x", "label": ""},
        {"value": "x", "info": 1},
        {"value": "x", "folder": "a//b"},
    )
    for malformed in malformed_options:
        candidate = json.loads(json.dumps(current))
        widget = cast(
            "dict[str, Any]",
            cast("list[dict[str, Any]]", candidate["interface"])[0]["widget"],
        )
        widget["options"] = [malformed]
        with pytest.raises(ValueError):
            schema_from_wire(candidate)

    old_candidate = json.loads(json.dumps(wire22))
    old_widget = cast(
        "dict[str, Any]",
        cast("list[dict[str, Any]]", old_candidate["interface"])[0]["widget"],
    )
    old_widget["options"] = [{"value": "x", "label": "X"}]
    with pytest.raises(ValueError, match="must be a non-empty string"):
        schema_from_wire(old_candidate)


def test_combo_remote_policy_wire20_roundtrip_downgrade_and_signature_exclusion() -> None:
    plain = ComboWidget(
        options=("alpha", "beta"),
        remote_route="/api/choices/test.remote",
        refresh_button=True,
    )
    policy = dataclasses.replace(
        plain,
        control_after_refresh="last",
        remote_timeout_ms=60_000,
        remote_max_retries=2,
        remote_refresh_ms=0,
    )
    schema = NodeSchema(
        node_type="test.remote-policy",
        inputs=(InputSpec("choice", TypeExpr.concrete("core.combo"), widget=policy),),
    )
    entry20 = cast("list[dict[str, Any]]", schema_to_wire(schema)["interface"])[0]
    assert entry20["widget"] == {
        "type": "COMBO",
        "options": ["alpha", "beta"],
        "remote": {
            "route": "/api/choices/test.remote",
            "refreshButton": True,
            "controlAfterRefresh": "last",
            "timeoutMs": 60_000,
            "maxRetries": 2,
            "refreshMs": 0,
        },
    }
    assert schema_from_wire(schema_to_wire(schema)) == schema

    without_policy = dataclasses.replace(
        schema,
        inputs=(dataclasses.replace(schema.inputs[0], widget=plain),),
    )
    assert schema_signature(schema) == schema_signature(without_policy)
    for old_version in (18, 19):
        old_wire = schema_to_wire(schema, wire_version=old_version)
        old_entry = cast("list[dict[str, Any]]", old_wire["interface"])[0]
        assert old_entry["widget"] == {
            "type": "COMBO",
            "options": ["alpha", "beta"],
            "remote": {
                "route": "/api/choices/test.remote",
                "refreshButton": True,
            },
        }
        assert json.dumps(old_wire, separators=(",", ":")) == json.dumps(
            schema_to_wire(without_policy, wire_version=old_version),
            separators=(",", ":"),
        )

    # Absence stays omission. The effective client defaults are 4096 ms,
    # two retries after the initial attempt, and no automatic expiry.
    plain_remote = cast(
        "dict[str, Any]",
        cast(
            "dict[str, Any]",
            cast("list[dict[str, Any]]", schema_to_wire(without_policy)["interface"])[0]["widget"],
        )["remote"],
    )
    assert plain_remote == {
        "route": "/api/choices/test.remote",
        "refreshButton": True,
    }


def test_multicombo_widget_wire21_roundtrip_refusal_and_signature_exclusion() -> None:
    widget = MultiComboWidget(
        options=(
            ComboOption(
                value="beta",
                label="Beta provider",
                info="Preferred provider",
                folder="Featured/Providers",
            ),
            "alpha",
            "beta",
        ),
        remote_route="/api/choices/test.providers",
        refresh_button=True,
        control_after_refresh="last",
        remote_timeout_ms=4096,
        remote_max_retries=2,
        remote_refresh_ms=0,
        placeholder="Select providers",
        chip=False,
    )
    combo_list = TypeExpr.list_of(TypeExpr.concrete("core.combo"))
    schema = NodeSchema(
        node_type="test.multicombo",
        inputs=(
            InputSpec(
                "providers",
                combo_list,
                required=False,
                default=["beta", "alpha", "beta"],
                widget=widget,
            ),
        ),
        outputs=(OutputSpec("providers", combo_list),),
    )
    wire = schema_to_wire(schema)
    assert wire["schemaVersion"] == 40
    (entry, output) = cast("list[dict[str, Any]]", wire["interface"])
    assert entry["type"] == {
        "kind": "list",
        "element": {"kind": "concrete", "types": ["core.combo"]},
    }
    assert entry["default"] == ["beta", "alpha", "beta"]
    assert entry["widget"] == {
        "type": "MULTI_COMBO",
        "options": [
            {
                "value": "beta",
                "label": "Beta provider",
                "info": "Preferred provider",
                "folder": "Featured/Providers",
            },
            "alpha",
            "beta",
        ],
        "remote": {
            "route": "/api/choices/test.providers",
            "refreshButton": True,
            "controlAfterRefresh": "last",
            "timeoutMs": 4096,
            "maxRetries": 2,
            "refreshMs": 0,
        },
        "placeholder": "Select providers",
        "chip": False,
    }
    assert output["type"] == entry["type"]
    assert "widget" not in output
    assert schema_from_wire(wire) == schema

    without_widget = dataclasses.replace(
        schema,
        inputs=(dataclasses.replace(schema.inputs[0], widget=None),),
    )
    assert schema_signature(schema) == schema_signature(without_widget)

    for old_version in (15, 16, 17, 18, 19, 20):
        with pytest.raises(
            SchemaWireVersionRequirement,
            match="MULTI_COMBO widget requires schema wire 21",
        ) as excinfo:
            schema_to_wire(schema, wire_version=old_version)
        assert excinfo.value.required_version == 21
        assert excinfo.value.code == "schema-wire-required"


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


def test_combo_remote_policy_wire_decode_rejects_type_range_dependency_and_unknown() -> None:
    schema = NodeSchema(
        node_type="test.remote-policy-decode",
        inputs=(
            InputSpec(
                "choice",
                TypeExpr.concrete("core.combo"),
                widget=ComboWidget(
                    remote_route="/api/choices/test.remote",
                    refresh_button=True,
                    control_after_refresh="first",
                    remote_timeout_ms=4096,
                    remote_max_retries=2,
                    remote_refresh_ms=0,
                ),
            ),
        ),
    )

    def malformed_remote(**changes: object) -> dict[str, object]:
        wire = schema_to_wire(schema)
        entry = cast("list[dict[str, Any]]", wire["interface"])[0]
        remote = cast("dict[str, Any]", cast("dict[str, Any]", entry["widget"])["remote"])
        remote.update(changes)
        return wire

    for changes, message in (
        ({"timeoutMs": True}, "timeoutMs"),
        ({"maxRetries": 6}, "maxRetries"),
        ({"refreshMs": -1}, "refreshMs"),
        ({"controlAfterRefresh": "middle"}, "controlAfterRefresh"),
        ({"unknown": 1}, "unknown fields"),
    ):
        with pytest.raises(ValueError, match=message):
            schema_from_wire(malformed_remote(**changes))

    no_button = malformed_remote(refreshButton=False)
    with pytest.raises(ValueError, match="requires a refresh button"):
        schema_from_wire(no_button)

    wire21 = schema_to_wire(schema, wire_version=21)
    entry21 = cast("list[dict[str, Any]]", wire21["interface"])[0]
    remote21 = cast("dict[str, Any]", cast("dict[str, Any]", entry21["widget"])["remote"])
    remote21["unknown"] = 4096
    with pytest.raises(ValueError, match="unknown fields"):
        schema_from_wire(wire21)


def test_boolean_widget_wire_roundtrip_and_signature_exclusion() -> None:
    """BOOLEAN is the fourth closed widget kind (wire v10): custom toggle
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
    """NUMBER is the fifth closed widget kind (wire v11): min/max/step
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


def test_number_widget_display_is_closed_and_downgrades_to_wire17() -> None:
    displays = ("number", "slider", "knob", "gradientslider")
    for display in displays:
        assert NumberWidget(display=display).display == display
    for malformed in ("", "Slider", "gradient_slider", 1, True, ["slider"]):
        with pytest.raises(ValueError, match="display must be one of"):
            NumberWidget(display=malformed)  # type: ignore[arg-type]

    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="display must be one of"):
        NumberWidget(display=StringSubclass("slider"))  # type: ignore[arg-type]

    plain = dataclasses.replace(SCHEMA, inputs=(SCHEMA.inputs[0],) + SCHEMA.inputs[1:])
    display_only = dataclasses.replace(
        plain,
        inputs=(dataclasses.replace(plain.inputs[0], widget=NumberWidget(display="number")),)
        + plain.inputs[1:],
    )
    wire18 = schema_to_wire(display_only)
    entry18 = cast("list[dict[str, Any]]", wire18["interface"])[0]
    assert entry18["widget"] == {"type": "NUMBER", "display": "number"}
    assert schema_from_wire(wire18) == display_only
    assert schema_signature(display_only) == schema_signature(plain)

    wire17 = schema_to_wire(display_only, wire_version=17)
    entry17 = cast("list[dict[str, Any]]", wire17["interface"])[0]
    assert "widget" not in entry17
    with pytest.raises(ValueError, match="unsupported schemaVersion: 17"):
        schema_from_wire(wire17)
    assert json.dumps(wire17, separators=(",", ":")) == json.dumps(
        schema_to_wire(plain, wire_version=17), separators=(",", ":")
    )

    constrained = dataclasses.replace(
        display_only,
        inputs=(
            dataclasses.replace(
                display_only.inputs[0], widget=NumberWidget(min=0, step=1, display="slider")
            ),
        )
        + display_only.inputs[1:],
    )
    constrained17 = schema_to_wire(constrained, wire_version=17)
    constrained_entry = cast("list[dict[str, Any]]", constrained17["interface"])[0]
    assert constrained_entry["widget"] == {"type": "NUMBER", "min": 0, "step": 1}
    constrained_without_display = dataclasses.replace(
        constrained,
        inputs=(dataclasses.replace(constrained.inputs[0], widget=NumberWidget(min=0, step=1)),)
        + constrained.inputs[1:],
    )
    assert json.dumps(constrained17, separators=(",", ":")) == json.dumps(
        schema_to_wire(constrained_without_display, wire_version=17), separators=(",", ":")
    )
    for old_version in (17,):
        old_wire = schema_to_wire(constrained, wire_version=old_version)
        old_entry = cast("list[dict[str, Any]]", old_wire["interface"])[0]
        cast("dict[str, Any]", old_entry["widget"])["display"] = "slider"
        with pytest.raises(ValueError, match="unsupported schemaVersion: 17"):
            schema_from_wire(old_wire)


def test_number_widget_representations_filter_display_only_wire17_members() -> None:
    def schema_with(widget: WidgetRepresentations) -> NodeSchema:
        return NodeSchema(
            node_type="test.number-representations",
            inputs=(InputSpec("value", TypeExpr.concrete("core.float"), widget=widget),),
        )

    bounded = WidgetRepresentation(
        "bounded",
        NumberWidget(min=0.0, max=1.0, step=0.1),
        display_name="Bounded",
    )
    slider = WidgetRepresentation("slider", NumberWidget(display="slider"))
    knob = WidgetRepresentation("knob", NumberWidget(display="knob"))

    retained_default = schema_with(
        WidgetRepresentations(
            representations=(bounded, slider),
            default="bounded",
            user_switchable=True,
        )
    )
    wire18 = schema_to_wire(retained_default)
    assert schema_from_wire(wire18) == retained_default
    wire17 = schema_to_wire(retained_default, wire_version=17)
    entry17 = cast("list[dict[str, Any]]", wire17["interface"])[0]
    assert entry17["widget"] == {
        "type": "REPRESENTATIONS",
        "default": "bounded",
        "userSwitchable": True,
        "representations": [
            {
                "id": "bounded",
                "displayName": "Bounded",
                "widget": {"type": "NUMBER", "min": 0.0, "max": 1.0, "step": 0.1},
            }
        ],
    }

    dropped_default = schema_with(
        WidgetRepresentations(
            representations=(slider, bounded, WidgetRepresentation("wide", NumberWidget(min=-1.0))),
            default="slider",
            user_switchable=True,
        )
    )
    dropped_wire17 = schema_to_wire(dropped_default, wire_version=17)
    dropped_widget = cast(
        "dict[str, Any]", cast("list[dict[str, Any]]", dropped_wire17["interface"])[0]["widget"]
    )
    assert dropped_widget["default"] == "bounded"
    assert [item["id"] for item in dropped_widget["representations"]] == ["bounded", "wide"]

    all_display_only = schema_with(
        WidgetRepresentations(
            representations=(slider, knob),
            default="slider",
            user_switchable=True,
        )
    )
    all_dropped_entry = cast(
        "list[dict[str, Any]]", schema_to_wire(all_display_only, wire_version=17)["interface"]
    )[0]
    assert "widget" not in all_dropped_entry


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


def test_number_widget_round_is_float_only_and_downgrades_to_wire18() -> None:
    assert NumberWidget(0, 1, 0.1, "fixed", "slider") == NumberWidget(
        min=0,
        max=1,
        step=0.1,
        control_after_generate="fixed",
        display="slider",
    )
    plain = dataclasses.replace(
        SCHEMA,
        inputs=(InputSpec("value", TypeExpr.concrete("core.float")),) + SCHEMA.inputs[1:],
    )
    rounded = dataclasses.replace(
        plain,
        inputs=(dataclasses.replace(plain.inputs[0], widget=NumberWidget(round=0.01)),)
        + plain.inputs[1:],
    )
    entry19 = cast("list[dict[str, Any]]", schema_to_wire(rounded)["interface"])[0]
    assert entry19["widget"] == {"type": "NUMBER", "round": 0.01}
    assert schema_from_wire(schema_to_wire(rounded)) == rounded
    without_widget = dataclasses.replace(
        rounded,
        inputs=(dataclasses.replace(rounded.inputs[0], widget=None),) + rounded.inputs[1:],
    )
    assert schema_signature(rounded) == schema_signature(without_widget)

    entry18 = cast("list[dict[str, Any]]", schema_to_wire(rounded, wire_version=18)["interface"])[0]
    assert "widget" not in entry18

    class NumberSubclass(float):
        pass

    for malformed in (
        0,
        -1,
        float("nan"),
        float("inf"),
        True,
        "0.01",
        NumberSubclass(0.01),
    ):
        with pytest.raises(ValueError, match="round must be a finite positive number"):
            NumberWidget(round=malformed)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="round requires a concrete core.float"):
        InputSpec("value", TypeExpr.concrete("core.int"), widget=NumberWidget(round=1))


def test_number_widget_unsafe_integer_bounds_use_decimal_wire() -> None:
    safe = 2**53 - 1
    unsafe = safe + 1
    uint64_max = 0xFFFFFFFFFFFFFFFF
    assert NumberWidget(min=0, max=safe).max == safe
    assert NumberWidget(min=-safe, max=0).min == -safe
    schema = NodeSchema(
        "test.uint64",
        inputs=(
            InputSpec(
                "seed",
                TypeExpr.concrete("core.int"),
                widget=NumberWidget(min=0, max=uint64_max, step=unsafe),
            ),
        ),
    )
    wire33 = schema_to_wire(schema)
    widget33 = cast("list[dict[str, Any]]", wire33["interface"])[0]["widget"]
    assert widget33 == {
        "type": "NUMBER",
        "min": 0,
        "max": str(uint64_max),
        "step": str(unsafe),
    }
    assert schema_from_wire(wire33) == schema

    widget32 = cast("list[dict[str, Any]]", schema_to_wire(schema, wire_version=32)["interface"])[
        0
    ]["widget"]
    assert widget32 == {"type": "NUMBER", "min": 0}

    for malformed in (
        "9007199254740991",
        "+9007199254740992",
        "09007199254740992",
        "1e16",
        "18446744073709551616",
        0xFFFFFFFFFFFFFFFF,
        float(0xFFFFFFFFFFFFFFFF),
    ):
        broken = schema_to_wire(schema)
        cast("dict[str, Any]", cast("list[dict[str, Any]]", broken["interface"])[0]["widget"])[
            "max"
        ] = malformed
        with pytest.raises(ValueError):
            schema_from_wire(broken)

    for bad in (-(2**63) - 1, uint64_max + 1):
        with pytest.raises(ValueError, match="supported integer range"):
            NumberWidget(min=bad)
    with pytest.raises(ValueError, match="JSON-double-safe"):
        InputSpec(
            "value",
            TypeExpr.concrete("core.float"),
            widget=NumberWidget(max=unsafe),
        )
    # Floats are doubles already: large float bounds stay legal, exactly
    # as they will arrive.
    assert NumberWidget(max=1e300).max == 1e300


def test_string_widget_wire_roundtrip_and_signature_exclusion() -> None:
    """STRING is the sixth closed widget kind (wire v11). Wire v17 makes
    the established inferred single-line presentation explicit while
    retaining the original multiline declaration unchanged."""
    plain = dataclasses.replace(
        SCHEMA,
        inputs=(InputSpec("text", TypeExpr.concrete("core.string"), required=False),)
        + SCHEMA.inputs[1:],
    )
    multiline = dataclasses.replace(
        plain,
        inputs=(dataclasses.replace(plain.inputs[0], widget=StringWidget()),) + plain.inputs[1:],
    )
    wire = schema_to_wire(multiline)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert interface[0]["widget"] == {"type": "STRING", "multiline": True}
    assert schema_from_wire(wire) == multiline
    assert schema_signature(multiline) == schema_signature(plain)

    single_line = dataclasses.replace(
        plain,
        inputs=(dataclasses.replace(plain.inputs[0], widget=StringWidget(multiline=False)),)
        + plain.inputs[1:],
    )
    single_wire = schema_to_wire(single_line)
    single_interface = cast("list[dict[str, Any]]", single_wire["interface"])
    assert single_interface[0]["widget"] == {"type": "STRING", "multiline": False}
    assert schema_from_wire(single_wire) == single_line
    assert (
        "widget"
        not in cast(
            "list[dict[str, Any]]", schema_to_wire(single_line, wire_version=16)["interface"]
        )[0]
    )
    assert schema_signature(single_line) == schema_signature(plain)


def test_string_widget_v19_fields_preserve_tristate_identity_and_downgrade() -> None:
    plain = NodeSchema(
        node_type="test.string-v19",
        inputs=(InputSpec("text", TypeExpr.concrete("core.string")),),
    )
    signatures: set[str] = set()
    for dynamic_prompts in (None, False, True):
        widget = StringWidget(
            multiline=None,
            placeholder="Describe an image",
            dynamic_prompts=dynamic_prompts,
        )
        schema = dataclasses.replace(
            plain,
            inputs=(dataclasses.replace(plain.inputs[0], widget=widget),),
        )
        wire = schema_to_wire(schema)
        entry = cast("list[dict[str, Any]]", wire["interface"])[0]
        expected: dict[str, object] = {
            "type": "STRING",
            "placeholder": "Describe an image",
        }
        if dynamic_prompts is not None:
            expected["dynamicPrompts"] = dynamic_prompts
        assert entry["widget"] == expected
        assert schema_from_wire(wire) == schema
        entry18 = cast(
            "list[dict[str, Any]]", schema_to_wire(schema, wire_version=18)["interface"]
        )[0]
        assert "widget" not in entry18
        signatures.add(schema_signature(schema))
    assert len(signatures) == 3

    placeholder_only = dataclasses.replace(
        plain,
        inputs=(
            dataclasses.replace(
                plain.inputs[0],
                widget=StringWidget(multiline=None, placeholder=""),
            ),
        ),
    )
    assert schema_signature(placeholder_only) == schema_signature(plain)

    dynamic_representations = WidgetRepresentations(
        representations=(
            WidgetRepresentation(
                "literal",
                StringWidget(multiline=False, dynamic_prompts=False),
            ),
            WidgetRepresentation(
                "expanded",
                StringWidget(multiline=True, dynamic_prompts=True),
            ),
        ),
        default="literal",
        user_switchable=True,
    )
    literal_default = dataclasses.replace(
        plain,
        inputs=(
            dataclasses.replace(
                plain.inputs[0],
                widget=dynamic_representations,
            ),
        ),
    )
    expanded_default = dataclasses.replace(
        literal_default,
        inputs=(
            dataclasses.replace(
                literal_default.inputs[0],
                widget=dataclasses.replace(
                    dynamic_representations,
                    default="expanded",
                ),
            ),
        ),
    )
    assert schema_signature(literal_default) != schema_signature(expanded_default)

    for malformed in (0, False, [], {}):
        with pytest.raises(ValueError, match="placeholder must be a string"):
            StringWidget(multiline=None, placeholder=malformed)  # type: ignore[arg-type]
    for malformed in (0, 1, "true", []):
        with pytest.raises(ValueError, match="dynamic_prompts must be a bool"):
            StringWidget(multiline=None, dynamic_prompts=malformed)  # type: ignore[arg-type]


def test_string_widget_validated() -> None:
    for malformed in (0, 1, "false", []):
        with pytest.raises(ValueError, match="multiline must be a bool"):
            StringWidget(multiline=malformed)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one presentation field"):
        StringWidget(multiline=None)


def test_string_completions_wire36_roundtrip_downlevel_and_signature_exclusion() -> None:
    completions = TextCompletions(
        items=(
            TextCompletionItem("sin", label="sin()", insert_text="sin()", detail="Function"),
            TextCompletionItem("**", detail="Operator", kind="operator"),
        ),
        input_families=("values",),
    )
    schema = NodeSchema(
        node_type="test.text-completions",
        inputs=(
            InputSpec(
                "expression",
                TypeExpr.concrete("core.string"),
                widget=StringWidget(multiline=True, completions=completions),
            ),
        ),
    )
    entry = cast("list[dict[str, Any]]", schema_to_wire(schema)["interface"])[0]
    assert entry["widget"] == {
        "type": "STRING",
        "multiline": True,
        "completions": {
            "items": [
                {
                    "value": "sin",
                    "label": "sin()",
                    "insertText": "sin()",
                    "detail": "Function",
                },
                {"value": "**", "detail": "Operator", "kind": "operator"},
            ],
            "inputFamilies": ["values"],
        },
    }
    assert schema_from_wire(schema_to_wire(schema)) == schema
    downlevel_wire = schema_to_wire(schema, wire_version=35)
    downlevel_entry = cast("list[dict[str, Any]]", downlevel_wire["interface"])[0]
    assert downlevel_entry["widget"] == {"type": "STRING", "multiline": True}
    downlevel_schema = schema_from_wire(downlevel_wire)
    assert downlevel_schema.inputs[0].widget == StringWidget(multiline=True)

    mislabeled_v35 = json.loads(json.dumps(schema_to_wire(schema)))
    mislabeled_v35["schemaVersion"] = 35
    with pytest.raises(ValueError, match="unknown fields: \\[?'completions'\\]?"):
        schema_from_wire(mislabeled_v35)
    without_completions = dataclasses.replace(
        schema,
        inputs=(
            dataclasses.replace(
                schema.inputs[0],
                widget=StringWidget(multiline=True),
            ),
        ),
    )
    assert schema_signature(schema) == schema_signature(without_completions)

    malformed = json.loads(json.dumps(schema_to_wire(schema)))
    malformed_entry = cast("list[dict[str, Any]]", malformed["interface"])[0]
    cast("dict[str, Any]", malformed_entry["widget"])["completions"] = None
    with pytest.raises(ValueError, match="completions must be an object"):
        schema_from_wire(malformed)


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


def test_widget_representations_roundtrip_downlevel_and_signature_exclusion() -> None:
    representations = WidgetRepresentations(
        representations=(
            WidgetRepresentation(
                "single-line", StringWidget(multiline=False), display_name="Single line"
            ),
            WidgetRepresentation(
                "multiline", StringWidget(multiline=True), display_name="Multiline"
            ),
        ),
        default="multiline",
        user_switchable=True,
    )
    plain = NodeSchema(
        node_type="test.representations",
        inputs=(InputSpec("text", TypeExpr.concrete("core.string")),),
    )
    schema = dataclasses.replace(
        plain,
        inputs=(dataclasses.replace(plain.inputs[0], widget=representations),),
    )
    wire = schema_to_wire(schema)
    entry = cast("list[dict[str, Any]]", wire["interface"])[0]
    assert entry["widget"] == {
        "type": "REPRESENTATIONS",
        "default": "multiline",
        "userSwitchable": True,
        "representations": [
            {
                "id": "single-line",
                "displayName": "Single line",
                "widget": {"type": "STRING", "multiline": False},
            },
            {
                "id": "multiline",
                "displayName": "Multiline",
                "widget": {"type": "STRING", "multiline": True},
            },
        ],
    }
    assert schema_from_wire(wire) == schema

    downlevel = schema_to_wire(schema, wire_version=16)
    downlevel_entry = cast("list[dict[str, Any]]", downlevel["interface"])[0]
    assert downlevel_entry["widget"] == {"type": "STRING", "multiline": True}
    with pytest.raises(ValueError, match="unsupported schemaVersion: 16"):
        schema_from_wire(downlevel)

    changed_presentation = dataclasses.replace(
        schema,
        inputs=(
            dataclasses.replace(
                schema.inputs[0],
                widget=dataclasses.replace(
                    representations,
                    default="single-line",
                    user_switchable=False,
                ),
            ),
        ),
    )
    # Without dynamicPrompts, the wrapper default and switching authority
    # remain presentation-only and retain the established signature.
    assert schema_signature(schema) == schema_signature(changed_presentation)
    assert schema_signature(schema) == schema_signature(plain)


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


def test_widget_representations_wire_decode_is_strict() -> None:
    schema = NodeSchema(
        node_type="test.representations-strict",
        inputs=(
            InputSpec(
                "text",
                TypeExpr.concrete("core.string"),
                widget=WidgetRepresentations(
                    (
                        WidgetRepresentation("single-line", StringWidget(multiline=False)),
                        WidgetRepresentation("multiline", StringWidget(multiline=True)),
                    ),
                    default="multiline",
                    user_switchable=True,
                ),
            ),
        ),
    )

    def widget_wire() -> dict[str, Any]:
        wire = schema_to_wire(schema)
        return cast("dict[str, Any]", cast("list[dict[str, Any]]", wire["interface"])[0]["widget"])

    malformed: list[dict[str, Any]] = []
    unknown_top = widget_wire()
    unknown_top["future"] = True
    malformed.append(unknown_top)
    unknown_entry = widget_wire()
    unknown_entry["representations"][0]["future"] = True
    malformed.append(unknown_entry)
    unknown_descriptor = widget_wire()
    unknown_descriptor["representations"][0]["widget"]["future"] = True
    malformed.append(unknown_descriptor)
    for field, value in (
        ("default", 1),
        ("userSwitchable", "yes"),
        ("representations", {}),
    ):
        broken = widget_wire()
        broken[field] = value
        malformed.append(broken)
    empty = widget_wire()
    empty["representations"] = []
    malformed.append(empty)
    duplicate = widget_wire()
    duplicate["representations"][1]["id"] = "single-line"
    malformed.append(duplicate)
    missing_default = widget_wire()
    missing_default["default"] = "missing"
    malformed.append(missing_default)

    for bad in malformed:
        wire = schema_to_wire(schema)
        cast("list[dict[str, Any]]", wire["interface"])[0]["widget"] = bad
        with pytest.raises(ValueError):
            schema_from_wire(wire)

    downlevel = schema_to_wire(schema, wire_version=16)
    cast("list[dict[str, Any]]", downlevel["interface"])[0]["widget"] = widget_wire()
    with pytest.raises(ValueError, match="unsupported schemaVersion: 16"):
        schema_from_wire(downlevel)


def test_widget_socket_binding_enforced() -> None:
    """NUMBER/STRING descriptors bind to the socket type and constraints
    interpret in the socket's domain (frontend amendment, wire v11): NUMBER
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
    """Per-input displayName (wire v11) is a presentation label: empty
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
    """Wire v14 replaces v13's suggestion marker with core.combo identity.
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


def test_combo_controller_and_color_are_v19_only_presentation() -> None:
    schema = NodeSchema(
        node_type="test.widget-v19",
        inputs=(
            InputSpec(
                "choice",
                TypeExpr.concrete("core.combo"),
                widget=ComboWidget(
                    options=("a", "b"),
                    control_after_generate="increment",
                ),
            ),
            InputSpec("color", TypeExpr.concrete("core.string"), widget=ColorWidget()),
        ),
    )
    wire19 = schema_to_wire(schema)
    entries19 = cast("list[dict[str, Any]]", wire19["interface"])
    assert entries19[0]["widget"] == {
        "type": "COMBO",
        "options": ["a", "b"],
        "controlAfterGenerate": "increment",
    }
    assert entries19[1]["widget"] == {"type": "COLOR"}
    assert schema_from_wire(wire19) == schema

    wire18 = schema_to_wire(schema, wire_version=18)
    entries18 = cast("list[dict[str, Any]]", wire18["interface"])
    assert entries18[0]["widget"] == {"type": "COMBO", "options": ["a", "b"]}
    assert "widget" not in entries18[1]
    without_v19 = dataclasses.replace(
        schema,
        inputs=(
            dataclasses.replace(
                schema.inputs[0],
                widget=ComboWidget(options=("a", "b")),
            ),
            dataclasses.replace(schema.inputs[1], widget=None),
        ),
    )
    assert json.dumps(wire18, separators=(",", ":")) == json.dumps(
        schema_to_wire(without_v19, wire_version=18), separators=(",", ":")
    )
    assert schema_signature(schema) == schema_signature(without_v19)


def test_wire18_recursively_filters_v19_only_representations_and_repairs_default() -> None:
    schema = NodeSchema(
        node_type="test.widget-v19-representations",
        inputs=(
            InputSpec(
                "value",
                TypeExpr.concrete("core.string"),
                widget=WidgetRepresentations(
                    representations=(
                        WidgetRepresentation("color", ColorWidget()),
                        WidgetRepresentation(
                            "placeholder",
                            StringWidget(multiline=None, placeholder="Describe an image"),
                        ),
                        WidgetRepresentation("multiline", StringWidget(multiline=True)),
                    ),
                    default="color",
                    user_switchable=True,
                ),
            ),
        ),
    )
    wire18 = schema_to_wire(schema, wire_version=18)
    entry = cast("list[dict[str, Any]]", wire18["interface"])[0]
    assert entry["widget"] == {
        "type": "REPRESENTATIONS",
        "default": "multiline",
        "userSwitchable": True,
        "representations": [
            {
                "id": "multiline",
                "widget": {"type": "STRING", "multiline": True},
            }
        ],
    }


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


def test_emits_previews_rides_the_wire_from_v24() -> None:
    """The live-preview capability flag is additive wire data: emitted only
    when declared and only at wire 24+, so older clients never see it and
    older backends simply leave it absent (decoded as False)."""
    schema = dataclasses.replace(SCHEMA, emits_previews=True)
    wire = schema_to_wire(schema)
    assert wire["emitsPreviews"] is True
    assert schema_from_wire(wire) == schema

    frozen = schema_to_wire(schema, wire_version=23)
    assert "emitsPreviews" not in frozen

    plain = schema_to_wire(SCHEMA)
    assert "emitsPreviews" not in plain
    assert schema_from_wire(plain).emits_previews is False


def test_emits_previews_never_joins_the_signature() -> None:
    """Whether a node ships live previews while running never changes what
    it computes - declaring the capability must not invalidate caches."""
    flagged = dataclasses.replace(SCHEMA, emits_previews=True)
    assert schema_signature(flagged) == schema_signature(SCHEMA)


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


def test_output_represents_rides_wire_v31_and_refuses_older_wires() -> None:
    schema = _represented_schema()
    for wire_version in (31, 32, 33, 34):
        wire = schema_to_wire(schema, wire_version=wire_version)
        output = cast("list[dict[str, Any]]", wire["interface"])[1]
        assert output["represents"] == {
            "input": "image",
            "rendition": "decoded-image",
        }
        assert schema_from_wire(wire) == schema

    with pytest.raises(
        SchemaWireVersionRequirement,
        match="output representation requires schema wire 31",
    ) as required:
        schema_to_wire(schema, wire_version=30)
    assert required.value.required_version == 31
    assert required.value.code == "schema-wire-required"

    mislabeled = schema_to_wire(schema)
    mislabeled["schemaVersion"] = 30
    with pytest.raises(ValueError, match="output.represents requires schema wire 31"):
        schema_from_wire(mislabeled)


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


def test_output_known_value_rides_wire_v34_and_validates_primitive_identity() -> None:
    integer = TypeExpr.concrete("core.int")
    known = OutputKnownValue("value")
    schema = NodeSchema(
        node_type="test.known-output",
        inputs=(InputSpec("value", integer, required=False, default=0),),
        outputs=(OutputSpec("value", integer, known_value=known),),
    )
    wire = schema_to_wire(schema)
    output = cast("list[dict[str, Any]]", wire["interface"])[1]
    assert output["knownValue"] == {"input": "value"}
    assert schema_from_wire(wire) == schema

    with pytest.raises(
        SchemaWireVersionRequirement,
        match="output known value requires schema wire 34",
    ) as required:
        schema_to_wire(schema, wire_version=33)
    assert required.value.required_version == 34

    mislabeled = schema_to_wire(schema)
    mislabeled["schemaVersion"] = 33
    with pytest.raises(ValueError, match="output.knownValue requires schema wire 34"):
        schema_from_wire(mislabeled)

    with pytest.raises(ValueError, match="input must be a non-empty string"):
        OutputKnownValue("")
    with pytest.raises(ValueError, match="known_value must be an OutputKnownValue"):
        OutputSpec("value", integer, known_value=cast("Any", {}))
    with pytest.raises(ValueError, match="is not a declared top-level input"):
        dataclasses.replace(
            schema,
            outputs=(OutputSpec("value", integer, known_value=OutputKnownValue("missing")),),
        )
    with pytest.raises(ValueError, match="same concrete primitive type"):
        dataclasses.replace(
            schema,
            outputs=(OutputSpec("value", TypeExpr.concrete("core.float"), known_value=known),),
        )
    with pytest.raises(ValueError, match="same concrete primitive type"):
        dataclasses.replace(
            schema,
            inputs=(InputSpec("value", TypeExpr.concrete("dinkster.image")),),
            outputs=(
                OutputSpec(
                    "value",
                    TypeExpr.concrete("dinkster.image"),
                    known_value=known,
                ),
            ),
        )

    without_known_value = dataclasses.replace(
        schema,
        outputs=(dataclasses.replace(schema.outputs[0], known_value=None),),
    )
    assert schema_signature(schema) == schema_signature(without_known_value)


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


def test_output_represents_wire_decode_rejects_unknown_fields_and_bad_shapes() -> None:
    def malformed(represents: object) -> dict[str, object]:
        wire = schema_to_wire(_represented_schema())
        cast("list[dict[str, Any]]", wire["interface"])[1]["represents"] = represents
        return wire

    with pytest.raises(ValueError, match="output.represents must be an object"):
        schema_from_wire(malformed(1))
    with pytest.raises(ValueError, match="output.represents has unknown fields"):
        schema_from_wire(
            malformed({"input": "image", "rendition": "decoded-image", "surprise": True})
        )
    with pytest.raises(ValueError, match="output.represents.input must be a string"):
        schema_from_wire(malformed({"input": 1, "rendition": "decoded-image"}))
    with pytest.raises(ValueError, match="output.represents.applies must be an object"):
        schema_from_wire(malformed({"input": "image", "rendition": "decoded-image", "applies": []}))
    with pytest.raises(ValueError, match="must be an array of strings"):
        schema_from_wire(
            malformed(
                {
                    "input": "image",
                    "rendition": "decoded-image",
                    "applies": {"mode": [1]},
                }
            )
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


def test_mirror_rides_the_wire_from_v29() -> None:
    """The mirror declaration is additive presentation metadata: emitted only
    when declared and only at wire 29+, so older clients never see it and
    older backends simply leave it absent (decoded as None)."""
    expression = dataclasses.replace(SCHEMA, mirror=EXPRESSION_MIRROR)
    wire = schema_to_wire(expression)
    assert wire["mirror"] == {
        "kind": "expression",
        "precision": "bounded",
        "tolerance": {"relative": 1e-12},
        "grammarVersion": 1,
    }
    assert schema_from_wire(wire) == expression

    glsl = dataclasses.replace(SCHEMA, mirror=GLSL_MIRROR)
    wire = schema_to_wire(glsl)
    assert wire["mirror"] == {
        "kind": "glsl",
        "precision": "bounded",
        "tolerance": {"perChannel": 1.0 / 255.0},
        "source": "void main() {}",
    }
    assert schema_from_wire(wire) == glsl

    exact = dataclasses.replace(
        SCHEMA, mirror=MirrorSpec(kind="expression", precision="exact", grammar_version=2)
    )
    wire = schema_to_wire(exact)
    assert wire["mirror"] == {"kind": "expression", "precision": "exact", "grammarVersion": 2}
    assert schema_from_wire(wire) == exact

    frozen = schema_to_wire(expression, wire_version=28)
    assert "mirror" not in frozen

    plain = schema_to_wire(SCHEMA)
    assert "mirror" not in plain
    assert schema_from_wire(plain).mirror is None


def test_mirror_never_joins_the_signature() -> None:
    """A mirror is a presentation-only estimate renderer - declaring,
    changing, or removing one never changes what the node computes, so
    caches must survive."""
    mirrored = dataclasses.replace(SCHEMA, mirror=EXPRESSION_MIRROR)
    assert schema_signature(mirrored) == schema_signature(SCHEMA)
    retuned = dataclasses.replace(SCHEMA, mirror=GLSL_MIRROR)
    assert schema_signature(retuned) == schema_signature(SCHEMA)


def test_mirror_validated() -> None:
    with pytest.raises(ValueError, match="unknown mirror kind"):
        MirrorSpec(kind="wasm", precision="exact")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="unknown mirror precision"):
        MirrorSpec(kind="expression", precision="close")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="bounded mirrors require a tolerance"):
        MirrorSpec(kind="expression", precision="bounded", grammar_version=1)
    with pytest.raises(ValueError, match="must not declare a tolerance"):
        MirrorSpec(
            kind="expression",
            precision="exact",
            tolerance=MirrorTolerance(relative=1e-12),
            grammar_version=1,
        )
    with pytest.raises(ValueError, match="require a positive grammar_version"):
        MirrorSpec(kind="expression", precision="exact")
    with pytest.raises(ValueError, match="require a positive grammar_version"):
        MirrorSpec(kind="expression", precision="exact", grammar_version=0)
    with pytest.raises(ValueError, match="must not carry shader source"):
        MirrorSpec(kind="expression", precision="exact", grammar_version=1, source="x")
    with pytest.raises(ValueError, match="require shader source"):
        MirrorSpec(kind="glsl", precision="exact")
    with pytest.raises(ValueError, match="must not carry a grammar_version"):
        MirrorSpec(kind="glsl", precision="exact", grammar_version=1, source="x")
    with pytest.raises(ValueError, match="exceeds 16384 bytes"):
        MirrorSpec(kind="glsl", precision="exact", source="x" * 16385)
    with pytest.raises(ValueError, match="unknown mirror kind"):
        MirrorSpec(kind=cast("Any", {"kind": "expression"}), precision="exact")
    with pytest.raises(ValueError, match="unknown mirror precision"):
        MirrorSpec(kind="expression", precision=cast("Any", {"precision": "exact"}))
    with pytest.raises(ValueError, match="tolerance must be a MirrorTolerance"):
        MirrorSpec(
            kind="expression",
            precision="bounded",
            tolerance=cast("Any", {"relative": 1e-12}),
            grammar_version=1,
        )
    with pytest.raises(ValueError, match="shader source must be a string"):
        MirrorSpec(kind="glsl", precision="exact", source=cast("Any", 1))
    with pytest.raises(ValueError, match="mirror must be a MirrorSpec"):
        dataclasses.replace(SCHEMA, mirror=cast("Any", {}))
    with pytest.raises(ValueError, match="mirror must be a MirrorSpec"):
        dataclasses.replace(SCHEMA, mirror=cast("Any", 1))
    with pytest.raises(ValueError, match="requires relative or per_channel"):
        MirrorTolerance()
    with pytest.raises(ValueError, match="finite positive float"):
        MirrorTolerance(relative=0.0)
    with pytest.raises(ValueError, match="finite positive float"):
        MirrorTolerance(per_channel=float("inf"))
    with pytest.raises(ValueError, match="finite positive float"):
        MirrorTolerance(relative=cast("Any", 1))


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


def test_mirror_decode_requires_wire_v29() -> None:
    """The v29 gate holds on decode as well as encode: a mirror carried by an
    older-versioned payload is a mislabeled wire, not data to accept."""
    wire = schema_to_wire(dataclasses.replace(SCHEMA, mirror=EXPRESSION_MIRROR))
    wire["schemaVersion"] = 28
    with pytest.raises(ValueError, match="mirror requires schema wire 29"):
        schema_from_wire(wire)
    del wire["mirror"]
    assert schema_from_wire(wire) == SCHEMA


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


def test_mirror_applies_rides_the_wire_from_v30() -> None:
    """An applies scope is emitted from wire 30. Below 30 the WHOLE scoped
    declaration is withheld: emitting the mirror without its scope would hand
    older clients a mirror they would run for combo values it does not cover
    (wrong estimates, not missing ones)."""
    wire = schema_to_wire(SCOPED_SCHEMA)
    assert wire["mirror"] == {
        "kind": "glsl",
        "precision": "bounded",
        "tolerance": {"perChannel": 1.0 / 255.0},
        "source": "void main() {}",
        "applies": {"operation": ["gaussian_blur", "sharpen"]},
    }
    assert schema_from_wire(wire) == SCOPED_SCHEMA

    frozen = schema_to_wire(SCOPED_SCHEMA, wire_version=29)
    assert "mirror" not in frozen

    unscoped = schema_to_wire(dataclasses.replace(SCHEMA, mirror=GLSL_MIRROR), wire_version=29)
    assert "applies" not in cast("dict[str, Any]", unscoped["mirror"])


def test_mirror_applies_never_joins_the_signature() -> None:
    """Scoping a mirror is as presentation-only as declaring one."""
    assert schema_signature(SCOPED_SCHEMA) == schema_signature(
        dataclasses.replace(SCOPED_SCHEMA, mirror=None)
    )


def test_mirror_applies_decode_requires_wire_v30() -> None:
    """The v30 gate holds on decode as well as encode: dropping an applies
    scope would over-apply the mirror after a decode/encode round trip."""
    wire = schema_to_wire(SCOPED_SCHEMA)
    wire["schemaVersion"] = 29
    with pytest.raises(ValueError, match="mirror.applies requires schema wire 30"):
        schema_from_wire(wire)
    del cast("dict[str, Any]", wire["mirror"])["applies"]
    assert schema_from_wire(wire) == dataclasses.replace(SCOPED_SCHEMA, mirror=GLSL_MIRROR)


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


def test_widget_groups_wire_validation_signature_and_downencode() -> None:
    from dinkster_schema import ConditionalWidgetCondition, ConditionalWidgetGroup

    with pytest.raises(ValueError, match="values must be unique"):
        ConditionalWidgetCondition("a", (1, 1.0))
    with pytest.raises(ValueError, match="values must be an immutable tuple"):
        ConditionalWidgetCondition(
            "a", cast("tuple[str | int | float | bool | None, ...]", ["show"])
        )
    with pytest.raises(ValueError, match="members must be an immutable tuple"):
        ConditionalWidgetGroup("a", ("show",), cast("tuple[str, ...]", ["b"]))
    with pytest.raises(ValueError, match="requires must be an immutable tuple"):
        ConditionalWidgetGroup(
            "a",
            ("show",),
            ("b",),
            cast(
                "tuple[ConditionalWidgetCondition, ...]",
                [ConditionalWidgetCondition("c", (True,))],
            ),
        )
    plain = NodeSchema(
        "test.widget-groups",
        inputs=(
            InputSpec("a", TypeExpr.concrete("core.int"), widget=NumberWidget(display="number")),
            InputSpec("b", TypeExpr.concrete("core.int"), widget=NumberWidget(display="number")),
        ),
    )
    with pytest.raises(ValueError, match="widget_groups must be an immutable tuple"):
        dataclasses.replace(
            plain,
            widget_groups=cast("tuple[ConditionalWidgetGroup, ...]", []),
        )
    grouped = dataclasses.replace(
        plain,
        widget_groups=(
            ConditionalWidgetGroup(
                "a",
                ("show", 1, True, None),
                ("b",),
            ),
        ),
    )
    wire = schema_to_wire(grouped, wire_version=27)
    assert wire["widgetGroups"] == [
        {"input": "a", "values": ["show", 1, True, None], "members": ["b"]}
    ]
    assert schema_from_wire(wire) == grouped
    assert "widgetGroups" not in schema_to_wire(grouped, wire_version=26)
    assert schema_signature(grouped) == schema_signature(plain)
    with pytest.raises(ValueError, match="cannot control themselves"):
        dataclasses.replace(
            grouped,
            widget_groups=(ConditionalWidgetGroup("a", (1,), ("a",)),),
        )
    wire["schemaVersion"] = 26
    with pytest.raises(ValueError, match="requires schema wire 27"):
        schema_from_wire(wire)


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


def test_variable_type_expr_wire_spells_allowed_not_types() -> None:
    # Wire-15 contract pin (docs/wire15-contract.md "Variables at every
    # depth"): variable is {kind, templateId, allowed?}; "types" is reserved
    # for concrete/union kinds. Regression: ResizeImageMaskNode's variable
    # input was emitted with "types" and failed the frontend's fail-closed
    # decoder (2026-07-29).
    unconstrained = TypeExpr.variable("input_type")
    assert type_expr_to_wire(unconstrained) == {
        "kind": "variable",
        "templateId": "input_type",
    }
    constrained = TypeExpr.variable("input_type", ("comfy.IMAGE", "comfy.MASK"))
    assert type_expr_to_wire(constrained) == {
        "kind": "variable",
        "templateId": "input_type",
        "allowed": ["comfy.IMAGE", "comfy.MASK"],
    }
    for expr in (
        unconstrained,
        constrained,
        TypeExpr.list_of(constrained),
        TypeExpr.asset_of(constrained),
    ):
        assert type_expr_from_wire(type_expr_to_wire(expr)) == expr


def test_variable_type_expr_wire_decode_is_fail_closed_on_spelling() -> None:
    with pytest.raises(ValueError, match="not a wire-15 variable field"):
        type_expr_from_wire(
            {
                "kind": "variable",
                "templateId": "input_type",
                "types": ["comfy.IMAGE", "comfy.MASK"],
            }
        )
    for kind, extra in (
        ("concrete", {"types": ["core.int"]}),
        ("union", {"types": ["core.int", "core.float"]}),
        ("wildcard", {}),
    ):
        with pytest.raises(ValueError, match="only legal on variable"):
            type_expr_from_wire({"kind": kind, **extra, "allowed": ["comfy.IMAGE"]})


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
