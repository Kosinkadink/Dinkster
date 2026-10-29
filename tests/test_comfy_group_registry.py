from __future__ import annotations

import copy
import dataclasses

import pytest
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    ComboWidget,
    ComfyAliasConfidence,
    ComfyAliasFamily,
    ComfyAliasSource,
    ComfyGroupEdge,
    ComfyGroupNode,
    ComfyGroupPattern,
    ComfyGroupRecord,
    ComfyGroupRegistry,
    ComfyGroupSource,
    ComfyGroupSourceSchema,
    InputSpec,
    MappingSource,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    comfy_group_collision_problems,
    comfy_group_registry_from_wire,
    comfy_group_registry_problems,
    comfy_group_registry_to_wire,
)

IMAGE = TypeExpr.concrete("core.image")
INT = TypeExpr.concrete("core.int")
COMBO = TypeExpr.concrete("core.combo")


def value_input(id: str, type: TypeExpr, default: object) -> InputSpec:
    return InputSpec(id, type, default=default, widget=NumberWidget(step=1))


PREPROCESSOR = NodeSchema(
    node_type="comfy.controlnet-aux.CannyPreprocessor",
    inputs=(
        InputSpec("image", IMAGE),
        value_input("low", INT, 100),
        InputSpec("mask", IMAGE, required=False),
    ),
    outputs=(OutputSpec("image", IMAGE),),
)
RESIZE = NodeSchema(
    node_type="comfy.ImageScale",
    inputs=(
        InputSpec("image", IMAGE),
        value_input("width", INT, 512),
        InputSpec("mode", COMBO, default="fit", widget=ComboWidget(("fit",))),
    ),
    outputs=(OutputSpec("image", IMAGE),),
)
GROUP = NodeSchema(
    node_type="comfy-group.comfyui-controlnet-aux.canny-resize",
    inputs=(
        InputSpec("image", IMAGE),
        value_input("low", INT, 100),
        value_input("width", INT, 512),
    ),
    outputs=(OutputSpec("image", IMAGE),),
)
TARGET = NodeSchema(
    node_type="dinkster.preprocess.canny",
    inputs=GROUP.inputs,
    outputs=GROUP.outputs,
)


def registry(*, carrier: str = TARGET.node_type) -> ComfyGroupRegistry:
    source = ComfyGroupSource(
        pack="comfyui-controlnet-aux",
        name="canny-resize",
        revision="0123456789abcdef",
    )
    pattern = ComfyGroupPattern(
        group_type=GROUP.node_type,
        anchor="preprocess",
        nodes=(
            (
                "preprocess",
                ComfyGroupNode(
                    source=ComfyAliasSource(
                        pack="comfyui-controlnet-aux",
                        node_class="CannyPreprocessor",
                        node_type=PREPROCESSOR.node_type,
                        revision="0123456789abcdef",
                    ),
                    mode="active",
                ),
            ),
            (
                "resize",
                ComfyGroupNode(
                    source=ComfyAliasSource(
                        pack="comfy-core",
                        node_class="ImageScale",
                        node_type=RESIZE.node_type,
                        revision="b78cec87",
                    ),
                    mode="active",
                ),
            ),
        ),
        edges=(ComfyGroupEdge("preprocess:image", "resize:image"),),
        inputs=(("image", "preprocess:image"),),
        parameters=(("low", "preprocess:low"), ("width", "resize:width")),
        constants=(("resize:mode", "fit"),),
        outputs=(("image", "resize:image"),),
        disconnected=("preprocess:mask",),
    )
    replacement = ReplacementRule(
        from_type=GROUP.node_type,
        cases=(
            ReplacementCase.build(
                carrier,
                inputs={
                    "image": MappingSource.copy("image"),
                    "low": MappingSource.copy("low"),
                    "width": MappingSource.copy("width"),
                },
                outputs={"image": "image"},
            ),
        ),
    )
    return ComfyGroupRegistry(
        source_schemas=(
            ComfyGroupSourceSchema(PREPROCESSOR, SCHEMA_WIRE_VERSION),
            ComfyGroupSourceSchema(RESIZE, SCHEMA_WIRE_VERSION),
        ),
        group_schemas=(ComfyGroupSourceSchema(GROUP, SCHEMA_WIRE_VERSION),),
        records=(
            ComfyGroupRecord(
                id="comfy_group:comfyui-controlnet-aux/canny-resize",
                mapping_kind="op",
                carrier=carrier,
                source=source,
                pattern=pattern,
                replacement=replacement,
                confidence=ComfyAliasConfidence(
                    tier="grouped",
                    evidence=(
                        "tests/test_comfy_group_registry.py::"
                        "test_group_registry_roundtrip_is_canonical_and_import_only",
                    ),
                ),
            ),
        ),
    )


def test_group_registry_roundtrip_is_canonical_and_import_only() -> None:
    declared = registry()
    wire = comfy_group_registry_to_wire(declared)

    assert wire["format"] == "dinkster-comfy-group/1"
    assert wire["records"][0]["mappingKind"] == "op"  # type: ignore[index]
    assert wire["records"][0]["pattern"]["anchor"] == "preprocess"  # type: ignore[index]
    assert wire["records"][0]["pattern"]["disconnected"] == [  # type: ignore[index]
        "preprocess:mask"
    ]
    assert "replacements" not in wire["groupSchemas"][0]  # type: ignore[index]
    assert comfy_group_registry_from_wire(wire) == declared
    assert comfy_group_registry_to_wire(comfy_group_registry_from_wire(wire)) == wire


def test_group_registry_reads_v1_patterns_without_disconnected_inputs() -> None:
    wire = comfy_group_registry_to_wire(registry())
    legacy = copy.deepcopy(wire)
    del legacy["records"][0]["pattern"]["disconnected"]  # type: ignore[index]

    parsed = comfy_group_registry_from_wire(legacy)

    assert parsed.records[0].pattern.disconnected == ()
    upgraded = comfy_group_registry_to_wire(parsed)
    assert upgraded["records"][0]["pattern"]["disconnected"] == []  # type: ignore[index]


def test_group_registry_validates_patterns_and_replacement_references() -> None:
    declared = registry()
    assert comfy_group_registry_problems(declared, {TARGET.node_type: TARGET}) == ()

    missing = registry(carrier="dinkster.preprocess.missing")
    problems = comfy_group_registry_problems(missing, {TARGET.node_type: TARGET})
    assert any("unknown carrier" in problem for problem in problems)

    record = declared.records[0]
    bad_pattern = dataclasses.replace(
        record.pattern,
        constants=(),
    )
    problems = comfy_group_registry_problems(
        dataclasses.replace(declared, records=(dataclasses.replace(record, pattern=bad_pattern),)),
        {TARGET.node_type: TARGET},
    )
    assert any("every source input" in problem for problem in problems)


def test_group_pattern_accepts_null_for_an_unconnected_optional_socket() -> None:
    declared = registry()
    source = declared.source_schemas[0]
    optional_source = dataclasses.replace(
        source,
        schema=dataclasses.replace(
            source.schema,
            inputs=source.schema.inputs + (InputSpec("optional_socket", IMAGE, required=False),),
        ),
    )
    record = declared.records[0]
    pattern = dataclasses.replace(
        record.pattern,
        constants=record.pattern.constants + (("preprocess:optional_socket", None),),
    )
    declared = dataclasses.replace(
        declared,
        source_schemas=(optional_source, declared.source_schemas[1]),
        records=(dataclasses.replace(record, pattern=pattern),),
    )

    assert comfy_group_registry_problems(declared, {TARGET.node_type: TARGET}) == ()


def test_group_registry_rejects_noncanonical_and_inconsistent_data() -> None:
    wire = comfy_group_registry_to_wire(registry())

    unknown = copy.deepcopy(wire)
    unknown["records"][0]["pattern"]["unexpected"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown fields: unexpected"):
        comfy_group_registry_from_wire(unknown)

    wrong_tier = copy.deepcopy(wire)
    wrong_tier["records"][0]["confidence"]["tier"] = "exact"  # type: ignore[index]
    with pytest.raises(ValueError, match="tier must be 'grouped'"):
        comfy_group_registry_from_wire(wrong_tier)

    missing_schema = copy.deepcopy(wire)
    missing_schema["sourceSchemas"] = []
    with pytest.raises(ValueError, match="lack source schemas"):
        comfy_group_registry_from_wire(missing_schema)

    noncanonical_rule = copy.deepcopy(wire)
    noncanonical_rule["records"][0]["replacement"]["extra"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="not canonical ReplacementRule"):
        comfy_group_registry_from_wire(noncanonical_rule)


def test_group_registry_requires_uniform_modes_and_bounded_patterns() -> None:
    declared = registry()
    pattern = declared.records[0].pattern
    nodes = list(pattern.nodes)
    nodes[1] = (nodes[1][0], dataclasses.replace(nodes[1][1], mode="muted"))
    with pytest.raises(ValueError, match="uniform mode"):
        dataclasses.replace(pattern, nodes=tuple(nodes))

    with pytest.raises(ValueError, match="2 to 16"):
        dataclasses.replace(pattern, nodes=pattern.nodes[:1])

    record = declared.records[0]
    disconnected = dataclasses.replace(pattern, edges=())
    problems = comfy_group_registry_problems(
        dataclasses.replace(declared, records=(dataclasses.replace(record, pattern=disconnected),)),
        {TARGET.node_type: TARGET},
    )
    assert any("must be connected" in problem for problem in problems)

    wrong_type = dataclasses.replace(
        pattern,
        edges=(ComfyGroupEdge("preprocess:image", "resize:width"),),
    )
    problems = comfy_group_registry_problems(
        dataclasses.replace(declared, records=(dataclasses.replace(record, pattern=wrong_type),)),
        {TARGET.node_type: TARGET},
    )
    assert any("mismatched types" in problem for problem in problems)


def test_group_registry_rejects_cross_role_input_mappings_and_unknown_core_revision() -> None:
    declared = registry()
    record = declared.records[0]
    duplicate_group_input = dataclasses.replace(
        record.pattern,
        inputs=(("image", "preprocess:image"), ("low", "preprocess:low")),
        parameters=(("low", "resize:width"),),
    )
    problems = comfy_group_registry_problems(
        dataclasses.replace(
            declared,
            records=(dataclasses.replace(record, pattern=duplicate_group_input),),
        ),
        {TARGET.node_type: TARGET},
    )
    assert any(
        "group schema input 'low' has multiple pattern mappings" in problem for problem in problems
    )

    required_disconnected = dataclasses.replace(
        record.pattern,
        inputs=(),
        disconnected=("preprocess:image", "preprocess:mask"),
    )
    problems = comfy_group_registry_problems(
        dataclasses.replace(
            declared,
            records=(dataclasses.replace(record, pattern=required_disconnected),),
        ),
        {TARGET.node_type: TARGET},
    )
    assert any(
        "disconnected source input 'preprocess:image' must be optional" in problem
        for problem in problems
    )

    widget_disconnected = dataclasses.replace(
        record.pattern,
        parameters=(("width", "resize:width"),),
        disconnected=("preprocess:low", "preprocess:mask"),
    )
    optional_widget_source = dataclasses.replace(
        declared.source_schemas[0],
        schema=dataclasses.replace(
            PREPROCESSOR,
            inputs=(
                PREPROCESSOR.inputs[0],
                dataclasses.replace(PREPROCESSOR.inputs[1], required=False),
                PREPROCESSOR.inputs[2],
            ),
        ),
    )
    problems = comfy_group_registry_problems(
        dataclasses.replace(
            declared,
            source_schemas=(optional_widget_source, declared.source_schemas[1]),
            records=(dataclasses.replace(record, pattern=widget_disconnected),),
        ),
        {TARGET.node_type: TARGET},
    )
    assert any(
        "disconnected source input 'preprocess:low' must be a socket, not a widget" in problem
        for problem in problems
    )

    with pytest.raises(ValueError, match="duplicate disconnected inputs"):
        dataclasses.replace(
            record.pattern,
            disconnected=("preprocess:mask", "preprocess:mask"),
        )

    with pytest.raises(ValueError, match="comfy-core groups must use a maintained revision"):
        ComfyGroupSource("comfy-core", "canny-resize", "deadbeef")


def test_family_group_record_and_cross_pack_collisions() -> None:
    declared = registry()
    record = declared.records[0]
    family = dataclasses.replace(
        record,
        mapping_kind="family",
        family=ComfyAliasFamily("dinkster.wan21", provider="dinkster-model-wan"),
    )
    family_registry = dataclasses.replace(declared, records=(family,))
    assert (
        comfy_group_registry_from_wire(comfy_group_registry_to_wire(family_registry))
        == family_registry
    )

    problems = comfy_group_collision_problems({"native-a": declared, "native-b": declared})
    assert len(problems) == 3
    assert any("record id" in problem for problem in problems)
    assert any("source key" in problem for problem in problems)
    assert any("groupType" in problem for problem in problems)
