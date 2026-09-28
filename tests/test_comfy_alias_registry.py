from __future__ import annotations

import copy
import dataclasses
from typing import Any, cast

import pytest
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    ComfyAliasConfidence,
    ComfyAliasFamily,
    ComfyAliasRecord,
    ComfyAliasRegistry,
    ComfyAliasSource,
    ComfyAliasSourceSchema,
    ComfyAliasTolerance,
    InputSpec,
    MappingSource,
    NodeSchema,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    comfy_alias_collision_problems,
    comfy_alias_registry_from_wire,
    comfy_alias_registry_problems,
    comfy_alias_registry_to_wire,
)

IMAGE = TypeExpr.concrete("core.image")

SOURCE_SCHEMA = NodeSchema(
    node_type="comfy.ImageScale",
    inputs=(InputSpec("image", IMAGE), InputSpec("mode", TypeExpr.concrete("core.string"))),
    outputs=(OutputSpec("image", IMAGE),),
)
TARGET_SCHEMA = NodeSchema(
    node_type="dinkster.image.resize",
    inputs=(InputSpec("image", IMAGE), InputSpec("mode", TypeExpr.concrete("core.string"))),
    outputs=(OutputSpec("image", IMAGE),),
)


def registry(*, carrier: str = "dinkster.image.resize") -> ComfyAliasRegistry:
    source = ComfyAliasSource(
        pack="comfy-core",
        node_class="ImageScale",
        node_type="comfy.ImageScale",
        revision="b78cec87",
    )
    rule = ReplacementRule(
        from_type=source.node_type,
        cases=(
            ReplacementCase.build(
                carrier,
                inputs={
                    "image": MappingSource.copy("image"),
                    "mode": MappingSource.copy("mode"),
                },
                outputs={"image": "image"},
            ),
        ),
    )
    return ComfyAliasRegistry(
        source_schemas=(ComfyAliasSourceSchema(SOURCE_SCHEMA, SCHEMA_WIRE_VERSION),),
        records=(
            ComfyAliasRecord(
                id="comfy_alias:comfy-core/ImageScale",
                mapping_kind="op",
                carrier=carrier,
                source=source,
                replacement=rule,
                confidence=ComfyAliasConfidence(
                    tier="exact",
                    evidence=("tests/workflows/image-scale.json",),
                ),
            ),
        ),
    )


def test_registry_wire_roundtrip_is_canonical_and_separate_from_native_schema() -> None:
    declared = registry()
    wire = comfy_alias_registry_to_wire(declared)

    assert wire["format"] == "dinkster-comfy-alias/1"
    assert wire["records"][0]["mappingKind"] == "op"  # type: ignore[index]
    assert wire["records"][0]["carrier"] == "dinkster.image.resize"  # type: ignore[index]
    assert "replacements" not in wire["sourceSchemas"][0]  # type: ignore[index]
    parsed = comfy_alias_registry_from_wire(wire)
    assert parsed == declared
    assert comfy_alias_registry_to_wire(parsed) == wire
    assert TARGET_SCHEMA.replacements == ()


def test_family_records_and_confidence_tolerances_roundtrip() -> None:
    declared = registry()
    record = dataclasses.replace(
        declared.records[0],
        mapping_kind="family",
        family=ComfyAliasFamily("dinkster.wan21", provider="dinkster-model-wan"),
        confidence=ComfyAliasConfidence(
            tier="equivalent",
            evidence=("tests/workflows/wan.json",),
            tolerances=(ComfyAliasTolerance("max_abs", "<=", 1e-5),),
        ),
    )
    declared = dataclasses.replace(declared, records=(record,))
    wire = comfy_alias_registry_to_wire(declared)

    assert wire["records"][0]["family"] == {  # type: ignore[index]
        "id": "dinkster.wan21",
        "provider": "dinkster-model-wan",
    }
    assert comfy_alias_registry_from_wire(wire) == declared


def test_registry_rejects_unknown_noncanonical_and_inconsistent_data() -> None:
    wire = comfy_alias_registry_to_wire(registry())

    unknown = copy.deepcopy(wire)
    unknown["records"][0]["surprise"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown fields: surprise"):
        comfy_alias_registry_from_wire(unknown)

    noncanonical_rule = copy.deepcopy(wire)
    noncanonical_rule["records"][0]["replacement"]["extra"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="not canonical ReplacementRule"):
        comfy_alias_registry_from_wire(noncanonical_rule)

    exact_tolerance = copy.deepcopy(wire)
    exact_tolerance["records"][0]["confidence"]["tolerances"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="forbidden for exact"):
        comfy_alias_registry_from_wire(exact_tolerance)

    empty_parametric_tolerances = copy.deepcopy(wire)
    empty_parametric_tolerances["records"][0]["confidence"] = {  # type: ignore[index]
        "tier": "parametric",
        "evidence": ["test"],
        "tolerances": [],
    }
    with pytest.raises(ValueError, match="not canonical wire data"):
        comfy_alias_registry_from_wire(empty_parametric_tolerances)

    oversized_scale = copy.deepcopy(wire)
    oversized_scale["records"][0]["replacement"]["cases"][0]["inputs"]["mode"] = {  # type: ignore[index]
        "kind": "value",
        "input": "mode",
        "transform": {"kind": "scale", "factor": 10**400},
    }
    with pytest.raises(ValueError, match="finite numbers"):
        comfy_alias_registry_from_wire(oversized_scale)

    missing_snapshot = copy.deepcopy(wire)
    missing_snapshot["sourceSchemas"] = []
    with pytest.raises(ValueError, match="lack source schemas"):
        comfy_alias_registry_from_wire(missing_snapshot)

    family_without_metadata = copy.deepcopy(wire)
    family_without_metadata["records"][0]["mappingKind"] = "family"  # type: ignore[index]
    with pytest.raises(ValueError, match="family mappings require"):
        comfy_alias_registry_from_wire(family_without_metadata)


def test_registry_requires_exact_record_identity_and_confidence_contract() -> None:
    declared = registry()
    record = declared.records[0]
    with pytest.raises(ValueError, match="id must be"):
        dataclasses.replace(record, id="comfy_alias:wrong/ImageScale")
    with pytest.raises(ValueError, match="replacement.from"):
        dataclasses.replace(
            record,
            replacement=ReplacementRule(
                from_type="comfy.Other",
                cases=(ReplacementCase.build("dinkster.image.resize"),),
            ),
        )
    with pytest.raises(ValueError, match="requires tolerances"):
        ComfyAliasConfidence(tier="equivalent", evidence=("test",))
    with pytest.raises(ValueError, match="forbids tolerances"):
        ComfyAliasConfidence(
            tier="exact",
            evidence=("test",),
            tolerances=(ComfyAliasTolerance("max_abs", "<=", 0.0),),
        )
    with pytest.raises(ValueError, match="finite number"):
        ComfyAliasTolerance("max_abs", "<=", float("nan"))
    assert dataclasses.replace(record.source, revision="8a33128f").revision == "8a33128f"
    with pytest.raises(ValueError, match="must use one of revisions"):
        dataclasses.replace(record.source, revision="main")


def test_source_schema_rejects_an_unsupported_wire_version() -> None:
    with pytest.raises(ValueError, match="unsupported schemaVersion: 15"):
        ComfyAliasSourceSchema(SOURCE_SCHEMA, 15)


def test_source_node_class_keeps_the_exact_legacy_name() -> None:
    declared = registry()
    source = dataclasses.replace(declared.records[0].source, node_class="Image/Scale")
    record = dataclasses.replace(
        declared.records[0],
        id="comfy_alias:comfy-core/Image/Scale",
        source=source,
    )
    updated = dataclasses.replace(declared, records=(record,))
    assert comfy_alias_registry_from_wire(comfy_alias_registry_to_wire(updated)) == updated


def test_registry_validates_carrier_and_replacement_references() -> None:
    assert comfy_alias_registry_problems(registry(), {TARGET_SCHEMA.node_type: TARGET_SCHEMA}) == ()

    missing = registry(carrier="dinkster.image.missing")
    problems = comfy_alias_registry_problems(missing, {TARGET_SCHEMA.node_type: TARGET_SCHEMA})
    assert "unknown carrier" in problems[0]

    declared = registry()
    bad_rule = ReplacementRule(
        from_type="comfy.ImageScale",
        cases=(
            ReplacementCase.build(
                "dinkster.image.resize",
                inputs={"missing_target": MappingSource.copy("missing_source")},
            ),
        ),
    )
    bad_record = dataclasses.replace(declared.records[0], replacement=bad_rule)
    problems = comfy_alias_registry_problems(
        dataclasses.replace(declared, records=(bad_record,)),
        {TARGET_SCHEMA.node_type: TARGET_SCHEMA},
    )
    assert any("missing_source" in problem for problem in problems)
    assert any("missing_target" in problem for problem in problems)


def test_registry_collisions_are_detected_across_packs() -> None:
    problems = comfy_alias_collision_problems({"native-a": registry(), "native-b": registry()})
    assert len(problems) == 3
    assert any("record id" in problem for problem in problems)
    assert any("source key" in problem for problem in problems)
    assert any("source nodeType" in problem for problem in problems)


def test_registry_rejects_unversioned_source_media_extension() -> None:
    wire = cast("dict[str, Any]", comfy_alias_registry_to_wire(registry()))
    wire["records"][0]["sourceMedia"] = [
        {"port": "image", "direction": "input", "alphaPolicy": "preserve"}
    ]
    with pytest.raises(ValueError, match="unknown fields: sourceMedia"):
        comfy_alias_registry_from_wire(wire)
