from __future__ import annotations

import base64
import hashlib
import json
import zlib
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any, cast

from dinkster_nodes_image import IMAGE_NODES
from dinkster_schema import (
    ComboWidget,
    DynamicComboSpec,
    DynamicEntry,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    MappingSource,
    NodeSchema,
    ReplacementPredicate,
    ReplacementRule,
    schema_from_wire,
    schema_to_wire,
    validate_replacement_references,
)

# Generated from commit 681f529b5eb80554d28ef836016bf78a7c4654f3 at schema wire 26.
FIXTURE = Path(__file__).parent / "fixtures" / "image_native_v1_wire26.json.zlib.b85"
FIXTURE_SHA256 = "5cec48d215446308d1ae6c857e1cd87edff9de7fbd218a2dad403fe5bfcda52f"

CASE_COUNTS = {
    "dinkster.image.resize": 101,
    "dinkster.image.transform": 7,
    "dinkster.image.crop": 9,
    "dinkster.image.composite": 9,
    "dinkster.image.adjust": 4,
    "dinkster.image.filter": 11,
    "dinkster.mask.make": 13,
    "dinkster.mask.text": 2,
    "dinkster.mask.morphology": 34,
    "dinkster.image.to_mask": 7,
    "dinkster.mask.to_image": 3,
    "dinkster.image.generate": 8,
    "dinkster.image.draw_region": 2,
    "dinkster.image.batch.edit": 6,
    "dinkster.mask.batch.edit": 6,
    "dinkster.image.batch.combine": 8,
    "dinkster.mask.batch.combine": 4,
    "dinkster.image.list.to_batch": 2,
    "dinkster.image.transition": 6,
}

MOVED_INPUTS = {
    "dinkster.image.resize": {
        "factor",
        "final_anchor",
        "final_pad_color",
        "final_pad_value",
        "final_padding",
        "height",
        "megapixels",
        "mode_anchor",
        "mode_padding",
        "multiple_of",
        "pad_color",
        "pad_value",
        "reference",
        "resolution_steps",
        "size",
        "width",
    },
    "dinkster.image.transform": {
        "angle",
        "bottom",
        "expand",
        "feathering",
        "fill",
        "interpolation",
        "left",
        "right",
        "steps",
        "top",
        "units",
        "x",
        "y",
    },
    "dinkster.image.crop": {
        "fill",
        "height",
        "mask",
        "mask_blur",
        "mask_threshold",
        "placement",
        "region",
        "width",
        "x",
        "y",
    },
    "dinkster.image.composite": {"interpolation", "mask", "mask_polarity"},
    "dinkster.image.adjust": {"factor", "mean", "standard_deviation"},
    "dinkster.image.filter": {
        "colors",
        "dither",
        "kernel_size",
        "radius",
        "seed",
        "sigma",
        "strength",
    },
    "dinkster.mask.make": {
        "angle",
        "background",
        "center_x",
        "center_y",
        "end_frame",
        "foreground",
        "grow",
        "noise_density",
        "noise_mode",
        "points_x",
        "points_y",
        "radius",
        "region",
        "seed",
        "shape_height",
        "shape_origin",
        "shape_width",
        "start_frame",
        "tile_size",
        "timing_function",
        "transition_type",
        "x",
        "y",
    },
    "dinkster.mask.text": {"foreground"},
    "dinkster.mask.morphology": {
        "block_mode",
        "blur_amount",
        "blur_radius",
        "bottom",
        "clamp",
        "decay_factor",
        "edge_policy",
        "edge_value",
        "fill_holes",
        "flip_input",
        "height",
        "incremental_expandrate",
        "input_high",
        "input_low",
        "iterations",
        "kernel_size",
        "left",
        "lerp_alpha",
        "output_high",
        "output_low",
        "radius",
        "right",
        "sigma",
        "tapered_corners",
        "threshold",
        "top",
        "width",
        "x",
        "y",
    },
    "dinkster.image.to_mask": {
        "blue",
        "channel",
        "color",
        "color_source",
        "color_value",
        "green",
        "metric",
        "red",
        "tolerance",
    },
    "dinkster.mask.to_image": {"mask_polarity", "color"},
    "dinkster.image.generate": {
        "angle",
        "center_x",
        "center_y",
        "color_a",
        "color_b",
        "color_value",
        "radius",
        "tile_size",
    },
    "dinkster.image.draw_region": {"line_width"},
    "dinkster.image.batch.edit": {
        "amount",
        "count",
        "expansion",
        "generate_repeat_marker",
        "range_end",
        "seed",
        "size",
        "start",
    },
    "dinkster.mask.batch.edit": {
        "amount",
        "count",
        "expansion",
        "range_end",
        "seed",
        "size",
        "start",
    },
    "dinkster.image.batch.combine": {"index", "indexes", "interpolation"},
    "dinkster.mask.batch.combine": {"index", "interpolation"},
    "dinkster.image.list.to_batch": {"interpolation"},
    "dinkster.image.transition": {"resize_interpolation", "start_index"},
}


def _fixture_bytes() -> bytes:
    encoded = b"".join(FIXTURE.read_bytes().split())
    return zlib.decompress(base64.b85decode(encoded))


@cache
def _v1_schemas() -> dict[str, NodeSchema]:
    wires = json.loads(_fixture_bytes())
    return {schema.node_type: schema for schema in map(schema_from_wire, wires)}


@cache
def _current_schemas() -> dict[str, NodeSchema]:
    return {
        schema.node_type: schema
        for node in IMAGE_NODES
        if (schema := node.schema()).node_type in CASE_COUNTS
    }


def _migration(schema: NodeSchema) -> ReplacementRule:
    rules = tuple(rule for rule in schema.replacements if rule.migration is not None)
    if schema.node_type in {"dinkster.image.composite", "dinkster.mask.to_image"}:
        rules = rules[-1:]
    if schema.node_type == "dinkster.image.resize":
        rules = tuple(
            rule
            for rule in rules
            if rule.migration is not None and "target" in rule.migration.historical_inputs
        )
    (rule,) = rules
    return rule


def _resize_migrations() -> tuple[ReplacementRule, ReplacementRule]:
    rules = tuple(
        rule
        for rule in _current_schemas()["dinkster.image.resize"].replacements
        if rule.migration is not None
    )
    assert len(rules) == 2
    dotted = next(
        rule
        for rule in rules
        if rule.migration is not None and "compatibility.target" in rule.migration.historical_inputs
    )
    flat = next(
        rule
        for rule in rules
        if rule.migration is not None and "target" in rule.migration.historical_inputs
    )
    return dotted, flat


def _dynamic_ids(entries: Sequence[DynamicEntry]) -> set[str]:
    found: set[str] = set()
    for entry in entries:
        found.add(entry.id)
        if isinstance(entry, InputFamilySpec):
            found.update(_dynamic_ids(entry.template))
        elif isinstance(entry, DynamicComboSpec):
            for option in entry.options:
                found.update(_dynamic_ids(option.inputs))
        elif isinstance(entry, DynamicSlotSpec):
            found.update(_dynamic_ids(entry.inputs))
            for variant in entry.variants or ():
                found.update(_dynamic_ids(variant.inputs))
    return found


def _nested_entries(entries: Sequence[DynamicEntry]) -> list[DynamicEntry]:
    found: list[DynamicEntry] = []
    for entry in entries:
        found.append(entry)
        if isinstance(entry, InputFamilySpec):
            found.extend(_nested_entries(entry.template))
        elif isinstance(entry, DynamicComboSpec):
            for option in entry.options:
                found.extend(_nested_entries(option.inputs))
        elif isinstance(entry, DynamicSlotSpec):
            found.extend(_nested_entries(entry.inputs))
            for variant in entry.variants or ():
                found.extend(_nested_entries(variant.inputs))
    return found


def _default_choices(entries: Sequence[DynamicEntry], parent: str = "") -> dict[str, str]:
    choices: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, DynamicComboSpec):
            continue
        path = f"{parent}.{entry.id}" if parent else entry.id
        assert entry.default is not None
        choices[path] = entry.default
        option = entry.option(entry.default)
        assert option is not None
        choices.update(_default_choices(option.inputs, path))
    return choices


def _matches(predicate: ReplacementPredicate, kind: str, input_id: str) -> bool:
    return (predicate.kind == kind and predicate.input == input_id) or any(
        _matches(child, kind, input_id) for child in predicate.of
    )


def _matches_value(predicate: ReplacementPredicate, input_id: str, value: object) -> bool:
    return (
        predicate.kind == "valueEquals" and predicate.input == input_id and predicate.value == value
    ) or any(_matches_value(child, input_id, value) for child in predicate.of)


def _rejects_connected(predicate: ReplacementPredicate, input_id: str) -> bool:
    return any(
        child.kind == "inputConnected" and child.input == input_id
        for negation in _predicates(predicate, "not")
        for child in negation.of
    )


def _predicates(predicate: ReplacementPredicate, kind: str) -> tuple[ReplacementPredicate, ...]:
    return (
        *((predicate,) if predicate.kind == kind else ()),
        *(match for child in predicate.of for match in _predicates(child, kind)),
    )


def test_frozen_v1_fixture_is_exact_and_test_only() -> None:
    raw = _fixture_bytes()
    assert len(raw) == 100_314
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256

    old = _v1_schemas()
    current = _current_schemas()
    assert old.keys() == current.keys() == CASE_COUNTS.keys()
    assert all(schema.version == 1 for schema in old.values())
    assert current["dinkster.image.resize"].version == 4
    assert all(
        schema.version
        == (3 if node_type in {"dinkster.image.composite", "dinkster.mask.to_image"} else 2)
        for node_type, schema in current.items()
        if node_type != "dinkster.image.resize"
    )

    for schema in current.values():
        wire = schema_to_wire(schema)
        assert wire["version"] == schema.version
        assert json.dumps(wire).count('"nodeType"') == 1
        assert "sourceSchemas" not in wire
        assert "nativeMigrations" not in wire
        replacements = cast("list[dict[str, Any]]", wire["replacements"])
        migration = next(
            replacement["migration"] for replacement in replacements if "migration" in replacement
        )
        assert isinstance(migration, dict)
        assert set(migration) == {"historicalInputs"}


def test_historical_inputs_are_exactly_the_retired_static_interface() -> None:
    historical_count = 0
    for node_type, old in _v1_schemas().items():
        current = _current_schemas()[node_type]
        current_inputs = {spec.id for spec in current.inputs}
        retired = {spec.id for spec in old.inputs if spec.id not in current_inputs}
        migration = _migration(current)
        assert migration.from_type == node_type
        assert migration.migration is not None
        assert set(migration.migration.historical_inputs) == retired
        historical_count += len(retired)
    assert historical_count == 169


def test_moved_static_input_inventory_is_the_audited_148() -> None:
    actual = {
        node_type: _dynamic_ids((*schema.combos, *schema.slots))
        - {combo.id for combo in schema.combos}
        for node_type, schema in _current_schemas().items()
    }
    assert {"minimum_area", "connectivity"} <= actual["dinkster.mask.morphology"]
    actual["dinkster.mask.morphology"] -= {"minimum_area", "connectivity"}
    assert actual == MOVED_INPUTS
    assert sum(map(len, actual.values())) == 148
    assert {"foreground", "background"} <= actual["dinkster.mask.make"]
    assert actual["dinkster.mask.text"] == {"foreground"}
    assert "color" not in actual["dinkster.mask.text"]
    assert "color" in {spec.id for spec in _current_schemas()["dinkster.mask.text"].inputs}


def test_dynamic_inputs_preserve_v1_types_defaults_and_options() -> None:
    for node_type, current in _current_schemas().items():
        if node_type == "dinkster.image.resize":
            continue
        old_inputs = {spec.id: spec for spec in _v1_schemas()[node_type].inputs}
        entries = _nested_entries((*current.combos, *current.slots))
        for entry in entries:
            if isinstance(entry, InputSpec) and entry.id == "mask_polarity":
                assert isinstance(entry.widget, ComboWidget)
                assert entry.widget.options == ("coverage", "transparency")
                assert entry.default == "coverage"
                continue
            old = old_inputs.get(entry.id)
            if old is None:
                assert (node_type, entry.id) in {
                    ("dinkster.image.crop", "source"),
                    ("dinkster.mask.morphology", "minimum_area"),
                    ("dinkster.mask.morphology", "connectivity"),
                }
                continue
            if isinstance(entry, InputSpec):
                assert (entry.type, entry.default, entry.widget) == (
                    old.type,
                    old.default,
                    old.widget,
                )
            elif isinstance(entry, DynamicComboSpec):
                assert entry.default == old.default
                assert isinstance(old.widget, ComboWidget)
                current_only = {option.key for option in entry.options} - set(old.widget.options)
                assert current_only <= {
                    "fill_holes",
                    "pad_edge",
                    "pad_edge_pixel",
                    "pillarbox_blur",
                    "pingpong",
                    "remove_small_components",
                    "total_pixels",
                }
            elif isinstance(entry, DynamicSlotSpec):
                assert entry.variants is not None
                assert {variant.type for variant in entry.variants} == {old.type}


def test_migration_case_inventory_and_identity_mappings() -> None:
    total_cases = 0
    for node_type, expected_count in CASE_COUNTS.items():
        schema = _current_schemas()[node_type]
        rule = _migration(schema)
        assert len(rule.cases) == expected_count
        assert sum(case.unconditional for case in rule.cases) == 1
        assert rule.cases[-1].unconditional
        assert dict(rule.cases[-1].slot_variants) == _default_choices(
            (*schema.combos, *schema.slots)
        )
        expected_outputs = {output.id: output.id for output in schema.outputs}
        for case in rule.cases:
            assert case.to == node_type
            assert dict(case.outputs) == expected_outputs
            assert {
                target: (mapping.kind, mapping.source_family)
                for target, mapping in case.output_families
            } == {family.id: ("copy", family.id) for family in schema.output_families}
        total_cases += len(rule.cases)
    assert total_cases == 242
    assert validate_replacement_references(_current_schemas()) == ()

    resize_rules = _resize_migrations()
    assert all(
        not path.startswith("compatibility.")
        for rule in resize_rules
        for case in rule.cases
        for path, _choice in case.slot_variants
    )
    assert {
        "pad_edge",
        "pad_edge_pixel",
        "pillarbox_blur",
    }.isdisjoint(
        {
            choice
            for rule in resize_rules
            for case in rule.cases
            for path, choice in case.slot_variants
            if path == "mode"
        }
    )
    assert {"constant", "edge_average", "edge_pixel", "blurred_background"} <= {
        choice
        for rule in resize_rules
        for case in rule.cases
        for path, choice in case.slot_variants
        if path in {"mode.mode_padding", "divisibility.final_padding"}
    }


def test_resize_v3_and_v1_migrations_map_legacy_mechanics_to_intents() -> None:
    dotted, flat = _resize_migrations()
    assert len(dotted.cases) == 129
    assert len(flat.cases) == 101
    assert dotted.note == flat.note
    assert dotted.note is not None and "center-fit uses centered fill" in dotted.note

    v3_pad_edge = next(
        case
        for case in dotted.cases
        if case.when is not None
        and _matches_value(case.when, "compatibility", "kjnodes_v2")
        and _matches_value(case.when, "compatibility.mode", "pad_edge")
        and _matches_value(case.when, "compatibility.anchor", "disabled")
    )
    assert dict(v3_pad_edge.slot_variants) == {
        "target": "dimensions",
        "mode": "pad",
        "divisibility": "crop",
        "mode.mode_padding": "edge_average",
    }
    assert dict(v3_pad_edge.inputs)["mode.mode_anchor"] == MappingSource.constant("center")

    center_fit_cases = [
        case
        for case in dotted.cases
        if case.when is not None
        and _matches_value(case.when, "compatibility", "kjnodes_v1")
        and _matches_value(case.when, "compatibility.mode", "fit")
        and _matches_value(case.when, "compatibility.anchor", "center")
    ]
    assert len(center_fit_cases) == 2
    assert all(dict(case.slot_variants)["mode"] == "fill" for case in center_fit_cases)

    pixel_cases = [
        case
        for case in dotted.cases
        if case.when is not None and _matches_value(case.when, "compatibility.mode", "total_pixels")
    ]
    assert len(pixel_cases) == 4
    assert all("pixels:values" in dict(case.input_families) for case in pixel_cases)
    assert all(dict(case.slot_variants)["target"] == "total_pixels" for case in pixel_cases)

    flat_source_modes = {
        predicate.value
        for case in flat.cases
        if case.when is not None
        for predicate in _predicates(case.when, "valueEquals")
        if predicate.input == "mode"
    }
    assert flat_source_modes == {"stretch", "fit", "fill", "pad"}
    assert all("pixels:values" not in dict(case.input_families) for case in flat.cases)


def test_known_dynamic_selector_cases_fail_closed_on_connections() -> None:
    for node_type, schema in _current_schemas().items():
        if node_type == "dinkster.image.resize":
            continue
        for case in _migration(schema).cases[:-1]:
            assert case.when is not None
            for path, _choice in case.slot_variants:
                input_id = path.rsplit(".", 1)[-1]
                if (node_type, input_id) in {
                    ("dinkster.image.crop", "source"),
                    ("dinkster.image.composite", "mask"),
                }:
                    continue
                assert _rejects_connected(case.when, input_id), (node_type, path)


def test_image_to_mask_has_seven_cases_and_safe_default_fallback() -> None:
    rule = _migration(_current_schemas()["dinkster.image.to_mask"])
    assert len(rule.cases) == 7
    for case in rule.cases[:-1]:
        assert case.when is not None
        assert _rejects_connected(case.when, "policy")
        assert _rejects_connected(case.when, "color_source")

    fallback = rule.cases[-1]
    assert dict(fallback.slot_variants) == {"policy": "channel"}
    assert not {"policy", "color_source"} & {
        mapping.input for _target, mapping in fallback.inputs if mapping.kind != "constant"
    }


def test_crop_prefers_region_then_mask_then_coordinates() -> None:
    cases = _migration(_current_schemas()["dinkster.image.crop"]).cases
    sources = [dict(case.slot_variants)["source"] for case in cases]
    assert sources == [
        "region",
        "region",
        "region",
        "mask",
        "mask",
        "mask",
        "coordinates",
        "coordinates",
        "coordinates",
    ]
    assert cases[0].when is not None and _matches(cases[0].when, "inputConnected", "region")
    assert cases[3].when is not None and _matches(cases[3].when, "inputConnected", "mask")


def test_composite_optional_mask_is_consumed_once_when_selected() -> None:
    cases = _migration(_current_schemas()["dinkster.image.composite"]).cases
    selected = [case for case in cases if dict(case.slot_variants).get("mask") == "mask"]
    assert len(selected) == 6
    for case in selected:
        consumers = [
            target
            for target, mapping in case.inputs
            if mapping.input == "mask" and mapping.kind in ("copy", "link")
        ]
        assert consumers == ["mask"]
    for case in cases:
        assert not any(link.to == "mask" for link in case.links)
