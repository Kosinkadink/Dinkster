"""Scalar conversion and curve standard library coverage."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_foundation import (
    CURVE_TYPE,
    FOUNDATION_NODES,
    Curve,
    CurveEditor,
    CurveEvaluate,
    CurveFromValues,
    CurveSample,
    ValueConvert,
    register_foundation_types,
)
from dinkster_nodes_foundation.conversion import parse_schedule
from dinkster_schema import (
    ABSENT,
    build_node_types,
    build_schemas,
    comfy_alias_registry_from_wire,
    comfy_alias_registry_problems,
    comfy_alias_registry_to_wire,
    schema_from_wire,
    use_reporter,
    validate_replacement_references,
)
from dinkster_schema.replace import rule_from_wire, rule_to_wire
from dinkster_values import RenditionUnavailable, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from tools.generate_foundation_comfy_aliases import CONVERSION_EVIDENCE, build_aliases

ALIAS_PATH = (
    Path(__file__).parents[1] / "packages" / "dinkster-nodes-foundation" / "comfy-aliases.json"
)


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_foundation_types(registry)
    return Engine(
        schemas=build_schemas(FOUNDATION_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(FOUNDATION_NODES), registry),
        cache=MemoryLRUCache(),
    )


@pytest.mark.parametrize(
    ("value", "target", "expected"),
    [
        (True, "int", 1),
        (-4, "float", -4.0),
        (" 2.5 ", "float", 2.5),
        ("9007199254740993.0", "int", 9007199254740993),
        (1.0, "boolean", True),
        ("off", "boolean", False),
        (3.5, "string", "3.5"),
        (" unchanged ", "string", " unchanged "),
    ],
)
def test_value_conversion_populates_only_selected_output(
    value: object, target: str, expected: object
) -> None:
    result = ValueConvert.execute(value=value, target=target, force_lossy=False)
    assert result[target] == expected
    assert all(
        result[other] is ABSENT for other in {"int", "float", "string", "boolean"} - {target}
    )


@pytest.mark.parametrize(
    ("value", "target"),
    [
        (1.5, "int"),
        (9007199254740993, "float"),
        (2, "boolean"),
        (0.5, "boolean"),
    ],
)
def test_lossy_conversions_require_force(value: object, target: str) -> None:
    with pytest.raises(ValueError, match="would be lossy"):
        ValueConvert.execute(value=value, target=target, force_lossy=False)


def test_forced_conversion_matches_numeric_casting() -> None:
    assert ValueConvert.execute(value=-1.9, target="int", force_lossy=True)["int"] == -1
    assert ValueConvert.execute(value=2, target="boolean", force_lossy=True)["boolean"] is True
    assert ValueConvert.execute(value=9007199254740993, target="float", force_lossy=True)[
        "float"
    ] == float(9007199254740993)
    with pytest.raises(ValueError, match="integer magnitude"):
        ValueConvert.execute(value="1e999999999", target="int", force_lossy=False)


def test_numeric_pair_conversion_populates_int_and_float_outputs() -> None:
    result = ValueConvert.execute(value=" 4.9 ", target="number", force_lossy=True)
    assert result == {
        "int": 4,
        "float": 4.9,
        "string": ABSENT,
        "boolean": ABSENT,
    }


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "nan", "-inf"])
def test_conversion_rejects_non_finite_numbers(value: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        ValueConvert.execute(value=value, target="float", force_lossy=True)


def test_conversion_alias_registry_is_canonical_and_valid() -> None:
    encoded = ALIAS_PATH.read_text(encoding="utf-8")
    payload = build_aliases()
    assert encoded == json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    registry = comfy_alias_registry_from_wire(payload)
    assert comfy_alias_registry_to_wire(registry) == payload
    native_schemas = build_schemas(FOUNDATION_NODES)
    assert comfy_alias_registry_problems(registry, native_schemas) == ()
    source_schemas = {
        schema.node_type: schema
        for schema in (
            schema_from_wire(cast("dict[str, Any]", wire))
            for wire in cast("list[dict[str, object]]", payload["sourceSchemas"])
        )
    }
    records = [
        record
        for record in cast("list[dict[str, Any]]", payload["records"])
        if record["confidence"]["evidence"] == list(CONVERSION_EVIDENCE)
    ]
    assert {record["source"]["nodeClass"] for record in records} == {
        "ComfyNumberConvert",
        "Int To Bool (mtb)",
    }
    assert {record["carrier"] for record in records} == {"dinkster.value.convert"}
    for record in records:
        rule = rule_from_wire(record["replacement"])
        assert rule_to_wire(rule) == record["replacement"]
        carrier = native_schemas[record["carrier"]]
        schemas = {
            **native_schemas,
            **source_schemas,
            carrier.node_type: dataclasses.replace(carrier, replacements=(rule,)),
        }
        assert validate_replacement_references(schemas) == ()


def test_conversion_alias_operation_mappings() -> None:
    records = {
        record["id"]: record for record in cast("list[dict[str, Any]]", build_aliases()["records"])
    }
    core_rule = rule_from_wire(records["comfy_alias:comfy-core/ComfyNumberConvert"]["replacement"])
    core_case = core_rule.cases[0]
    assert core_case.to == "dinkster.value.convert"
    assert core_case.nodes is None
    core_inputs = dict(core_case.inputs)
    assert core_inputs["value"].input == "value"
    assert core_inputs["target"].value == "number"
    assert core_inputs["force_lossy"].value is True
    assert dict(core_case.outputs) == {"float": "FLOAT", "int": "INT"}

    for value, expected_float, expected_int in (
        (True, 1.0, 1),
        (-3, -3.0, -3),
        (2.75, 2.75, 2),
        (" 4.9 ", 4.9, 4),
        ("9007199254740993", float("9007199254740993"), 9007199254740993),
        (
            "9007199254740993.9",
            float("9007199254740993.9"),
            int(float("9007199254740993.9")),
        ),
    ):
        converted = ValueConvert.execute(value=value, target="number", force_lossy=True)
        assert converted["float"] == expected_float
        assert converted["int"] == expected_int

    mtb_rule = rule_from_wire(records["comfy_alias:comfy-mtb/Int To Bool (mtb)"]["replacement"])
    mtb_inputs = dict(mtb_rule.cases[0].inputs)
    assert mtb_inputs["value"].input == "int"
    assert mtb_inputs["target"].value == "boolean"
    assert mtb_inputs["force_lossy"].value is True
    assert ValueConvert.execute(value=-2, target="boolean", force_lossy=True)["boolean"] is True


def test_curve_is_frozen_validated_and_interpolates() -> None:
    curve = Curve(((-2, 10), (0, 0), (4, 8)))
    assert curve.points == ((-2.0, 10.0), (0.0, 0.0), (4.0, 8.0))
    assert curve.start == -2.0
    assert curve.end == 4.0
    assert curve.evaluate(-10) == 10.0
    assert curve.evaluate(-1) == 5.0
    assert curve.evaluate(2) == 4.0
    assert curve.evaluate(10) == 8.0
    assert Curve(((-1e308, -1e308), (1e308, 1e308))).evaluate(0.0) == 0.0
    high = 1e308
    high_end = math.nextafter(math.nextafter(high, math.inf), math.inf)
    high_middle = math.nextafter(high, math.inf)
    assert Curve(((high, 0), (high_end, 2))).evaluate(high_middle) == 1.0
    with pytest.raises(FrozenInstanceError):
        curve.points = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "points",
    [
        (),
        ((0, 1), (0, 2)),
        ((1, 1), (0, 2)),
        ((math.inf, 1),),
        ((0, math.nan),),
        ((True, 1),),
    ],
)
def test_curve_rejects_invalid_points(points: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Curve(points)  # type: ignore[arg-type]


def test_curve_codec_is_canonical_and_bounded() -> None:
    registry = TypeRegistry()
    register_foundation_types(registry)
    register_foundation_types(registry)
    spec = registry.spec(CURVE_TYPE)
    curve = Curve(((0, 1), (2.5, -3)))
    encoded = (
        b'{"interpolation":"linear","points":[{"position":0.0,"value":1.0},'
        b'{"position":2.5,"value":-3.0}]}'
    )
    assert spec.encode(curve) == encoded
    assert spec.decode(encoded) == curve
    assert spec.coerce is not None
    assert spec.coerce(json.loads(encoded)) == curve
    assert spec.decode(b'{"points":[{"position":0,"value":1}]}') == Curve(((0, 1),))
    assert spec.meta is not None
    assert spec.meta(curve) == {"pointCount": 2, "start": 0.0, "end": 2.5}
    with pytest.raises(ValueError, match="invalid dinkster.curve payload"):
        spec.decode(b"not json")
    with pytest.raises(ValueError, match="strictly increasing"):
        spec.decode(b'{"points":[{"position":1,"value":0},{"position":1,"value":2}]}')
    with pytest.raises(ValueError, match="between 1 and 4096"):
        Curve(tuple((index, index) for index in range(4097)))


def test_curve_points_rendition_declares_and_enforces_point_bound() -> None:
    registry = TypeRegistry()
    register_foundation_types(registry)
    rendition = registry.renditions_of(CURVE_TYPE)[0]
    assert rendition.kind == "curve-points"
    assert rendition.mime == "application/json"
    assert rendition.limits == {"points": 4096}

    curve = Curve(
        tuple((index / 8, (-1.0 if index == 2047 else index / 4096)) for index in range(4096))
    )
    record = json.loads(registry.render(registry.wrap(CURVE_TYPE, curve), rendition.kind).data)
    assert len(record["points"]) == 4096
    assert record["points"][2047] == {"position": 255.875, "value": -1.0}

    render = cast(Callable[[object], bytes], rendition.render)
    oversized = {"points": [{"position": index, "value": index} for index in range(4097)]}
    with pytest.raises(RenditionUnavailable, match="between 1 and 4096"):
        render(oversized)


def test_curve_interpolation_is_explicit_and_shape_preserving() -> None:
    with pytest.raises(ValueError, match="interpolation"):
        Curve(((0, 0),), "spline")  # type: ignore[arg-type]
    linear = Curve(((0, 0), (1, 2), (3, 3)))
    smooth = Curve(linear.points, "monotone_cubic")
    assert linear.evaluate(0.5) == 1.0
    assert smooth.evaluate(0.5) == pytest.approx(1.09375)
    samples = [smooth.evaluate(index / 20 * 3) for index in range(21)]
    assert samples == sorted(samples)
    assert min(samples) == 0.0 and max(samples) == 3.0
    assert Curve(((0, 1), (1, 1), (2, 3)), "monotone_cubic").evaluate(0.5) == 1.0
    assert Curve(((0, 1),), "monotone_cubic").evaluate(100) == 1.0
    assert Curve(((-1e308, 0), (1e308, 1)), "monotone_cubic").evaluate(0) == 0.5
    assert Curve(((0, -1e308), (1, 1e308)), "monotone_cubic").evaluate(0.5) == 0.0


@pytest.mark.parametrize(
    ("points", "position", "expected"),
    [
        (((0, 0), (1, 2), (3, 3)), 0.5, 1.09375),
        (((0, 0), (1, 2), (3, 3)), 2.0, 2.6875),
        (((0, 0), (1, 2), (2, 0)), 0.5, 1.25),
        (((0, 1), (1, 1), (2, 3)), 1.5, 1.75),
        (((-2, -4), (-1, -1), (2, 2)), 0.0, 0.4444444444444444),
    ],
)
def test_monotone_curve_matches_the_maintained_editor_reference(
    points: tuple[tuple[int, int], ...], position: float, expected: float
) -> None:
    assert Curve(points, "monotone_cubic").evaluate(position) == pytest.approx(expected)


def test_curve_editor_schema_execution_and_histogram_event() -> None:
    schema = CurveEditor.define_schema()
    curve_input, histogram_input = schema.inputs
    assert curve_input.default == Curve(((0, 0), (1, 1)), "monotone_cubic").to_record()
    assert histogram_input.required is False
    assert schema.emits_previews is True
    curve = Curve(((0, 2), (1, 4)), "monotone_cubic")
    events: list[tuple[str, object, bytes | None]] = []
    with use_reporter(lambda name, data, blob: events.append((name, data, blob))):
        assert CurveEditor.execute(curve=curve, histogram=[0] * 256)["curve"] is curve
    assert events == [("dinkster.curve.histogram", {"histogram": [0] * 256}, None)]
    for invalid in ([0] * 255, [0] * 255 + [-1], [0] * 255 + [True]):
        with pytest.raises(ValueError, match="exactly 256"):
            CurveEditor.execute(curve=curve, histogram=invalid)


def test_curve_nodes_build_evaluate_and_sample() -> None:
    curve = cast(
        "Curve",
        CurveFromValues.execute(values=[2.0, 4.0, 8.0], start=-1.0, step=2.0)["curve"],
    )
    assert curve == Curve(((-1, 2), (1, 4), (3, 8)))
    assert CurveEvaluate.execute(curve=curve, position=0.0) == {"value": 3.0}
    assert CurveSample.execute(curve=curve, count=5) == {"values": [2.0, 3.0, 4.0, 6.0, 8.0]}
    assert CurveSample.execute(curve=curve, count=1) == {"values": [2.0]}
    with pytest.raises(ValueError, match="greater than zero"):
        CurveFromValues.execute(values=[1.0], start=0.0, step=0.0)
    with pytest.raises(ValueError, match="between 1 and"):
        CurveSample.execute(curve=curve, count=0)


def test_numeric_schedule_parser_is_strict_and_orders_points() -> None:
    assert parse_schedule("10:(1.0), 0: (0.5), 20:-2,") == Curve(
        ((0.0, 0.5), (10.0, 1.0), (20.0, -2.0))
    )
    for invalid in ("", "0:1,,2:3", "0:(1", "0:sin(1)", "0:1:2", "0:1,0:2"):
        with pytest.raises(ValueError):
            parse_schedule(invalid)


def test_conversion_and_curve_nodes_execute_through_engine() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "editor": GraphNode("dinkster.curve.editor", {}),
                "schedule": GraphNode("dinkster.string.schedule_parse", {"text": "0:(1), 10:(3)"}),
                "position": GraphNode("dinkster.float", {"value": 2.5}),
                "evaluate": GraphNode(
                    "dinkster.curve.evaluate",
                    {
                        "curve": Link("schedule", "curve"),
                        "position": Link("position", "value"),
                    },
                ),
                "convert": GraphNode(
                    "dinkster.value.convert",
                    {"value": Link("evaluate", "value"), "target": "int", "force_lossy": True},
                ),
            }
        )
        result = await engine.run(graph, ["editor", "schedule", "evaluate", "convert"])
        assert result.outputs["editor"]["curve"].resolve() == Curve(
            ((0, 0), (1, 1)), "monotone_cubic"
        )
        assert result.outputs["schedule"]["curve"].type_id == CURVE_TYPE
        assert result.outputs["evaluate"]["value"].resolve() == 1.5
        assert result.outputs["convert"]["int"].resolve() == 1
        assert result.outputs["convert"]["float"].type_id == "core.absent"

    asyncio.run(scenario())


def test_conversion_curve_schemas_are_consolidated() -> None:
    schemas = build_schemas(FOUNDATION_NODES)
    expected = {
        "dinkster.value.convert",
        "dinkster.curve.editor",
        "dinkster.curve.from_values",
        "dinkster.curve.evaluate",
        "dinkster.curve.sample",
        "dinkster.string.schedule_parse",
    }
    assert expected <= set(schemas)
    for node_type in expected:
        assert schemas[node_type].aliases == ()
    convert = schemas["dinkster.value.convert"]
    assert convert.inputs[0].type.types == ("core.int", "core.float", "core.string", "core.boolean")
    assert {output.id for output in convert.outputs if output.optional} == {
        "int",
        "float",
        "string",
        "boolean",
    }
