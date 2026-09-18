"""Numeric expression and value-operation standard library coverage."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import struct
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import (
    Graph,
    GraphNode,
    Link,
    TypedLiteral,
    graph_from_wire,
    graph_to_wire,
    validate,
)
from dinkster_nodes_foundation import (
    FOUNDATION_NODES,
    BoolLogic,
    MathExpression,
    ValueClamp,
    ValueCompare,
    ValueRandom,
    ValueRemap,
    ValueSelect,
)
from dinkster_nodes_foundation.expression import evaluate_expression
from dinkster_schema import (
    ABSENT,
    InputSpec,
    StringWidget,
    TypeExpr,
    build_node_types,
    build_schemas,
    comfy_alias_registry_from_wire,
    comfy_alias_registry_problems,
    comfy_alias_registry_to_wire,
    schema_from_wire,
    schema_to_wire,
    validate_replacement_references,
)
from dinkster_schema.replace import rule_from_wire, rule_to_wire
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from hypothesis import given
from hypothesis import strategies as st

from tools.generate_foundation_comfy_aliases import NUMERIC_EVIDENCE, build_aliases
from tools.generate_math_expression_vector import (
    CASES,
    COMFYUI_REFERENCE_COMMIT,
    COMFYUI_REFERENCE_PATH,
    COMFYUI_SOURCE_SHA256,
    SIMPLEEVAL_REFERENCE_VERSION,
    _assert_reference_parity,
    build_vector,
    regeneration_stable_view,
)

VECTOR_PATH = Path(__file__).parent / "fixtures" / "math_expression_v1.json"
ALIAS_PATH = (
    Path(__file__).parents[1] / "packages" / "dinkster-nodes-foundation" / "comfy-aliases.json"
)


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=build_schemas(FOUNDATION_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(FOUNDATION_NODES), registry),
        cache=MemoryLRUCache(),
    )


def test_expression_vector_is_current_and_executable() -> None:
    vector = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
    assert vector["comfyui_reference"] == {
        "commit": COMFYUI_REFERENCE_COMMIT,
        "path": COMFYUI_REFERENCE_PATH,
        "source_sha256": COMFYUI_SOURCE_SHA256,
        "executed_symbol": "MathExpressionNode.execute",
        "simpleeval_version": SIMPLEEVAL_REFERENCE_VERSION,
        "generator": "tools/generate_math_expression_vector.py",
    }
    # Bounded-case floats are the generator platform's libm values; every
    # other part of the corpus must be regeneration-stable on this platform.
    assert regeneration_stable_view(vector) == regeneration_stable_view(build_vector())

    def bits(value: float) -> str:
        return struct.pack(">d", value).hex()

    def within_tolerance(actual: float, expected: float, tolerance: float) -> bool:
        return abs(actual - expected) <= tolerance * max(abs(actual), abs(expected))

    tolerance = cast("float", vector["mirror_relative_tolerance"])
    for case in cast("list[dict[str, Any]]", vector["cases"]):
        expected = cast("dict[str, Any]", case["outputs"])
        actual = dict(
            zip(
                ("float", "int", "boolean"),
                evaluate_expression(
                    cast("str", case["expression"]), cast("dict[str, object]", case["inputs"])
                ),
                strict=True,
            )
        )
        assert tuple(expected) == ("float", "int", "boolean")
        actual_bits = bits(cast("float", actual["float"]))
        if case["mirror_class"] == "exact":
            assert actual_bits == case["float_bits"], case["id"]
            assert actual == expected, case["id"]
        else:
            assert case["mirror_class"] == "bounded", case["id"]
            actual_float = actual.pop("float")
            expected_float = expected.pop("float")
            assert actual == expected, case["id"]
            assert within_tolerance(
                cast("float", actual_float), cast("float", expected_float), tolerance
            ), case["id"]
        if "int_strings" in case:
            assert str(actual["int"]) == case["int_strings"]


def test_expression_vector_enforces_captured_comfyui_parity() -> None:
    vector = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
    specs = {spec["id"]: spec for spec in CASES}
    for case in cast("list[dict[str, Any]]", vector["cases"]):
        if case["compatibility"] != "ComfyMathExpression":
            continue
        outputs = cast("dict[str, Any]", case["outputs"])
        reference = cast(
            "tuple[float, int, bool]",
            tuple(outputs[name] for name in ("float", "int", "boolean")),
        )
        dinkster = evaluate_expression(
            cast("str", case["expression"]), cast("dict[str, object]", case["inputs"])
        )
        _assert_reference_parity(specs[cast("str", case["id"])], dinkster, reference)

    spec = CASES[0]
    dinkster = evaluate_expression(spec["expression"], spec["inputs"])
    with pytest.raises(ValueError, match="integer or boolean output differs"):
        _assert_reference_parity(spec, dinkster, (dinkster[0], dinkster[1] + 1, dinkster[2]))
    with pytest.raises(ValueError, match="float output differs"):
        _assert_reference_parity(
            spec,
            dinkster,
            (math.nextafter(dinkster[0], math.inf), dinkster[1], dinkster[2]),
        )

    bounded_spec = next(
        spec
        for spec in CASES
        if spec["compatibility"] == "ComfyMathExpression" and spec["mirror_class"] == "bounded"
    )
    bounded = evaluate_expression(bounded_spec["expression"], bounded_spec["inputs"])
    with pytest.raises(ValueError, match="float output exceeds"):
        _assert_reference_parity(
            bounded_spec,
            bounded,
            (bounded[0] + 1e-6, bounded[1], bounded[2]),
        )


def test_expression_schema_declares_the_mirror_the_vector_verifies() -> None:
    vector = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
    schema = MathExpression.define_schema()
    mirror = schema.mirror
    assert mirror is not None
    assert mirror.kind == "expression"
    assert mirror.precision == "bounded"
    assert mirror.grammar_version == vector["grammar_version"]
    assert mirror.tolerance is not None
    assert mirror.tolerance.relative == vector["mirror_relative_tolerance"]
    assert mirror.source is None

    expression_input = schema.inputs[0]
    assert expression_input.id == "expression"
    assert isinstance(expression_input.widget, StringWidget)
    completions = expression_input.widget.completions
    assert completions is not None
    assert completions.input_families == ("values",)
    by_value = {item.value: item for item in completions.items}
    assert by_value["sin"].insert_text == "sin("
    assert by_value["sin"].detail == "Function"
    assert by_value["**"].kind == "operator"
    assert by_value["and"].kind == "identifier"
    assert by_value["or"].kind == "identifier"
    assert by_value["not"].kind == "identifier"
    assert {item.value for item in completions.items if item.detail == "Operator"} == {
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "**",
        "<<",
        ">>",
        "|",
        "&",
        "^",
        "~",
        "and",
        "or",
        "not",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
    }
    assert {"True", "False", "values"} <= set(by_value)

    family = schema.input_families[0]
    assert family.id == "values"
    assert family.member_names == tuple("abcdefghijklmnopqrstuvwxyz")
    assert family.min_members == 1
    family_input = family.template[0]
    assert isinstance(family_input, InputSpec)
    assert family_input.type == TypeExpr.union("core.float", "core.int", "core.boolean")
    assert family_input.force_input is True
    assert tuple((output.id, output.type, output.optional) for output in schema.outputs) == (
        ("float", TypeExpr.concrete("core.float"), False),
        ("int", TypeExpr.concrete("core.int"), False),
        ("boolean", TypeExpr.concrete("core.boolean"), False),
    )

    wire = schema_to_wire(schema)
    assert schema_from_wire(wire) == schema
    interface = cast("list[dict[str, Any]]", wire["interface"])
    family_wire = next(entry for entry in interface if entry["role"] == "inputFamily")
    assert family_wire["template"][0]["type"] == {
        "kind": "union",
        "types": ["core.float", "core.int", "core.boolean"],
    }
    assert family_wire["template"][0]["forceInput"] is True
    output_wire = [entry for entry in interface if entry["role"] == "output"]
    assert [(entry["id"], entry["type"]) for entry in output_wire] == [
        ("float", {"kind": "concrete", "types": ["core.float"]}),
        ("int", {"kind": "concrete", "types": ["core.int"]}),
        ("boolean", {"kind": "concrete", "types": ["core.boolean"]}),
    ]
    assert all("optional" not in entry for entry in output_wire)


def test_foundation_comfy_alias_records() -> None:
    encoded = ALIAS_PATH.read_text(encoding="utf-8")
    assert encoded == json.dumps(build_aliases(), ensure_ascii=True, separators=(",", ":")) + "\n"
    payload = json.loads(encoded)
    assert payload == build_aliases()
    registry = comfy_alias_registry_from_wire(payload)
    assert comfy_alias_registry_to_wire(registry) == payload

    source_schemas = {
        schema.node_type: schema
        for schema in (
            schema_from_wire(cast("dict[str, Any]", wire))
            for wire in cast("list[dict[str, object]]", payload["sourceSchemas"])
        )
    }
    native_schemas = build_schemas(FOUNDATION_NODES)
    assert comfy_alias_registry_problems(registry, native_schemas) == ()
    all_records = cast("list[dict[str, Any]]", payload["records"])
    assert len({record["id"] for record in all_records}) == len(all_records)
    assert {"ComfyAndNode", "ComfyOrNode"}.issubset(
        record["source"]["nodeClass"] for record in all_records
    )
    records = [
        record
        for record in all_records
        if record["confidence"]["evidence"] == list(NUMERIC_EVIDENCE)
    ]
    assert {record["carrier"] for record in records} == {
        "dinkster.boolean",
        "dinkster.math.expression",
        "dinkster.bool.logic",
        "dinkster.value.compare",
        "dinkster.value.select",
        "dinkster.value.remap",
    }

    for record in records:
        assert record["id"].startswith("comfy_alias:")
        assert record["mappingKind"] == "op"
        source = record["source"]
        assert source["nodeType"] in source_schemas
        assert source_schemas[source["nodeType"]].replacements == ()
        rule = rule_from_wire(record["replacement"])
        assert rule_to_wire(rule) == record["replacement"]
        assert rule.from_type == source["nodeType"]
        assert record["carrier"] in {case.to for case in rule.cases}
        carrier = native_schemas[record["carrier"]]
        schemas = {
            **native_schemas,
            **source_schemas,
            carrier.node_type: dataclasses.replace(carrier, replacements=(rule,)),
        }
        assert validate_replacement_references(schemas) == ()
        assert record["confidence"]["evidence"] == [
            "tests/test_numeric_stdlib.py::test_foundation_comfy_alias_records",
            "tests/test_numeric_stdlib.py::test_foundation_comfy_alias_behavior",
        ]

    for node_type in (
        "dinkster.math.expression",
        "dinkster.value.compare",
        "dinkster.value.select",
        "dinkster.bool.logic",
        "dinkster.value.clamp",
        "dinkster.value.remap",
        "dinkster.value.random",
    ):
        assert native_schemas[node_type].aliases == ()


def test_foundation_comfy_alias_behavior() -> None:
    payload = build_aliases()
    records = {record["id"]: record for record in cast("list[dict[str, Any]]", payload["records"])}
    source_schemas = {
        schema.node_type: schema
        for schema in (
            schema_from_wire(cast("dict[str, Any]", wire))
            for wire in cast("list[dict[str, object]]", payload["sourceSchemas"])
        )
    }

    def rule(identifier: str):
        return rule_from_wire(records[identifier]["replacement"])

    def fallback_enum(identifier: str, target_input: str) -> dict[str, str]:
        source = dict(rule(identifier).cases[-1].inputs)[target_input]
        assert source.transform is not None
        return dict(source.transform.map)

    boolean_record = records["comfy_alias:comfy-core/PrimitiveBoolean"]
    assert boolean_record["source"]["revision"] == "b78cec87"
    boolean_case = rule("comfy_alias:comfy-core/PrimitiveBoolean").cases[0]
    assert dict(boolean_case.inputs)["value"].input == "value"
    assert dict(boolean_case.outputs) == {"value": "boolean"}

    math_record = records["comfy_alias:comfy-core/ComfyMathExpression"]
    assert math_record["source"]["revision"] == "b78cec87"
    math_source = source_schemas["comfy.ComfyMathExpression"]
    assert math_source.aliases == ("ComfyMathExpression",)
    assert math_source.input_families[0].member_names == tuple("abcdefghijklmnopqrstuvwxyz")
    math_family_input = math_source.input_families[0].template[0]
    assert isinstance(math_family_input, InputSpec)
    assert math_family_input.type == TypeExpr.union("core.float", "core.int", "core.boolean")
    assert tuple((output.id, output.optional) for output in math_source.outputs) == (
        ("FLOAT", False),
        ("INT", False),
        ("BOOL", False),
    )
    math_case = rule("comfy_alias:comfy-core/ComfyMathExpression").cases[0]
    math_family = dict(math_case.input_families)["values"]
    assert math_family.kind == "copy" and math_family.source_family == "values"
    assert dict(math_family.inputs)["value"].input == "value"
    assert dict(math_case.outputs) == {"float": "FLOAT", "int": "INT", "boolean": "BOOL"}

    for node_class, operation, result in (
        ("ComfyAndNode", "and", False),
        ("ComfyOrNode", "or", True),
    ):
        identifier = f"comfy_alias:comfy-core/{node_class}"
        record = records[identifier]
        assert record["confidence"]["tier"] == "parametric"
        source_schema = source_schemas[f"comfy.{node_class}"]
        assert source_schema.input_families[0].max_members == 10
        assert source_schema.input_families[0].member_prefix == "value"
        assert record["replacement"]["cases"][0]["inputFamilies"] == {
            "values": {
                "kind": "copy",
                "sourceFamily": "values",
                "inputs": {"value": {"kind": "copy", "input": "value"}},
            }
        }
        logic_case = rule(identifier).cases[0]
        assert dict(logic_case.inputs)["operation"].value == operation
        assert dict(logic_case.outputs) == {"result": "boolean"}
        assert BoolLogic.execute(operation=operation, values={"value0": True, "value1": False}) == {
            "result": result
        }

    essentials_id = "comfy_alias:comfyui_essentials/SimpleComparison+"
    essentials = fallback_enum(essentials_id, "operation")
    assert essentials == {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
    expected = {"==": False, "!=": True, "<": True, "<=": True, ">": False, ">=": False}
    for source_operation, result in expected.items():
        assert ValueCompare.execute(
            a=2, b=3, operation=essentials[source_operation], epsilon=0.0
        ) == {"result": result}

    easy_id = "comfy_alias:comfyui-easy-use/easy compare"
    easy = fallback_enum(easy_id, "operation")
    assert easy == {
        "a == b": "eq",
        "a != b": "ne",
        "a < b": "lt",
        "a <= b": "le",
        "a > b": "gt",
        "a >= b": "ge",
    }
    easy_rule = rule(easy_id)
    first_unary_inputs = dict(easy_rule.cases[0].inputs)
    omitted_unary_inputs = dict(easy_rule.cases[1].inputs)
    assert first_unary_inputs["b"].value == 0
    assert omitted_unary_inputs["a"].value == omitted_unary_inputs["b"].value == 0
    assert ValueCompare.execute(a=0, b=0, operation="gt", epsilon=0.0) == {"result": False}
    assert ValueCompare.execute(a=0, b=0, operation="le", epsilon=0.0) == {"result": True}

    impact_id = "comfy_alias:comfyui-impact-pack/ImpactCompare"
    impact_rule = rule(impact_id)
    assert [case.to for case in impact_rule.cases[:2]] == ["dinkster.boolean", "dinkster.boolean"]
    assert dict(impact_rule.cases[0].inputs)["value"].value is True
    assert dict(impact_rule.cases[1].inputs)["value"].value is False
    assert fallback_enum(impact_id, "operation") == {
        "a = b": "eq",
        "a <> b": "ne",
        "a < b": "lt",
        "a <= b": "le",
        "a > b": "gt",
        "a >= b": "ge",
    }

    switch = rule("comfy_alias:was-node-suite-comfyui/Number Input Switch")
    switch_inputs = dict(switch.cases[0].inputs)
    assert switch_inputs["condition"].input == "boolean"
    assert switch_inputs["on_false"].input == "number_b"
    assert switch_inputs["on_true"].input == "number_a"

    fit = rule("comfy_alias:comfy-mtb/Fit Number (mtb)")
    fit_inputs = dict(fit.cases[0].inputs)
    assert fit_inputs["curve"].transform is not None
    assert dict(fit_inputs["curve"].transform.map) == {"Linear": "linear"}
    assert ValueRemap.execute(
        value=5.0,
        input_minimum=0.0,
        input_maximum=10.0,
        output_minimum=-1.0,
        output_maximum=1.0,
        clamp=False,
        curve="linear",
    ) == {"value": 0.0}


@given(a=st.integers(-1_000_000, 1_000_000), b=st.integers(-1_000_000, 1_000_000))
def test_expression_integer_arithmetic_property(a: int, b: int) -> None:
    assert evaluate_expression("a + b * 2", {"a": a, "b": b}) == (
        float(a + b * 2),
        a + b * 2,
        bool(a + b * 2),
    )


@given(
    value=st.floats(allow_nan=False, allow_infinity=False, width=32),
    minimum=st.floats(allow_nan=False, allow_infinity=False, width=32),
    span=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_clamp_result_is_in_closed_range_property(
    value: float, minimum: float, span: float
) -> None:
    maximum = minimum + span
    result = ValueClamp.execute(value=value, minimum=minimum, maximum=maximum)["value"]
    assert minimum <= cast("float", result) <= maximum


def test_expression_validation_and_lazy_conditional() -> None:
    assert evaluate_expression("1 if a else 1 / 0", {"a": True}) == (1.0, 1, True)
    assert evaluate_expression("min(values)", {"a": 3, "b": -2}) == (-2.0, -2, True)

    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_expression("", {"a": 1})
    with pytest.raises(NameError, match="unknown expression name"):
        evaluate_expression("b", {"a": 1})
    with pytest.raises(ValueError, match="4000"):
        evaluate_expression("2 ** 4001", {"a": 1})
    with pytest.raises(TypeError, match="int, float, or bool"):
        evaluate_expression("a", {"a": "1"})
    with pytest.raises(TypeError, match="int, float, or bool"):
        evaluate_expression("a", {"a": [1, 2]})
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_expression("1e309", {"a": 1})
    with pytest.raises(ValueError, match="one lowercase letter"):
        evaluate_expression("st + 1", {"st": 5})
    with pytest.raises(ValueError, match="16384 bits"):
        evaluate_expression("(2 ** 4000) ** 4000", {"a": 1})


def test_expression_outputs_required_scalar_ports() -> None:
    result = MathExpression.execute(expression="a / 2", values={"a": 5})
    assert result == {
        "float": 2.5,
        "int": 2,
        "boolean": True,
    }


def test_existing_scalar_math_workflow_needs_no_migration() -> None:
    expression = GraphNode(
        "dinkster.math.expression",
        {
            "expression": "a / b",
            "values.a": TypedLiteral("core.int", 7),
            "values.b": TypedLiteral("core.int", 2),
        },
    )
    nodes: dict[str, GraphNode] = {"expression": expression}
    for output, type_id, expected in (
        ("float", "core.float", 3.5),
        ("int", "core.int", 3),
        ("boolean", "core.boolean", True),
    ):
        nodes[f"compare_{output}"] = GraphNode(
            "dinkster.value.compare",
            {
                "a": Link("expression", output),
                "b": TypedLiteral(type_id, expected),
                "operation": "eq",
                "epsilon": 0.0,
            },
        )
    graph = Graph(nodes=nodes)
    restored = graph_from_wire(graph_to_wire(graph))
    assert restored == graph
    diagnostics = validate(
        restored,
        build_schemas(FOUNDATION_NODES),
        ("compare_float", "compare_int", "compare_boolean"),
    )
    assert [diagnostic for diagnostic in diagnostics if diagnostic.severity == "error"] == []


@pytest.mark.parametrize(
    ("operation", "expected"),
    (("eq", True), ("ne", False), ("lt", False), ("le", True), ("gt", False), ("ge", True)),
)
def test_compare_numeric_epsilon(operation: str, expected: bool) -> None:
    result = ValueCompare.execute(a=1.0, b=1.05, operation=operation, epsilon=0.1)
    assert result == {"result": expected}


def test_compare_select_logic_and_clamp() -> None:
    assert ValueCompare.execute(a={"x": 1}, b={"x": 1}, operation="eq", epsilon=0.0) == {
        "result": True
    }
    assert ValueCompare.execute(a="a", b="b", operation="lt", epsilon=0.0) == {"result": True}
    assert ValueCompare.execute(a=2**100, b=2**100 + 1, operation="lt", epsilon=0.0) == {
        "result": True
    }
    assert ValueSelect.execute(condition=True, on_false="no", on_true="yes") == {"value": "yes"}
    assert BoolLogic.execute(operation="and", values={"a": True, "b": True}) == {"result": True}
    assert BoolLogic.execute(operation="or", values={"a": False, "b": False}) == {"result": False}
    assert BoolLogic.execute(operation="not", values={"a": False}) == {"result": True}
    assert BoolLogic.execute(operation="xor", values={"a": True, "b": True, "c": True}) == {
        "result": True
    }
    assert ValueClamp.execute(value=8, minimum=0, maximum=5) == {"value": 5}
    with pytest.raises(ValueError, match="exactly one"):
        BoolLogic.execute(operation="not", values={"a": True, "b": False})


@pytest.mark.parametrize(
    ("curve", "expected"),
    (
        ("linear", 0.25),
        ("smoothstep", 0.15625),
        ("smootherstep", 0.103515625),
        ("ease_in", 0.0625),
        ("ease_out", 0.4375),
        ("ease_in_out", 0.125),
    ),
)
def test_remap_curves(curve: str, expected: float) -> None:
    result = ValueRemap.execute(
        value=0.25,
        input_minimum=0.0,
        input_maximum=1.0,
        output_minimum=0.0,
        output_maximum=1.0,
        clamp=True,
        curve=curve,
    )
    assert result == {"value": expected}


def test_seeded_random_is_reproducible_and_bounded() -> None:
    first = ValueRandom.execute(seed=42, minimum=0.0, maximum=1.0, number_type="float")
    assert first == {"int": ABSENT, "float": 0.7415648787718233}
    assert ValueRandom.execute(seed=42, minimum=0.0, maximum=1.0, number_type="float") == first
    assert ValueRandom.execute(seed=42, minimum=-3.0, maximum=3.0, number_type="int") == {
        "int": 2,
        "float": ABSENT,
    }
    with pytest.raises(ValueError, match="integral endpoints"):
        ValueRandom.execute(seed=0, minimum=0.5, maximum=2.0, number_type="int")


def test_random_float_range_remains_half_open_at_adjacent_bounds() -> None:
    minimum = 1.0
    maximum = math.nextafter(minimum, math.inf)
    for seed in (0, 42, (1 << 64) - 1):
        result = ValueRandom.execute(
            seed=seed, minimum=minimum, maximum=maximum, number_type="float"
        )
        assert result["float"] == minimum


def test_numeric_nodes_execute_through_engine() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "a": GraphNode("dinkster.int", {"value": 3}),
                "b": GraphNode("dinkster.float", {"value": 2.5}),
                "expected": GraphNode("dinkster.float", {"value": 8.0}),
                "expression": GraphNode(
                    "dinkster.math.expression",
                    {
                        "expression": "a + b * 2",
                        "values.a": Link("a", "value"),
                        "values.b": Link("b", "value"),
                    },
                ),
                "compare": GraphNode(
                    "dinkster.value.compare",
                    {
                        "a": Link("expression", "float"),
                        "b": Link("expected", "value"),
                        "operation": "eq",
                        "epsilon": 0.0,
                    },
                ),
            }
        )
        result = await engine.run(graph, ["expression", "compare"])
        assert result.outputs["expression"]["float"].resolve() == 8.0
        assert result.outputs["compare"]["result"].resolve() is True

    asyncio.run(scenario())


def test_numeric_node_schemas_are_well_formed() -> None:
    schemas = build_schemas(FOUNDATION_NODES)
    assert {
        "dinkster.math.expression",
        "dinkster.value.compare",
        "dinkster.value.select",
        "dinkster.bool.logic",
        "dinkster.value.clamp",
        "dinkster.value.remap",
        "dinkster.value.random",
    } <= set(schemas)
    assert schemas["dinkster.int"].inputs[0].widget.control_after_generate == "fixed"  # type: ignore[union-attr]
    assert ValueRandom.schema().inputs[0].widget.control_after_generate == "randomize"  # type: ignore[union-attr]
    assert math.isfinite(
        cast(
            "float",
            ValueRandom.execute(seed=0, minimum=-1e308, maximum=1e308, number_type="float")[
                "float"
            ],
        )
    )
