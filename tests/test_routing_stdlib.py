"""Lazy routing nodes and maintained ComfyUI aliases."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_foundation import (
    FOUNDATION_NODES,
    RouteGate,
    RouteSwitch,
    RouteSwitchByName,
    ValueSelect,
)
from dinkster_schema import (
    AbsentOutput,
    ComboWidget,
    InputSpec,
    SelectorSpec,
    build_node_types,
    build_schemas,
    comfy_alias_registry_from_wire,
    comfy_alias_registry_problems,
    comfy_alias_registry_to_wire,
    schema_from_wire,
    validate_replacement_references,
)
from dinkster_schema.replace import rule_from_wire, rule_to_wire
from dinkster_values import TypeRegistry, is_absent, register_core_types
from dinkster_workers import InProcessWorker

from tools.generate_foundation_comfy_aliases import ROUTING_EVIDENCE, build_aliases

ALIAS_PATH = (
    Path(__file__).parents[1] / "packages" / "dinkster-nodes-foundation" / "comfy-aliases.json"
)


def test_routing_schemas_and_operations() -> None:
    select = ValueSelect.schema()
    assert [spec.lazy for spec in select.inputs] == [False, True, True]
    assert select.selector == SelectorSpec("condition", {"false": "on_false", "true": "on_true"})
    assert ValueSelect.check_lazy_status(condition=True, on_false=None, on_true=None) == (
        "on_true",
    )
    assert ValueSelect.check_lazy_status(condition=False, on_false="off", on_true=None) == ()

    switch = RouteSwitch.schema()
    family = switch.input_families[0]
    assert family.id == "values"
    assert family.min_members == 1 and family.max_members == 512
    template = family.template[0]
    assert isinstance(template, InputSpec) and template.lazy is True
    assert RouteSwitch.check_lazy_status(index=1, values={"a": None, "b": None}) == ("values.b",)
    assert RouteSwitch.check_lazy_status(index=1, values={"a": None, "b": "B"}) == ()
    assert RouteSwitch.execute(index=1, values={"a": None, "b": "B"}) == {"value": "B"}
    with pytest.raises(ValueError, match="index must be in"):
        RouteSwitch.check_lazy_status(index=2, values={"a": None, "b": None})

    named = RouteSwitchByName.schema()
    assert named.node_type == "dinkster.route.switch_by_name"
    assert isinstance(named.inputs[0].widget, ComboWidget)
    assert named.inputs[0].widget.option_source is not None
    assert named.inputs[0].widget.option_source.input_family == "values"
    assert RouteSwitchByName.check_lazy_status(
        choice="stable_b", values={"stable_a": None, "stable_b": None}
    ) == ("values.stable_b",)
    assert (
        RouteSwitchByName.check_lazy_status(
            choice="stable_b", values={"stable_a": None, "stable_b": "B"}
        )
        == ()
    )
    assert RouteSwitchByName.execute(
        choice="stable_b", values={"stable_a": None, "stable_b": "B"}
    ) == {"value": "B"}
    with pytest.raises(ValueError, match="must name a values member"):
        RouteSwitchByName.check_lazy_status(
            choice="display label", values={"stable_a": None, "stable_b": None}
        )

    gate = RouteGate.schema()
    assert gate.inputs[1].lazy is True
    assert gate.outputs[0].optional is True
    assert RouteGate.check_lazy_status(condition=False, value=None) == ()
    assert RouteGate.check_lazy_status(condition=True, value=None) == ("value",)
    assert isinstance(RouteGate.execute(condition=False, value=None)["value"], AbsentOutput)
    assert RouteGate.execute(condition=True, value="open") == {"value": "open"}


def test_value_select_and_gate_do_not_execute_unselected_sources() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(FOUNDATION_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(FOUNDATION_NODES), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            {
                "off": GraphNode("dinkster.string", {"value": "off"}),
                "on": GraphNode("dinkster.string", {"value": "on"}),
                "closed": GraphNode("dinkster.string", {"value": "closed"}),
                "left": GraphNode("dinkster.string", {"value": "left"}),
                "right": GraphNode("dinkster.string", {"value": "right"}),
                "select": GraphNode(
                    "dinkster.value.select",
                    {
                        "condition": True,
                        "on_false": Link("off", "value"),
                        "on_true": Link("on", "value"),
                    },
                ),
                "gate": GraphNode(
                    "dinkster.route.gate",
                    {"condition": False, "value": Link("closed", "value")},
                ),
                "named": GraphNode(
                    "dinkster.route.switch_by_name",
                    {
                        "choice": "stable_right",
                        "values.stable_left": Link("left", "value"),
                        "values.stable_right": Link("right", "value"),
                    },
                ),
            }
        )

        result = await engine.run(graph, ["select", "gate", "named"])
        assert result.outputs["select"]["value"].resolve() == "on"
        assert is_absent(result.outputs["gate"]["value"])
        assert result.outputs["named"]["value"].resolve() == "right"
        assert set(result.executed) == {"on", "select", "gate", "right", "named"}

    asyncio.run(scenario())


def test_named_route_graph_wire_persists_only_stable_member_ids() -> None:
    graph = Graph(
        {
            "named": GraphNode(
                "dinkster.route.switch_by_name",
                {
                    "choice": "stable_right",
                    "values.stable_left": Link("left", "value"),
                    "values.stable_right": Link("right", "value"),
                },
            ),
        }
    )
    wire = graph_to_wire(graph)
    assert wire == {
        "nodes": {
            "named": {
                "nodeType": "dinkster.route.switch_by_name",
                "inputs": {
                    "choice": "stable_right",
                    "values.stable_left": {"$link": {"node": "left", "output": "value"}},
                    "values.stable_right": {"$link": {"node": "right", "output": "value"}},
                },
            },
        },
    }
    assert graph_from_wire(wire) == graph

    registry = TypeRegistry()
    register_core_types(registry)
    with pytest.raises(TypeError, match="core.combo expects a string"):
        registry.wrap(
            "core.combo",
            {"value": "stable_right", "label": "Right branch"},
        )


def _routing_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for record in cast("list[dict[str, Any]]", payload["records"])
        if record["confidence"]["evidence"] == list(ROUTING_EVIDENCE)
    ]


def test_routing_alias_registry_is_canonical_and_valid() -> None:
    encoded = ALIAS_PATH.read_text(encoding="utf-8")
    assert encoded == json.dumps(build_aliases(), ensure_ascii=True, separators=(",", ":")) + "\n"
    payload = cast("dict[str, Any]", json.loads(encoded))
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
    for record in _routing_records(payload):
        rule = rule_from_wire(record["replacement"])
        assert rule_to_wire(rule) == record["replacement"]
        carrier = native_schemas[record["carrier"]]
        schemas = {
            **native_schemas,
            **source_schemas,
            carrier.node_type: dataclasses.replace(carrier, replacements=(rule,)),
        }
        assert validate_replacement_references(schemas) == ()


def test_routing_alias_operation_mappings() -> None:
    payload = cast("dict[str, Any]", build_aliases())
    records = _routing_records(payload)
    expected = {
        "ComfySwitchNode": ("switch", "on_false", "on_true", "output"),
        "LazySwitchKJ": ("switch", "on_false", "on_true", "*"),
        "easy ifElse": ("boolean", "on_false", "on_true", "*"),
        "ImpactConditionalBranch": ("cond", "ff_value", "tt_value", "*"),
    }
    assert {record["source"]["nodeClass"] for record in records} == set(expected)
    for record in records:
        assert record["carrier"] == "dinkster.value.select"
        assert record["confidence"]["tier"] == "exact"
        rule = rule_from_wire(record["replacement"])
        case = rule.cases[0]
        inputs = dict(case.inputs)
        condition, on_false, on_true, source_output = expected[record["source"]["nodeClass"]]
        assert inputs["condition"].input == condition
        assert inputs["on_false"].input == on_false
        assert inputs["on_true"].input == on_true
        assert dict(case.outputs) == {"value": source_output}
