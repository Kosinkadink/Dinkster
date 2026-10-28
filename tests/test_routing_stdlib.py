"""Lazy routing nodes and graph persistence."""

from __future__ import annotations

import asyncio

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
)
from dinkster_values import TypeRegistry, is_absent, register_core_types
from dinkster_workers import InProcessWorker


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
    with pytest.raises(TypeError, match="COMBO expects a JSON scalar"):
        registry.wrap(
            "COMBO",
            {"value": "stable_right", "label": "Right branch"},
        )
