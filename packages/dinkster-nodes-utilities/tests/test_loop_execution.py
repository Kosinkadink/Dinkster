"""Loop execution uses the ordinary utility pack and preserves typed children."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_utilities.loop_lists import ListElement, MakeList
from dinkster_values import TypeRegistry, ValueMeta, make_list_value, register_core_types

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("mode", ("simple", "For", "List"))
@pytest.mark.parametrize("accumulate", (True, False))
def test_loop_modes_execute_twice(in_process: bool, mode: str, accumulate: bool) -> None:
    options = {
        "simple": {"mode.num_iterations": 3},
        "For": {"mode.start_iteration_index": 3, "mode.max_iteration": 10, "mode.step": 3},
        "List": {"mode.list": TypedLiteral("list<INT>", [9, 2, 5])},
    }[mode]
    output = "list_item" if mode == "List" else "iteration_index"
    expected = {"simple": [0, 2, 4], "For": [6, 12, 18], "List": [18, 4, 10]}[mode]
    graph = Graph(
        {
            "start": GraphNode(
                "dinkster.utilities.start_loop", options, slot_variants={"mode": mode}
            ),
            "body": GraphNode(
                "dinkster.utilities.comfy_math_expression",
                {"expression": "a * 2", "values.a": Link("start", output)},
            ),
            "control": GraphNode("dinkster.utilities.primitive_boolean", {"value": accumulate}),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {"output_value": Link("body", "INT"), "accumulate": Link("control", "BOOLEAN")},
            ),
        }
    )

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(
                PackSpec(
                    Path(__file__).resolve().parents[1] / "dinkster-pack.toml",
                    in_process=in_process,
                    trust_reserved=True,
                )
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await asyncio.wait_for(engine.run(graph, ["end"]), 20)
                assert result.outputs["end"]["outputs"].resolve() == (
                    expected if accumulate else expected[-1:]
                )
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("in_process", (True, False))
def test_make_list_preserves_document_order_across_cached_runs(in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(
                PackSpec(
                    Path(__file__).resolve().parents[1] / "dinkster-pack.toml",
                    in_process=in_process,
                    trust_reserved=True,
                )
            )
            engine = composer.composition.make_engine(lambda _event: None)
            values = {"z": 17, "a": -4, "m": 29}
            for order in (("z", "a", "m"), ("m", "z", "a"), ("z", "a", "m")):
                graph = Graph(
                    {
                        "list": GraphNode(
                            "dinkster.utilities.make_list",
                            {f"items.{key}": TypedLiteral("INT", values[key]) for key in order},
                        )
                    }
                )
                for _ in range(2):
                    result = await asyncio.wait_for(engine.run(graph, ["list"]), 20)
                    assert result.outputs["list"]["list"].resolve() == [
                        values[key] for key in order
                    ]
        finally:
            await composer.close()

    run_scenario(scenario())


def test_list_support_keeps_whole_typed_elements_and_unbounded_inputs() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    first = replace(registry.wrap("INT", 17), meta=ValueMeta({"provenance": "first"}))
    second = registry.wrap("INT", -4)
    nested = make_list_value("INT", [first, second])
    result = MakeList.execute(input_values={"items.a": nested})
    assert result["list"] == [nested]
    for index, expected in ((0, first), (-1, second)):
        assert ListElement.execute(index=index, input_values={"list": nested})["item"] is expected
    with pytest.raises(ValueError, match="index 2 out of range for a list of 2"):
        ListElement.execute(index=2, input_values={"list": nested})
