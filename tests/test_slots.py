"""Dynamic slots (hazard H10): type-driven variant specialization.

The schema declares per-type variants; the DOCUMENT stores which one is
active; elaboration turns the choice into ordinary concrete inputs. Covers
the model guardrails, elaboration semantics, wire round trips, graph wire,
worker kwarg grouping (SlotValue), cache identity, and validation of the
connected type against the chosen variant.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, GraphValidationError, Invocation
from dinkster_graph import (
    Graph,
    GraphNode,
    Link,
    graph_from_wire,
    graph_to_wire,
    validate,
)
from dinkster_schema import (
    DynamicSlotSpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    SlotValue,
    SlotVariant,
    TypeExpr,
    build_node_types,
    build_schemas,
    elaborate,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_text.text import StringConcatenate
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types

from tests.async_scenarios import run_scenario

STRING = TypeExpr.concrete("STRING")
FLOAT = TypeExpr.concrete("FLOAT")
MATCHED = TypeExpr.variable("T", ("STRING", "FLOAT"))

SOURCE_SLOT = DynamicSlotSpec(
    id="source",
    variants=(
        SlotVariant(
            key="text",
            type=STRING,
            inputs=(InputSpec("prefix", STRING, default=""),),
        ),
        SlotVariant(
            key="number",
            type=FLOAT,
            inputs=(InputSpec("scale", FLOAT, default=1.0),),
        ),
    ),
)

STYLIZE_SCHEMA = NodeSchema(
    node_type="test.stylize",
    slots=(SOURCE_SLOT,),
    outputs=(OutputSpec("text", STRING),),
)

OPEN_SLOT = DynamicSlotSpec(
    "source",
    slot_type=STRING,
    inputs=(InputSpec("scale", FLOAT),),
    force_input=True,
)
OPEN_SCHEMA = NodeSchema(
    node_type="test.open-slot",
    slots=(OPEN_SLOT,),
    outputs=(OutputSpec("out", STRING),),
)

MATCHED_SLOT = DynamicSlotSpec(
    id="source",
    variants=(
        SlotVariant("text", STRING, inputs=(InputSpec("copy", MATCHED),)),
        SlotVariant("number", FLOAT, inputs=(InputSpec("copy", MATCHED),)),
    ),
    type_template_id="T",
)
MATCHED_SCHEMA = NodeSchema(
    node_type="test.matched-slot",
    slots=(MATCHED_SLOT,),
    outputs=(
        OutputSpec("out", MATCHED),
        OutputSpec("batch", TypeExpr.list_of(MATCHED)),
        OutputSpec("asset", TypeExpr.asset_of(MATCHED)),
    ),
)


# --- model guardrails ---


# --- elaboration ---


def test_absent_open_slot_is_bare_optional_socket_and_validates() -> None:
    effective = elaborate(OPEN_SCHEMA, [])
    assert [(spec.id, spec.required) for spec in effective.inputs] == [("source", False)]
    assert effective.slot_choices == ()
    graph = Graph(nodes={"n": GraphNode("test.open-slot")})
    assert validate(graph, {"test.open-slot": OPEN_SCHEMA}, ["n"]) == []


def test_inactive_open_slot_dependent_is_unknown_input() -> None:
    graph = Graph(nodes={"n": GraphNode("test.open-slot", {"source.scale": 2.0})})
    diagnostics = validate(graph, {"test.open-slot": OPEN_SCHEMA}, ["n"])
    assert any(
        diagnostic.code == "unknown-input" and diagnostic.input_id == "source.scale"
        for diagnostic in diagnostics
    )


# --- schema wire ---


def test_slots_round_trip_through_wire() -> None:
    wire = schema_to_wire(STYLIZE_SCHEMA)
    entries = [e for e in wire["interface"] if e["role"] == "dynamicSlot"]  # type: ignore[index, union-attr]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "source"
    assert [v["key"] for v in entry["variants"]] == ["text", "number"]
    assert schema_from_wire(wire) == STYLIZE_SCHEMA


# --- graph model + wire ---


def test_graph_wire_round_trips_slot_variants() -> None:
    graph = Graph(
        nodes={
            "n": GraphNode(
                "test.stylize",
                {"source": "hi", "source.prefix": ">> "},
                slot_variants={"source": "text"},
            ),
            "m": GraphNode("test.other", {}),
        }
    )
    wire = graph_to_wire(graph)
    assert wire["nodes"]["n"]["slotVariants"] == {"source": "text"}  # type: ignore[index]
    assert "slotVariants" not in wire["nodes"]["m"]  # type: ignore[operator]
    back = graph_from_wire(wire)
    node = back.nodes["n"]
    assert isinstance(node, GraphNode)
    assert dict(node.slot_variants) == {"source": "text"}


# --- engine + worker end-to-end ---


class Stylize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return STYLIZE_SCHEMA

    @classmethod
    def execute(cls, *, source: SlotValue) -> Mapping[str, object]:
        # Explicit dispatch on the variant key - never runtime type sniffing.
        if source.variant == "text":
            prefix = source.options["prefix"]
            return cls.outputs(text=f"{prefix}{source.value}")
        assert source.variant == "number"
        assert isinstance(source.value, float)
        scale = source.options["scale"]
        assert isinstance(scale, float)
        return cls.outputs(text=str(source.value * scale))


class OpenSlotProbe(Node):
    seen: list[dict[str, object]] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return OPEN_SCHEMA

    @classmethod
    def execute(cls, **kwargs: object) -> Mapping[str, object]:
        cls.seen.append(kwargs)
        return cls.outputs(out="ok")


class MatchedSlotPass(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.matched-slot-pass",
            slots=(MATCHED_SLOT,),
            outputs=(OutputSpec("out", MATCHED),),
        )

    @classmethod
    def execute(cls, *, source: SlotValue) -> Mapping[str, object]:
        return cls.outputs(out=source.value)


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    nodes: list[type[Node]] = list(SCAFFOLD_NODES) + [Stylize]
    return Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )


def stylize_graph(variant: str, inputs: dict[str, object]) -> Graph:
    return Graph(nodes={"n": GraphNode("test.stylize", inputs, slot_variants={"source": variant})})


def test_slot_executes_with_grouped_slot_value() -> None:
    async def scenario() -> None:
        engine = make_engine()

        text = await engine.run(
            stylize_graph("text", {"source": "hello", "source.prefix": ">> "}), ["n"]
        )
        assert text.outputs["n"]["text"].resolve() == ">> hello"

        number = await engine.run(
            stylize_graph("number", {"source": 2.0, "source.scale": 3.0}), ["n"]
        )
        assert number.outputs["n"]["text"].resolve() == "6.0"

    run_scenario(scenario())


def test_worker_wraps_matched_slot_output_as_the_selected_type() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        schema = MatchedSlotPass.define_schema()
        worker = InProcessWorker({schema.node_type: MatchedSlotPass}, registry)
        for variant, type_id, value in (
            ("text", "STRING", "hello"),
            ("number", "FLOAT", 2.5),
        ):
            effective = elaborate(
                schema,
                ["source", "source.copy"],
                slot_variants={"source": variant},
            )
            result = await worker.invoke(
                Invocation(
                    invocation_id=f"matched-{variant}",
                    node_id="n",
                    node_type=schema.node_type,
                    inputs={
                        "source": registry.wrap(type_id, value),
                        "source.copy": registry.wrap(type_id, value),
                    },
                    effective_schema=effective,
                )
            )
            assert result.error is None
            assert result.outputs is not None
            assert result.outputs["out"].type_id == type_id
            assert result.outputs["out"].resolve() == value

    run_scenario(scenario())


def test_slot_variant_switch_is_a_cache_miss_even_with_same_shape() -> None:
    class Pick(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.pick",
                slots=(
                    DynamicSlotSpec(
                        id="s",
                        variants=(SlotVariant("a", STRING), SlotVariant("b", STRING)),
                    ),
                ),
                outputs=(OutputSpec("out", STRING),),
            )

        @classmethod
        def execute(cls, *, s: SlotValue) -> Mapping[str, object]:
            return cls.outputs(out=f"{s.variant}:{s.value}")

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas([Pick]),
            registry=registry,
            worker=InProcessWorker(build_node_types([Pick]), registry),
            cache=MemoryLRUCache(),
        )

        def graph(variant: str) -> Graph:
            return Graph(
                nodes={"n": GraphNode("test.pick", {"s": "x"}, slot_variants={"s": variant})}
            )

        first = await engine.run(graph("a"), ["n"])
        assert first.executed == ("n",)
        assert first.outputs["n"]["out"].resolve() == "a:x"

        # Same values, same interface SHAPE, different variant: must re-run.
        switched = await engine.run(graph("b"), ["n"])
        assert switched.executed == ("n",)
        assert switched.outputs["n"]["out"].resolve() == "b:x"

        # And back: the original entry is still valid and hits.
        back = await engine.run(graph("a"), ["n"])
        assert back.cached == ("n",)

    run_scenario(scenario())


def test_connected_type_mismatch_is_diagnosed_never_redispatched() -> None:
    # The stored choice is authoritative for the interface: a producer whose
    # type disagrees gets the SAME advisory type-mismatch diagnostic as any
    # concrete input (the backend never silently re-dispatches to another
    # variant), because after elaboration the slot IS an ordinary input.
    schemas = build_schemas([*SCAFFOLD_NODES, StringConcatenate, Stylize])
    graph = Graph(
        nodes={
            "c": GraphNode(
                "dinkster.text.string_concatenate",
                {"string_a": "he", "string_b": "llo", "delimiter": ""},
            ),
            "n": GraphNode(
                "test.stylize",
                {"source": Link("c", "STRING"), "source.scale": 2.0},
                slot_variants={"source": "number"},
            ),
        }
    )
    diags = validate(graph, schemas, ["n"])
    mismatches = [d for d in diags if d.code == "type-mismatch"]
    assert mismatches and mismatches[0].input_id == "source"


def test_missing_choice_is_a_validation_error_at_run() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(nodes={"n": GraphNode("test.stylize", {"source": "hi"})})
        with pytest.raises(GraphValidationError):
            await engine.run(graph, ["n"])

    run_scenario(scenario())


def test_worker_delivers_open_slot_plain_and_omits_it_when_absent() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker({"test.open-slot": OpenSlotProbe}, registry)
        OpenSlotProbe.seen.clear()

        connected = await worker.invoke(
            Invocation(
                invocation_id="open-connected",
                node_id="n",
                node_type="test.open-slot",
                inputs={
                    "source": registry.wrap("STRING", "value"),
                    "source.scale": registry.wrap("FLOAT", 2.0),
                },
                effective_schema=elaborate(OPEN_SCHEMA, ["source", "source.scale"]),
            )
        )
        assert connected.error is None
        assert OpenSlotProbe.seen[-1] == {
            "source": "value",
            "source.scale": 2.0,
        }

        absent = await worker.invoke(
            Invocation(
                invocation_id="open-absent",
                node_id="n",
                node_type="test.open-slot",
                inputs={},
                effective_schema=elaborate(OPEN_SCHEMA, []),
            )
        )
        assert absent.error is None
        assert OpenSlotProbe.seen[-1] == {}

    run_scenario(scenario())
