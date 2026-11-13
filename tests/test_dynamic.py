"""Dynamic interfaces (hazard H10): elaboration, cache determinism, worker
validation against the effective schema, and output-family representability."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError, GraphValidationError, Invocation
from dinkster_graph import Graph, GraphNode, Link, validate
from dinkster_schema import (
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputCountSpec,
    OutputFamilySpec,
    OutputInterface,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    elaborate,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from scaffold_nodes import SCAFFOLD_NODES as BASE_NODES
from scaffold_nodes import register_scaffold_types

from tests.async_scenarios import run_scenario

STRING = TypeExpr.concrete("STRING")
FLOAT = TypeExpr.concrete("FLOAT")

JOIN_SCHEMA = NodeSchema(
    node_type="test.join",
    inputs=(InputSpec("separator", STRING, default=", "),),
    input_families=(InputFamilySpec("pieces", STRING, min_members=1),),
    outputs=(OutputSpec("text", STRING),),
)


class JoinStrings(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "std.string.join",
            inputs=JOIN_SCHEMA.inputs,
            input_families=JOIN_SCHEMA.input_families,
            outputs=JOIN_SCHEMA.outputs,
        )

    @classmethod
    def execute(cls, *, separator: str, pieces: Mapping[str, str]) -> Mapping[str, object]:
        return cls.outputs(text=separator.join(pieces.values()))


class ConcatStrings(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "std.string.concat",
            inputs=(
                InputSpec("a", STRING),
                InputSpec("b", STRING),
                InputSpec("separator", STRING, default=" "),
            ),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(cls, *, a: str, b: str, separator: str) -> Mapping[str, object]:
        return cls.outputs(text=a + separator + b)


class SplitString(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "std.string.split",
            inputs=(InputSpec("text", STRING), InputSpec("separator", STRING, default=" ")),
            output_families=(OutputFamilySpec("parts", STRING, min_members=1),),
        )

    @classmethod
    def execute(
        cls, *, text: str, separator: str, output_spec: OutputInterface
    ) -> Mapping[str, object]:
        members = output_spec.family("parts")
        pieces = text.split(separator, len(members) - 1)
        pieces += [""] * (len(members) - len(pieces))
        return cls.outputs(parts=dict(zip(members, pieces, strict=True)))


SCAFFOLD_NODES = (*BASE_NODES, JoinStrings, ConcatStrings, SplitString)


# --- elaboration is pure, stable, and order-preserving ---


def test_positional_family_validation_requires_first_slots_not_tail() -> None:
    schema = NodeSchema(
        "test.positional",
        input_families=(
            InputFamilySpec(
                "pieces",
                STRING,
                min_members=2,
                max_members=4,
                member_prefix="piece",
                optional_after_minimum=True,
            ),
        ),
    )

    def diagnostics(inputs: dict[str, object]) -> list[str]:
        graph = Graph(nodes={"n": GraphNode(schema.node_type, inputs)})
        return [
            d.message
            for d in validate(graph, {schema.node_type: schema}, ["n"])
            if d.severity == "error"
        ]

    assert (
        diagnostics({"pieces.piece0": "first", "pieces.piece1": "second", "pieces.piece2": "third"})
        == []
    )
    assert diagnostics(
        {"pieces.piece0": "first", "pieces.piece1": "second", "pieces.piece2": None}
    ) == ["input 'pieces.piece2' literal does not match STRING"]
    missing = diagnostics({"pieces.piece2": "third", "pieces.piece3": "fourth"})
    assert len(missing) == 2
    assert any("pieces.piece0" in message for message in missing)
    assert any("pieces.piece1" in message for message in missing)


# --- schema-level guardrails ---


# --- wire format: input and output families round-trip ---


# --- graph validation on the effective interface ---


def _schemas() -> dict[str, NodeSchema]:
    schemas = build_schemas(SCAFFOLD_NODES)
    schemas[JOIN_SCHEMA.node_type] = JOIN_SCHEMA
    return schemas


FANOUT_SCHEMA = NodeSchema(
    node_type="test.fanout",
    inputs=(InputSpec("seed", STRING),),
    output_families=(OutputFamilySpec("results", STRING),),
)


def test_dangling_dynamic_member_is_a_document_time_error() -> None:
    schemas = _schemas()
    schemas["test.fanout"] = FANOUT_SCHEMA
    graph = Graph(
        nodes={
            # Document stores only member "0", but the consumer links to "2".
            "p": GraphNode("test.fanout", {"seed": "s"}, output_members={"results": ("0",)}),
            "j": GraphNode("test.join", {"pieces.a": Link("p", "results.2"), "separator": "-"}),
        }
    )
    diags = validate(graph, schemas, ["j"])
    dangling = [d for d in diags if d.code == "dangling-output"]
    assert dangling and "currently has members" in dangling[0].message


# --- engine + worker end-to-end ---


def make_engine(extra_nodes: list[type[Node]] | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    nodes = list(SCAFFOLD_NODES) + (extra_nodes or [])
    return Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )


def join_graph(*pieces: tuple[str, str], separator: str = " ") -> Graph:
    inputs: dict[str, object] = {f"pieces.{k}": v for k, v in pieces}
    inputs["separator"] = separator
    return Graph(nodes={"j": GraphNode("std.string.join", inputs)})


def test_dynamic_inputs_execute_and_cache_deterministically() -> None:
    async def scenario() -> None:
        engine = make_engine()

        first = await engine.run(join_graph(("a", "x"), ("b", "y")), ["j"])
        assert first.outputs["j"]["text"].resolve() == "x y"
        assert first.executed == ("j",)

        # Same member set and values -> cache hit.
        rerun = await engine.run(join_graph(("a", "x"), ("b", "y")), ["j"])
        assert rerun.cached == ("j",)

        # Growing the family -> different effective interface -> re-executes.
        grown = await engine.run(join_graph(("a", "x"), ("b", "y"), ("c", "z")), ["j"])
        assert grown.executed == ("j",)
        assert grown.outputs["j"]["text"].resolve() == "x y z"

        # Shrinking back -> the original entry is still valid and hits.
        back = await engine.run(join_graph(("a", "x"), ("b", "y")), ["j"])
        assert back.cached == ("j",)

    run_scenario(scenario())


def test_min_members_enforced_at_run() -> None:
    async def scenario() -> None:
        engine = make_engine()
        with pytest.raises(GraphValidationError):
            await engine.run(join_graph(), ["j"])

    run_scenario(scenario())


class WrongOutputs(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.wrong_outputs",
            inputs=(InputSpec("a", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    def execute(cls, *, a: str) -> Mapping[str, object]:
        return {"not_out": a}  # bypasses cls.outputs() on purpose


def test_worker_validates_against_the_effective_schema() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker({"test.wrong_outputs": WrongOutputs}, registry)
        wrapped = registry.wrap("STRING", "v")
        effective = elaborate(WrongOutputs.define_schema(), ["a"])
        result = await worker.invoke(
            Invocation(
                invocation_id="i1",
                node_id="n",
                node_type="test.wrong_outputs",
                inputs={"a": wrapped},
                effective_schema=effective,
            )
        )
        assert result.error is not None
        assert "not_out" in result.error.message

        # And through the engine: the same violation surfaces as ExecutionError.
        engine = make_engine([WrongOutputs])
        graph = Graph(nodes={"n": GraphNode("test.wrong_outputs", {"a": "v"})})
        with pytest.raises(ExecutionError):
            await engine.run(graph, ["n"])

    run_scenario(scenario())


# --- output families: elaboration is document-determined ---


# --- output families: end to end through engine and worker ---


def split_graph(members: tuple[str, ...], text: str = "a b c") -> Graph:
    return Graph(
        nodes={
            "s": GraphNode(
                "std.string.split",
                {"text": text, "separator": " "},
                output_members={"parts": members},
            )
        }
    )


def test_dynamic_outputs_execute_and_cache_deterministically() -> None:
    async def scenario() -> None:
        engine = make_engine()

        first = await engine.run(split_graph(("a", "b")), ["s"])
        assert first.executed == ("s",)
        assert first.outputs["s"]["parts.a"].resolve() == "a"
        assert first.outputs["s"]["parts.b"].resolve() == "b c"

        # Same membership and inputs -> cache hit.
        rerun = await engine.run(split_graph(("a", "b")), ["s"])
        assert rerun.cached == ("s",)

        # Different membership -> different effective interface -> re-executes.
        grown = await engine.run(split_graph(("a", "b", "c")), ["s"])
        assert grown.executed == ("s",)
        assert grown.outputs["s"]["parts.c"].resolve() == "c"

    run_scenario(scenario())


def test_dynamic_outputs_link_into_downstream_nodes() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "s": GraphNode(
                    "std.string.split",
                    {"text": "x y z", "separator": " "},
                    output_members={"parts": ("p", "q", "r")},
                ),
                "j": GraphNode(
                    "std.string.join",
                    {
                        "pieces.a": Link("s", "parts.r"),
                        "pieces.b": Link("s", "parts.p"),
                        "separator": "-",
                    },
                ),
            }
        )
        result = await engine.run(graph, ["j"])
        assert result.outputs["j"]["text"].resolve() == "z-x"

    run_scenario(scenario())


class SpecProbe(Node):
    """Records the OutputInterface it receives, for boundary assertions."""

    seen: OutputInterface | None = None

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.spec_probe",
            inputs=(InputSpec("seed", STRING),),
            output_families=(OutputFamilySpec("results", STRING),),
        )

    @classmethod
    def execute(cls, *, seed: str, output_spec: OutputInterface) -> Mapping[str, object]:
        SpecProbe.seen = output_spec
        return cls.outputs(results={s: seed for s in output_spec.family("results")})


class CountSpecProbe(Node):
    seen: OutputInterface | None = None

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.count_spec_probe",
            inputs=(
                InputSpec("count", TypeExpr.concrete("INT")),
                InputSpec("seed", STRING),
            ),
            output_families=(OutputFamilySpec("results", STRING, count=OutputCountSpec("count")),),
        )

    @classmethod
    def execute(
        cls, *, count: int, seed: str, output_spec: OutputInterface
    ) -> Mapping[str, object]:
        CountSpecProbe.seen = output_spec
        assert len(output_spec.family("results")) == count
        return cls.outputs(results={suffix: seed for suffix in output_spec.family("results")})


def test_counted_output_members_reach_worker_output_spec() -> None:
    async def scenario() -> None:
        CountSpecProbe.seen = None
        engine = make_engine([CountSpecProbe])
        graph = Graph(
            nodes={
                "n": GraphNode(
                    "test.count_spec_probe",
                    {"count": 3, "seed": "v"},
                    output_members={"results": ("0", "1", "2")},
                )
            }
        )
        result = await engine.run(graph, ["n"])
        assert CountSpecProbe.seen is not None
        assert CountSpecProbe.seen.family("results") == ("0", "1", "2")
        assert set(result.outputs["n"]) == {"results.0", "results.1", "results.2"}

    run_scenario(scenario())


class BadFanout(Node):
    """Returns members that do not match the document's membership."""

    return_suffixes: tuple[str, ...] = ("wrong",)

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.bad_fanout",
            inputs=(InputSpec("seed", STRING),),
            output_families=(OutputFamilySpec("results", STRING),),
        )

    @classmethod
    def execute(cls, *, seed: str, output_spec: OutputInterface) -> Mapping[str, object]:
        return {"results": {s: seed for s in cls.return_suffixes}}


def test_wrong_grouped_return_members_fail_at_the_worker_boundary() -> None:
    async def scenario() -> None:
        engine = make_engine([BadFanout])
        graph = Graph(
            nodes={
                "n": GraphNode("test.bad_fanout", {"seed": "v"}, output_members={"results": ("a",)})
            }
        )
        # Extra member "wrong", missing member "a".
        BadFanout.return_suffixes = ("wrong",)
        with pytest.raises(ExecutionError, match="results.a"):
            await engine.run(graph, ["n"])
        # Missing member only.
        BadFanout.return_suffixes = ()
        with pytest.raises(ExecutionError, match="missing outputs: results.a"):
            await engine.run(graph, ["n"])

    run_scenario(scenario())


# --- the run-scoped snapshot: topology is fixed at run() entry ---
