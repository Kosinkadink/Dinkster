"""First-class absence (DESIGN 3.15): typed no-value with provenance.

The anti-ExecutionBlocker slice: deliberate absence from optional outputs,
per-input policies (skip/omit/accept/fail), engine-level skip cascades with
root-cause provenance, cache replay, boundary crossing, and the maybe-absent
document diagnostic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import Graph, GraphNode, Link, validate
from dinkster_schema import (
    ABSENT,
    AbsentOutput,
    AbsentPolicy,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_values import (
    ABSENT_ORIGIN_META_KEY,
    ABSENT_REASON_META_KEY,
    ABSENT_STANDS_FOR_META_KEY,
    CORE_ABSENT,
    TypeRegistry,
    is_absent,
    make_absent_value,
    register_core_types,
)
from dinkster_workers import InProcessWorker
from dinkster_workers.boundary import ValueCodec

# -- schema declarations ----------------------------------------------------


def test_omit_policy_requires_optional_input() -> None:
    with pytest.raises(ValueError, match="omit"):
        InputSpec("x", TypeExpr.concrete("core.int"), required=True, on_absent="omit")


def test_absent_policy_defaults() -> None:
    required = InputSpec("a", TypeExpr.concrete("core.int"))
    optional = InputSpec("b", TypeExpr.concrete("core.int"), required=False)
    assert required.absent_policy() == "skip"
    assert optional.absent_policy() == "omit"
    declared = InputSpec("c", TypeExpr.concrete("core.int"), on_absent="accept")
    assert declared.absent_policy() == "accept"


def test_schema_wire_roundtrip_with_absence_fields() -> None:
    schema = NodeSchema(
        node_type="test.maybe",
        inputs=(
            InputSpec("strict", TypeExpr.concrete("core.int"), on_absent="fail"),
            InputSpec("loose", TypeExpr.concrete("core.int"), on_absent="accept"),
        ),
        outputs=(
            OutputSpec("value", TypeExpr.concrete("core.int"), optional=True),
            OutputSpec("always", TypeExpr.concrete("core.int")),
        ),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema
    wire = schema_to_wire(schema)
    interface = wire["interface"]
    assert isinstance(interface, list)
    strict = next(e for e in interface if e["id"] == "strict")
    assert strict["onAbsent"] == "fail"
    value_out = next(e for e in interface if e["id"] == "value")
    assert value_out["optional"] is True
    always_out = next(e for e in interface if e["id"] == "always")
    assert "optional" not in always_out


# -- absent envelopes -------------------------------------------------------


def test_absent_value_is_interrogable_and_deterministic() -> None:
    value = make_absent_value(origin="n1/vae", reason="no VAE", stands_for="core.int")
    assert is_absent(value)
    assert value.type_id == CORE_ABSENT
    assert value.meta.get(ABSENT_ORIGIN_META_KEY) == "n1/vae"
    assert value.meta.get(ABSENT_REASON_META_KEY) == "no VAE"
    assert value.meta.get(ABSENT_STANDS_FOR_META_KEY) == "core.int"
    assert value.resolve() is None
    same = make_absent_value(origin="n1/vae", reason="no VAE", stands_for="core.int")
    assert same.fingerprint == value.fingerprint
    other = make_absent_value(origin="n2/vae", reason="no VAE", stands_for="core.int")
    assert other.fingerprint != value.fingerprint


def test_absent_crosses_the_boundary() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    sender = ValueCodec(registry, use_shm=False, accept_shm=False)
    receiver = ValueCodec(registry, use_shm=False, accept_shm=False)
    value = make_absent_value(origin="n1/out", reason="nothing", stands_for="core.int")
    blobs: list[bytes] = []
    wire, _ = sender.encode(value, blobs, [])
    decoded, _ = receiver.decode(wire, blobs, [])
    assert is_absent(decoded)
    assert decoded.fingerprint == value.fingerprint
    assert decoded.meta.get(ABSENT_ORIGIN_META_KEY) == "n1/out"
    assert decoded.resolve() is None


# -- nodes ------------------------------------------------------------------


class MaybeProduce(Node):
    """Optional output: ABSENT when produce is false."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.maybe_produce",
            inputs=(InputSpec("produce", TypeExpr.concrete("core.boolean")),),
            outputs=(OutputSpec("value", TypeExpr.concrete("core.int"), optional=True),),
        )

    @classmethod
    def execute(cls, produce: bool) -> Mapping[str, object]:
        if produce:
            return cls.outputs(value=42)
        return cls.outputs(value=AbsentOutput("nothing to produce"))


class IllegalAbsent(Node):
    """Returns ABSENT for a non-optional output: contract error."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.illegal_absent",
            inputs=(),
            outputs=(OutputSpec("value", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(value=ABSENT)


class AddOne(Node):
    """Default policy (skip) on a required input."""

    ran: list[int] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.add_one",
            inputs=(InputSpec("value", TypeExpr.concrete("core.int")),),
            outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls, value: int) -> Mapping[str, object]:
        cls.ran.append(value)
        return cls.outputs(out=value + 1)


class Coalesce(Node):
    """accept policy: receives None and substitutes a fallback."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.coalesce",
            inputs=(
                InputSpec("value", TypeExpr.concrete("core.int"), on_absent="accept"),
                InputSpec("fallback", TypeExpr.concrete("core.int")),
            ),
            outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls, value: int | None, fallback: int) -> Mapping[str, object]:
        return cls.outputs(out=fallback if value is None else value)


class WithOptional(Node):
    """omit policy (the optional-input default): runs without the input."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.with_optional",
            inputs=(
                InputSpec("base", TypeExpr.concrete("core.int")),
                InputSpec("extra", TypeExpr.concrete("core.int"), required=False),
            ),
            outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls, base: int, extra: int | None = None) -> Mapping[str, object]:
        return cls.outputs(out=base + (extra if extra is not None else 0))


class WithDefault(Node):
    """omit policy where the optional input carries a schema default."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.with_default",
            inputs=(
                InputSpec("base", TypeExpr.concrete("core.int")),
                InputSpec("extra", TypeExpr.concrete("core.int"), required=False, default=100),
            ),
            outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls, base: int, extra: int = -1) -> Mapping[str, object]:
        return cls.outputs(out=base + extra)


class Strict(Node):
    """fail policy: absence is a loud run failure."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.strict",
            inputs=(InputSpec("value", TypeExpr.concrete("core.int"), on_absent="fail"),),
            outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls, value: int) -> Mapping[str, object]:
        return cls.outputs(out=value)


NODES: tuple[type[Node], ...] = (
    MaybeProduce,
    IllegalAbsent,
    AddOne,
    Coalesce,
    WithOptional,
    WithDefault,
    Strict,
)


def make_engine(events: list | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=build_schemas(NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
        on_event=None if events is None else events.append,
    )


# -- deliberate absence -----------------------------------------------------


def test_optional_output_carries_provenance_and_caches() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(nodes={"m": GraphNode("test.maybe_produce", {"produce": False})})
        result = await engine.run(graph, ["m"])
        value = result.outputs["m"]["value"]
        assert is_absent(value)
        assert value.meta.get(ABSENT_ORIGIN_META_KEY) == "m/value"
        assert value.meta.get(ABSENT_REASON_META_KEY) == "nothing to produce"
        assert value.meta.get(ABSENT_STANDS_FOR_META_KEY) == "core.int"

        # Deliberate absence is an ordinary cached value.
        again = await engine.run(graph, ["m"])
        assert again.executed == ()
        assert again.cached == ("m",)
        assert is_absent(again.outputs["m"]["value"])

    asyncio.run(scenario())


def test_absent_on_non_optional_output_is_contract_error() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(nodes={"i": GraphNode("test.illegal_absent", {})})
        with pytest.raises(ExecutionError, match="not.*declared optional"):
            await engine.run(graph, ["i"])

    asyncio.run(scenario())


# -- skip cascade -----------------------------------------------------------


def test_skip_cascades_with_root_provenance() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        events: list = []
        engine = make_engine(events)
        graph = Graph(
            nodes={
                "m": GraphNode("test.maybe_produce", {"produce": False}),
                "a": GraphNode("test.add_one", {"value": Link("m", "value")}),
                "b": GraphNode("test.add_one", {"value": Link("a", "out")}),
            }
        )
        result = await engine.run(graph, ["b"])
        # Neither consumer executed; node code never saw an absent envelope.
        assert AddOne.ran == []
        assert set(result.skipped) == {"a", "b"}
        assert result.executed == ("m",)
        # The cascade preserves the ROOT origin, not the previous hop.
        final = result.outputs["b"]["out"]
        assert is_absent(final)
        assert final.meta.get(ABSENT_ORIGIN_META_KEY) == "m/value"
        assert final.meta.get(ABSENT_REASON_META_KEY) == "nothing to produce"
        # Skips surface as events with the blocking input named.
        skip_events = [e for e in events if e.kind == "node_skipped"]
        assert {e.node_id for e in skip_events} == {"a", "b"}
        assert all(e.detail["origin"] == "m/value" for e in skip_events)

        # Skips are recomputed, never cached: only m replays from cache.
        again = await engine.run(graph, ["b"])
        assert again.cached == ("m",)
        assert set(again.skipped) == {"a", "b"}

    asyncio.run(scenario())


def test_skip_does_not_block_produced_branch() -> None:
    async def scenario() -> None:
        AddOne.ran.clear()
        engine = make_engine()
        graph = Graph(
            nodes={
                "m": GraphNode("test.maybe_produce", {"produce": True}),
                "a": GraphNode("test.add_one", {"value": Link("m", "value")}),
            }
        )
        result = await engine.run(graph, ["a"])
        assert result.skipped == ()
        assert result.outputs["a"]["out"].resolve() == 43
        assert AddOne.ran == [42]

    asyncio.run(scenario())


# -- accept, omit, fail -----------------------------------------------------


def test_accept_policy_receives_plain_none() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "m": GraphNode("test.maybe_produce", {"produce": False}),
                "c": GraphNode("test.coalesce", {"value": Link("m", "value"), "fallback": 7}),
            }
        )
        result = await engine.run(graph, ["c"])
        assert result.skipped == ()
        assert result.outputs["c"]["out"].resolve() == 7

    asyncio.run(scenario())


def test_omit_policy_matches_unconnected_cache_shape() -> None:
    async def scenario() -> None:
        engine = make_engine()
        linked = Graph(
            nodes={
                "m": GraphNode("test.maybe_produce", {"produce": False}),
                "w": GraphNode("test.with_optional", {"base": 5, "extra": Link("m", "value")}),
            }
        )
        first = await engine.run(linked, ["w"])
        assert first.skipped == ()
        assert first.outputs["w"]["out"].resolve() == 5

        # The same node with the input truly unconnected hits the same entry:
        # omission produced an identical cache key.
        unconnected = Graph(nodes={"w": GraphNode("test.with_optional", {"base": 5})})
        second = await engine.run(unconnected, ["w"])
        assert second.cached == ("w",)

    asyncio.run(scenario())


def test_demanded_lazy_absence_uses_existing_policies_and_producer_cache() -> None:
    def lazy_consumer(policy: AbsentPolicy) -> type[Node]:
        class LazyConsumer(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    node_type=f"test.lazy-absent-{policy}",
                    inputs=(
                        InputSpec(
                            "value",
                            TypeExpr.concrete("core.int"),
                            required=policy != "omit",
                            on_absent=policy,
                            lazy=True,
                        ),
                    ),
                    outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
                )

            @classmethod
            def check_lazy_status(cls, **_inputs: object) -> tuple[str, ...]:
                return ("value",)

            @classmethod
            def execute(cls, **inputs: object) -> Mapping[str, object]:
                value = inputs.get("value", -1)
                return cls.outputs(out=-2 if value is None else value)

        return LazyConsumer

    async def scenario() -> None:
        consumers = tuple(lazy_consumer(policy) for policy in ("skip", "omit", "accept", "fail"))
        registry = TypeRegistry()
        register_core_types(registry)
        cache = MemoryLRUCache()
        nodes = (MaybeProduce, *consumers)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=cache,
        )

        def graph(policy: str) -> Graph:
            return Graph(
                {
                    "producer": GraphNode("test.maybe_produce", {"produce": False}),
                    "consumer": GraphNode(
                        f"test.lazy-absent-{policy}",
                        {"value": Link("producer", "value")},
                    ),
                }
            )

        skipped = await engine.run(graph("skip"), ["consumer"])
        assert skipped.skipped == ("consumer",)
        assert is_absent(skipped.outputs["consumer"]["out"])

        omitted = await engine.run(graph("omit"), ["consumer"])
        assert omitted.outputs["consumer"]["out"].resolve() == -1

        accepted = await engine.run(graph("accept"), ["consumer"])
        assert accepted.outputs["consumer"]["out"].resolve() == -2
        accepted_again = await engine.run(graph("accept"), ["consumer"])
        assert accepted_again.executed == ()
        assert accepted_again.cached == ("producer", "consumer")

        with pytest.raises(ExecutionError) as failed:
            await engine.run(graph("fail"), ["consumer"])
        assert failed.value.error.node_id == "consumer"
        assert "on_absent='fail'" in failed.value.error.message

        # The demanded producer and idempotent lazy consumers cache only
        # after the hook fixes the final demand/visibility identity. Skips and
        # failures still create no consumer entry.
        assert len(cache) == 3

    asyncio.run(scenario())


def test_omit_policy_substitutes_schema_default() -> None:
    """An absent arriving at an optional input with a schema default behaves
    exactly like the unconnected shape: the schema default (not execute()'s
    Python default) is used, and the cache key matches unconnected."""

    async def scenario() -> None:
        engine = make_engine()
        linked = Graph(
            nodes={
                "m": GraphNode("test.maybe_produce", {"produce": False}),
                "w": GraphNode("test.with_default", {"base": 5, "extra": Link("m", "value")}),
            }
        )
        first = await engine.run(linked, ["w"])
        assert first.skipped == ()
        assert first.outputs["w"]["out"].resolve() == 105  # schema default, not -1

        unconnected = Graph(nodes={"w": GraphNode("test.with_default", {"base": 5})})
        second = await engine.run(unconnected, ["w"])
        assert second.cached == ("w",)
        assert second.outputs["w"]["out"].resolve() == 105

    asyncio.run(scenario())


def test_fail_policy_names_the_origin() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "m": GraphNode("test.maybe_produce", {"produce": False}),
                "s": GraphNode("test.strict", {"value": Link("m", "value")}),
            }
        )
        with pytest.raises(ExecutionError) as excinfo:
            await engine.run(graph, ["s"])
        message = str(excinfo.value)
        assert "m/value" in message  # the producer, not a downstream victim
        assert "nothing to produce" in message

    asyncio.run(scenario())


# -- document-time diagnostics ----------------------------------------------


def test_maybe_absent_warning_on_optional_into_fail() -> None:
    graph = Graph(
        nodes={
            "m": GraphNode("test.maybe_produce", {"produce": True}),
            "s": GraphNode("test.strict", {"value": Link("m", "value")}),
        }
    )
    diags = validate(graph, build_schemas(NODES), ["s"])
    codes = {d.code for d in diags}
    assert "maybe-absent" in codes
    warning = next(d for d in diags if d.code == "maybe-absent")
    assert warning.severity == "warning"
    assert "test.maybe_produce" in warning.message


def test_no_warning_for_default_skip_consumer() -> None:
    graph = Graph(
        nodes={
            "m": GraphNode("test.maybe_produce", {"produce": True}),
            "a": GraphNode("test.add_one", {"value": Link("m", "value")}),
        }
    )
    diags = validate(graph, build_schemas(NODES), ["a"])
    assert "maybe-absent" not in {d.code for d in diags}
